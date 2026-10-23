#!/usr/bin/env python
"""
Choose the substations for the substation / feeder forecasting test (whole of 2025) and freeze them in a spec.

Reads every district of the enriched dataset v1.4 once (bounded reads, one district per worker process; only
substation / feeder aggregates leave a worker, never a customer key), describes every substation, applies the
eligibility rules, and draws a two-stage stratified sample with a fixed seed:

  stage 1  test districts: one per region (14) that has an eligible district, the rest of --test-districts allocated to
           regions by their share of eligible households; inside a region a district is drawn with probability
           proportional to its eligible households. The guide's 5 sample districts (used for all development so far)
           are never test districts.
  stage 2  --per-district substations in each test district, one per size stratum (eligible substations of the
           district split at their size quantiles: one large, one mid-sized for 2).
  dev      --dev substations for model choice and tuning, from districts that hold no test substation: first the sample
           districts (up to --per-district each), then one per district from the regions least represented so far.

Eligibility (all must hold; the reason is written for every substation that fails):
  households      >= --min-households distinct customers under it with a row in 2025
  mapping         >= --min-path-share of them on a full, non-placeholder path (transformer, feeder >= 0, transformer
                  not a > 1,000-customer placeholder)
  stability       <= --max-moved of them seen under another substation between --start and --end
  tree            <= --max-foreign of its rows on a transformer whose usual substation is another one
  coverage 2025   >= --min-coverage of its in-span customer-days in 2025 observed or estimated (the target's days)
  history         >= --min-history of its in-span customer-days observed or estimated in 2024-08 .. 2024-12 (the new
                  regime before the first refit)
  feeders         >= 1 feeder id under it
Reported, not filtered: silent transformer-days (a transformer with >= 10 households in span and < 30 % of them labelled)
in 2025, the apartment share and the no-gas share ('neither' heating class: mostly electric heating), coverage of 2026-H1
(the reserve), CAS node type and coordinates.

Outputs (<out-dir>):
  substations_all.csv     every substation: sizes, quality, coverage, segments, region, coordinates, eligible + reason
  selection.csv           the chosen substations: role (test / dev), stratum, the same columns
  feeders_selected.csv    every feeder under a chosen substation: households, coverage, size band
  selection_report.txt    eligibility funnel, allocation by region, sample vs eligible population
  figures/                map.png (if coordinates), distributions.png, regions.png
and the frozen spec --spec-out (default configs/test_set_substations_2025.yaml): version, data, target, splits, refits,
horizons, rules, the seed and the substation lists with their feeders; the scorer will check its hash.

  python scripts/test_set/select_substations.py --out-dir ~/work/substation_selection --workers 12
"""
import argparse
import concurrent.futures as cf
import contextlib
import hashlib
import importlib.util
import io
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CORE = os.path.join(REPO, "core")
sys.path.insert(0, CORE)
MISSING = -(2 ** 62)
SAMPLE = ["03203", "10240", "22202", "26266", "33220"]          # configs/test_set_2025.yaml sample_districts
REF_ROOT = "s3://phase1/electricity/analytics/reference"
LABELLED = ("complete_observed", "complete_with_estimated")
DEFAULTS = dict(min_households=300, min_path_share=0.85, max_moved=0.10, max_foreign=0.10, min_coverage=0.70,
                min_history=0.50)


# ---------------------------------------------------------------------------------------------------- district worker
class _Idx:
    """Growing id -> row index map with a (rows, T) int / float matrix per measure."""

    def __init__(self, T, measures):
        self.T, self.of, self.ids = T, {}, []
        self.m = {k: np.zeros((0, T), dt) for k, dt in measures.items()}

    def index(self, ids):
        u, inv = np.unique(ids, return_inverse=True)
        li = np.empty(len(u), np.int64)
        for j, v in enumerate(u.tolist()):
            x = self.of.get(v)
            if x is None:
                x = self.of[v] = len(self.ids)
                self.ids.append(v)
            li[j] = x
        n = len(self.ids)
        for k, a in self.m.items():
            if a.shape[0] < n:
                self.m[k] = np.vstack([a, np.zeros((max(n, 2 * a.shape[0], 8) - a.shape[0], self.T), a.dtype)])
        return li[inv]

    def add(self, k, rows, day, w=None):
        a = self.m[k]                                                # bincount: far faster than np.add.at
        a += np.bincount(rows * self.T + day, weights=w, minlength=a.size).reshape(a.shape).astype(a.dtype)

    def get(self, k):
        return self.m[k][:len(self.ids)]


def _codes(col, table):
    """Arrow string column -> int codes through a growing dict (one per district: no key leaves the worker)."""
    import pyarrow.compute as pc
    de = pc.dictionary_encode(col)
    uq = de.dictionary.to_pylist()
    m = np.empty(len(uq), np.int64)
    for j, s in enumerate(uq):
        v = table.get(s)
        if v is None:
            v = table[s] = len(table)
        m[j] = v
    idx = de.indices
    return m[pc.fill_null(idx, 0).to_numpy(zero_copy_only=False)] if len(uq) else np.zeros(len(col), np.int64)


def district_worker(code, cfg):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as pads
    sys.path.insert(0, cfg["core"])
    import enriched_io as eio
    t0 = time.time()
    fs, root = eio.open_root(cfg["root"])
    d0, d1 = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    dates = pd.date_range(d0, d1)
    T = len(dates)
    e0 = int((d0 - pd.Timestamp("1970-01-01")).days)
    dset = eio.district_daily(fs, root, code)
    want = ["customer_key", "date", "transformer_id", "feeder_id", "substation_id", "placeholder_transformer",
            "label_state", "kwh", "region_code", "region_name", "admin_location_name", "operational_segment"]
    cols = [c for c in want if c in dset.schema.names]
    flt = (pads.field("date") >= d0.date()) & (pads.field("date") <= d1.date())
    lab = pa.array(list(LABELLED), type=pa.string())
    S = _Idx(T, dict(span=np.int32, lab=np.int32, obs=np.int32, kwh=np.float64, full=np.int32))
    F = _Idx(T, dict(span=np.int32, lab=np.int32, kwh=np.float64))
    TR = _Idx(T, dict(span=np.int32, lab=np.int32))
    combos, tr_sub, fe_sub, names = [], [], [], {}
    key_of, seg_of = {}, {"": 0}
    i25 = ((pd.Timestamp("2025-01-01") - d0).days, (pd.Timestamp("2025-12-31") - d0).days)
    for b in dset.scanner(columns=cols, filter=flt, batch_size=262_144, use_threads=True).to_batches():
        n = b.num_rows
        if not n:
            continue

        def ints(col):
            if col not in cols:
                return np.full(n, MISSING, np.int64)
            return pc.fill_null(pc.cast(b.column(col), pa.int64()), MISSING).to_numpy(zero_copy_only=False)

        def strs(col):
            if col not in cols:
                return np.full(n, "", dtype=object)
            return np.asarray(pc.fill_null(pc.cast(b.column(col), pa.string()), "").to_pylist(), dtype=object)
        day = pc.cast(pc.cast(b.column("date"), pa.date32()), pa.int32()).to_numpy(zero_copy_only=False).astype(np.int64) - e0
        tr, fe, su = ints("transformer_id"), ints("feeder_id"), ints("substation_id")
        ph = (pc.fill_null(b.column("placeholder_transformer"), False).to_numpy(zero_copy_only=False)
              if "placeholder_transformer" in cols else np.zeros(n, bool))
        islab = pc.and_(pc.fill_null(pc.is_in(b.column("label_state"), value_set=lab), False),
                        pc.is_valid(b.column("kwh"))).to_numpy(zero_copy_only=False)
        isobs = pc.fill_null(pc.equal(b.column("label_state"), "complete_observed"), False).to_numpy(zero_copy_only=False)
        isobs &= islab
        kw = pc.fill_null(pc.cast(b.column("kwh"), pa.float64()), 0.0).to_numpy(zero_copy_only=False)
        full = (tr >= 0) & (fe >= 0) & (su >= 0) & ~ph
        for c in ("region_code", "region_name", "admin_location_name"):
            if c in cols and c not in names:
                v = [x for x in strs(c)[:1000] if x]
                if v:
                    names[c] = pd.Series(v).mode().iloc[0]
        ok = su >= 0
        if ok.any():
            r = S.index(su[ok])
            dd = day[ok]
            S.add("span", r, dd)
            S.add("lab", r[islab[ok]], dd[islab[ok]])
            S.add("obs", r[isobs[ok]], dd[isobs[ok]])
            S.add("kwh", r[islab[ok]], dd[islab[ok]], kw[ok][islab[ok]])
            S.add("full", r[full[ok]], dd[full[ok]])
        okf = (fe >= 0) & ok
        if okf.any():
            r = F.index(fe[okf])
            dd = day[okf]
            F.add("span", r, dd)
            F.add("lab", r[islab[okf]], dd[islab[okf]])
            F.add("kwh", r[islab[okf]], dd[islab[okf]], kw[okf][islab[okf]])
            fe_sub.append(pd.DataFrame({"f": fe[okf], "s": su[okf]}).value_counts().rename("n").reset_index())
        okt = (tr >= 0) & ~ph & ok
        if okt.any():
            r = TR.index(tr[okt])
            dd = day[okt]
            TR.add("span", r, dd)
            TR.add("lab", r[islab[okt]], dd[islab[okt]])
            tr_sub.append(pd.DataFrame({"t": tr[okt], "s": su[okt]}).value_counts().rename("n").reset_index())
        cust = _codes(b.column("customer_key"), key_of)
        segc = (_codes(pc.fill_null(pc.cast(b.column("operational_segment"), pa.string()), ""), seg_of)
                if "operational_segment" in cols else np.zeros(n, np.int64))
        y25 = (day >= i25[0]) & (day <= i25[1])
        combos.append(pd.DataFrame({"c": cust, "s": su, "full": full, "seg": segc, "y25": y25}).drop_duplicates())
        if len(combos) >= 40:                                             # keep the memory flat on big districts
            combos = [pd.concat(combos, ignore_index=True).drop_duplicates()]
    res = dict(code=str(code), seconds=time.time() - t0, **names)
    if not S.ids:
        return res
    year = dates.year.to_numpy()
    in25 = year == 2025
    hist = (dates >= pd.Timestamp("2024-08-01")) & (dates <= pd.Timestamp("2024-12-31"))
    res26 = (dates >= pd.Timestamp("2026-01-01")) & (dates <= pd.Timestamp("2026-06-30"))
    cmb = pd.concat(combos, ignore_index=True).drop_duplicates()
    del combos
    seg_name = np.empty(len(seg_of), dtype=object)
    for k, v in seg_of.items():
        seg_name[v] = k
    cmb["seg"] = seg_name[cmb["seg"].to_numpy()]
    # customers per substation (2025), path quality, moves between substations, segments
    c25 = cmb[cmb["y25"] & (cmb["s"] >= 0)]
    moved = cmb[cmb["s"] >= 0].groupby("c")["s"].nunique()
    seg = cmb[cmb["seg"] != ""].drop_duplicates("c", keep="last").set_index("c")["seg"]
    per = c25.groupby("s").agg(households=("c", "nunique")).reset_index()
    fullc = c25[c25["full"]].groupby("s")["c"].nunique().rename("households_full_path")
    per = per.merge(fullc, on="s", how="left").fillna({"households_full_path": 0})
    cs = c25.drop_duplicates(["c", "s"])[["c", "s"]]
    cs["moved"] = cs["c"].map(moved).fillna(1).to_numpy() > 1
    cs["seg"] = cs["c"].map(seg).fillna("")
    cs["apartment"] = cs["seg"].str.startswith("apartment")
    cs["no_gas"] = cs["seg"].str.endswith("__neither")
    cs["known_seg"] = cs["seg"] != ""
    g = cs.groupby("s").agg(moved=("moved", "mean"), apartment=("apartment", "sum"), no_gas=("no_gas", "sum"),
                            known_seg=("known_seg", "sum")).reset_index()
    per = per.merge(g, on="s", how="left")
    # tree: rows on a transformer whose usual substation is another one
    if tr_sub:
        ts_ = pd.concat(tr_sub).groupby(["t", "s"])["n"].sum().reset_index()
        mode = ts_.sort_values("n", ascending=False).drop_duplicates("t").set_index("t")["s"]
        ts_["foreign"] = ts_["t"].map(mode) != ts_["s"]
        fr = ts_.groupby("s").apply(lambda q: q.loc[q["foreign"], "n"].sum() / max(q["n"].sum(), 1),
                                    include_groups=False).rename("foreign_share")
        per = per.merge(fr, left_on="s", right_index=True, how="left")
        t_of_s = ts_[~ts_["foreign"]].groupby("s")["t"].nunique().rename("transformers")
        per = per.merge(t_of_s, left_on="s", right_index=True, how="left")
    fe_rows = []
    if fe_sub:
        fs_ = pd.concat(fe_sub).groupby(["f", "s"])["n"].sum().reset_index()
        fmode = fs_.sort_values("n", ascending=False).drop_duplicates("f").set_index("f")["s"]
        per = per.merge(fmode.reset_index().groupby("s")["f"].nunique().rename("feeders"), left_on="s",
                        right_index=True, how="left")
        fspan, flab, fkwh = F.get("span"), F.get("lab"), F.get("kwh")
        fid = np.asarray(F.ids)
        for i, f in enumerate(fid):
            sp = fspan[i, in25].sum()
            fe_rows.append(dict(coato_code=str(code), feeder_id=int(f), substation_id=int(fmode.get(f, MISSING)),
                                customer_days_2025=int(sp),
                                households_2025=float(fspan[i, in25].mean()),
                                coverage_2025=float(flab[i, in25].sum() / sp) if sp else np.nan,
                                kwh_per_day_2025=float(fkwh[i, in25].sum() / max(in25.sum(), 1))))
    # coverage and kWh per substation from the day matrices
    sid = np.asarray(S.ids)
    sp, lb, ob, kwh, fu = (S.get(k) for k in ("span", "lab", "obs", "kwh", "full"))
    rows = []
    for i, s in enumerate(sid):
        def cov(mask):
            a = sp[i, mask].sum()
            return float(lb[i, mask].sum() / a) if a else np.nan
        rows.append(dict(s=int(s), coverage_2025=cov(in25), observed_2025=float(ob[i, in25].sum() / max(sp[i, in25].sum(), 1)),
                         history_coverage=cov(hist), reserve_coverage=cov(res26),
                         full_path_rows_2025=float(fu[i, in25].sum() / max(sp[i, in25].sum(), 1)),
                         households_in_span_2025=float(sp[i, in25].mean()),
                         kwh_per_day_2025=float(kwh[i, in25].sum() / max(in25.sum(), 1)),
                         days_2025_low_coverage=int(((lb[i, in25] < 0.7 * sp[i, in25]) & (sp[i, in25] > 0)).sum())))
    per = per.merge(pd.DataFrame(rows), on="s", how="left")
    # silent transformer-days in 2025 (reported)
    if TR.ids:
        tsp, tlb = TR.get("span")[:, in25], TR.get("lab")[:, in25]
        sil = (tsp >= 10) & (tlb < 0.3 * tsp)
        tdf = pd.DataFrame({"t": np.asarray(TR.ids), "silent": sil.sum(1), "elig": (tsp >= 10).sum(1)})
        tdf["s"] = tdf["t"].map(mode) if tr_sub else MISSING
        q = tdf.groupby("s")[["silent", "elig"]].sum()
        per = per.merge((q["silent"] / q["elig"].clip(lower=1)).rename("silent_share_2025"), left_on="s",
                        right_index=True, how="left")
    per = per.rename(columns={"s": "substation_id"})
    per.insert(0, "coato_code", str(code))
    for c in ("region_code", "region_name", "admin_location_name"):
        per[c] = names.get(c, "")
    per["path_share"] = per["households_full_path"] / per["households"].clip(lower=1)
    per["apartment_share"] = per["apartment"] / per["known_seg"].clip(lower=1)
    per["no_gas_share"] = per["no_gas"] / per["known_seg"].clip(lower=1)
    per = per.drop(columns=["apartment", "no_gas"])
    res["substations"] = per
    res["feeders"] = pd.DataFrame(fe_rows)
    res["seconds"] = time.time() - t0
    return res


# ---------------------------------------------------------------------------------------------------- eligibility
def eligibility(sub, rules):
    checks = [("households", sub["households"] >= rules["min_households"], f"< {rules['min_households']} households"),
              ("mapping", sub["path_share"] >= rules["min_path_share"], f"path share < {rules['min_path_share']:.0%}"),
              ("stability", sub["moved"].fillna(1) <= rules["max_moved"], f"moved > {rules['max_moved']:.0%}"),
              ("tree", sub["foreign_share"].fillna(0) <= rules["max_foreign"], f"foreign > {rules['max_foreign']:.0%}"),
              ("coverage", sub["coverage_2025"].fillna(0) >= rules["min_coverage"],
               f"2025 coverage < {rules['min_coverage']:.0%}"),
              ("history", sub["history_coverage"].fillna(0) >= rules["min_history"],
               f"2024-H2 coverage < {rules['min_history']:.0%}"),
              ("feeders", sub["feeders"].fillna(0) >= 1, "no feeder")]
    reason = pd.Series("", index=sub.index, dtype=object)
    ok = pd.Series(True, index=sub.index)
    funnel = []
    for name, c, why in checks:
        c = c.fillna(False)
        reason = reason.where(~(ok & ~c), why)                     # the first rule a substation fails
        ok &= c
        funnel.append((name, int(ok.sum())))
    return ok, reason, funnel


# ---------------------------------------------------------------------------------------------------- sampling
def _pps(rng, items, weights):
    w = np.asarray(weights, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
    return items[int(rng.choice(len(items), p=w))]


def stratified_pick(rng, subs, k):
    """k substations from one district: eligible ones split into k size strata (largest first), one random pick each."""
    s = subs.sort_values("households", ascending=False).reset_index(drop=True)
    lab = (lambda i: ("large", "mid")[i]) if k == 2 else (lambda i: "one" if k == 1 else f"size {i + 1}/{k}")
    if len(s) <= k:
        return s.assign(stratum=[lab(i) for i in range(len(s))])
    parts = np.array_split(np.arange(len(s)), k)
    return pd.DataFrame([s.iloc[int(rng.choice(p))].to_dict() | {"stratum": lab(i)} for i, p in enumerate(parts)])


def select(sub, args, rng):
    el = sub[sub["eligible"]].copy()
    dist = el.groupby("coato_code").agg(region=("region_code", "first"), n=("substation_id", "size"),
                                        hh=("households", "sum")).reset_index()
    exclude = set(args.exclude_test.split(",")) if args.exclude_test else set()
    cand = dist[(dist["n"] >= args.per_district) & ~dist["coato_code"].isin(exclude)]
    regions = sorted(cand["region"].unique())
    picked = []
    notes = []
    # stage 1a: one district per region
    for reg in regions:
        c = cand[(cand["region"] == reg) & ~cand["coato_code"].isin(picked)]
        if len(c):
            picked.append(_pps(rng, c["coato_code"].to_numpy(), c["hh"].to_numpy()))
    if len(picked) > args.test_districts:
        notes.append(f"{len(picked)} regions have an eligible district but --test-districts is {args.test_districts}: "
                     f"keeping a random {args.test_districts}")
        picked = list(rng.choice(picked, args.test_districts, replace=False))
    # stage 1b: the rest by each region's share of eligible households (largest shortfall first)
    share = cand.groupby("region")["hh"].sum() / cand["hh"].sum()
    while len(picked) < args.test_districts:
        have = pd.Series([cand.set_index("coato_code").loc[c, "region"] for c in picked]).value_counts()
        target = share * args.test_districts
        gap = (target - have.reindex(target.index).fillna(0)).sort_values(ascending=False)
        done = False
        for reg in gap.index:
            c = cand[(cand["region"] == reg) & ~cand["coato_code"].isin(picked)]
            if len(c):
                picked.append(_pps(rng, c["coato_code"].to_numpy(), c["hh"].to_numpy()))
                done = True
                break
        if not done:
            notes.append(f"only {len(picked)} candidate test districts (asked {args.test_districts})")
            break
    # stage 2: substations per test district
    test = pd.concat([stratified_pick(rng, el[el["coato_code"] == c], args.per_district) for c in picked],
                     ignore_index=True).assign(role="test")
    # dev: districts without a test substation; the sample districts first
    free = el[~el["coato_code"].isin(picked)]
    dev = []
    for c in [c for c in args.exclude_test.split(",") if c] if args.exclude_test else []:
        s = free[free["coato_code"] == c]
        if len(s) and len(dev) < args.dev:
            dev.append(stratified_pick(rng, s, min(args.per_district, args.dev - len(dev))))
    used = set(pd.concat(dev)["coato_code"]) if dev else set()
    while sum(len(d) for d in dev) < args.dev:
        rest = free[~free["coato_code"].isin(used)]
        if not len(rest):
            notes.append(f"only {sum(len(d) for d in dev)} dev substations available (asked {args.dev})")
            break
        cur = pd.concat(dev)["region_code"].value_counts() if dev else pd.Series(dtype=int)
        regs = rest["region_code"].unique()
        reg = min(regs, key=lambda r: (cur.get(r, 0), rng.random()))
        dd = rest[rest["region_code"] == reg]
        dsum = dd.groupby("coato_code")["households"].sum()
        c = _pps(rng, dsum.index.to_numpy(), dsum.to_numpy())
        dev.append(stratified_pick(rng, dd[dd["coato_code"] == c], 1).assign(stratum="one per district"))
        used.add(c)
    dev = pd.concat(dev, ignore_index=True).assign(role="dev") if dev else pd.DataFrame()
    return pd.concat([test, dev], ignore_index=True), picked, notes


# ---------------------------------------------------------------------------------------------------- reference (CAS)
def cas_nodes(ref_root):
    if not ref_root or ref_root.lower() == "none":
        return None
    spec = importlib.util.spec_from_file_location("hierarchy_check", os.path.join(REPO, "scripts", "analysis",
                                                                                  "hierarchy_check.py"))
    hc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hc)
    nn, _, msg = hc.load_reference(ref_root)
    print(f"[cas] {msg}")
    if nn is None or "network_node_id" not in nn:
        return None
    nn = nn.dropna(subset=["network_node_id"]).copy()
    nn["network_node_id"] = nn["network_node_id"].astype(np.int64)
    typ = next((c for c in ("node_type_name_ru", "node_type_name_uz", "node_type_code", "node_type_id") if c in nn), None)
    keep = ["network_node_id"] + [c for c in ("latitude", "longitude") if c in nn] + ([typ] if typ else [])
    out = nn.drop_duplicates("network_node_id")[keep].rename(columns={typ: "cas_node_type"} if typ else {})
    for c in ("latitude", "longitude"):
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ---------------------------------------------------------------------------------------------------- spec + report
def write_spec(path, args, sel, feeders, sel_sha):
    import yaml
    def lst(role):
        out = []
        for r in sel[sel["role"] == role].sort_values(["coato_code", "substation_id"]).itertuples():
            fd = feeders[(feeders["coato_code"] == r.coato_code) & (feeders["substation_id"] == r.substation_id)]
            out.append(dict(coato_code=str(r.coato_code), substation_id=int(r.substation_id), region=str(r.region_name or
                            r.region_code), district=str(r.admin_location_name), stratum=str(r.stratum),
                            households=int(r.households), feeders=sorted(int(f) for f in fd["feeder_id"])))
        return out
    spec = dict(
        version=args.version,
        description=("Substation / feeder forecasting test: residential load (households under the node, node-level T2), "
                     "issue dates 2025, rolling quarterly refits. Model choice on `dev` only; `test` is scored once."),
        data=dict(uri=args.root, start="2022-01-01", end="2026-06-29"),
        target=dict(name="T2_node", definition="households in span x (kwh observed + estimated) / (n observed + n "
                    "estimated), per node-day", min_coverage=0.7, levels=["substation", "feeder"],
                    note="household (b2c) load only: not the metered substation or feeder load (no business customers, "
                         "no losses)"),
        history_gap_days=1,
        splits=dict(dev=dict(issue_start="2025-01-01", issue_end="2025-12-31"),
                    test=dict(issue_start="2025-01-01", issue_end="2025-12-31"),
                    reserve="issue dates 2026 on the test substations: not used"),
        refits=["2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01"],
        horizons=60, control_horizons=[1, 7, 14, 30, 60], reference_model="B0_node", weather="none",
        selection=dict(seed=args.seed, rules={k: getattr(args, k) for k in DEFAULTS},
                       test_districts=args.test_districts, per_district=args.per_district, dev=args.dev,
                       excluded_from_test=args.exclude_test, selection_csv_sha256=sel_sha,
                       caveat="dev and test cover the same period (2025): model choice sees the same weather and "
                              "national shocks as the test, which makes it slightly optimistic"),
        test_substations=lst("test"), dev_substations=lst("dev"))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("# Generated by scripts/test_set/select_substations.py; do not edit by hand (re-run with a new version)\n")
        yaml.safe_dump(spec, fh, sort_keys=False, allow_unicode=True, width=120)


def report(sub, sel, feeders, funnel, picked, notes, args):
    print(f"SUBSTATION SELECTION {args.version} | seed {args.seed}\n")
    print(f"districts read {sub['coato_code'].nunique()} | substations {len(sub):,} | households (2025) "
          f"{sub['households'].sum():,.0f}")
    print("\n1. ELIGIBILITY FUNNEL (substations left after each rule)")
    prev = len(sub)
    for name, n in funnel:
        print(f"  {name:10s} {n:6,d}   (-{prev - n:,})")
        prev = n
    el = sub[sub["eligible"]]
    print(f"  eligible: {len(el):,} substations, {el['households'].sum():,.0f} households "
          f"({el['households'].sum() / max(sub['households'].sum(), 1):.0%} of all) in {el['coato_code'].nunique()} districts")
    print("  most common reasons: " + "; ".join(f"{k} {v:,}" for k, v in
                                                sub.loc[~sub["eligible"], "reason"].value_counts().head(6).items()))
    print("\n2. BY REGION (eligible households share | test districts | test / dev substations)")
    reg = el.groupby(["region_code", "region_name"])["households"].sum()
    reg = (reg / reg.sum()).rename("eligible_share").reset_index()
    t = sel[sel["role"] == "test"].groupby("region_code").agg(test=("substation_id", "size"),
                                                               districts=("coato_code", "nunique"))
    d = sel[sel["role"] == "dev"].groupby("region_code").size().rename("dev")
    reg = reg.merge(t, on="region_code", how="left").merge(d, on="region_code", how="left").fillna(0)
    reg["eligible_share"] = reg["eligible_share"].map("{:.1%}".format)
    for c in ("test", "districts", "dev"):
        if c in reg:
            reg[c] = reg[c].astype(int)
    print(reg.to_string(index=False))
    print("\n3. SAMPLE vs ELIGIBLE POPULATION (median [p10, p90])")
    for c, f in (("households", "{:,.0f}"), ("coverage_2025", "{:.0%}"), ("observed_2025", "{:.0%}"),
                 ("apartment_share", "{:.0%}"), ("no_gas_share", "{:.0%}"), ("silent_share_2025", "{:.1%}"),
                 ("feeders", "{:.0f}"), ("kwh_per_day_2025", "{:,.0f}")):
        def q(x):
            x = x.dropna()
            return "n/a" if not len(x) else f"{f.format(x.median())} [{f.format(x.quantile(.1))}, {f.format(x.quantile(.9))}]"
        print(f"  {c:20s} eligible {q(el[c]):32s} test {q(sel.loc[sel['role'] == 'test', c]):32s} "
              f"dev {q(sel.loc[sel['role'] == 'dev', c])}")
    print("\n4. CHOSEN")
    for role in ("test", "dev"):
        s = sel[sel["role"] == role]
        fd = feeders.merge(s[["coato_code", "substation_id"]], on=["coato_code", "substation_id"])
        print(f"  {role}: {len(s)} substations in {s['coato_code'].nunique()} districts, {s['region_code'].nunique()} "
              f"regions | {s['households'].sum():,.0f} households | {len(fd)} feeders (size bands: "
              + ", ".join(f"{k} {v}" for k, v in fd["size_band"].value_counts().sort_index().items()) + ")")
    for n in notes:
        print(f"  NOTE: {n}")
    print("\nCaveat: dev and test cover the same period (2025); choose models on dev, score test once.")


def charts(sub, sel, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = os.path.join(out, "figures")
    os.makedirs(fd, exist_ok=True)
    el = sub[sub["eligible"]]
    if {"latitude", "longitude"} <= set(sub.columns):
        box = lambda d: d[d["latitude"].between(37, 46) & d["longitude"].between(55, 74)]   # noqa: E731
        fig, ax = plt.subplots(figsize=(10, 7))
        a = box(sub)
        ax.scatter(a["longitude"], a["latitude"], s=4, c="#e5e7eb", label="all substations")
        a = box(el)
        ax.scatter(a["longitude"], a["latitude"], s=6, c="#9ca3af", label="eligible")
        for role, col in (("dev", "#f97316"), ("test", "#2563eb")):
            a = box(sel[sel["role"] == role])
            ax.scatter(a["longitude"], a["latitude"], s=40, c=col, edgecolor="white", label=f"{role} ({len(a)})")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_title("Substations: eligible population and the chosen test / dev sample")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fd, "map.png"), dpi=120)
        plt.close()
    cols = ["households", "coverage_2025", "apartment_share", "no_gas_share"]
    fig, axes = plt.subplots(1, len(cols), figsize=(15, 4))
    for ax, c in zip(axes, cols):
        data = [el[c].dropna(), sel.loc[sel["role"] == "test", c].dropna(), sel.loc[sel["role"] == "dev", c].dropna()]
        ax.boxplot(data, tick_labels=["eligible", "test", "dev"], showfliers=False)
        if c == "households":
            ax.set_yscale("log")
        ax.set_title(c)
    plt.suptitle("Sample vs eligible substations")
    plt.tight_layout()
    plt.savefig(os.path.join(fd, "distributions.png"), dpi=120)
    plt.close()
    reg = el.groupby("region_name")["households"].sum()
    reg = reg / reg.sum()
    tc = sel[sel["role"] == "test"].groupby("region_name").size()
    tc = tc / max(tc.sum(), 1)
    df = pd.DataFrame({"eligible households": reg, "test substations": tc}).fillna(0).sort_values("eligible households")
    ax = df.plot(kind="barh", figsize=(9, 6), color=["#9ca3af", "#2563eb"])
    ax.set_xlabel("share")
    ax.set_title("Regions: share of eligible households vs share of test substations")
    plt.tight_layout()
    plt.savefig(os.path.join(fd, "regions.png"), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------------------------------- main
def main(argv=None):
    import enriched_io as eio
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--root", default=eio.ENRICHED_ROOT)
    p.add_argument("--ref-root", default=REF_ROOT, help="CAS reference (network_nodes) for coordinates; 'none' skips")
    p.add_argument("--start", default="2024-08-01", help="read from (the new regime; also the stability window)")
    p.add_argument("--end", default=eio.WINDOW_END)
    p.add_argument("--districts", default=None, help="comma list (default: every finished district)")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--test-districts", type=int, default=20)
    p.add_argument("--per-district", type=int, default=2)
    p.add_argument("--dev", type=int, default=20)
    p.add_argument("--exclude-test", default=",".join(SAMPLE), help="districts never used for test (dev first)")
    p.add_argument("--version", default="ss-2026-10-23-v1")
    p.add_argument("--spec-out", default=os.path.join(REPO, "configs", "test_set_substations_2025.yaml"))
    for k, v in DEFAULTS.items():
        p.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    args = p.parse_args(argv)
    out = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out, exist_ok=True)
    cfg = dict(core=CORE, root=args.root, start=args.start, end=args.end)
    fs, root = eio.open_root(args.root)
    codes = eio.finished_districts(fs, root, want=set(args.districts.split(",")) if args.districts else None)
    t0 = time.time()
    res = []
    if args.workers <= 1:
        for c in codes:
            res.append(district_worker(c, cfg))
    else:
        thr = max(1, (os.cpu_count() or 4) // args.workers)
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[v] = str(thr)
        with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
            pending = {ex.submit(district_worker, c, cfg): c for c in codes}
            while pending:
                done, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
                for f in done:
                    code = pending.pop(f)
                    try:
                        res.append(f.result())
                    except Exception as e:                                   # noqa: BLE001
                        raise RuntimeError(f"district {code} failed: {type(e).__name__}: {e}") from e
                    k = len(res)
                    if k % 10 == 0 or k == len(codes) or k <= 3:
                        el = time.time() - t0
                        print(f"[read] {k}/{len(codes)} districts, {el / 60:.1f} min, about "
                              f"{el / k * (len(codes) - k) / 60:.1f} min left", flush=True)
    sub = pd.concat([r["substations"] for r in res if "substations" in r], ignore_index=True)
    feeders = pd.concat([r["feeders"] for r in res if "feeders" in r and len(r["feeders"])], ignore_index=True)
    feeders["size_band"] = pd.cut(feeders["households_2025"], [0, 50, 200, 1000, 10 ** 9],
                                  labels=["< 50", "50-200", "200-1000", ">= 1000"]).astype(str)
    cas = cas_nodes(args.ref_root)
    if cas is not None:
        sub = sub.merge(cas, left_on="substation_id", right_on="network_node_id", how="left").drop(columns="network_node_id")
    sub["eligible"], sub["reason"], funnel = eligibility(sub, {k: getattr(args, k) for k in DEFAULTS})
    rng = np.random.default_rng(args.seed)
    sel, picked, notes = select(sub, args, rng)
    sub.to_csv(os.path.join(out, "substations_all.csv"), index=False)
    sel_path = os.path.join(out, "selection.csv")
    sel.to_csv(sel_path, index=False)
    sel_sha = hashlib.sha256(open(sel_path, "rb").read()).hexdigest()
    fsel = feeders.merge(sel[["coato_code", "substation_id", "role"]], on=["coato_code", "substation_id"])
    fsel.to_csv(os.path.join(out, "feeders_selected.csv"), index=False)
    write_spec(args.spec_out, args, sel, feeders, sel_sha)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(sub, sel, feeders, funnel, picked, notes, args)
    txt = buf.getvalue()
    print(txt)
    open(os.path.join(out, "selection_report.txt"), "w").write(txt)
    try:
        charts(sub, sel, out)
    except Exception as e:                                                   # noqa: BLE001
        print(f"[charts] WARNING: not drawn ({type(e).__name__}: {e})")
    json.dump(dict(version=args.version, seed=args.seed, spec=args.spec_out, selection_sha256=sel_sha,
                   seconds=round(time.time() - t0, 1)), open(os.path.join(out, "selection.json"), "w"), indent=2)
    print(f"[spec] {args.spec_out}\n[done] {out} in {(time.time() - t0) / 60:.1f} min")
    return out


if __name__ == "__main__":
    main()
