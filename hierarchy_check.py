#!/usr/bin/env python3
"""
Is household -> transformer -> feeder -> substation -> district (coato) -> region a hierarchy we can rely on?

Three kinds of evidence, all from the national enriched household dataset v1.4 (every finished district,
2022-01-01 .. 2026-06-29) and, when readable, the CAS network-node table reference.network_nodes:

STRUCTURE (does it behave like a tree?)
  A. Completeness   households with a missing / negative (sentinel) node, or on a placeholder transformer
  B. Stability      households whose (transformer, feeder, substation) changes within the window
  C. Nesting        children with more than one parent, over ALL rows (not one value per customer):
                    transformer -> feeder, feeder -> substation, node -> district, district -> region
  D. Uniqueness     node ids found in more than one district
  E. Sizes          households per node, transformers per feeder, feeders per substation

PHYSICS (do siblings share problems, as they would on a real shared line?)
  F. For every transformer-day with a sudden drop, the share of OTHER transformers on the SAME feeder dropping that
     day vs the share of transformers on OTHER feeders of the same district (lift = ratio; 1 = the grouping means
     nothing). Same for feeders under one substation. Two kinds of drop:
       reporting drop  observed share of in-span households < 30% (node's usual median >= 70%)
       kWh drop        kWh per observed household < 50% of its trailing 28-day median (>= 10 households observed)
     Days on which > 50% of the district's nodes drop are left out (district-wide events). A control with the
     labels shuffled among the district's nodes should give a lift close to 1.

INDEPENDENT SOURCE (does the CAS network table agree?)  -- reads --ref-root, skipped if not readable
  G. For each level: share of households whose node id exists in network_nodes, the node TYPE it has there
     (a transformer column should hold transformer-type nodes, etc.), whether the CAS parent chain of the
     transformer reaches the enriched feeder (and in how many steps), feeder -> substation likewise, whether the
     CAS node's coato_code equals the district, and the distance transformer -> its substation from CAS coordinates.
  H. District -> region: is each district's region_code its parent in reference.coato_branch_hierarchy?

Outputs (<out-dir>), aggregates only: hierarchy_checks.txt (the printed report), hierarchy_nodes.csv (one row per
distinct transformer/feeder/substation path per district with its household count), hierarchy_physics.csv.

  python hierarchy_check.py --out-dir ~/work/hierarchy --workers 12
"""
import argparse
import concurrent.futures as cf
import contextlib
import io
import multiprocessing as mp
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MISSING = -(2 ** 62)
REF_ROOT = "s3://phase1/electricity/analytics/reference"


# ---------------------------------------------------------------------------------------------------- physics helpers
def _drop_flags(S, O, K, min_hh=10):
    """S, O, K: (nodes, T) in-span households, observed households, observed kWh -> (eligible_rep, drop_rep,
    eligible_kwh, drop_kwh), all (nodes, T) bool."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cov = np.where(S >= min_hh, O / np.maximum(S, 1), np.nan)
        med = np.nanmedian(cov, axis=1)
        el_r = np.isfinite(cov) & (np.nan_to_num(med) >= 0.7)[:, None]
        dr_r = el_r & (cov < 0.3)
        m = np.where(O >= min_hh, K / np.maximum(O, 1), np.nan)
        tm = pd.DataFrame(m.T).shift(1).rolling(28, min_periods=10).median().to_numpy().T
        el_k = np.isfinite(m) & np.isfinite(tm) & (tm > 0)
        dr_k = el_k & (m < 0.5 * np.nan_to_num(tm))
    return el_r, dr_r, el_k, dr_k


def _sibling_counts(el, dr, group, max_share=0.5):
    """For every dropping node-day: eligible siblings (same group) and how many of them drop, against eligible nodes in
    OTHER groups and how many of them drop. Days with > max_share of eligible nodes dropping are skipped.
    -> [events, same_n, same_drop, other_n, other_drop, days_skipped]"""
    g = np.asarray(group)
    G = int(g.max()) + 1 if len(g) else 0
    out = np.zeros(6)
    for d in range(el.shape[1]):
        e, x = el[:, d], dr[:, d]
        ne, nx = int(e.sum()), int(x.sum())
        if nx == 0 or ne < 3:
            continue
        if nx > max_share * ne:
            out[5] += 1
            continue
        Eg = np.bincount(g[e], minlength=G)
        Xg = np.bincount(g[x], minlength=G)
        gi = g[x]
        out[0] += nx
        out[1] += (Eg[gi] - 1).sum()
        out[2] += (Xg[gi] - 1).sum()
        out[3] += (ne - Eg[gi]).sum()
        out[4] += (nx - Xg[gi]).sum()
    return out


def _physics(S, O, K, child_group, rng):
    out = {}
    er, xr, ek, xk = _drop_flags(S, O, K)
    sh = rng.permutation(child_group)
    for kind, el, dr in (("reporting", er, xr), ("kwh", ek, xk)):
        out[kind] = _sibling_counts(el, dr, child_group)
        out[kind + " (shuffled control)"] = _sibling_counts(el, dr, sh)
    return out


# ---------------------------------------------------------------------------------------------------- district worker
def district_worker(code, cfg):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as pads
    sys.path.insert(0, cfg["here"])
    import enriched_io as eio
    t0 = time.time()
    fs, root = eio.open_root(cfg["root"])
    d0, d1 = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    T = (d1 - d0).days + 1
    e0 = int((d0 - pd.Timestamp("1970-01-01")).days)
    dset = eio.district_daily(fs, root, code)
    want = ["customer_key", "date", "transformer_id", "feeder_id", "substation_id", "region_code",
            "placeholder_transformer", "label_state", "kwh"]
    cols = [c for c in want if c in dset.schema.names]
    flt = (pads.field("date") >= d0.date()) & (pads.field("date") <= d1.date())
    labels = pa.array(list(cfg["labels"]), type=pa.string())
    key_of, tr_idx, tr_ids = {}, {}, []
    S = np.zeros((0, T), np.int32)
    O = np.zeros((0, T), np.int32)
    K = np.zeros((0, T), np.float64)
    combos, regions, phs = [], {}, set()
    rows = 0
    for b in dset.scanner(columns=cols, filter=flt, batch_size=262_144, use_threads=True).to_batches():
        n = b.num_rows
        if not n:
            continue
        rows += n

        def ints(col):
            if col not in cols:
                return np.full(n, MISSING, np.int64)
            return pc.fill_null(pc.cast(b.column(col), pa.int64()), MISSING).to_numpy(zero_copy_only=False)
        de = pc.dictionary_encode(b.column("customer_key"))
        uq = de.dictionary.to_pylist()
        ck = np.empty(len(uq), dtype=np.int64)
        for j, s in enumerate(uq):
            v = key_of.get(s)
            if v is None:
                v = key_of[s] = len(key_of)
            ck[j] = v
        cust = ck[de.indices.to_numpy(zero_copy_only=False)]
        tr, fe, su = ints("transformer_id"), ints("feeder_id"), ints("substation_id")
        combos.append(pd.DataFrame({"c": cust, "t": tr, "f": fe, "s": su}).drop_duplicates())
        if "region_code" in cols:
            vc = pc.value_counts(pc.cast(b.column("region_code"), pa.string()))
            for v, k in zip(vc.field("values").to_pylist(), vc.field("counts").to_pylist()):
                if v is not None:
                    regions[str(v)] = regions.get(str(v), 0) + k
        if "placeholder_transformer" in cols:
            pv = pc.fill_null(b.column("placeholder_transformer"), False).to_numpy(zero_copy_only=False)
            phs.update(int(t) for t in np.unique(tr[pv]))
        tu, tinv = np.unique(tr, return_inverse=True)
        li = np.empty(len(tu), dtype=np.int64)
        for j, t in enumerate(tu):
            v = tr_idx.get(int(t))
            if v is None:
                v = tr_idx[int(t)] = len(tr_ids)
                tr_ids.append(int(t))
            li[j] = v
        tl = li[tinv]
        if len(tr_ids) > S.shape[0]:
            grow = max(len(tr_ids), 2 * S.shape[0], 16) - S.shape[0]
            S = np.vstack([S, np.zeros((grow, T), np.int32)])
            O = np.vstack([O, np.zeros((grow, T), np.int32)])
            K = np.vstack([K, np.zeros((grow, T))])
        day = pc.cast(pc.cast(b.column("date"), pa.date32()), pa.int32()).to_numpy(zero_copy_only=False).astype(np.int64) - e0
        kv = b.column("kwh")
        ok = pc.and_(pc.fill_null(pc.is_in(b.column("label_state"), value_set=labels), False),
                     pc.is_valid(kv)).to_numpy(zero_copy_only=False)
        size = S.shape[0] * T
        idx = tl * T + day
        S += np.bincount(idx, minlength=size).reshape(S.shape).astype(np.int32)
        O += np.bincount(idx[ok], minlength=size).reshape(S.shape).astype(np.int32)
        kw = pc.fill_null(pc.cast(kv, pa.float64()), 0.0).to_numpy(zero_copy_only=False)
        K += np.bincount(idx[ok], weights=kw[ok], minlength=size).reshape(S.shape)
    res = dict(code=str(code), rows=rows, regions=regions)
    if not tr_ids:
        res["seconds"] = time.time() - t0
        return res
    cmb = pd.concat(combos, ignore_index=True).drop_duplicates()
    del combos
    per_c = cmb.groupby("c").size()
    res["customers"] = int(len(per_c))
    res["customers_multi_mapping"] = int((per_c > 1).sum())
    for lvl, col in (("transformer", "t"), ("feeder", "f"), ("substation", "s")):
        res[f"customers_multi_{lvl}"] = int((cmb.groupby("c")[col].nunique() > 1).sum())
    # one row per distinct path; households = distinct customers ever on it (a moved customer counts on both)
    nodes = (cmb.groupby(["t", "f", "s"])["c"].nunique().rename("households").reset_index()
             .rename(columns={"t": "transformer_id", "f": "feeder_id", "s": "substation_id"}))
    nodes["coato_code"] = str(code)
    nodes["placeholder"] = nodes["transformer_id"].isin(phs)
    res["nodes"] = nodes

    # ---- physics: transformer -> its main feeder; feeder -> its main substation
    nt = len(tr_ids)
    S, O, K = S[:nt], O[:nt], K[:nt]
    good = nodes[(nodes["transformer_id"] >= 0) & (nodes["feeder_id"] >= 0) & (nodes["substation_id"] >= 0)
                 & ~nodes["placeholder"]]
    main_f = good.sort_values("households", ascending=False).drop_duplicates("transformer_id") \
        .set_index("transformer_id")["feeder_id"]
    tf = np.array([main_f.get(t, MISSING) for t in tr_ids])
    okt = tf != MISSING
    rng = np.random.default_rng(sum(map(ord, str(code))) * 7919)
    phys = {}
    if okt.sum() >= 6:
        fu, fg = np.unique(tf[okt], return_inverse=True)
        if fg.max() >= 1:
            for k, v in _physics(S[okt], O[okt], K[okt], fg, rng).items():
                phys[("transformer within feeder", k)] = v
        main_s = good.groupby(["feeder_id", "substation_id"])["households"].sum().reset_index() \
            .sort_values("households", ascending=False).drop_duplicates("feeder_id").set_index("feeder_id")["substation_id"]
        Sf = np.zeros((len(fu), T))
        Of, Kf = np.zeros_like(Sf), np.zeros_like(Sf)
        np.add.at(Sf, fg, S[okt])
        np.add.at(Of, fg, O[okt])
        np.add.at(Kf, fg, K[okt])
        sub = np.array([main_s.get(f, MISSING) for f in fu])
        oks = sub != MISSING
        if oks.sum() >= 6:
            _, sg = np.unique(sub[oks], return_inverse=True)
            if sg.max() >= 1:
                for k, v in _physics(Sf[oks], Of[oks], Kf[oks], sg, rng).items():
                    phys[("feeder within substation", k)] = v
    res["physics"] = phys
    res["seconds"] = time.time() - t0
    return res


# ---------------------------------------------------------------------------------------------------- main
def main(argv=None):
    import enriched_io as eio
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--root", default=eio.ENRICHED_ROOT)
    p.add_argument("--ref-root", default=REF_ROOT, help="folder holding network_nodes/ and coato_branch_hierarchy/ "
                                                        "('none' skips the CAS checks)")
    p.add_argument("--start", default=eio.WINDOW_START)
    p.add_argument("--end", default=eio.WINDOW_END)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--labels", default="complete_observed")
    p.add_argument("--districts", default=None, help="comma-separated coato codes (default: all finished)")
    args = p.parse_args(argv)
    out = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out, exist_ok=True)
    cfg = dict(here=HERE, root=args.root, start=args.start, end=args.end, labels=args.labels.split(","))
    fs, root = eio.open_root(args.root)
    codes = eio.finished_districts(fs, root, want=set(args.districts.split(",")) if args.districts else None)
    thr = max(1, (os.cpu_count() or 4) // args.workers)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(thr)
    t0 = time.time()
    results = []
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        pending = {ex.submit(district_worker, c, cfg): c for c in codes}
        k = 0
        while pending:
            done, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
            for f in done:
                code = pending.pop(f)
                try:
                    results.append(f.result())
                except Exception as e:                                       # noqa: BLE001
                    raise RuntimeError(f"district {code} failed: {type(e).__name__}: {e}") from e
                del f
                k += 1
                if k % 10 == 0 or k == len(codes) or k <= 3:
                    el = time.time() - t0
                    print(f"[read] {k}/{len(codes)} districts, {el / 60:.1f} min, about "
                          f"{el / k * (len(codes) - k) / 60:.1f} min left", flush=True)
    ref = load_reference(args.ref_root)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(results, out, ref)
    txt = buf.getvalue()
    print(txt)
    with open(os.path.join(out, "hierarchy_checks.txt"), "w") as fh:
        fh.write(txt)
    print(f"[done] {out} in {(time.time() - t0) / 60:.1f} min")


def load_reference(ref_root):
    """(network_nodes DataFrame or None, coato_branch_hierarchy DataFrame or None, message)"""
    if not ref_root or ref_root.lower() == "none":
        return None, None, "skipped (--ref-root none)"
    import pyarrow.dataset as pads
    import enriched_io as eio
    try:
        fs, root = eio.open_root(ref_root)
    except SystemExit as e:
        return None, None, f"not readable: {e}"
    want = {"network_nodes": ["network_node_id", "coato_code", "node_type_id", "node_type_code", "node_type_name_ru",
                              "node_type_name_uz", "parent_network_node_id", "latitude", "longitude", "status",
                              "is_deleted"],
            "coato_branch_hierarchy": ["coato_code", "parent_coato_code", "name_uz", "level_id", "status", "is_deleted"]}
    out, msgs = [], []
    for name, cols in want.items():
        try:
            d = pads.dataset(f"{root}/{name}/data", filesystem=fs, format="parquet")
            have = [c for c in cols if c in d.schema.names]
            out.append(d.to_table(columns=have).to_pandas())
            msgs.append(f"{name} {len(out[-1]):,} rows")
        except Exception as e:                                               # noqa: BLE001
            out.append(None)
            msgs.append(f"{name} not readable ({type(e).__name__}: {str(e)[:120]})")
    return out[0], out[1], "; ".join(msgs)


# ---------------------------------------------------------------------------------------------------- report
def _q(x):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return "n/a"
    v = np.percentile(x, [5, 25, 50, 75, 95])
    return " | ".join(f"{a} {b:,.0f}" for a, b in zip(("p5", "p25", "median", "p75", "p95"), v)) + f" | max {x.max():,.0f}"


def report(results, out, ref):
    results = [r for r in results if "nodes" in r]
    nodes = pd.concat([r["nodes"] for r in results], ignore_index=True)
    nodes.replace({MISSING: np.nan}).to_csv(os.path.join(out, "hierarchy_nodes.csv"), index=False)
    hh = nodes["households"]
    H = int(hh.sum())
    cust = sum(r["customers"] for r in results)
    print(f"Households: {cust:,} in {len(results)} districts ({sum(r['rows'] for r in results):,} customer-day rows)")

    print("\n=== A. Completeness (share of households) ===")
    for lvl in ("transformer_id", "feeder_id", "substation_id"):
        miss = nodes[lvl] == MISSING
        neg = (nodes[lvl] < 0) & ~miss
        print(f"  {lvl:15s} missing {hh[miss].sum() / H:7.2%} | negative id (sentinel) {hh[neg].sum() / H:7.2%}")
    print(f"  on a placeholder transformer (> 1,000 customers): {hh[nodes['placeholder']].sum() / H:.2%}")
    ok = (nodes["transformer_id"] >= 0) & (nodes["feeder_id"] >= 0) & (nodes["substation_id"] >= 0) & ~nodes["placeholder"]
    clean = nodes[ok]
    print(f"  complete, non-placeholder path: {clean['households'].sum() / H:.2%}  <- the part the checks below use")

    print("\n=== B. Stability within the window (households whose mapping changes) ===")
    for k, lab in (("customers_multi_mapping", "any change"), ("customers_multi_transformer", "transformer"),
                   ("customers_multi_feeder", "feeder"), ("customers_multi_substation", "substation")):
        m = sum(r[k] for r in results)
        print(f"  {lab:12s} {m:9,} ({m / max(cust, 1):.3%})")
    print("  The dataset applies the LAST-KNOWN mapping to every date, so ~0 here is by construction: it does not show "
          "that the 2022 network matched today's.")

    print("\n=== C. Nesting: children with more than one parent ===")
    for name, child, parent in (("transformer -> feeder", ["coato_code", "transformer_id"], "feeder_id"),
                                ("feeder -> substation", ["coato_code", "feeder_id"], "substation_id"),
                                ("transformer -> district", ["transformer_id"], "coato_code"),
                                ("feeder -> district", ["feeder_id"], "coato_code"),
                                ("substation -> district", ["substation_id"], "coato_code")):
        g = clean.groupby(child).agg(parents=(parent, "nunique"), households=("households", "sum"))
        bad = g[g["parents"] > 1]
        print(f"  {name:24s} {len(bad):7,} of {len(g):8,} children ({len(bad) / max(len(g), 1):6.2%}); households on them "
              f"{bad['households'].sum() / max(g['households'].sum(), 1):6.2%}; max parents {int(g['parents'].max())}")
    multi_reg = {r["code"]: r["regions"] for r in results if len(r["regions"]) > 1}
    no_reg = [r["code"] for r in results if not r["regions"]]
    print(f"  district -> region       {len(multi_reg)} districts with more than one region_code"
          + (f" (e.g. {dict(list(multi_reg.items())[:3])})" if multi_reg else "")
          + (f"; {len(no_reg)} districts without region_code" if no_reg else ""))
    reg_of = {r["code"]: max(r["regions"], key=r["regions"].get) for r in results if r["regions"]}
    if reg_of:
        rc = pd.Series(reg_of).value_counts()
        print(f"  regions: {len(rc)} | districts per region: {_q(rc.to_numpy())}")

    print("\n=== D. Uniqueness: ids found in more than one district ===")
    for lvl in ("transformer_id", "feeder_id", "substation_id"):
        g = clean.groupby(lvl).agg(d=("coato_code", "nunique"), households=("households", "sum"))
        b = g[g["d"] > 1]
        print(f"  {lvl:15s} {len(b):,} of {len(g):,} ids; {b['households'].sum() / max(g['households'].sum(), 1):.2%} "
              "of households")

    print("\n=== E. Sizes (complete paths, nodes counted per district) ===")
    t = clean.groupby(["coato_code", "transformer_id"])["households"].sum()
    f = clean.groupby(["coato_code", "feeder_id"]).agg(households=("households", "sum"),
                                                       transformers=("transformer_id", "nunique"))
    s = clean.groupby(["coato_code", "substation_id"]).agg(households=("households", "sum"), feeders=("feeder_id", "nunique"))
    d = clean.groupby("coato_code").agg(substations=("substation_id", "nunique"))
    print(f"  households per transformer ({len(t):,}): {_q(t)}")
    print(f"  households per feeder      ({len(f):,}): {_q(f['households'])}")
    print(f"  households per substation  ({len(s):,}): {_q(s['households'])}")
    print(f"  transformers per feeder:   {_q(f['transformers'])}")
    print(f"  feeders per substation:    {_q(s['feeders'])}")
    print(f"  substations per district:  {_q(d['substations'])}")
    print(f"  feeders with 1 transformer: {(f['transformers'] == 1).mean():.1%} | substations with 1 feeder: "
          f"{(s['feeders'] == 1).mean():.1%}")

    physics_report(results, out)
    nn, br, msg = ref
    print(f"\n=== G. Agreement with the CAS network table ({msg}) ===")
    if nn is not None:
        cas_report(clean, nodes, nn)
    else:
        print("  skipped")
    print("\n=== H. District -> region against reference.coato_branch_hierarchy ===")
    if br is not None and reg_of:
        region_report(reg_of, br)
    else:
        print("  skipped (no hierarchy table or no region_code in the data)")


def physics_report(results, out):
    print("\n=== F. Physics: when a node drops, do its siblings drop with it? ===")
    agg = {}
    for r in results:
        for key, v in (r.get("physics") or {}).items():
            agg[key] = agg.get(key, 0) + v
    rows = []
    for (level, kind), v in sorted(agg.items()):
        ev, sn, sd, on, od, skip = v
        ps, po = sd / max(sn, 1), od / max(on, 1)
        rows.append(dict(level=level, drop=kind, drop_events=int(ev), p_sibling=ps, p_other=po,
                         lift=ps / po if po > 0 else np.nan, district_wide_days_skipped=int(skip)))
    ph = pd.DataFrame(rows)
    ph.to_csv(os.path.join(out, "hierarchy_physics.csv"), index=False)
    if not len(ph):
        print("  no district had enough nodes")
        return
    print("  p_sibling = share of the dropping node's siblings (same parent) dropping the same day; p_other = share of the "
          "district's nodes under other parents; lift = p_sibling / p_other (about 1 = the grouping carries no information)")
    print(ph.to_string(index=False, formatters={"p_sibling": "{:.2%}".format, "p_other": "{:.2%}".format,
                                                "lift": "{:.1f}".format, "drop_events": "{:,}".format}))
    # per district: a mapping that is right in some districts and arbitrary in others shows up here
    per = []
    for r in results:
        for (level, kind), v in (r.get("physics") or {}).items():
            if kind in ("kwh", "reporting") and v[0] >= 20 and v[3] > 0 and v[4] > 0:
                per.append(dict(coato_code=r["code"], level=level, drop=kind, drop_events=int(v[0]),
                                lift=(v[2] / max(v[1], 1)) / (v[4] / v[3])))
    per = pd.DataFrame(per)
    if len(per):
        per.to_csv(os.path.join(out, "hierarchy_physics_by_district.csv"), index=False)
        print("  per district (districts with >= 20 drop events):")
        for (level, kind), g in per.groupby(["level", "drop"]):
            x = g["lift"]
            print(f"    {level:26s} {kind:9s} {len(g):4d} districts | lift median {x.median():.1f} | p10 {x.quantile(.1):.1f} | "
                  f"share of districts with lift < 1.5: {(x < 1.5).mean():.0%}")


def _hav_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def cas_report(clean, nodes, nn):
    nn = nn.dropna(subset=["network_node_id"]).copy()
    nn["network_node_id"] = nn["network_node_id"].astype(np.int64)
    nn = nn.drop_duplicates("network_node_id").set_index("network_node_id")
    typ = next((c for c in ("node_type_name_ru", "node_type_name_uz", "node_type_code", "node_type_id") if c in nn), None)
    H = nodes["households"].sum()
    for lvl in ("transformer_id", "feeder_id", "substation_id"):
        g = nodes[nodes[lvl] >= 0].groupby(lvl)["households"].sum()
        found = g.index.isin(nn.index)
        print(f"  {lvl:15s} ids in CAS: {found.mean():6.2%} of ids, {g[found].sum() / max(g.sum(), 1):6.2%} of their "
              f"households ({g.sum() / H:.1%} of all households have a non-negative id)")
        if typ and found.any():
            tt = pd.Series(g[found].to_numpy(), index=nn.loc[g.index[found], typ].astype(str).to_numpy()).groupby(level=0).sum()
            tt = (tt / tt.sum()).sort_values(ascending=False).head(5)
            print("      CAS node type (share of households): " + "; ".join(f"{k} {v:.1%}" for k, v in tt.items()))
        if "is_deleted" in nn and found.any():
            dl = nn.loc[g.index[found], "is_deleted"].fillna(False).astype(bool).to_numpy()
            print(f"      deleted in CAS: {g[found][dl].sum() / max(g[found].sum(), 1):.2%} of households")
    if "parent_network_node_id" not in nn:
        return
    parent = nn["parent_network_node_id"].dropna().astype(np.int64).to_dict()

    def steps(child, target, max_up=6):
        x = child
        for k in range(1, max_up + 1):
            x = parent.get(x)
            if x is None:
                return "chain ends first"
            if x == target:
                return f"{k} step" + ("s" if k > 1 else "")
        return f"not within {max_up} steps"

    for name, a, b_ in (("transformer -> feeder", "transformer_id", "feeder_id"),
                        ("feeder -> substation", "feeder_id", "substation_id"),
                        ("transformer -> substation", "transformer_id", "substation_id")):
        pairs = clean.groupby([a, b_])["households"].sum().reset_index()
        pairs = pairs[pairs[a].isin(nn.index)]
        if not len(pairs):
            continue
        pairs["r"] = [steps(int(x), int(y)) for x, y in zip(pairs[a], pairs[b_])]
        sh = pairs.groupby("r")["households"].sum()
        sh = (sh / sh.sum()).sort_values(ascending=False)
        print(f"  CAS parent chain reaches it, {name:26s}: " + "; ".join(f"{k} {v:.1%}" for k, v in sh.items()))
    if "coato_code" in nn:
        tc = clean.groupby(["transformer_id", "coato_code"])["households"].sum().reset_index()
        tc = tc[tc["transformer_id"].isin(nn.index)]
        if len(tc):
            cas_c = nn.loc[tc["transformer_id"], "coato_code"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            same = cas_c.to_numpy() == tc["coato_code"].to_numpy()
            print(f"  CAS coato_code of the transformer == its district in the data: "
                  f"{tc['households'][same].sum() / tc['households'].sum():.2%} of households")
    if {"latitude", "longitude"} <= set(nn.columns):
        la = pd.to_numeric(nn["latitude"], errors="coerce")
        lo = pd.to_numeric(nn["longitude"], errors="coerce")
        okc = la.between(37, 46) & lo.between(55, 74)                         # Uzbekistan's bounding box
        print(f"  CAS nodes with coordinates inside Uzbekistan's box: {okc.mean():.1%}")
        ts = clean.groupby(["coato_code", "transformer_id", "substation_id"])["households"].sum().reset_index()
        inside = set(okc.index[okc])
        ts = ts[ts["transformer_id"].isin(inside) & ts["substation_id"].isin(inside)]
        if len(ts):
            km = _hav_km(la[ts["transformer_id"]].to_numpy(), lo[ts["transformer_id"]].to_numpy(),
                         la[ts["substation_id"]].to_numpy(), lo[ts["substation_id"]].to_numpy())
            print(f"  distance transformer -> its substation ({len(ts):,} with coordinates): {_q(km)} km; "
                  f"> 30 km: {(km > 30).mean():.1%}")
            subs = ts.drop_duplicates(["coato_code", "substation_id"])
            ranks = []
            for code, g in ts.groupby("coato_code"):
                ss = subs.loc[subs["coato_code"] == code, "substation_id"].to_numpy()
                if len(ss) < 2:
                    continue
                D = _hav_km(la[g["transformer_id"]].to_numpy()[:, None], lo[g["transformer_id"]].to_numpy()[:, None],
                            la[ss].to_numpy()[None, :], lo[ss].to_numpy()[None, :])
                own = ss[None, :] == g["substation_id"].to_numpy()[:, None]
                ranks.append((D < D[own][:, None]).sum(1))                     # substations closer than its own
            if ranks:
                rk = np.concatenate(ranks)
                print(f"  its own substation is the nearest of the district's substations for {(rk == 0).mean():.1%} of "
                      f"transformers; among the 3 nearest for {(rk <= 2).mean():.1%}")


def region_report(reg_of, br):
    br = br.copy()
    clean = lambda s: s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)     # noqa: E731
    br["coato_code"], br["parent_coato_code"] = clean(br["coato_code"]), clean(br["parent_coato_code"])
    parent = br.set_index("coato_code")["parent_coato_code"].to_dict()
    d = pd.DataFrame({"coato_code": list(reg_of), "region_code": [str(v).strip() for v in reg_of.values()]})
    d["in_table"] = d["coato_code"].isin(parent)
    d["parent"] = d["coato_code"].map(parent)
    print(f"  districts found in the branch hierarchy: {d['in_table'].sum()} of {len(d)}")
    m = d[d["in_table"]]
    if len(m):
        eq = (m["parent"] == m["region_code"]).mean()
        x = m.groupby("parent")["region_code"].nunique()
        y = m.groupby("region_code")["parent"].nunique()
        print(f"  region_code == hierarchy parent: {eq:.1%} | hierarchy parents with > 1 region_code: {(x > 1).sum()} | "
              f"region_codes with > 1 hierarchy parent: {(y > 1).sum()}")
        if eq < 1:
            ex = m[m["parent"] != m["region_code"]][["coato_code", "region_code", "parent"]].head(6)
            print("  examples (coato, region_code, hierarchy parent): "
                  + ", ".join(f"({a}, {b}, {c})" for a, b, c in ex.itertuples(index=False)))


if __name__ == "__main__":
    main()
