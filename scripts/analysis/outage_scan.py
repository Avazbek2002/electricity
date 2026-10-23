#!/usr/bin/env python
"""
outage_scan.py - unrecorded supply outages from the household data: a dip that is SHARED by the households of one
transformer, feeder or substation, and not by their neighbours, is a network event, not demand.

New regime only (default from 2024-08-01: before it most readings come every 4-5 days in whole kWh, so a one-day dip is
invisible). Reads the enriched dataset v1.4 district by district (bounded reads, one district per worker process) and
returns aggregates of >= --min-hh observed households only; no customer key leaves a worker.

1. Household ratio      r = kWh(d) / median of the household's own observed kWh over the 28 days before d
                        (>= 8 of them, median >= --min-level kWh), divided by the district's median r that day, so weather,
                        holidays and weekdays (shared by the whole district) cancel. Only complete_observed days count.
2. Node-day panel       for every transformer, feeder and substation: households in span (S) and observed (O), the node's
                        kWh ratio sum(kWh) / sum(expected), the share of its households with r < --hh-dip, and the
                        reporting share O / the node's usual O (28-day median before d).
3. Flags                dip     O >= --min-hh, node ratio < --node-dip, >= --hh-share of its households dipping and
                                reporting share >= --min-reporting (else the few households left are not the node)
                        partial silent  the same dip with reporting share < --min-reporting (and not silent)
                        silent  usual O >= --min-hh and O < --silent x usual O (readings missing: a collection or
                                communication outage, not low demand)
4. Top-down assignment  for each day, an event is put at the HIGHEST level that explains it:
                          substation  it dips, >= --coherence of its eligible feeders dip, < --siblings of the other
                                      substations of the district dip
                          feeder      the same against its transformers and the other feeders of its substation
                          transformer it dips and < --siblings of the other transformers on its feeder dip
                        (a node with fewer than 2 eligible children passes the child test on its own dip). Nodes inside
                        an assigned parent event are not counted again; a dip that passes no test is 'unresolved'.
                        Days on which the district median ratio drops suddenly (< 0.8 x its 7-day centred median) are
                        listed as district-wide (holiday, data system or a district outage): they cancel inside the
                        district, so they need the national view.
5. Classification       each assigned run of dip days becomes one event:
                          level_shift    >= --shift-days days long (a vacancy, meter swap or mapping change, not an outage)
                          clustered_dip  touches a clustered day: >= --cluster-share of the district's eligible
                                         transformers (and >= --cluster-min) dip that day, spread over >= --cluster-spread
                                         of its substations (so one substation's outage is not a cluster); holiday /
                                         Ramadan flags joined
                          supply_dip     the rest, in depth tiers: shallow (< --deep-depth) and deep
                        A node with >= --chronic supply dips is flagged chronic (unreliable supply or a volatile node:
                        review, do not clean automatically).
6. Shuffle control      households re-assigned at random to the district's transformers (sizes kept, feeder and
                        substation follow the transformer): no physics left, so the event rate there is the false-alarm
                        floor. Lift = real rate / shuffled rate, per type, level and depth tier; a supply-dip tier with
                        lift >= --usable-lift is marked usable for cleaning.
7. Synthetic injection  known cuts planted in the real data (transformers and feeders; depth 30 % or 60 %, 1 or 2 days, on
                        node-days without a real event), then the whole scan again: recall at the right level and at any
                        level.

Only households with a full, non-placeholder path (transformer, feeder, substation >= 0; transformer not a > 1,000-customer
placeholder) are used, the same set in every run. A household is put under its transformer's most common feeder and a
feeder under its most common substation, so the nodes form a tree (the count re-parented is reported). A district
where fewer than --min-path-share of the households have such a path is skipped (its mapping is mostly placeholders).

Outputs (<out-dir>):
  events.csv              one row per event: coato_code, level, node_id, parent ids, type (supply_dip / clustered_dip /
                          level_shift / partial_silent / silent), tier, start, end, days, depth (1 - mean node ratio),
                          min_ratio, lost_kwh_observed, lost_kwh_est (scaled to the households in span, at most
                          x --max-scale; blank for silent types), households_observed, reporting_share,
                          children_dipping, siblings_dipping, rebound (ratio >= 1.15 the day after), clustered_share,
                          holiday, ramadan, chronic, node_supply_dips
  node_day.parquet        the node-day panel (eligible node-days only)
  district_days.csv       district median ratio, district-wide flag, share of transformers dipping, clustered flag,
                          holiday name, Ramadan
  control.csv             per district, type, level and tier: eligible node-days, events, node-days in events (real,
                          shuffled)
  injection.csv           one row per planted cut: level, node, depth, days, detected at the same level / at any level
  outage_report.txt       the printed summary
  figures/                events_by_month.png, lift.png, injection_recall.png, top_events.png

  python scripts/analysis/outage_scan.py --out-dir ~/work/outage_scan                     # the 5 sample districts
  python scripts/analysis/outage_scan.py --out-dir ~/work/outage_scan_nat --all --workers 12
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

HERE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core"))
sys.path.insert(0, HERE)
MISSING = -(2 ** 62)
SAMPLE = ["03203", "10240", "22202", "26266", "33220"]          # configs/test_set_2025.yaml sample_districts
LEVELS = ("substation", "feeder", "transformer")                # top-down order
DEFAULTS = dict(min_level=1.0, base_days=28, base_min=8, hh_dip=0.8, node_dip=0.75, hh_share=0.5, min_hh=10,
                silent=0.3, coherence=0.6, siblings=0.3, district_dip=0.8, rebound=1.15, min_reporting=0.7,
                deep_depth=0.45, shift_days=8, cluster_share=0.05, cluster_min=5, cluster_spread=0.5, chronic=5,
                max_scale=2.0, usable_lift=5.0, min_path_share=0.5)
SUPPLY_TYPES = ("supply_dip", "clustered_dip", "level_shift")
SILENT_TYPES = ("partial_silent", "silent")
TIERS = ("shallow", "deep")


# ---------------------------------------------------------------------------------------------------- reading
def read_district(code, cfg):
    """-> K (H, T) float32 observed kWh (NaN elsewhere), span (H, T) bool (a row exists), path (H, 3) int64
    [transformer, feeder, substation] (last seen; MISSING if unknown), ph (H,) bool placeholder transformer, dates."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as pads
    import enriched_io as eio
    fs, root = eio.open_root(cfg["root"])
    d0, d1 = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    dates = pd.date_range(d0, d1, freq="D")
    T = len(dates)
    e0 = int((d0 - pd.Timestamp("1970-01-01")).days)
    dset = eio.district_daily(fs, root, code)
    want = ["customer_key", "date", "transformer_id", "feeder_id", "substation_id", "placeholder_transformer",
            "label_state", "kwh", "is_public_holiday", "is_extra_day_off", "is_ramadan", "holiday_name"]
    cols = [c for c in want if c in dset.schema.names]
    flt = (pads.field("date") >= d0.date()) & (pads.field("date") <= d1.date())
    labels = pa.array(list(cfg["labels"]), type=pa.string())
    key_of = {}
    cap = 1024
    K = np.full((cap, T), np.nan, np.float32)
    span = np.zeros((cap, T), bool)
    path = np.full((cap, 3), MISSING, np.int64)
    last = np.full(cap, -1, np.int64)
    ph = np.zeros(cap, bool)
    hol, ram = np.zeros(T, bool), np.zeros(T, bool)
    hname = np.full(T, "", dtype=object)
    for b in dset.scanner(columns=cols, filter=flt, batch_size=262_144, use_threads=True).to_batches():
        n = b.num_rows
        if not n:
            continue

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
        if len(key_of) > K.shape[0]:
            grow = max(len(key_of), 2 * K.shape[0]) - K.shape[0]
            K = np.vstack([K, np.full((grow, T), np.nan, np.float32)])
            span = np.vstack([span, np.zeros((grow, T), bool)])
            path = np.vstack([path, np.full((grow, 3), MISSING, np.int64)])
            last = np.concatenate([last, np.full(grow, -1, np.int64)])
            ph = np.concatenate([ph, np.zeros(grow, bool)])
        day = pc.cast(pc.cast(b.column("date"), pa.date32()), pa.int32()).to_numpy(zero_copy_only=False).astype(np.int64) - e0
        span[cust, day] = True
        for col, arr in (("is_public_holiday", hol), ("is_extra_day_off", hol), ("is_ramadan", ram)):
            if col in cols:
                v = pc.fill_null(b.column(col), False).to_numpy(zero_copy_only=False)
                arr[np.unique(day[v])] = True
        if "holiday_name" in cols:
            hn = b.column("holiday_name")
            v = pc.is_valid(hn).to_numpy(zero_copy_only=False)
            if v.any():
                for dd, name in zip(*np.unique(np.c_[day[v], np.asarray(hn.to_pylist(), dtype=object)[v]].astype(str),
                                               axis=0).T):
                    hname[int(dd)] = name
        kv = b.column("kwh")
        ok = pc.and_(pc.fill_null(pc.is_in(b.column("label_state"), value_set=labels), False),
                     pc.is_valid(kv)).to_numpy(zero_copy_only=False)
        kw = pc.fill_null(pc.cast(kv, pa.float64()), np.nan).to_numpy(zero_copy_only=False)
        K[cust[ok], day[ok]] = kw[ok]
        # last-seen path per customer
        tr, fe, su = ints("transformer_id"), ints("feeder_id"), ints("substation_id")
        pv = (pc.fill_null(b.column("placeholder_transformer"), False).to_numpy(zero_copy_only=False)
              if "placeholder_transformer" in cols else np.zeros(n, bool))
        o = np.lexsort((day, cust))
        keep = np.r_[cust[o][1:] != cust[o][:-1], True]                  # the latest row of each customer in the batch
        i = o[keep]
        newer = day[i] > last[cust[i]]
        i = i[newer]
        c = cust[i]
        last[c] = day[i]
        path[c, 0], path[c, 1], path[c, 2] = tr[i], fe[i], su[i]
        ph[c] = pv[i]
    H = len(key_of)
    cal = pd.DataFrame(dict(date=dates, holiday=hol, ramadan=ram, holiday_name=hname))
    return K[:H], span[:H], path[:H], ph[:H], dates, cal


# ---------------------------------------------------------------------------------------------------- the tree
def build_tree(path, ph):
    """Households with a full non-placeholder path -> (use mask, household transformer index, t_par (nT,) feeder index,
    f_par (nF,) substation index, ids per level, re-parented household count)."""
    use = (path >= 0).all(axis=1) & ~ph
    p = path[use]
    t_ids, ht = np.unique(p[:, 0], return_inverse=True)
    # transformer -> its most common feeder; feeder -> its most common substation
    tf = pd.DataFrame({"t": ht, "f": p[:, 1]}).value_counts().reset_index().drop_duplicates("t").sort_values("t")
    f_of_t = tf["f"].to_numpy()
    f_ids, t_par = np.unique(f_of_t, return_inverse=True)
    hf = t_par[ht]                                                  # every feeder has households this way
    fs = pd.DataFrame({"f": hf, "s": p[:, 2]}).value_counts().reset_index().drop_duplicates("f").sort_values("f")
    s_ids, f_par = np.unique(fs["s"].to_numpy(), return_inverse=True)
    moved = int(((f_ids[t_par[ht]] != p[:, 1]) | (s_ids[f_par[t_par[ht]]] != p[:, 2])).sum())
    return dict(use=use, ht=ht.astype(np.int64), t_par=t_par.astype(np.int64), f_par=f_par.astype(np.int64),
                ids=dict(transformer=t_ids, feeder=f_ids, substation=s_ids), moved=moved)


# ---------------------------------------------------------------------------------------------------- the scan
def ratios(K, cfg):
    """-> r_adj (H, T) district-normalised ratio (NaN where not usable), exp_adj (H, T) expected kWh, m (T,) the
    district median ratio."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base = (pd.DataFrame(K.T).shift(1).rolling(cfg["base_days"], min_periods=cfg["base_min"]).median()
                .to_numpy(np.float32).T)
        ok = np.isfinite(K) & np.isfinite(base) & (base >= cfg["min_level"])
        r = np.where(ok, K / np.where(ok, base, 1.0), np.nan).astype(np.float32)
        n = ok.sum(0)
        m = np.where(n >= 30, np.nanmedian(r, axis=0), np.nan)
    m = np.where(np.isfinite(m) & (m > 0), m, 1.0).astype(np.float32)
    return r / m[None, :], np.where(ok, base * m[None, :], np.nan).astype(np.float32), m


def _onehot(idx, n):
    from scipy import sparse
    H = len(idx)
    return sparse.csr_matrix((np.ones(H, np.float32), (idx, np.arange(H))), shape=(n, H))


def node_panel(K, span, r, exp, hidx, n, cfg):
    """Sums per node and day for households with node index hidx -> dict of (n, T) arrays."""
    A = _onehot(hidx, n)
    ok = np.isfinite(r)
    O = np.asarray(A @ ok.astype(np.float32))
    S = np.asarray(A @ span.astype(np.float32))
    sk = np.asarray(A @ np.where(ok, K, 0).astype(np.float32))
    se = np.asarray(A @ np.where(ok, exp, 0).astype(np.float32))
    nd = np.asarray(A @ (ok & (r < cfg["hh_dip"])).astype(np.float32))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        usual = pd.DataFrame(O.T).shift(1).rolling(cfg["base_days"], min_periods=cfg["base_min"]).median().to_numpy().T
        ratio = np.where(se > 0, sk / np.maximum(se, 1e-9), np.nan)
        share = np.where(O > 0, nd / np.maximum(O, 1), np.nan)
        rep = np.where(usual > 0, O / np.maximum(usual, 1e-9), np.nan)
    elig = O >= cfg["min_hh"]
    raw = elig & (ratio < cfg["node_dip"]) & (share >= cfg["hh_share"])
    repok = ~(rep < cfg["min_reporting"])                            # NaN (no usual count yet) passes
    selig = np.nan_to_num(usual) >= cfg["min_hh"]
    silent = selig & (O < cfg["silent"] * np.nan_to_num(usual))
    return dict(O=O, S=S, sk=sk, se=se, ratio=ratio, share=share, rep=rep, elig=elig, dip=raw & repok, selig=selig,
                silent=silent, psil=raw & ~repok & ~silent)


def _group_sum(x, par, n_par):
    out = np.zeros((n_par, x.shape[1]), np.int64)
    np.add.at(out, par, x.astype(np.int64))
    return out


def attribute(flag, elig, t_par, f_par, cfg):
    """Top-down assignment for one flag type. flag / elig: dicts level -> (n, T) bool. -> dict level -> (att, unresolved,
    children_share, siblings_share), each (n, T)."""
    nS = flag["substation"].shape[0]
    pars = dict(transformer=t_par, feeder=f_par, substation=np.zeros(nS, np.int64))
    child = dict(substation="feeder", feeder="transformer")
    out, cov = {}, {}
    for lv in LEVELS:
        fl, el = flag[lv], elig[lv]
        par = pars[lv]
        n_par = int(par.max()) + 1 if len(par) else 0
        E, D = _group_sum(el, par, n_par), _group_sum(fl & el, par, n_par)
        sE, sD = E[par] - el, D[par] - (fl & el)                    # siblings under the same parent
        with np.errstate(invalid="ignore", divide="ignore"):
            sib = np.where(sE >= 1, sD / np.maximum(sE, 1), np.nan)
            if lv in child:
                c = child[lv]
                cpar = pars[c]
                cE, cD = _group_sum(elig[c], cpar, fl.shape[0]), _group_sum(flag[c] & elig[c], cpar, fl.shape[0])
                coh = np.where(cE >= 2, cD / np.maximum(cE, 1), np.nan)
            else:
                coh = np.full(fl.shape, np.nan)
        ok_child = ~(coh < cfg["coherence"])                        # NaN (too few children) passes
        ok_sib = ~(sib >= cfg["siblings"])                          # NaN (no eligible sibling) passes
        covered = cov.get(lv, np.zeros(fl.shape, bool))
        att = fl & el & ok_child & ok_sib & ~covered
        unres = fl & el & ~att & ~covered
        out[lv] = (att, unres, coh, sib)
        if lv in child:                                             # children inside an assigned event are covered
            c = child[lv]
            cov[c] = (att | covered)[pars[c]]
    return out


def _runs(x):
    """(start, end exclusive) of the True runs of a 1-D bool array."""
    d = np.diff(np.r_[0, x.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def classify(res, cfg):
    """Every assigned run of event days -> dict(level, k, s, e, type, tier, depth)."""
    out = []
    clus = res["clustered"]
    for kind, src in (("dip", "supply_dip"), ("psil", "partial_silent"), ("silent", "silent")):
        att = res["att"][src]
        for lv in LEVELS:
            a = att[lv][0].copy()
            el = res["panels"][lv]["selig" if src == "silent" else "elig"]
            # bridge one ineligible day inside an event (too few households observed that day): one event, not two
            gap = ~a[:, 1:-1] & ~el[:, 1:-1] & a[:, :-2] & a[:, 2:]
            a[:, 1:-1] |= gap
            ratio = res["panels"][lv]["ratio"]
            for k in np.flatnonzero(a.any(axis=1)):
                for s, e in _runs(a[k]):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        depth = float(1 - np.nanmean(ratio[k, s:e]))
                    if kind == "dip":
                        tier = "deep" if depth >= cfg["deep_depth"] else "shallow"
                        typ = ("level_shift" if e - s >= cfg["shift_days"] else
                               "clustered_dip" if clus[s:e].any() else "supply_dip")
                    else:
                        tier, typ = "all", src
                    out.append(dict(level=lv, k=int(k), s=int(s), e=int(e), type=typ, tier=tier, depth=depth))
    return out


def events_table(code, runs, res, tree, dates, cal, cfg):
    with warnings.catch_warnings():                     # silent events have no observed kWh: NaN depth is expected
        warnings.simplefilter("ignore", RuntimeWarning)
        return _events_table(code, runs, res, tree, dates, cal, cfg)


def _events_table(code, runs, res, tree, dates, cal, cfg):
    rows = []
    ids = tree["ids"]
    src = dict(supply_dip="supply_dip", clustered_dip="supply_dip", level_shift="supply_dip",
               partial_silent="partial_silent", silent="silent")
    for x in runs:
        lv, k, s, e = x["level"], x["k"], x["s"], x["e"]
        _, _, coh, sib = res["att"][src[x["type"]]][lv]
        P = res["panels"][lv]
        sl = slice(s, e)
        O, S = P["O"][k, sl], P["S"][k, sl]
        supply = x["type"] in SUPPLY_TYPES
        lost = float(np.nansum(P["se"][k, sl] - P["sk"][k, sl])) if supply else np.nan
        scale = min(float(np.nanmean(S) / max(np.nanmean(O), 1)), cfg["max_scale"])
        nxt = P["ratio"][k, e] if e < len(dates) else np.nan
        c0, c1 = max(s - 1, 0), min(e + 1, len(dates))                       # the event and the day on each side
        names = sorted({n for n in cal["holiday_name"].iloc[c0:c1] if n})
        row = dict(coato_code=str(code), level=lv, node_id=int(ids[lv][k]), type=x["type"], tier=x["tier"],
                   start=dates[s].date(), end=dates[e - 1].date(), days=int(e - s),
                   depth=x["depth"] if supply else np.nan,
                   min_ratio=float(np.nanmin(P["ratio"][k, sl])) if supply else np.nan,
                   lost_kwh_observed=lost, lost_kwh_est=lost * scale if supply else np.nan,
                   households_observed=float(np.nanmean(O)), households_in_span=float(np.nanmean(S)),
                   reporting_share=float(np.nanmean(P["rep"][k, sl])),
                   children_dipping=float(np.nanmean(coh[k, sl])) if np.isfinite(coh[k, sl]).any() else np.nan,
                   siblings_dipping=float(np.nanmean(sib[k, sl])) if np.isfinite(sib[k, sl]).any() else np.nan,
                   rebound=bool(supply and np.isfinite(nxt) and nxt >= cfg["rebound"]),
                   clustered_share=float(np.nanmax(res["cluster_share"][sl])),
                   holiday="; ".join(names) if names else ("holiday" if cal["holiday"].iloc[c0:c1].any() else ""),
                   ramadan=bool(cal["ramadan"].iloc[sl].any()))
        if lv == "transformer":
            row["feeder_id"] = int(ids["feeder"][tree["t_par"][k]])
            row["substation_id"] = int(ids["substation"][tree["f_par"][tree["t_par"][k]]])
        elif lv == "feeder":
            row["substation_id"] = int(ids["substation"][tree["f_par"][k]])
        rows.append(row)
    ev = pd.DataFrame(rows)
    if len(ev):
        n = ev[ev["type"] == "supply_dip"].groupby(["level", "node_id"]).size().rename("node_supply_dips")
        ev = ev.merge(n.reset_index(), on=["level", "node_id"], how="left")
        ev["node_supply_dips"] = ev["node_supply_dips"].fillna(0).astype(int)
        ev["chronic"] = ev["node_supply_dips"] >= cfg["chronic"]
    return ev.to_dict("records")


def scan(K, span, ht, tree, dates, cfg, rat=None):
    """One full pass for households assigned to transformers ht -> dict(panels, att (supply, silent), m, rat).
    rat = ratios(K) from an earlier pass over the same K (the shuffle control re-uses it)."""
    r, exp, m = rat if rat is not None else ratios(K, cfg)
    hidx = dict(transformer=ht, feeder=tree["t_par"][ht], substation=tree["f_par"][tree["t_par"][ht]])
    n = dict(transformer=len(tree["t_par"]), feeder=len(tree["f_par"]), substation=int(tree["f_par"].max()) + 1)
    panels = {lv: node_panel(K, span, r, exp, hidx[lv], n[lv], cfg) for lv in LEVELS}
    # district-wide days: the district's median ratio falls suddenly (vs the 7 days around it). Slow weather swings are
    # not flagged; inside the district they cancel anyway, because every household ratio is divided by that median.
    ms = pd.Series(m).rolling(7, center=True, min_periods=4).median().to_numpy()
    dist_bad = m < cfg["district_dip"] * ms
    att = dict(supply_dip=attribute({lv: panels[lv]["dip"] for lv in LEVELS}, {lv: panels[lv]["elig"] for lv in LEVELS},
                                    tree["t_par"], tree["f_par"], cfg),
               partial_silent=attribute({lv: panels[lv]["psil"] for lv in LEVELS},
                                        {lv: panels[lv]["elig"] for lv in LEVELS}, tree["t_par"], tree["f_par"], cfg),
               silent=attribute({lv: panels[lv]["silent"] for lv in LEVELS}, {lv: panels[lv]["selig"] for lv in LEVELS},
                                tree["t_par"], tree["f_par"], cfg))
    # clustered days: many of the district's transformers dip at once (holiday travel, a heat or cold day felt unevenly)
    Pt = panels["transformer"]
    nd, ne = Pt["dip"].sum(0), Pt["elig"].sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cshare = np.where(ne > 0, nd / np.maximum(ne, 1), 0.0)
    s_of_t = tree["f_par"][tree["t_par"]]
    nS = int(tree["f_par"].max()) + 1
    has_d = _group_sum(Pt["dip"], s_of_t, nS) > 0
    has_e = _group_sum(Pt["elig"], s_of_t, nS) > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = np.where(has_e.sum(0) > 0, (has_d & has_e).sum(0) / np.maximum(has_e.sum(0), 1), 0.0)
    # spread over the district (not one substation's outage pulling many transformers down at once)
    clustered = (cshare >= cfg["cluster_share"]) & (nd >= cfg["cluster_min"]) & (spread >= cfg["cluster_spread"])
    return dict(panels=panels, att=att, m=m, dist_bad=dist_bad, rat=(r, exp, m), cluster_share=cshare,
                clustered=clustered)


def counts(code, res, run, cfg):
    runs = pd.DataFrame(classify(res, cfg))
    rows = []
    combos = [(ty, ti) for ty in SUPPLY_TYPES for ti in TIERS] + [(ty, "all") for ty in SILENT_TYPES]
    for ty, ti in combos:
        for lv in LEVELS:
            P = res["panels"][lv]
            el = P["selig"] if ty == "silent" else P["elig"]
            q = runs[(runs["type"] == ty) & (runs["tier"] == ti) & (runs["level"] == lv)] if len(runs) else runs
            rows.append(dict(coato_code=str(code), run=run, type=ty, tier=ti, level=lv,
                             eligible_node_days=int(el.sum()), events=int(len(q)),
                             event_node_days=int((q["e"] - q["s"]).sum()) if len(q) else 0))
    return rows


# ---------------------------------------------------------------------------------------------------- injection
def inject(K, ht, tree, real, cfg, rng):
    """Plant cuts in a copy of K on node-days without a real event. -> (K', list of injections)."""
    Ki = K.copy()
    T = K.shape[1]
    P = real["panels"]
    busy = {lv: real["att"]["supply_dip"][lv][0] | real["att"]["supply_dip"][lv][1] | P[lv]["dip"] for lv in LEVELS}
    hf = tree["t_par"][ht]
    out = []
    for lv, n_inj in (("transformer", cfg["inject_transformers"]), ("feeder", cfg["inject_feeders"])):
        good = np.flatnonzero(P[lv]["elig"].mean(axis=1) >= 0.5)
        if not len(good):
            continue
        for j in range(n_inj):
            k = int(rng.choice(good))
            depth, days = (0.3, 0.6)[j % 2], (1, 2)[(j // 2) % 2]
            for _ in range(20):
                s = int(rng.integers(cfg["base_days"] + 10, T - days - 1))
                if P[lv]["elig"][k, s:s + days].all() and not busy[lv][k, s - 1:s + days + 1].any():
                    break
            else:
                continue
            hh = np.flatnonzero((ht if lv == "transformer" else hf) == k)
            Ki[np.ix_(hh, np.arange(s, s + days))] *= (1 - depth)
            busy[lv][k, s - 1:s + days + 1] = True
            out.append(dict(level=lv, node=k, start=s, days=days, depth=depth))
    return Ki, out


def score_injections(inj, res, tree):
    A = {lv: res["att"]["supply_dip"][lv][0] for lv in LEVELS}
    U = {lv: res["att"]["supply_dip"][lv][1] for lv in LEVELS}
    rows = []
    for x in inj:
        k, sl = x["node"], slice(x["start"], x["start"] + x["days"])
        if x["level"] == "transformer":
            f = tree["t_par"][k]
            s = tree["f_par"][f]
            same = A["transformer"][k, sl].any()
            anyl = same or A["feeder"][f, sl].any() or A["substation"][s, sl].any() or U["transformer"][k, sl].any()
        else:
            s = tree["f_par"][k]
            kids = np.flatnonzero(tree["t_par"] == k)
            same = A["feeder"][k, sl].any()
            anyl = (same or A["substation"][s, sl].any() or U["feeder"][k, sl].any()
                    or (len(kids) and A["transformer"][kids, sl].any()))
        rows.append(dict(x, detected_same_level=bool(same), detected_any_level=bool(anyl)))
    return rows


# ---------------------------------------------------------------------------------------------------- worker
def district_worker(code, cfg):
    sys.path.insert(0, cfg["here"])
    t0 = time.time()
    rng = np.random.default_rng([cfg["seed"], int(code) if str(code).isdigit() else 0])
    K, span, path, ph, dates, cal = read_district(code, cfg)
    H0 = len(K)
    tree = build_tree(path, ph)
    use = tree["use"]
    K, span = K[use], span[use]
    info = dict(coato_code=str(code), households_read=H0, households_used=int(use.sum()),
                households_reparented=tree["moved"], **{f"{lv}s": len(tree["ids"][lv]) for lv in LEVELS})
    share = use.sum() / max(H0, 1)
    if use.sum() < 50 or share < cfg["min_path_share"]:
        info["skipped"] = (f"only {share:.0%} of households have a usable network path "
                           f"(< {cfg['min_path_share']:.0%}: mostly placeholder transformers)")
        return dict(info=info, events=[], control=[], injection=[], node_day=None, district=None,
                    seconds=time.time() - t0)
    info["skipped"] = ""
    ht = tree["ht"]
    real = scan(K, span, ht, tree, dates, cfg)
    events = events_table(code, classify(real, cfg), real, tree, dates, cal, cfg)
    control = counts(code, real, "real", cfg)
    for i in range(cfg["shuffles"]):
        sh = scan(K, span, rng.permutation(ht), tree, dates, cfg, rat=real["rat"])
        control += counts(code, sh, f"shuffled_{i}", cfg)
        del sh
    real.pop("rat")
    Ki, inj = inject(K, ht, tree, real, cfg, rng)
    injection = []
    if inj:
        ri = scan(Ki, span, ht, tree, dates, cfg)
        injection = [dict(coato_code=str(code), node_id=int(tree["ids"][x["level"]][x["node"]]),
                          start=dates[x["start"]].date(), **{k: v for k, v in x.items() if k not in ("node", "start")})
                     for x in score_injections(inj, ri, tree)]
        del ri, Ki
    # node-day panel, eligible node-days only (>= min_hh observed households)
    frames = []
    for lv in LEVELS:
        P = real["panels"][lv]
        k, d = np.nonzero(P["elig"] | P["selig"])
        keep = P["O"][k, d] >= cfg["min_hh"]
        k, d = k[keep], d[keep]
        par = (tree["t_par"] if lv == "transformer" else tree["f_par"] if lv == "feeder"
               else np.zeros(len(tree["ids"]["substation"]), np.int64))
        pid = (tree["ids"]["feeder"] if lv == "transformer" else tree["ids"]["substation"] if lv == "feeder"
               else np.array([int(code) if str(code).isdigit() else 0]))
        a_sup, a_sil = real["att"]["supply_dip"][lv][0], real["att"]["silent"][lv][0]
        a_ps = real["att"]["partial_silent"][lv][0]
        frames.append(pd.DataFrame(dict(
            coato_code=str(code), level=lv, node_id=tree["ids"][lv][k], parent_id=pid[par[k]], date=dates[d],
            households_in_span=P["S"][k, d].astype(np.int32), households_observed=P["O"][k, d].astype(np.int32),
            ratio=P["ratio"][k, d].astype(np.float32), dipping_share=P["share"][k, d].astype(np.float32),
            reporting_share=P["rep"][k, d].astype(np.float32), kwh_observed=P["sk"][k, d],
            kwh_expected=P["se"][k, d], supply_event=a_sup[k, d], partial_silent_event=a_ps[k, d],
            silent_event=a_sil[k, d])))
    district = pd.DataFrame(dict(coato_code=str(code), date=dates, district_ratio=real["m"],
                                 district_wide=real["dist_bad"], transformers_dipping=real["cluster_share"],
                                 clustered=real["clustered"], holiday=cal["holiday"].to_numpy(),
                                 holiday_name=cal["holiday_name"].to_numpy(), ramadan=cal["ramadan"].to_numpy()))
    return dict(info=info, events=events, control=control, injection=injection, node_day=pd.concat(frames),
                district=district, seconds=time.time() - t0)


# ---------------------------------------------------------------------------------------------------- report
def lift_table(ctl, usable_lift):
    """Events per 1,000 eligible node-days, real vs shuffled, per type, level and tier."""
    if not len(ctl):
        return pd.DataFrame()
    c = ctl.copy()
    c["kind"] = np.where(c["run"] == "real", "real", "shuffled")
    g = c.groupby(["type", "tier", "level", "kind"])[["eligible_node_days", "events"]].sum().reset_index()
    g["rate"] = 1000 * g["events"] / g["eligible_node_days"].clip(lower=1)
    w = g.pivot_table(index=["type", "tier", "level"], columns="kind", values=["rate", "events"]).reset_index()
    w.columns = ["_".join(x).strip("_") for x in w.columns]
    for col in ("rate_shuffled", "events_shuffled"):
        if col not in w:
            w[col] = 0.0
    w["lift"] = w["rate_real"] / w["rate_shuffled"].replace(0, np.nan)
    w["usable"] = (w["type"] == "supply_dip") & ((w["lift"] >= usable_lift) |
                                                 (w["rate_shuffled"].eq(0) & (w["events_real"] >= 10)))
    order = {t: i for i, t in enumerate(SUPPLY_TYPES + SILENT_TYPES)}
    return w.sort_values(["type", "tier", "level"], key=lambda s: s.map(order) if s.name == "type" else s)


def report(info, ev, ctl, inj, dd, cfg):
    print("OUTAGE SCAN v2: shared dips by network level (new regime, district-normalised household ratios)\n")
    used = info[info["skipped"] == ""] if "skipped" in info else info
    print(f"districts {len(used)} scanned | households used {used['households_used'].sum():,} of "
          f"{used['households_read'].sum():,} read (full non-placeholder path) | transformers "
          f"{used['transformers'].sum():,}, feeders {used['feeders'].sum():,}, substations {used['substations'].sum():,}")
    for x in info[info.get("skipped", pd.Series("", index=info.index)) != ""].itertuples():
        print(f"SKIPPED {x.coato_code}: {x.skipped} ({x.households_used:,} of {x.households_read:,})")
    if dd is not None and len(dd):
        bad = dd[dd["district_wide"]]
        print(f"district-wide days (district median ratio < {cfg['district_dip']} x its 7-day median): {len(bad)} "
              "district-days" + (": " + ", ".join(f"{r.coato_code} {r.date:%Y-%m-%d} ({r.district_ratio:.2f})"
                                                   for r in bad.sort_values("district_ratio").head(8).itertuples())
                                 if len(bad) else ""))
        cl = dd[dd["clustered"]]
        print(f"clustered days (>= {cfg['cluster_share']:.0%} of the district's transformers dipping): {len(cl)} "
              f"district-days; on a holiday or the day around one: "
              f"{int((cl['holiday'] | cl['holiday_name'].astype(bool)).sum())}; in Ramadan: {int(cl['ramadan'].sum())}")
        if len(cl):
            print("  most clustered: " + ", ".join(
                f"{r.coato_code} {r.date:%Y-%m-%d} ({r.transformers_dipping:.0%}{', ' + r.holiday_name if r.holiday_name else ''}"
                f"{', Ramadan' if r.ramadan else ''})" for r in cl.sort_values("transformers_dipping", ascending=False)
                .head(10).itertuples()))
    print(f"\n1. REAL vs SHUFFLED per type, depth tier and level (events per 1,000 eligible node-days; lift = real / "
          f"shuffled; usable = supply_dip tier with lift >= {cfg['usable_lift']:g})")
    w = lift_table(ctl, cfg["usable_lift"])
    if len(w):
        print(w[["type", "tier", "level", "events_real", "events_shuffled", "rate_real", "rate_shuffled", "lift",
                 "usable"]].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
        if "coato_code" in ctl:
            c = ctl[ctl["type"] == "supply_dip"].copy()
            c["kind"] = np.where(c["run"] == "real", "real", "shuffled")
            g = c.groupby(["coato_code", "tier", "kind"])[["events", "eligible_node_days"]].sum()
            r = (1000 * g["events"] / g["eligible_node_days"].clip(lower=1)).unstack()
            if {"real", "shuffled"} <= set(r.columns):
                r["lift"] = r["real"] / r["shuffled"].replace(0, np.nan)
                print("\n  supply_dip lift by district and tier (all levels):")
                print(r["lift"].unstack().to_string(float_format=lambda v: f"{v:.1f}"))
    print("\n2. SYNTHETIC INJECTION (planted cuts found)")
    if len(inj):
        r = inj.groupby(["level", "depth", "days"]).agg(planted=("detected_same_level", "size"),
                                                        same_level=("detected_same_level", "mean"),
                                                        any_level=("detected_any_level", "mean")).reset_index()
        print(r.to_string(index=False, float_format=lambda v: f"{v:.0%}" if v <= 1 else f"{v}"))
    else:
        print("  none planted")
    print("\n3. EVENTS (real)")
    if len(ev):
        s = ev.groupby(["type", "tier", "level"]).agg(events=("days", "size"), node_days=("days", "sum"),
                                                     median_days=("days", "median"), median_depth=("depth", "median"),
                                                     lost_kwh_est=("lost_kwh_est", "sum"), rebound=("rebound", "mean"),
                                                     chronic=("chronic", "mean")).reset_index()
        print(s.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
        sd = ev[ev["type"] == "supply_dip"]
        nodes = sd.groupby(["coato_code", "level", "node_id"]).size()
        print(f"\n  chronic nodes (>= {cfg['chronic']} supply dips): {(nodes >= cfg['chronic']).sum()} of {len(nodes)} "
              f"nodes with a supply dip, holding {nodes[nodes >= cfg['chronic']].sum() / max(nodes.sum(), 1):.0%} of them")
        top = sd[sd["tier"] == "deep"].sort_values("lost_kwh_est", ascending=False).head(12)
        print("\n  largest DEEP supply dips by estimated kWh lost:")
        for x in top.itertuples():
            sib = f"{x.siblings_dipping:.0%}" if np.isfinite(x.siblings_dipping) else "n/a"
            print(f"    {x.coato_code} {x.level:11s} {x.node_id:>14} {x.start}..{x.end} ({x.days} d) depth {x.depth:.0%} "
                  f"| {x.households_observed:,.0f} hh observed | lost ~{x.lost_kwh_est:,.0f} kWh | reporting "
                  f"{x.reporting_share:.0%} | siblings dipping {sib}" + (" | rebound" if x.rebound else "")
                  + (" | CHRONIC" if x.chronic else "") + (f" | {x.holiday}" if x.holiday else ""))
    else:
        print("  none")
    print("\nReading: use for cleaning only the supply_dip tiers marked usable (lift well above 1 means the dips follow "
          "the network). clustered_dip = many transformers at once (holiday travel, uneven weather): demand, not supply. "
          "level_shift = a lasting change. partial_silent / silent = readings missing: mask as missing, never as low "
          "demand. Chronic nodes: review before cleaning.")


def charts(ev, ctl, inj, nd, out, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    col = {"substation": "#7c3aed", "feeder": "#2563eb", "transformer": "#93c5fd"}
    tcol = {"supply_dip": "#2563eb", "clustered_dip": "#f59e0b", "level_shift": "#9ca3af",
            "partial_silent": "#fca5a5", "silent": "#dc2626"}
    if len(ev):
        e = ev.copy()
        e["month"] = pd.to_datetime(e["start"]).dt.to_period("M").astype(str)
        t = e.pivot_table(index="month", columns="type", values="days", aggfunc="size", fill_value=0)
        t = t.reindex(columns=[c for c in SUPPLY_TYPES + SILENT_TYPES if c in t.columns])
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        sup = [c for c in SUPPLY_TYPES if c in t.columns]
        sil = [c for c in SILENT_TYPES if c in t.columns]
        if sup:
            t[sup].plot(kind="bar", stacked=True, ax=axes[0], color=[tcol[c] for c in sup])
        axes[0].set_title("Supply-side events by month")
        if sil:
            t[sil].plot(kind="bar", stacked=True, ax=axes[1], color=[tcol[c] for c in sil])
        axes[1].set_title("Missing-reading events by month")
        for ax in axes:
            ax.set_ylabel("events")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "events_by_month.png"), dpi=120)
        plt.close()
    w = lift_table(ctl, cfg["usable_lift"])
    if len(w):
        q = w[w["type"].isin(["supply_dip", "silent", "partial_silent"])].copy()
        q["case"] = q["type"] + " " + q["tier"].where(q["tier"] != "all", "") + " | " + q["level"]
        q = q.set_index("case")["lift"].fillna(0)
        fig, ax = plt.subplots(figsize=(9, 0.35 * len(q) + 1.5))
        ax.barh(q.index, q.to_numpy(), color=["#2563eb" if v >= cfg["usable_lift"] else "#9ca3af" for v in q.to_numpy()])
        ax.axvline(cfg["usable_lift"], color="#dc2626", ls="--", lw=1)
        ax.axvline(1, color="#6b7280", lw=0.8)
        ax.set_xlabel("lift = real event rate / shuffled event rate (blue: >= usable threshold)")
        ax.set_title("Do the events follow the network?")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "lift.png"), dpi=120)
        plt.close()
    if len(inj):
        r = inj.groupby(["level", "depth", "days"])[["detected_same_level", "detected_any_level"]].mean().reset_index()
        r["case"] = r["level"] + " " + (100 * r["depth"]).astype(int).astype(str) + "%, " + r["days"].astype(str) + " d"
        ax = r.set_index("case")[["detected_same_level", "detected_any_level"]].plot(kind="barh", figsize=(8, 5),
                                                                                     color=["#2563eb", "#93c5fd"])
        ax.set_xlim(0, 1)
        ax.set_title("Planted cuts found (recall)")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "injection_recall.png"), dpi=120)
        plt.close()
    if len(ev) and nd is not None and len(nd):
        top = ev[(ev["type"] == "supply_dip") & (ev["tier"] == "deep")].sort_values("lost_kwh_est", ascending=False).head(9)
        if len(top):
            fig, axes = plt.subplots(3, 3, figsize=(14, 9), squeeze=False)
            for ax, x in zip(axes.ravel(), top.itertuples()):
                s0, s1 = pd.Timestamp(x.start) - pd.Timedelta(days=14), pd.Timestamp(x.end) + pd.Timedelta(days=14)
                q = nd[(nd["coato_code"] == x.coato_code) & (nd["level"] == x.level) & nd["date"].between(s0, s1)]
                me = q[q["node_id"] == x.node_id]
                par = me["parent_id"].iloc[0] if len(me) else None
                for _, g in q[(q["parent_id"] == par) & (q["node_id"] != x.node_id)].groupby("node_id"):
                    ax.plot(g["date"], g["ratio"], color="#d1d5db", lw=1)
                ax.plot(me["date"], me["ratio"], color=col[x.level], lw=2)
                ax.axvspan(pd.Timestamp(x.start) - pd.Timedelta(hours=12), pd.Timestamp(x.end) + pd.Timedelta(hours=12),
                           color="#fecaca", alpha=0.5)
                ax.axhline(1, color="#9ca3af", lw=0.8, ls="--")
                ax.set_ylim(0, 1.6)
                ax.set_title(f"{x.coato_code} {x.level} {x.node_id}{' (chronic)' if x.chronic else ''}\n"
                             f"{x.start} ({x.days} d, -{x.depth:.0%})", fontsize=9)
                ax.tick_params(labelsize=7)
                for lab in ax.get_xticklabels():
                    lab.set_rotation(30)
            for ax in axes.ravel()[len(top):]:
                ax.axis("off")
            fig.suptitle("Largest deep supply dips: node kWh ratio (colour) vs its siblings (grey)")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "top_events.png"), dpi=110)
            plt.close()


# ---------------------------------------------------------------------------------------------------- main
def main(argv=None):
    import enriched_io as eio
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--root", default=eio.ENRICHED_ROOT)
    p.add_argument("--start", default="2024-08-01", help="new regime; the baseline needs ~2 weeks before events show")
    p.add_argument("--end", default=eio.WINDOW_END)
    p.add_argument("--districts", default=",".join(SAMPLE), help="comma list (default: the 5 sample districts)")
    p.add_argument("--all", action="store_true", help="every finished district")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--labels", default="complete_observed")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--shuffles", type=int, default=1)
    p.add_argument("--inject-transformers", type=int, default=40, help="planted transformer cuts per district")
    p.add_argument("--inject-feeders", type=int, default=12, help="planted feeder cuts per district")
    for k, v in DEFAULTS.items():
        p.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = p.parse_args(argv)
    out = os.path.abspath(os.path.expanduser(a.out_dir))
    os.makedirs(out, exist_ok=True)
    cfg = dict(here=HERE, root=a.root, start=a.start, end=a.end, labels=a.labels.split(","), seed=a.seed,
               shuffles=a.shuffles, inject_transformers=a.inject_transformers, inject_feeders=a.inject_feeders,
               **{k: getattr(a, k) for k in DEFAULTS})
    fs, root = eio.open_root(a.root)
    codes = eio.finished_districts(fs, root, want=None if a.all else set(a.districts.split(",")))
    thr = max(1, (os.cpu_count() or 4) // max(a.workers, 1))
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(thr)
    t0 = time.time()
    res = []

    def done_one(code, r):
        res.append(r)
        print(f"[scan] {len(res)}/{len(codes)} {code}: {r['info']['households_used']:,} households, "
              f"{len(r['events'])} events, {r['seconds']:.0f} s", flush=True)
    if a.workers <= 1:                                                       # in-process (debugging, tests)
        for c in codes:
            done_one(c, district_worker(c, cfg))
    else:
        with cf.ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
            pending = {ex.submit(district_worker, c, cfg): c for c in codes}
            while pending:
                done, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
                for f in done:
                    code = pending.pop(f)
                    try:
                        r = f.result()
                    except Exception as e:                                   # noqa: BLE001
                        raise RuntimeError(f"district {code} failed: {type(e).__name__}: {e}") from e
                    done_one(code, r)
    info = pd.DataFrame([r["info"] for r in res])
    ev = pd.DataFrame([e for r in res for e in r["events"]])
    ctl = pd.DataFrame([c for r in res for c in r["control"]])
    inj = pd.DataFrame([x for r in res for x in r["injection"]])
    nd = pd.concat([r["node_day"] for r in res if r["node_day"] is not None], ignore_index=True) if res else None
    dd = pd.concat([r["district"] for r in res if r["district"] is not None], ignore_index=True) if res else None
    if len(ev):
        ev = ev.sort_values(["coato_code", "type", "tier", "level", "start"])
    ev.to_csv(os.path.join(out, "events.csv"), index=False)
    ctl.to_csv(os.path.join(out, "control.csv"), index=False)
    inj.to_csv(os.path.join(out, "injection.csv"), index=False)
    info.to_csv(os.path.join(out, "districts.csv"), index=False)
    if nd is not None and len(nd):
        nd.to_parquet(os.path.join(out, "node_day.parquet"), index=False)
    if dd is not None and len(dd):
        dd.to_csv(os.path.join(out, "district_days.csv"), index=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(info, ev, ctl, inj, dd, cfg)
    txt = buf.getvalue()
    print(txt)
    open(os.path.join(out, "outage_report.txt"), "w").write(txt)
    try:
        charts(ev, ctl, inj, nd, out, cfg)
    except Exception as e:                                                   # noqa: BLE001
        print(f"[charts] WARNING: not drawn ({type(e).__name__}: {e})")
    print(f"[done] {out} in {(time.time() - t0) / 60:.1f} min")
    return out


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    from mlflow_utils import run_script
    run_script(main, "electricity/analysis")
