#!/usr/bin/env python
"""
Weather-period and season shares of the data, and calendar spikes at substation / feeder level.

    python scripts/analysis/node_calendar_spikes.py --workers 12 \
        --forecasts ~/work/eval/node_blend_test_5pct/forecasts/blend \
                    ~/work/eval/runs/<..._xgboost_nodes-dev-test>/forecasts/xgboost
    # + national household-day shares by weather period (reads 3 columns of every district):
    python scripts/analysis/node_calendar_spikes.py --workers 16 --all-districts

Part A. DATA SHARES. Each day is classed by the district's observed mean temperature (v8 breakpoints): heating
    t_mean < 20 C, dead band 20-26.5 C, cooling > 26.5 C, and by season (DJF, MAM, JJA, SON) and meter regime
    (old < 2024-08-01). Shares of node-days and of node energy (the 60 test-set substations and their feeders);
    with --all-districts also of observed household-days nationally (the training rows) by year and regime.
Part B. CALENDAR EFFECTS. Day type from the standing calendar: workday, Saturday, Sunday, holiday (incl. the
    extra day off), the day before / after a holiday; Ramadan as a flag. Per node-day:
      spike ratio   T2 / centred 15-day rolling median of T2 (robust local level)
      spike         robust z of log(ratio) > 2.5 (per node)
      T2 step       log T2(d) - log T2(d-1)
      same-household step   log of sum kWh(d) / sum kWh(d-1) over the households observed on BOTH days
    If T2 jumps on a day type but the same-household step does not, the spike comes from WHO is observed that day
    (composition / reading schedule), not from consumption: a data artefact, not a load event.
Part C. FORECAST ERROR BY TARGET-DAY TYPE (with --forecasts): pooled sysMAPE and bias of each model (and B0) by
    the day type of the target day.
Part D. FORECAST ERROR BY WEATHER PERIOD (with --forecasts): median per-node sysMAPE, pooled sysMAPE and bias by the
    weather period of the target day (heating / dead band / cooling, the district's observed mean) x horizon block.

Writes <out>/report.txt, CSV tables and PNG charts. Node aggregates only (nodes of >= 50 households; a same-household
step needs >= 10 households on both days); no customer keys.
"""
import argparse
import concurrent.futures as cf
import glob
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
import harness  # noqa: E402,F401
from harness import nodeset as ns  # noqa: E402
from harness import testset as ts  # noqa: E402

HDD, CDD = 20.0, 26.5
REGIME = pd.Timestamp("2024-08-01")
MIN_PANEL = 10
DAYTYPES = ["workday", "Saturday", "Sunday", "pre-holiday", "holiday", "post-holiday"]


def wclass(t):
    t = np.asarray(t, float)
    return np.where(~np.isfinite(t), "unknown", np.where(t < HDD, "heating", np.where(t > CDD, "cooling", "dead band")))


def season(d):
    m = pd.DatetimeIndex(d).month
    return np.select([np.isin(m, (12, 1, 2)), np.isin(m, (3, 4, 5)), np.isin(m, (6, 7, 8))], ["winter", "spring", "summer"],
                     "autumn")


def day_types(dates):
    """DataFrame date, daytype, ram, hname for the standing calendar (holiday includes the extra day off)."""
    d = pd.DatetimeIndex(dates)
    cal = ts.standing_calendar(pd.date_range(d.min() - pd.Timedelta(days=2), d.max() + pd.Timedelta(days=2)))
    cal = cal.set_index("date")
    off = (cal["hol"] + cal["xoff"]) > 0
    pre = off.shift(-1, fill_value=False) & ~off
    post = off.shift(1, fill_value=False) & ~off
    dow = cal.index.dayofweek
    dt = np.where(off, "holiday", np.where(pre & (dow < 5), "pre-holiday", np.where(post & (dow < 5), "post-holiday",
                  np.where(dow == 5, "Saturday", np.where(dow == 6, "Sunday", "workday")))))
    out = pd.DataFrame({"date": cal.index, "daytype": dt, "ram": cal["ram"].to_numpy(), "hname": cal["hname"].to_numpy()})
    return out[out["date"].isin(d)].reset_index(drop=True)


# ----------------------------------------------------------------------------- workers
def node_worker(code, cfg):
    """One district: its weather, and per node of the spec the households observed per day, their kWh sum, and the
    same-household day-over-day sums (households observed on d and d-1)."""
    from harness.data import read_district
    t0 = time.time()
    d = cfg["data"]
    D = read_district(code, [dict(uri=d["uri"], env_prefix=d.get("env_prefix", ""), start=d["start"], end=d["end"])],
                      d["start"], d["end"], ["complete_observed"])
    names = cfg["nodes"]
    K, T = len(names), len(D["dates"])
    out = dict(code=code, dates0=str(D["dates"][0].date()), T=T, wt=D["wt"].astype(np.float32), names=[n for n, _, _ in names])
    if not len(D["keys"]):
        return out
    sub = {nid: k for k, (_, lv, nid) in enumerate(names) if lv == "substation"}
    fed = {nid: k for k, (_, lv, nid) in enumerate(names) if lv == "feeder"}
    gs = np.array([sub.get(int(v), -1) for v in D["network"][:, 2]])
    gf = np.array([fed.get(int(v), -1) for v in D["network"][:, 1]])
    M = D["M"]
    acc = {k: np.zeros((K, T)) for k in ("n_obs", "kwh", "ll_n", "ll_now", "ll_prev")}
    for g in (gs, gf):
        for k in range(K):
            rows = np.flatnonzero(g == k)
            if not len(rows):
                continue
            X = M[rows].astype(np.float64)
            ob = np.isfinite(X)
            acc["n_obs"][k] += ob.sum(0)
            acc["kwh"][k] += np.where(ob, X, 0).sum(0)
            both = ob[:, 1:] & ob[:, :-1]
            acc["ll_n"][k, 1:] += both.sum(0)
            acc["ll_now"][k, 1:] += np.where(both, X[:, 1:], 0).sum(0)
            acc["ll_prev"][k, 1:] += np.where(both, X[:, :-1], 0).sum(0)
    small = acc["ll_n"] < MIN_PANEL
    for k in ("ll_now", "ll_prev"):
        acc[k][small] = np.nan
    acc["kwh"][acc["n_obs"] < MIN_PANEL] = np.nan
    out.update({k: v.astype(np.float32) for k, v in acc.items()}, seconds=time.time() - t0)
    return out


def weather_worker(code, cfg):
    """One district, 3 columns: observed household-days per date (complete_observed) and the district t_mean."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as pads
    import enriched_io as eio
    d = cfg["data"]
    fs, root = eio.open_root(d["uri"], d.get("env_prefix", ""))
    dset = eio.district_daily(fs, root, code)
    d0, d1 = pd.Timestamp(d["start"]), pd.Timestamp(d["end"])
    T = (d1 - d0).days + 1
    e0 = int((d0 - pd.Timestamp("1970-01-01")).days)
    n, t = np.zeros(T), np.full(T, np.nan)
    flt = (pads.field("date") >= d0.date()) & (pads.field("date") <= d1.date())
    for b in dset.scanner(columns=["date", "label_state", "t_mean"], filter=flt, batch_size=262_144).to_batches():
        if not b.num_rows:
            continue
        day = pc.cast(pc.cast(b.column("date"), pa.date32()), pa.int32()).to_numpy(zero_copy_only=False).astype(np.int64) - e0
        ob = pc.fill_null(pc.equal(b.column("label_state"), "complete_observed"), False).to_numpy(zero_copy_only=False)
        n += np.bincount(day[ob], minlength=T)
        np.fmax.at(t, day, np.asarray(b.column("t_mean").to_numpy(zero_copy_only=False), dtype=float))
    return dict(code=code, n=n, t=t)


def run_pool(fn, jobs, workers):
    if workers <= 1:
        return [fn(c, j) for c, j in jobs]
    out = []
    with cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(fn, c, j) for c, j in jobs]
        for k, fu in enumerate(cf.as_completed(futs), 1):
            out.append(fu.result())
            if k % 10 == 0 or k == len(futs):
                print(f"[read] {k}/{len(futs)}", flush=True)
    return out


# ----------------------------------------------------------------------------- analysis
def shares(df, by, w=None):
    g = df.groupby(by)
    s = g[w].sum() if w else g.size()
    return (s / s.groupby(level=0).transform("sum") if isinstance(by, list) and len(by) > 1 else s / s.sum())


def robust_z(x):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    return (x - med) / mad if mad > 0 else np.full_like(x, np.nan)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", default=os.path.join(REPO, "configs", "test_set_substations_2025.yaml"))
    p.add_argument("--test-dir", default=None)
    p.add_argument("--root", default=None, help="data root instead of the spec's (tests)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--forecasts", nargs="*", default=[], help="node forecast folders (node.csv.gz) for Part C")
    p.add_argument("--role", default="test", choices=["dev", "test"], help="nodes scored in Part C")
    p.add_argument("--all-districts", action="store_true", help="national household-day shares (3 columns per district)")
    p.add_argument("--out", default="~/work/analysis/node_calendar_spikes")
    a = p.parse_args(argv)
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    spec = ns.load_spec(a.spec)
    test_dir = os.path.expanduser(a.test_dir or os.path.join("~/work/test_set", spec["version"]))
    data = dict(spec["data"])
    if a.root:
        data["uri"] = a.root
    nodes, targets = ns.load_nodes(test_dir), ns.load_targets(test_dir)
    codes = sorted(nodes["coato"].unique())
    jobs = [(c, dict(data=data, nodes=[(r.node, r.level, int(r.node_id)) for r in nodes[nodes["coato"] == c].itertuples()]))
            for c in codes]
    print(f"[read] {len(codes)} districts, {len(nodes)} nodes", flush=True)
    res = run_pool(node_worker, jobs, a.workers)
    lines = []

    # ---- node-day frame
    parts, wx = [], []
    for r in res:
        dates = pd.date_range(r["dates0"], periods=r["T"], freq="D")
        wx.append(pd.DataFrame({"coato": r["code"], "date": dates, "t_mean": r["wt"][0]}))
        if "n_obs" not in r:
            continue
        for k, name in enumerate(r["names"]):
            parts.append(pd.DataFrame({"node": name, "date": dates, "n_obs": r["n_obs"][k], "kwh_obs": r["kwh"][k],
                                       "ll_now": r["ll_now"][k], "ll_prev": r["ll_prev"][k]}))
    wx = pd.concat(wx, ignore_index=True)
    hh = pd.concat(parts, ignore_index=True)
    df = targets.merge(nodes[["node", "level", "coato", "role", "households"]], on="node")
    df = df.merge(hh, on=["node", "date"], how="left").merge(wx, on=["coato", "date"], how="left")
    df = df.merge(day_types(df["date"].unique()), on="date", how="left")
    df["wclass"] = wclass(df["t_mean"])
    df["season"] = season(df["date"])
    df["regime"] = np.where(df["date"] < REGIME, "old", "new")
    df["year"] = df["date"].dt.year
    df = df.sort_values(["node", "date"]).reset_index(drop=True)
    clean = df[~df["flag"] & np.isfinite(df["T2"])].copy()

    # ---- Part A: shares
    A = []
    for name, q in (("node-days, 2025 (scored)", clean[clean["year"] == 2025]),
                    ("node-days, before 2025 (training period)", clean[clean["year"] < 2025]),
                    ("node-days, new regime", clean[clean["regime"] == "new"]),
                    ("node-days, old regime (T3 days)", df[(df["regime"] == "old") & np.isfinite(df["T3"])])):
        for dim in ("wclass", "season"):
            c = q.groupby(dim).size()
            e = q.groupby(dim)["T2" if "old" not in name else "T3"].sum()
            for k in c.index:
                A.append(dict(data=name, by=dim, group=k, days=int(c[k]), share_days=c[k] / c.sum(),
                              share_energy=e[k] / e.sum()))
    A = pd.DataFrame(A)
    A.to_csv(os.path.join(out, "shares_nodes.csv"), index=False, float_format="%.4f")
    wc = clean.groupby(["season", "wclass"]).size().unstack(fill_value=0)
    wc = wc.div(wc.sum(1), axis=0)
    lines += ["PART A. DATA SHARES (60 test-set substations + feeders; clean node-days)", ""]
    for name in A["data"].unique():
        q = A[A["data"] == name]
        lines.append(f"  {name}:")
        for dim in ("wclass", "season"):
            lines.append("    " + " | ".join(f"{r.group} {r.share_days:.1%} of days, {r.share_energy:.1%} of energy"
                                             for r in q[q["by"] == dim].itertuples()))
    lines += ["", "  weather period within each season (share of clean node-days):", wc.to_string(float_format="{:.1%}".format), ""]

    if a.all_districts:
        import enriched_io as eio
        fs, root = eio.open_root(data["uri"], data.get("env_prefix", ""))
        allc = eio.finished_districts(fs, root)
        W = run_pool(weather_worker, [(c, dict(data=data)) for c in allc], a.workers)
        dates = pd.date_range(data["start"], data["end"], freq="D")
        rows = []
        for r in W:
            rows.append(pd.DataFrame({"date": dates, "n": r["n"], "t": r["t"]}))
        nat = pd.concat(rows, ignore_index=True)
        nat = nat[nat["n"] > 0]
        nat["wclass"], nat["season"] = wclass(nat["t"]), season(nat["date"])
        nat["regime"] = np.where(nat["date"] < REGIME, "old", "new")
        nat["period"] = np.where(nat["date"] < REGIME, "old regime 2022-01..2024-07",
                                 np.where(nat["date"] < "2025-01-01", "new regime 2024-08..2024-12",
                                          np.where(nat["date"] < "2026-01-01", "2025", "2026-H1")))
        t1 = nat.pivot_table(index="period", columns="wclass", values="n", aggfunc="sum").fillna(0)
        t2 = nat.pivot_table(index="period", columns="season", values="n", aggfunc="sum").fillna(0)
        tot = t1.sum(1)
        t1, t2 = t1.div(tot, axis=0), t2.div(t2.sum(1), axis=0)
        t1.insert(0, "household-days (M)", tot / 1e6)
        t1.to_csv(os.path.join(out, "shares_national_wclass.csv"), float_format="%.4f")
        t2.to_csv(os.path.join(out, "shares_national_season.csv"), float_format="%.4f")
        lines += ["  NATIONAL observed household-days (the training rows), by weather period:",
                  t1.to_string(float_format="{:.3f}".format), "", "  ... by season:", t2.to_string(float_format="{:.1%}".format), ""]

    # ---- Part B: calendar effects
    g = clean.groupby("node", sort=False)
    clean["base"] = g["T2"].transform(lambda s: s.rolling(15, center=True, min_periods=8).median())
    clean["lratio"] = np.log(clean["T2"] / clean["base"])
    clean["z"] = clean.groupby("node")["lratio"].transform(lambda s: pd.Series(robust_z(s.to_numpy()), index=s.index))
    clean["spike"] = clean["z"] > 2.5
    clean["dip"] = clean["z"] < -2.5
    prev = clean.groupby("node")["T2"].shift(1)
    gap1 = (clean["date"] - clean.groupby("node")["date"].shift(1)).dt.days == 1
    clean["t2_step"] = np.where(gap1, np.log(clean["T2"] / prev), np.nan)
    clean["hh_step"] = np.log(clean["ll_now"] / clean["ll_prev"])
    clean["cov"] = clean["n_obs"] / clean["n_rows"]
    clean["mean_obs"] = clean["kwh_obs"] / clean["n_obs"]
    B = clean.groupby(["level", "daytype"]).agg(node_days=("T2", "size"), level_vs_local_pct=("lratio", "mean"),
                                                spike_rate=("spike", "mean"), dip_rate=("dip", "mean"),
                                                t2_step_pct=("t2_step", "mean"), same_household_step_pct=("hh_step", "mean"),
                                                observed_share=("cov", "mean")).reset_index()
    for c in ("level_vs_local_pct", "t2_step_pct", "same_household_step_pct"):
        B[c] = 100 * (np.exp(B[c]) - 1)
    B["daytype"] = pd.Categorical(B["daytype"], DAYTYPES, ordered=True)
    B = B.sort_values(["level", "daytype"])
    B.to_csv(os.path.join(out, "daytype_effects.csv"), index=False, float_format="%.4f")
    base_rate = clean["spike"].mean()
    lines += ["PART B. CALENDAR EFFECTS (clean node-days, all years)",
              "  level_vs_local = T2 vs its centred 15-day median; t2_step = change vs the day before; same-household step ="
              " the same change for households observed on both days (if T2 moves and the same households do not, the"
              " movement is who is observed, not consumption)", "",
              B.to_string(index=False, float_format="{:.2f}".format), "",
              f"  base spike rate (all days): {base_rate:.2%}", ""]
    S = clean.groupby(["level", "season", "daytype"]).agg(level_vs_local_pct=("lratio", "mean"), spike_rate=("spike", "mean"),
                                                          same_household_step_pct=("hh_step", "mean"),
                                                          t2_step_pct=("t2_step", "mean")).reset_index()
    for c in ("level_vs_local_pct", "same_household_step_pct", "t2_step_pct"):
        S[c] = 100 * (np.exp(S[c]) - 1)
    S.to_csv(os.path.join(out, "daytype_by_season.csv"), index=False, float_format="%.4f")
    sub = S[S["level"] == "substation"].pivot(index="daytype", columns="season", values="level_vs_local_pct")
    lines += ["  substations: T2 vs local level, %, by day type x season:",
              sub.reindex(DAYTYPES).to_string(float_format="{:+.1f}".format), ""]
    clean["dow"] = clean["date"].dt.day_name().str[:3]
    D = clean[clean["daytype"].isin(["workday", "Saturday", "Sunday"])].groupby(["level", "dow"]).agg(
        observed_share=("cov", "mean"), mean_obs_kwh=("mean_obs", "mean"), level_vs_local_pct=("lratio", "mean"),
        t2_step_pct=("t2_step", "mean"), same_household_step_pct=("hh_step", "mean")).reset_index()
    for c in ("level_vs_local_pct", "t2_step_pct", "same_household_step_pct"):
        D[c] = 100 * (np.exp(D[c]) - 1)
    D.to_csv(os.path.join(out, "day_of_week.csv"), index=False, float_format="%.4f")
    lines += ["  day of week (non-holiday days):", D.to_string(index=False, float_format="{:.3f}".format), ""]
    top = clean.groupby("date").agg(nodes=("spike", "size"), spiking=("spike", "mean"), dipping=("dip", "mean"),
                                    level_vs_local=("lratio", "mean"), daytype=("daytype", "first"), hname=("hname", "first"),
                                    ram=("ram", "first"), t_mean=("t_mean", "mean"))
    top = top[top["nodes"] >= 20]
    top["level_vs_local"] = 100 * (np.exp(top["level_vs_local"]) - 1)
    top.sort_values("spiking", ascending=False).to_csv(os.path.join(out, "spike_dates.csv"), float_format="%.4f")
    lines += ["  dates where most nodes spike together (share of nodes with z > 2.5):",
              top.sort_values("spiking", ascending=False).head(25).to_string(float_format="{:.2f}".format), "",
              "  dates where most nodes dip together:",
              top.sort_values("dipping", ascending=False).head(15).to_string(float_format="{:.2f}".format), ""]

    # ---- Part C: forecast error by target-day type
    if a.forecasts:
        models = {}
        for fd in a.forecasts:
            fd = os.path.expanduser(fd)
            models[os.path.basename(fd.rstrip("/"))] = ns.read_forecasts(fd)
        want = set(nodes.loc[nodes["role"] == a.role, "node"])
        b0 = ns.b0_forecasts(targets, spec, nodes=want, issue=ns.issue_dates(spec, a.role))
        models["B0"] = b0[["node", "origin", "horizon", "forecast"]]
        rows, _ = ns.common_rows(models, targets, spec, a.role, nodes)
        rows = rows.merge(day_types(rows["date"].unique())[["date", "daytype"]], on="date", how="left")
        C = []
        for (lv, dt), q in rows.groupby(["level", "daytype"]):
            for m in models:
                C.append(dict(level=lv, daytype=dt, model=m, node_days=len(q),
                              sysmape=float((q[m] - q["y"]).abs().sum() / q["y"].sum()),
                              bias=float((q[m] - q["y"]).sum() / q["y"].sum())))
        C = pd.DataFrame(C)
        C.to_csv(os.path.join(out, "error_by_daytype.csv"), index=False, float_format="%.5f")
        pv = C[C["level"] == "substation"].pivot(index="daytype", columns="model", values="sysmape").reindex(DAYTYPES)
        pb = C[C["level"] == "substation"].pivot(index="daytype", columns="model", values="bias").reindex(DAYTYPES)
        lines += [f"PART C. FORECAST ERROR BY TARGET-DAY TYPE ({a.role} substations, pooled over all horizons)",
                  "  sysMAPE:", pv.to_string(float_format="{:.2%}".format), "", "  bias:",
                  pb.to_string(float_format="{:+.2%}".format), ""]

        # ---- Part D: forecast error by weather period of the target day (district observed t_mean)
        rows = rows.merge(nodes[["node", "coato"]], on="node", how="left").merge(wx, on=["coato", "date"], how="left")
        rows["wclass"] = wclass(rows["t_mean"])
        rows["hblock"] = pd.cut(rows["horizon"], [0, 7, 14, 30, 60], labels=["h1-7", "h8-14", "h15-30", "h31-60"])
        Dw = []
        for (lv, wc_, hb), q in rows.groupby(["level", "wclass", "hblock"], observed=True):
            for m in models:
                pn = q.groupby("node").apply(lambda r: (r[m] - r["y"]).abs().sum() / r["y"].sum())
                Dw.append(dict(level=lv, wclass=wc_, hblock=hb, model=m, node_days=len(q), nodes=q["node"].nunique(),
                               median_node=float(pn.median()),
                               pooled=float((q[m] - q["y"]).abs().sum() / q["y"].sum()),
                               bias=float((q[m] - q["y"]).sum() / q["y"].sum())))
        Dw = pd.DataFrame(Dw)
        Dw.to_csv(os.path.join(out, "error_by_wclass.csv"), index=False, float_format="%.5f")
        lines += [f"PART D. FORECAST ERROR BY WEATHER PERIOD OF THE TARGET DAY ({a.role}; district observed mean temperature:"
                  f" heating < {HDD:g} C, dead band, cooling > {CDD:g} C)"]
        for lv in ("substation", "feeder"):
            q = Dw[Dw["level"] == lv]
            if not len(q):
                continue
            nd = q[q["model"] == "B0"].groupby("wclass")["node_days"].sum()
            lines += [f"  {lv}: node-days " + " | ".join(f"{k} {v:,} ({v / nd.sum():.0%})" for k, v in nd.items())]
            for col, fmt in (("median_node", "{:.2%}"), ("bias", "{:+.2%}")):
                t = q.pivot_table(index=["wclass", "hblock"], columns="model", values=col, observed=True)
                t = t[list(models)]
                t = t.reindex([(w, h) for w in ("heating", "dead band", "cooling", "unknown")
                               for h in ("h1-7", "h8-14", "h15-30", "h31-60") if (w, h) in t.index])
                lines += [f"  {lv} {'median per-node sysMAPE' if col == 'median_node' else 'pooled bias'}:",
                          t.to_string(float_format=fmt.format), ""]

    _charts(clean, B, D, top, out)
    text = "\n".join(lines)
    open(os.path.join(out, "report.txt"), "w").write(text + "\n")
    print(text)
    print(f"\n-> {out}")
    return out


def _charts(clean, B, D, top, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    b = B[B["level"] == "substation"].set_index("daytype").reindex(DAYTYPES)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(DAYTYPES))
    ax.bar(x - 0.2, b["t2_step_pct"], 0.4, label="T2 change vs day before", color="#2563eb")
    ax.bar(x + 0.2, b["same_household_step_pct"], 0.4, label="same households, change vs day before", color="#9ca3af")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, DAYTYPES)
    ax.set_ylabel("%")
    ax.set_title("Substations: day-type step in T2 vs in the same households", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "daytype_steps.png"), dpi=120)
    plt.close(fig)
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    d = D[D["level"] == "substation"].set_index("dow").reindex(order)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].bar(order, d["level_vs_local_pct"], color="#2563eb")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("T2 vs local 15-day level, %", loc="left", fontsize=10)
    axes[1].bar(order, 100 * d["observed_share"], color="#9ca3af")
    axes[1].set_title("households with an observed reading, %", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "day_of_week.png"), dpi=120)
    plt.close(fig)
    t = top.reset_index().sort_values("date")
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(t["date"], t["spiking"], lw=0.8, color="#2563eb", label="share of nodes spiking")
    ax.plot(t["date"], -t["dipping"], lw=0.8, color="#f97316", label="share of nodes dipping (negative)")
    for r in t[t["daytype"] == "holiday"].itertuples():
        ax.axvline(r.date, color="#9ca3af", lw=0.5, alpha=0.6)
    ax.axvline(REGIME, color="black", ls="--", lw=0.8)
    ax.set_title("Days when nodes spike or dip together (grey lines = holidays; dashed = regime change)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "spike_calendar.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(REPO, "core"))
    main()
