#!/usr/bin/env python
"""
How much of a household model's district error comes from scaling households up to the district, not from the model?

The harness forecasts a district as  present(o - 1) x mean forecast of the households with fresh history.
This script replaces the forecast by the TRUTH: the fresh households' own observed kWh on the target day d.
That "oracle" is the best any household model can do under this scaling, so its error is a floor:

    oracle(o, h)  = present(o - 1) x mean observed kWh on d of the fresh households
    T2(d)         = n_rows(d) x (kWh observed + estimated) / (n observed + estimated)

    oracle / T2   = [present / n_rows]  x  [fresh mean on d / T2 per customer]
                     customer count        representativeness of the fresh households

Reported per split, horizon and issue month, on the sum of the districts (clean rows, the 98 % rule), plus the same
for a model's forecasts when --forecasts is given (model error = model - oracle). From the targets alone: the mean kWh
of an estimated vs an observed client-day, and the estimated share, by month.

Outputs are district-sum aggregates only (no customer keys).

    python scripts/analysis/bottom_up_oracle.py --sample --workers 5 --out-dir ~/work/bottom_up_oracle
    python scripts/analysis/bottom_up_oracle.py --sample --workers 5 --out-dir ~/work/bottom_up_oracle \
        --forecasts ~/work/eval/runs/<run>/forecasts/gam
"""
import argparse
import concurrent.futures as cf
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
from harness import testset as ts  # noqa: E402

MIN_INPUT_DAYS, MIN_RECENT_DAYS, RECENT_WINDOW, LOOKBACK = 60, 20, 30, 90   # the harness's fresh-history rule


def worker(code, spec_path, labels):
    from harness.data import read_district
    spec = ts.load_spec(spec_path)
    d = spec["data"]
    D = read_district(code, [dict(uri=d["uri"], env_prefix=d.get("env_prefix", ""), start=d["start"], end=d["end"])],
                      d["start"], d["end"], labels)
    if not len(D["keys"]):
        return pd.DataFrame()
    M, dates = D["M"], D["dates"]
    T = M.shape[1]
    gap = int(spec["history_gap_days"])
    hs = np.asarray(spec["control_horizons"])
    cnt = np.zeros((len(M), T + 1), dtype=np.int32)
    np.cumsum(np.isfinite(M), axis=1, out=cnt[:, 1:])
    rows = []
    for split in ("validation", "test"):
        for o_day in ts.issue_dates(spec, split):
            o = ts.day_index(spec, o_day)
            last = o - gap
            if last + 1 - LOOKBACK < 0 or o + hs.max() > T - 1:
                continue
            n_in = cnt[:, last + 1] - cnt[:, last + 1 - LOOKBACK]
            n_rec = cnt[:, last + 1] - cnt[:, last + 1 - RECENT_WINDOW]
            fresh = np.flatnonzero((n_in >= MIN_INPUT_DAYS) & (n_rec >= MIN_RECENT_DAYS))
            present = int(((D["first"] <= last) & (D["last"] >= last)).sum())
            hist = np.nanmean(M[fresh, last + 1 - RECENT_WINDOW:last + 1]) if len(fresh) else np.nan
            for h in hs:
                a = M[fresh, o + h] if len(fresh) else np.empty(0)
                ob = np.isfinite(a)
                rows.append((str(code), dates[o], int(h), dates[o + h], present, len(fresh), int(ob.sum()),
                             float(a[ob].mean()) if ob.any() else np.nan, float(hist)))
    return pd.DataFrame(rows, columns=["node", "origin", "horizon", "date", "present", "n_fresh", "n_fresh_obs_d",
                                       "fresh_mean_d", "fresh_mean_last30"])


def collect(codes, a):
    with cf.ProcessPoolExecutor(max_workers=min(a.workers, len(codes)), mp_context=mp.get_context("spawn")) as ex:
        parts = list(ex.map(worker, codes, [a.test_set] * len(codes), [a.labels.split(",")] * len(codes)))
    return pd.concat(parts, ignore_index=True)


def ratio(num, den):
    return float(num / den) if den else np.nan


def result_metrics(out_dir):
    """MLflow metrics: <split>/<column>_h<h>, e.g. validation/oracle_bias_h30, test/model_minus_oracle_h7."""
    t = pd.read_csv(os.path.join(out_dir, "oracle_by_horizon.csv"))
    return {f"{r['split']}/{c}_h{int(r['horizon'])}": float(r[c]) for _, r in t.iterrows()
            for c in t.columns if c not in ("split", "horizon", "days") and pd.notna(r[c])}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-set", default=os.path.join(REPO, "configs", "test_set.yaml"))
    p.add_argument("--test-dir", default=None, help="default ~/work/test_set/<version>[-sample]")
    p.add_argument("--sample", action="store_true", help="the guide's 5 districts")
    p.add_argument("--coatos", default=None, help="comma-separated district codes")
    p.add_argument("--forecasts", default=None, help="a run's forecasts/<model> folder (district.csv.gz)")
    p.add_argument("--labels", default="complete_observed")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)
    spec = ts.load_spec(a.test_set)
    out = os.path.expanduser(a.out_dir)
    os.makedirs(out, exist_ok=True)
    tdir = a.test_dir or os.path.join("~/work/test_set", spec["version"] + ("-sample" if a.sample else ""))
    if not os.path.exists(os.path.join(os.path.expanduser(tdir), "manifest.json")) and a.sample:
        tdir = os.path.join("~/work/test_set", spec["version"])
    tg = ts.load_targets(tdir, spec)
    codes = spec["sample_districts"] if a.sample else (a.coatos.split(",") if a.coatos else sorted(tg["node"].unique()))
    tg = tg[tg["node"].isin(codes)]
    L = []

    def say(s=""):
        print(s)
        L.append(s)

    # ---- from the targets alone: are estimated client-days different from observed ones?
    m = tg.assign(month=tg["date"].dt.to_period("M")).groupby("month")[["n_obs", "n_est", "kwh_obs", "kwh_est"]].sum()
    m = m[m.index >= pd.Period("2023-07", "M")]
    tab = pd.DataFrame({"est_share": m["n_est"] / (m["n_obs"] + m["n_est"]),
                        "kwh_per_obs_day": m["kwh_obs"] / m["n_obs"], "kwh_per_est_day": m["kwh_est"] / m["n_est"]})
    tab["est_vs_obs"] = tab["kwh_per_est_day"] / tab["kwh_per_obs_day"] - 1
    say(f"districts {', '.join(codes)} | test set {spec['version']}")
    say("\n=== client-days: estimated share and kWh per estimated vs observed client-day (sum of the districts) ===")
    say(tab.to_string(float_format=lambda x: f"{x:8.3f}"))
    tab.to_csv(os.path.join(out, "estimated_vs_observed_by_month.csv"))

    # ---- the oracle (reads the districts)
    o = collect(codes, a)
    o = o.merge(tg[["node", "date", "y", "n_rows", "n_obs", "n_est", "kwh_obs", "kwh_est", "flag", "artefact"]],
                on=["node", "date"], how="left")
    o["clean"] = o["y"].notna() & ~o["flag"].fillna(True).astype(bool) & ~o["artefact"].fillna(True).astype(bool)
    o["per_cust"] = (o["kwh_obs"].fillna(0) + o["kwh_est"].fillna(0)) / (o["n_obs"].fillna(0) + o["n_est"].fillna(0))
    o["oracle"] = o["present"] * o["fresh_mean_d"]
    o["count_only"] = o["present"] * o["per_cust"]                  # right per-customer level, the harness's count
    o["repr_only"] = o["n_rows"] * o["fresh_mean_d"]                # right count, the fresh households' level
    if a.forecasts:
        f = ts.load_forecasts(a.forecasts)
        o = o.merge(f[["node", "origin", "horizon", "forecast"]], on=["node", "origin", "horizon"], how="left")
    cols = ["y", "oracle", "count_only", "repr_only"] + (["forecast"] if a.forecasts else [])
    ok = o["clean"] & o["oracle"].notna() & (o["forecast"].notna() if a.forecasts else True)
    need = int(np.floor(spec["national"]["min_district_share"] * len(codes)))
    g = o[ok].groupby(["origin", "horizon"])
    nat = g[cols].sum()
    nat["districts"] = g.size()
    nat = nat[nat["districts"] >= need].reset_index()
    nat["split"] = np.where(nat["origin"] < pd.Timestamp(spec["splits"]["test"]["issue_start"]), "validation", "test")
    vend = pd.Timestamp(spec["splits"]["validation"]["target_end"])
    nat = nat[(nat["split"] == "test") | (nat["origin"] + pd.to_timedelta(nat["horizon"], unit="D") <= vend)]
    nat.to_csv(os.path.join(out, "oracle_national_daily.csv"), index=False)
    dropped = o.assign(no_fresh=o["n_fresh"] == 0)
    say("\n=== district rows with no fresh household (no household forecast possible), by issue month ===")
    say(dropped.assign(month=dropped["origin"].dt.to_period("M")).groupby("month")["no_fresh"].mean()
        .loc[lambda s: s > 0].to_string(float_format=lambda x: f"{x:.0%}") or "none")

    def summary(d):
        r = {"days": len(d)}
        for c in cols[1:]:
            r[f"{c}_bias"] = ratio((d[c] - d["y"]).sum(), d["y"].sum())
            r[f"{c}_sysmape"] = ratio((d[c] - d["y"]).abs().sum(), d["y"].sum())
        if a.forecasts:
            r["model_minus_oracle"] = ratio((d["forecast"] - d["oracle"]).sum(), d["y"].sum())
        return pd.Series(r)

    fmt = lambda x: f"{x:+7.1%}"  # noqa: E731
    by_h = nat.groupby(["split", "horizon"]).apply(summary, include_groups=False).astype({"days": int})
    say("\n=== sum of the districts: bias / sysMAPE of the oracle and its two factors (vs T2) ===")
    say("oracle = present x fresh households' actual mean; count_only = present x T2 per customer; "
        "repr_only = n_rows x fresh actual mean")
    say(by_h.to_string(float_format=fmt))
    by_h.to_csv(os.path.join(out, "oracle_by_horizon.csv"))
    nat["month"] = nat["origin"].dt.to_period("M")
    by_m = (nat[nat["horizon"] == 30].groupby(["split", "month"]).apply(summary, include_groups=False)
            .astype({"days": int}))
    say("\n=== h30 by issue month ===")
    say(by_m.to_string(float_format=fmt))
    by_m.to_csv(os.path.join(out, "oracle_h30_by_month.csv"))
    say("\nReading it: oracle bias = the floor of a household model's district bias under this scaling. count_only "
        "~0 and repr_only < 0 -> the fresh households use less than the average customer (the estimated ones use "
        "more): scale by T2 per customer at o - 1 instead. model_minus_oracle = the model's own share.")
    with open(os.path.join(out, "report.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(REPO, "core"))
    from mlflow_utils import run_script
    run_script(main, "electricity/analysis", metrics_from=result_metrics)
