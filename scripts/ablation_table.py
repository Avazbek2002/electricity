#!/usr/bin/env python
"""
Side-by-side table of evaluate.py runs that differ in one setting (e.g. the old-data ablation, scripts/run_old_ablation.sh).

For every model it takes the LATEST run folder <out-root>/runs/<time>_<model>_validate (optionally only runs started at or
after --since) and reads scores/scores.csv (national and district sysMAPE per horizon, already on the common panel with
B0). Also the tree counts of the fits behind the run (boost.json of the v8 boosters, found by the config hash).

    python scripts/ablation_table.py                                   # the old-data ablation, latest runs
    python scripts/ablation_table.py --since 20261023-200000 --models xgboost abl_old_none abl_old_w050

Columns: national sysMAPE at the control horizons, headline (their mean), mean over ALL horizons 1..60 (less noisy than
five points), district headline, h60 bias, scored days, and the change of the headline / all-h mean vs the first model
(the baseline). Written as CSV next to the runs: <out-root>/ablations/<name>_<time>.csv.
"""
import argparse
import glob
import json
import os
import sys
import time

import pandas as pd
import yaml

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _p in (REPO, os.path.join(REPO, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OLD_ABLATION = ["xgboost", "abl_old_none", "abl_old_nonwinter", "abl_old_w050", "abl_old_w025", "abl_old_noflag"]
CHANGE = {"xgboost": "baseline: all, weight 1.0, flag on", "abl_old_none": "INCLUDE_OLD none",
          "abl_old_nonwinter": "INCLUDE_OLD non_winter", "abl_old_w050": "OLD_WEIGHT 0.5",
          "abl_old_w025": "OLD_WEIGHT 0.25", "abl_old_noflag": "OLD_REGIME_FLAG off"}


def latest_run(root, model, mode, since):
    runs = sorted(glob.glob(os.path.join(root, "runs", f"*_{model}_{mode}")))
    runs = [r for r in runs if os.path.basename(r).split("_")[0] >= (since or "")
            and os.path.exists(os.path.join(r, "scores", "scores.csv"))]
    return runs[-1] if runs else None


def trees(root, model):
    """'cb/xgb' final tree counts per refit of the fit behind this model's config (latest commit with that hash)."""
    try:
        import boost_harness as bh
        cfg = yaml.safe_load(open(os.path.join(REPO, "models", model, "config.yaml")))
        if not str(cfg.get("family", "")).startswith("v8_boost"):
            return ""
        sub = "_boost_wx" if cfg.get("family") == "v8_boost_wx" else "_boost"
        h = bh.config_hash(cfg) if sub == "_boost" else __import__("boost_wx_harness").config_hash(cfg)
        files = sorted(glob.glob(os.path.join(root, "artifacts", sub, "*", h, "refit-*", "boost.json")),
                       key=os.path.getmtime)
        latest = {}
        for f in files:                                   # newest file per refit wins
            latest[os.path.basename(os.path.dirname(f))] = json.load(open(f))
        return " | ".join(f"{k[6:]}: {v['catboost_final_iterations']}/{v['xgboost_final_iterations']}"
                          for k, v in sorted(latest.items()))
    except Exception as e:                                # noqa: BLE001  the table matters more than the tree counts
        return f"? ({type(e).__name__})"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=OLD_ABLATION, help="first = the baseline")
    p.add_argument("--mode", default="validate")
    p.add_argument("--split", default="validation")
    p.add_argument("--since", default=None, help="only runs started at/after this time, e.g. 20261023-200000")
    p.add_argument("--out-root", default="~/work/eval")
    p.add_argument("--name", default="old_data")
    a = p.parse_args(argv)
    root = os.path.expanduser(a.out_root)
    rows, b0 = [], None
    for m in a.models:
        run = latest_run(root, m, a.mode, a.since)
        if run is None:
            print(f"[skip] {m}: no finished {a.mode} run" + (f" since {a.since}" if a.since else ""))
            continue
        s = pd.read_csv(os.path.join(run, "scores", "scores.csv"))
        s = s[s["split"] == a.split]
        ctrl = [1, 7, 14, 30, 60]

        def get(name):
            q = s[s["model"] == name].set_index("horizon")
            return q if len(q) else None
        q = get(m)
        if q is None:
            print(f"[skip] {m}: no {a.split} scores in {run}")
            continue
        r = dict(model=m, change=CHANGE.get(m, ""), **{f"h{h}": 100 * q.loc[h, "sysmape"] for h in ctrl})
        r["headline"] = sum(r[f"h{h}"] for h in ctrl) / len(ctrl)
        r["all_h"] = 100 * q["sysmape"].mean()
        r["district_headline"] = 100 * q.loc[ctrl, "district_sysmape"].mean()
        r["bias_h60"] = 100 * q.loc[60, "bias"]
        r["days_h1"] = int(q.loc[1, "days"])
        r["trees_cb/xgb"] = trees(root, m)
        r["run"] = os.path.basename(run)
        rows.append(r)
        bq = get("B0")
        if bq is not None and b0 is None:
            b0 = dict(model="B0", change="reference, same rows", **{f"h{h}": 100 * bq.loc[h, "sysmape"] for h in ctrl})
            b0["headline"] = sum(b0[f"h{h}"] for h in ctrl) / len(ctrl)
            b0["all_h"] = 100 * bq["sysmape"].mean()
            b0["district_headline"] = 100 * bq.loc[ctrl, "district_sysmape"].mean()
            b0["bias_h60"] = 100 * bq.loc[60, "bias"]
            b0["days_h1"] = int(bq.loc[1, "days"])
    if not rows:
        raise SystemExit("no runs found")
    t = pd.DataFrame(rows + ([b0] if b0 else []))
    base = t.iloc[0]
    t["d_headline"] = t["headline"] - base["headline"]
    t["d_all_h"] = t["all_h"] - base["all_h"]
    days = t["days_h1"].dropna().unique()
    if len(days) > 1:
        print(f"WARNING: the runs scored different numbers of issue dates {sorted(days)}: compare like with like")
    cols = ["model", "change", "h1", "h7", "h14", "h30", "h60", "headline", "d_headline", "all_h", "d_all_h",
            "district_headline", "bias_h60", "days_h1", "trees_cb/xgb"]
    pd.set_option("display.width", 250)
    print(f"\nNational sysMAPE, % ({a.split}); d_* = change vs {base['model']} in points (below 0 = better)\n")
    print(t[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    out = os.path.join(root, "ablations", f"{a.name}_{time.strftime('%Y%m%d-%H%M%S')}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t.to_csv(out, index=False, float_format="%.4f")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
