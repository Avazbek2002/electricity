#!/usr/bin/env python
"""
The common scorer: any model's forecasts for the fixed test set (configs/test_set.yaml) against the frozen actuals.

    # harness models: evaluate.py --mode test writes and scores the forecasts itself; to re-score a run folder
    python scripts/test_set/score_test.py --forecasts ~/work/eval/runs/<run>/forecasts/gam

    # any other model (e.g. a direct feeder / substation model): get the rows to fill in, fill `forecast`, score
    python scripts/test_set/score_test.py --template ~/work/my_model_forecasts
    python scripts/test_set/score_test.py --forecasts ~/work/my_model_forecasts --reference ~/work/eval/runs/<run>/forecasts/persistence30_ref

Submission format (one file per level you forecast; any subset of levels):
    <dir>/forecast_meta.json    {"model": "...", "test_set_version": "...", "test_set_sha256": "..."} (the template writes it)
    <dir>/<level>.csv[.gz]      fold, node, origin, horizon, forecast [, forecast_observed_sum, observed]
        forecast = the node's coverage-adjusted total kWh on day origin + horizon (see harness/testset.py)
    Every row of the template must be there, once, with a finite forecast; otherwise the level is refused.
    Rules: a forecast for origin t may use data up to day t only (node history: history/<fold>/<level>.csv.gz rows
    with date <= t), and each fold's model may be trained only on data up to that fold's first_origin.
"""
import argparse
import os
import sys

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
from harness import testset as ts  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", default=os.path.join(REPO, "configs", "test_set.yaml"))
    p.add_argument("--test-dir", default=None, help="built test set (default ~/work/test_set/<version>)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--forecasts", help="folder with forecast_meta.json and <level>.csv[.gz]")
    g.add_argument("--template", help="write the rows a submission must contain into this folder")
    p.add_argument("--reference", default=None, help="forecast folder of a reference model, for relative scores")
    p.add_argument("--levels", default=None, help="comma list (default: every level present)")
    p.add_argument("--out-dir", default=None, help="where to write score_*.csv (default <forecasts>/scores)")
    a = p.parse_args()
    spec = ts.load_spec(a.spec)
    test_dir = os.path.expanduser(a.test_dir or os.path.join("~/work/test_set", spec["version"]))
    levels = a.levels.split(",") if a.levels else None
    if a.template:
        out = os.path.expanduser(a.template)
        man = ts.read_manifest(test_dir)
        ts.check_same_spec(spec, man)
        n = ts.write_template(test_dir, spec, out, levels)
        print(f"[template] {out}: " + ", ".join(f"{lv} {k:,} rows" for lv, k in n.items()))
        print("  fill `forecast` (coverage-adjusted node total, kWh), set `model` in forecast_meta.json, then score with "
              "--forecasts")
        return
    fdir = os.path.expanduser(a.forecasts)
    res = ts.score(fdir, test_dir, spec, levels)
    ref = ts.score(os.path.expanduser(a.reference), test_dir, spec, res["levels"]) if a.reference else None
    text = ts.report(res, spec, reference=ref)
    print(text)
    out = os.path.expanduser(a.out_dir or os.path.join(fdir, "scores"))
    ts.save_scores(res, out, text)
    print(f"\n[score] written to {out}")
    try:                                                           # metrics into the MLflow run run_script opened
        sys.path.insert(0, os.path.join(REPO, "core"))
        import mlflow_utils as mu
        tr = mu.Tracker("electricity/test", enabled=None)
        if tr.enabled and tr.mlflow.active_run():
            tr.metrics(ts.metrics_for_mlflow(res))
            tr.tags({"model": res["meta"].get("model", "?"), "test_set_version": spec["version"],
                     "spec_pass": str(bool(res["spec_check"]["pass"].all()))})
    except Exception as e:                                         # noqa: BLE001
        print(f"[mlflow] WARNING: metrics not logged ({type(e).__name__}: {e})")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(REPO, "core"))
    from mlflow_utils import run_script
    run_script(main, "electricity/test", stage="score_test", out_flags=("--out-dir",))
