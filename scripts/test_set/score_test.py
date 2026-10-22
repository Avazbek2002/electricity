#!/usr/bin/env python
"""
The common scorer (ML guide part 4): any model's district forecasts against the district target T2, with B0 on the
same rows, national = sum of districts on a common panel, both splits, the ТЗ check and the bias audit.

    # harness runs score themselves; to re-score a run folder
    python scripts/test_set/score_test.py --forecasts ~/work/eval/runs/<run>/forecasts/gam

    # any other model (district level, or client level summed to districts)
    python scripts/test_set/score_test.py --template ~/work/my_forecasts --sample      # the rows to fill in
    python scripts/test_set/score_test.py --forecasts ~/work/my_forecasts
    python scripts/test_set/score_test.py --forecasts ~/work/my_forecasts --compare gam=~/work/eval/runs/<run>/forecasts/gam

Submission: <dir>/forecast_meta.json and <dir>/district.csv[.gz] with node, origin, horizon, forecast.
    forecast = the district's T2 (kWh) on day origin + horizon, issued on `origin` with data up to origin - 1.
    Rules (guide parts 4-5): history up to o - 1; a model fitted at refit R trained on targets d <= R - 1; no
    target-day weather in the standard run; choose on validation, look at the test only for the chosen variants.
    Rows missing for a model are dropped for every compared model (reported); the national sum needs >= 98 % of the
    districts present.
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
    p.add_argument("--test-dir", default=None, help="built evaluation set (default ~/work/test_set/<version>)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--forecasts", help="folder with forecast_meta.json and district.csv[.gz]")
    g.add_argument("--template", help="write the rows a submission contains into this folder")
    p.add_argument("--sample", action="store_true", help="template: the guide's 5 districts only")
    p.add_argument("--districts", default=None, help="template: comma list of districts")
    p.add_argument("--target", default="T2", choices=["T2", "T3"], help="T3 = experimental (reported separately)")
    p.add_argument("--compare", action="append", default=[], help="name=forecast_dir of another model, same rows")
    p.add_argument("--out-dir", default=None, help="where to write the scores (default <forecasts>/scores)")
    a = p.parse_args()
    spec = ts.load_spec(a.spec)
    test_dir = os.path.expanduser(a.test_dir or os.path.join("~/work/test_set", spec["version"]))
    man = ts.read_manifest(test_dir)
    ts.check_same_spec(spec, man)
    if a.template:
        import pandas as pd
        all_d = pd.read_csv(os.path.join(test_dir, "districts.csv"), dtype=str)["district"].tolist()
        dist = spec["sample_districts"] if a.sample else (a.districts.split(",") if a.districts else all_d)
        n = ts.write_template(spec, os.path.expanduser(a.template), dist)
        print(f"[template] {a.template}: {n:,} rows ({len(dist)} districts x issue dates x horizons, both splits)")
        print("  fill `forecast` (district T2 in kWh), set `model` (and `weather`, `level`) in forecast_meta.json, "
              "then score with --forecasts")
        return
    others = {}
    for c in a.compare:
        name, _, path = c.partition("=")
        others[name] = os.path.expanduser(path)
    fdir = os.path.expanduser(a.forecasts)
    res = ts.score(fdir, test_dir, spec, target=a.target, others=others)
    text = ts.report(res, spec)
    print(text)
    out = os.path.expanduser(a.out_dir or os.path.join(fdir, "scores" if a.target == "T2" else f"scores_{a.target}"))
    ts.save_scores(res, out, text)
    print(f"\n[score] written to {out}")
    try:                                                           # metrics into the MLflow run run_script opened
        sys.path.insert(0, os.path.join(REPO, "core"))
        import mlflow_utils as mu
        tr = mu.Tracker("electricity/test", enabled=None)
        if tr.enabled and tr.mlflow.active_run():
            tr.metrics(ts.metrics_for_mlflow(res))
            tr.tags({"model": res["model"], "test_set_version": spec["version"], "target": a.target,
                     "weather": res["meta"].get("weather", "?"), "districts": str(len(res["districts"]))})
            for f in ("scores.csv", "spec_check.csv", "bias_audit.csv", "report.txt", "national_daily.csv"):
                tr.artifact(os.path.join(out, f), subdir="scores")
    except Exception as e:                                         # noqa: BLE001
        print(f"[mlflow] WARNING: metrics not logged ({type(e).__name__}: {e})")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(REPO, "core"))
    from mlflow_utils import run_script
    run_script(main, "electricity/test", stage="score_test", out_flags=("--no-out-dir",))
