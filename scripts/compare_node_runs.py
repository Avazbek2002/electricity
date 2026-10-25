#!/usr/bin/env python
"""Compare node runs (evaluate.py ... --test-set configs/test_set_substations_2025.yaml) on the SAME node-days.

    python scripts/compare_node_runs.py ~/work/eval/runs/<run1> ~/work/eval/runs/<run2> ...  [--unscaled] [--out DIR]
    python scripts/compare_node_runs.py --latest tide_agg tide_daily xgboost nlinear_x       # newest run of each model

Each run's own model forecasts (and with --unscaled also <model>_unscaled) plus the node B0 are scored on the rows every
one of them forecasts (median per-node sysMAPE, pooled, bias, share of nodes better than B0, by size band).
"""
import argparse
import glob
import json
import os
import sys

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO)
import harness  # noqa: E402,F401
from harness import nodeset as ns  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*")
    p.add_argument("--latest", nargs="*", default=None, help="model names: the newest node run of each")
    p.add_argument("--role", default="dev", choices=["dev", "test"])
    p.add_argument("--eval-dir", default="~/work/eval")
    p.add_argument("--spec", default=os.path.join(REPO, "configs", "test_set_substations_2025.yaml"))
    p.add_argument("--unscaled", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    runs = list(a.runs)
    for m in a.latest or []:
        found = sorted(glob.glob(os.path.join(os.path.expanduser(a.eval_dir), "runs", f"*_{m}_nodes-{a.role}")))
        if not found:
            raise SystemExit(f"no node run of {m} ({a.role}) under {a.eval_dir}/runs")
        runs.append(found[-1])
    if not runs:
        raise SystemExit("give run folders or --latest <models>")
    spec = ns.load_spec(a.spec)
    models, test_dir, role = {}, None, None
    for r in runs:
        r = os.path.expanduser(r)
        for fd in sorted(glob.glob(os.path.join(r, "forecasts", "*"))):
            name = os.path.basename(fd)
            if name.endswith("_unscaled") and not a.unscaled:
                continue
            meta_p = os.path.join(fd, "forecast_meta.json")
            if os.path.exists(meta_p):
                meta = json.load(open(meta_p))
                test_dir, role = test_dir or meta["test_dir"], role or meta["role"]
                if meta["role"] != role:
                    raise SystemExit(f"{r} is a {meta['role']} run, the others {role}")
            if name in models:
                name = f"{name}@{os.path.basename(r)[:15]}"
            models[name] = ns.read_forecasts(fd)
    ns.check_same_spec(spec, ns.read_manifest(test_dir))
    targets, nodes = ns.load_targets(test_dir), ns.load_nodes(test_dir)
    want = set(nodes.loc[nodes["role"] == role, "node"])
    b0 = ns.b0_forecasts(targets, spec, nodes=want, issue=ns.issue_dates(spec, role))
    models[ns.REFERENCE] = b0[["node", "origin", "horizon", "forecast"]]
    res = ns.score(models, targets, spec, role, nodes)
    text = ns.report(res, list(models))
    print(text)
    if a.out:
        ns.save(res, os.path.expanduser(a.out), text)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
