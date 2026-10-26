#!/usr/bin/env python
"""Per-horizon blend of node forecasts (household models + the node B0), chosen on the dev substations.

    python scripts/node_blend.py --latest xgboost_all_s25 nlinear_x --out ~/work/eval/node_blend_dev
    python scripts/node_blend.py --role test --latest xgboost_all_s25 nlinear_x \
        --apply ~/work/eval/node_blend_dev/weights.json --out ~/work/eval/node_blend_test        # once, on the test

Weights: for each horizon block (1-7, 8-14, 15-30, 31-45, 46-60) the convex combination of the models and B0 on a 0.05
grid that minimises the mean per-node sysMAPE (each node-day weighted by 1 / its node's actual total in the block, so a
large substation does not outweigh small feeders). Substations and feeders share one set of weights.

Honest dev score: leave-one-quarter-out (weights fitted on three quarters of the 2025 issue dates, scored on the fourth,
the four held-out quarters put together). The weights saved are the ones fitted on all of dev. The test run only
applies saved weights (--apply): nothing is fitted on the test substations.
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO)
import harness  # noqa: E402,F401
from harness import nodeset as ns  # noqa: E402

BLOCKS = [(1, 7), (8, 14), (15, 30), (31, 45), (46, 60)]
STEP = 0.05


def load(runs, latest, role, eval_dir):
    runs = list(runs)
    for m in latest or []:
        found = sorted(glob.glob(os.path.join(os.path.expanduser(eval_dir), "runs", f"*_{m}_nodes-{role}")))
        if not found:
            raise SystemExit(f"no node run of {m} ({role}) under {eval_dir}/runs")
        runs.append(found[-1])
    models, test_dir = {}, None
    for r in runs:
        fd = os.path.join(os.path.expanduser(r), "forecasts")
        for d in sorted(glob.glob(os.path.join(fd, "*"))):
            name = os.path.basename(d)
            if name.endswith("_unscaled"):
                continue
            meta = json.load(open(os.path.join(d, "forecast_meta.json")))
            if meta["role"] != role:
                raise SystemExit(f"{r} is a {meta['role']} run, not {role}")
            test_dir = test_dir or meta["test_dir"]
            models[name] = ns.read_forecasts(d)
    return models, test_dir


def simplex(k, step=STEP):
    n = int(round(1 / step))
    out = [np.array(c) / n for c in itertools.product(range(n + 1), repeat=k) if sum(c) == n]
    return np.stack(out)


def fit_weights(rows, names, W):
    """-> {block: weight vector} minimising the node-normalised absolute error."""
    out = {}
    for a, b in BLOCKS:
        q = rows[(rows["horizon"] >= a) & (rows["horizon"] <= b)]
        tot = q.groupby("node")["y"].transform("sum").to_numpy()
        wt = 1.0 / np.where(tot > 0, tot, np.nan)
        P = q[names].to_numpy(float)
        y = q["y"].to_numpy(float)
        best, bw = np.inf, None
        for i in range(0, len(W), 64):                         # chunks of weight vectors
            E = np.abs(P @ W[i:i + 64].T - y[:, None])
            s = np.nansum(E * wt[:, None], axis=0)
            j = int(np.argmin(s))
            if s[j] < best:
                best, bw = s[j], W[i + j]
        out[f"{a}-{b}"] = [float(x) for x in bw]
    return out


def apply(rows, names, weights):
    f = np.full(len(rows), np.nan)
    P = rows[names].to_numpy(float)
    for (a, b) in BLOCKS:
        m = ((rows["horizon"] >= a) & (rows["horizon"] <= b)).to_numpy()
        f[m] = P[m] @ np.asarray(weights[f"{a}-{b}"])
    return rows[["node", "origin", "horizon"]].assign(forecast=f)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*")
    p.add_argument("--latest", nargs="*", default=None)
    p.add_argument("--role", default="dev", choices=["dev", "test"])
    p.add_argument("--eval-dir", default="~/work/eval")
    p.add_argument("--spec", default=os.path.join(REPO, "configs", "test_set_substations_2025.yaml"))
    p.add_argument("--apply", default=None, help="weights.json from a dev run (required for --role test)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.role == "test" and not a.apply:
        raise SystemExit("on the test substations the blend is only applied (--apply <dev weights.json>)")
    spec = ns.load_spec(a.spec)
    models, test_dir = load(a.runs, a.latest, a.role, a.eval_dir)
    ns.check_same_spec(spec, ns.read_manifest(test_dir))
    targets, nodes = ns.load_targets(test_dir), ns.load_nodes(test_dir)
    want = set(nodes.loc[nodes["role"] == a.role, "node"])
    b0 = ns.b0_forecasts(targets, spec, nodes=want, issue=ns.issue_dates(spec, a.role))
    models[ns.REFERENCE] = b0[["node", "origin", "horizon", "forecast"]]
    names = list(models)
    rows, _ = ns.common_rows(models, targets, spec, a.role, nodes)
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    if a.apply:
        weights = json.load(open(os.path.expanduser(a.apply)))
        if weights["models"] != names:
            raise SystemExit(f"weights are for {weights['models']}, these runs give {names}")
        blends = {"blend": apply(rows, names, weights["weights"])}
    else:
        W = simplex(len(names))
        print(f"[blend] {len(names)} components {names}, {len(W)} weight vectors per horizon block, {len(rows):,} rows")
        full = fit_weights(rows, names, W)
        q = rows["origin"].dt.quarter
        parts = []
        for k in sorted(q.unique()):
            wk = fit_weights(rows[q != k], names, W)
            parts.append(apply(rows[q == k], names, wk))
            print(f"[blend] quarter {k} held out: " + " | ".join(
                f"h{b}: " + " ".join(f"{n} {w:.2f}" for n, w in zip(names, v) if w > 0) for b, v in wk.items()))
        blends = {"blend_cv": pd.concat(parts, ignore_index=True), "blend_insample": apply(rows, names, full)}
        weights = dict(models=names, weights=full, blocks=BLOCKS, step=STEP, role=a.role)
        json.dump(weights, open(os.path.join(out, "weights.json"), "w"), indent=2)
        print("\nWEIGHTS (all of dev; saved to weights.json for the one test run)")
        for b, v in full.items():
            print(f"  h{b:6s} " + "  ".join(f"{n} {w:.2f}" for n, w in zip(names, v)))
    res = ns.score({**blends, **models}, targets, spec, a.role, nodes)
    text = ns.report(res, list(blends) + names)
    print("\n" + text)
    ns.save(res, out, text)
    for k, v in blends.items():
        d = os.path.join(out, "forecasts", k)
        os.makedirs(d, exist_ok=True)
        v.to_csv(os.path.join(d, "node.csv.gz"), index=False, float_format="%.6f")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
