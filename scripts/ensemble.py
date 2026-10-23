#!/usr/bin/env python
"""
Ensemble of already evaluated models: the (weighted) mean of their district forecasts, scored by the common scorer
on exactly the same rows as every member and B0. Nothing is retrained.

    python scripts/ensemble.py --runs ~/work/eval/runs/<xgboost run> ~/work/eval/runs/<catboost run> \
        ~/work/eval/runs/<nlinear_x run> --test-set configs/test_set_2025.yaml
    python scripts/ensemble.py --runs ... --weights 0.4,0.2,0.4 --name ens_xgb_cb_nl_w

Each --runs folder is an evaluate.py run (forecasts/<model>/district.csv.gz + forecast_meta.json), all on the same test
set, split(s) and districts. The ensemble is defined on the rows every member forecast (inner join); members, B0 and
the ensemble are then scored on the same days. Validation runs give a validation ensemble; for the test, ensemble the
members' `--mode test` runs (choose weights on validation only).

Also averaged when every member wrote them: the supplementary substation / feeder bottom-up sums (same households, so
the actual sums agree). Household / transformer error sums (levels.csv.gz) cannot be averaged and are not reported.

Output: <out-root>/runs/<time>_<name>_ensemble/ with forecasts/<name>/ (district.csv.gz, network/, forecast_meta.json)
and scores/ (report.txt, scores.csv, ... and charts) -- a run folder that scripts/compare_models.py accepts. Unless
--no-wandb, the ensemble and its members are logged to W&B with compare_models.py.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO)
from harness import testset as ts  # noqa: E402

KEY = ["node", "origin", "horizon"]


def member(run, unscaled=False):
    fd = os.path.join(run, "forecasts")
    names = [d for d in sorted(os.listdir(fd)) if not d.endswith("_unscaled")]
    if len(names) != 1:
        raise SystemExit(f"{run}: expected one model folder in forecasts/, found {names}")
    d = os.path.join(fd, names[0] + ("_unscaled" if unscaled else ""))
    if not os.path.isdir(d):
        raise SystemExit(f"{run}: no {os.path.basename(d)} forecasts")
    meta = json.load(open(os.path.join(fd, names[0], "forecast_meta.json")))
    return names[0], d, meta


def combine(frames, weights, value):
    """Inner join on KEY of each member's `value`, weighted mean -> DataFrame KEY + value, n rows per member dropped."""
    m = None
    for i, f in enumerate(frames):
        g = f[KEY + [value]].rename(columns={value: f"m{i}"})
        m = g if m is None else m.merge(g, on=KEY, how="inner")
    cols = [f"m{i}" for i in range(len(frames))]
    m = m[np.isfinite(m[cols]).all(axis=1)]
    w = np.asarray(weights, dtype=float)
    m[value] = m[cols].to_numpy() @ (w / w.sum())
    return m[KEY + [value]]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="evaluate.py run folders of the member models")
    p.add_argument("--weights", default=None, help="comma list, one per run (default: equal); normalised to sum 1")
    p.add_argument("--name", default=None, help="ensemble name (default ens_<member1>_<member2>_...)")
    p.add_argument("--unscaled", action="store_true", help="average the members' unscaled forecasts instead")
    p.add_argument("--test-set", default=os.path.join(REPO, "configs", "test_set.yaml"))
    p.add_argument("--test-dir", default=None, help="default: the one recorded in the first run")
    p.add_argument("--config", default=os.path.join(REPO, "configs", "eval.yaml"))
    p.add_argument("--out-root", default="~/work/eval")
    p.add_argument("--no-wandb", action="store_true")
    a = p.parse_args(argv)
    spec = ts.load_spec(a.test_set)
    runs = [os.path.expanduser(r.rstrip("/")) for r in a.runs]
    if len(runs) < 2:
        raise SystemExit("an ensemble needs at least two runs")
    mem = [member(r, a.unscaled) for r in runs]
    names = [n for n, _, _ in mem]
    if len(set(names)) != len(names):
        raise SystemExit(f"the same model twice: {names}")
    weights = [1.0] * len(runs) if a.weights is None else [float(x) for x in a.weights.split(",")]
    if len(weights) != len(runs) or min(weights) < 0 or sum(weights) <= 0:
        raise SystemExit("--weights: one non-negative number per run")
    first = mem[0][2]
    for n, _, m in mem:
        if m.get("test_set_sha256") != spec["_sha256"]:
            raise SystemExit(f"{n} was made for another test set definition than {a.test_set}: pass the right --test-set")
        for k in ("districts", "splits", "target", "weather"):
            if m.get(k) != first.get(k):
                raise SystemExit(f"{n} differs from {names[0]} in `{k}` ({m.get(k)} vs {first.get(k)}): "
                                 f"ensemble runs of the same scope only")
    test_dir = os.path.expanduser(a.test_dir or first.get("test_dir") or "")
    if not test_dir or not os.path.exists(os.path.join(test_dir, "manifest.json")):
        raise SystemExit("pass --test-dir (the built test set these runs were scored on)")
    name = a.name or "ens_" + "_".join(names) + ("_unscaled" if a.unscaled else "")

    # ---- district forecasts
    frames = [ts.load_forecasts(d) for _, d, _ in mem]
    ens = combine(frames, weights, "forecast")
    lost = {n: int(len(f) - len(ens)) for n, f in zip(names, frames)}
    out = os.path.join(os.path.expanduser(a.out_root), "runs", f"{time.strftime('%Y%m%d-%H%M%S')}_{name}_ensemble")
    fdir = os.path.join(out, "forecasts", name)
    os.makedirs(fdir, exist_ok=True)
    e = ens.copy()
    e["origin"] = e["origin"].dt.date
    e.to_csv(os.path.join(fdir, "district.csv.gz"), index=False, float_format="%.6f")
    w = np.asarray(weights) / sum(weights)
    print(f"[ensemble] {name} = " + " + ".join(f"{wi:.3f} x {n}" for wi, n in zip(w, names)) +
          f" | {len(ens):,} district rows (rows only some members had: {lost})")

    # ---- supplementary network sums (same households in every member)
    net_done = []
    for lv in ("substation", "feeder"):
        parts = [ts._read(os.path.join(d, "network", f"{lv}.csv")) for _, d, _ in mem]
        if any(x is None for x in parts):
            continue
        k = KEY
        base = parts[0][k + ["actual_observed_sum", "observed"]]
        same = all(np.allclose(base.merge(x[k + ["actual_observed_sum"]], on=k, suffixes=("", "_b"))
                               .pipe(lambda q: q["actual_observed_sum"] - q["actual_observed_sum_b"]), 0, atol=1e-3)
                   for x in parts[1:])
        if not same:
            print(f"[ensemble] {lv}: members forecast different households; not averaged")
            continue
        avg = combine(parts, weights, "forecast_observed_sum").merge(base, on=k, how="left")
        os.makedirs(os.path.join(fdir, "network"), exist_ok=True)
        avg.to_csv(os.path.join(fdir, "network", f"{lv}.csv.gz"), index=False, float_format="%.6f")
        net_done.append(lv)

    meta = dict(model=name, test_set_version=spec["version"], test_set_sha256=spec["_sha256"], target=first.get("target"),
                weather=first.get("weather"), level="ensemble of " + ", ".join(names), scope=first.get("scope"),
                districts=first.get("districts"), splits=first.get("splits"), test_dir=test_dir,
                members=[dict(model=n, run=r, weight=float(wi), forecasts=d, commit=m.get("git_commit"),
                              substitution_test=(m.get("substitution_test") or {}).get("passed"))
                         for (n, d, m), r, wi in zip(mem, runs, w)],
                unscaled=a.unscaled, network_levels=net_done,
                notes=[f"ensemble: weighted mean of district forecasts ({', '.join(f'{n} {wi:.3f}' for n, wi in zip(names, w))}); "
                       "no retraining; each member passed its own substitution test"])
    json.dump(meta, open(os.path.join(fdir, "forecast_meta.json"), "w"), indent=2, default=str)

    # ---- score: ensemble, every member and B0 on the same rows
    res = ts.score(fdir, test_dir, spec, others={n: d for n, d, _ in mem})
    text = ts.report(res, spec)
    print("\n" + text)
    sdir = os.path.join(out, "scores")
    ts.save_scores(res, sdir, text)
    wf = os.path.join(runs[0], "scores", "weather_daily.csv")          # for the window charts' temperature panel
    if os.path.exists(wf):
        import shutil
        shutil.copyfile(wf, os.path.join(sdir, "weather_daily.csv"))
    rows = []
    for split, s in res["splits"].items():
        t = s["scores"]
        if not t["days"].sum():
            continue
        hd = {m: float(t[(t["model"] == m) & t["horizon"].isin(res["ctrl"])]["sysmape"].mean()) for m in res["models"]}
        rows += [dict(split=split, model=m, headline=v) for m, v in hd.items()]
        print(f"\n=== {split}: headline (national sysMAPE, mean of h {res['ctrl']}) on the same days ===")
        for m, v in sorted(hd.items(), key=lambda x: x[1]):
            print(f"  {m:28s} {v:7.2%}")
    pd.DataFrame(rows).to_csv(os.path.join(sdir, "headlines.csv"), index=False)
    try:
        from harness import plots
        plots.metrics_by_horizon(res, sdir)
        print(f"[charts] {sdir}: metrics by horizon, {len(plots.window_samples(res, sdir))} window charts")
    except Exception as ex:                                        # noqa: BLE001
        print(f"[charts] WARNING: not drawn ({type(ex).__name__}: {ex})")
    print(f"\n[done] {out}")
    if not a.no_wandb:
        import importlib.util
        sp = importlib.util.spec_from_file_location("compare_models", os.path.join(REPO, "scripts", "compare_models.py"))
        compare_models = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(compare_models)
        compare_models.main(["--runs", out, *runs, "--test-set", a.test_set, "--test-dir", test_dir, "--config",
                             a.config, "--name", f"{name} vs members"])
    return out


if __name__ == "__main__":
    main()
