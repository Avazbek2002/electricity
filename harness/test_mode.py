"""evaluate.py --mode test: the fixed test set (configs/test_set.yaml) for any harness model.

Per fold: fit on data before the fold (cached per model, commit and cutoff). Then, per district (one worker reads it
once for all folds): forecast the fold's fixed panel at every daily origin, and add per node and (origin, horizon)
the forecasts of the panel households observed on the target day. Forecast CSVs are written in the common format
(harness/testset.py) for the model and for the 30-day persistence reference, then scored by the common scorer.
"""
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import shutil
import time

import numpy as np
import pandas as pd

from harness import testset as ts
from harness.contract import HORIZON, Panel, TrainSpec
from harness.data import read_district, restore_credentials, scrub_credentials
from harness.runner import REFERENCE, load_model, persistence30
from harness.weather import district_weather, issue_index


# ----------------------------------------------------------------------------- worker: one district, every fold
def test_worker(code, cfg):
    t0 = time.time()
    import pyarrow.parquet as pq
    import whole_pipeline_v8 as wp
    spec = cfg["spec"]
    d = spec["data"]
    restore_credentials()
    D = read_district(code, [dict(uri=d["uri"], env_prefix=d.get("env_prefix", ""), start=d["start"], end=d["end"])],
                      d["start"], d["end"], d["labels"])
    res = dict(code=str(code), folds={}, gaps=0, windows=0, seconds=0.0)
    if not len(D["keys"]):
        return res
    M, wt, dates = D["M"], D["wt"], D["dates"]
    row_of = {k: i for i, k in enumerate(D["keys"])}
    clim, grid, grid_mm = district_weather(code, wt, dates, cfg["seed"], wp)
    region = cfg["regions"].get(str(code), "unknown")
    panels = {}
    for f in spec["folds"]:                                         # read the panels before S3 keys are hidden
        p = os.path.join(cfg["test_dir"], "panel", f["name"], f"{code}.parquet")
        keys = pq.read_table(p).column("customer_key").to_pylist() if os.path.exists(p) else []
        missing = [k for k in keys if k not in row_of]
        if missing:
            raise RuntimeError(f"district {code} fold {f['name']}: {len(missing)} panel households not in the data "
                               f"(the data changed since the test set was built)")
        panels[f["name"]] = np.asarray([row_of[k] for k in keys], dtype=np.int64)
    scrub_credentials()
    W, blk = int(cfg["history_days"]), int(cfg["block"])
    lv_h = {lv: np.asarray(ts.level_horizons(spec, lv)) - 1 for lv in cfg["levels"]}
    for f in spec["folds"]:
        r_all = panels[f["name"]]
        if not len(r_all):
            continue
        model = load_model(cfg["model_dir"], cfg["model_config"])
        model.load(cfg["artifacts"][f["name"]])
        test_idx, origins = ts.fold_days(spec, f)
        t_lo, t_hi = int(test_idx[0]), int(test_idx[-1])
        G = ts.groups(D["network"][r_all], code, region, cfg["levels"])
        acc = {}
        for lv, (g, names) in G.items():
            K, nh = len(names), len(lv_h[lv])
            acc[lv] = dict(names=names, N=np.bincount(g[g >= 0], minlength=K).astype(float),
                           FO=np.zeros((K, len(origins), nh)), FR=np.zeros((K, len(origins), nh)),
                           NO=np.zeros((K, len(origins), nh)))
        for oi, o in enumerate(origins):
            o = int(o)
            lo = max(0, o - W + 1)
            gi = issue_index(clim, dates[o])
            g1 = np.full_like(grid, np.nan)
            g1[:, gi] = grid[:, gi]                                 # only the forecast issued on this origin
            gm = {k: np.full_like(v, np.nan) for k, v in grid_mm.items()}
            for k in gm:
                gm[k][:, gi] = grid_mm[k][:, gi]
            tgt = o + 1 + np.arange(HORIZON)
            in_test = (tgt >= t_lo) & (tgt <= t_hi)
            for a in range(0, len(r_all), blk):
                pos = np.arange(a, min(a + blk, len(r_all)))
                r = r_all[pos]
                panel = Panel(coato=str(code), origin=dates[o], keys=D["keys"][r], kwh=M[r, lo:o + 1].copy(),
                              dates=dates[lo:o + 1], weather=wt[:, lo:o + 1].copy(), housing=D["housing"][r].copy(),
                              network=D["network"][r].copy(),
                              pipeline=dict(clim=clim, fc_grid=g1, fc_grid_mm=gm, seed=int(cfg["seed"])))
                P = np.asarray(model.predict(panel), dtype=np.float64)
                if P.shape != (len(r), HORIZON):
                    raise RuntimeError(f"{cfg['model_name']}.predict returned {P.shape}, expected {(len(r), HORIZON)}")
                ref = persistence30(panel.kwh)
                bad = ~np.isfinite(P) | (P < 0)
                res["gaps"] += int(bad[:, in_test].sum())
                P = np.where(bad, ref[:, None], P)
                obs = np.isfinite(M[r, o + 1:o + 1 + HORIZON]) & in_test[None, :]   # actuals: never shown to the model
                Pobs, Robs, Ob = np.where(obs, P, 0.0), np.where(obs, ref[:, None], 0.0), obs.astype(np.float64)
                for lv, (g, names) in G.items():
                    hz, K, gb = lv_h[lv], len(names), g[pos]
                    acc[lv]["FO"][:, oi, :] += ts.bincount2(gb, Pobs[:, hz], K)
                    acc[lv]["FR"][:, oi, :] += ts.bincount2(gb, Robs[:, hz], K)
                    acc[lv]["NO"][:, oi, :] += ts.bincount2(gb, Ob[:, hz], K)
                res["windows"] += len(r)
        res["folds"][f["name"]] = acc
    res["seconds"] = time.time() - t0
    res["peak_bytes"] = ts.peak_rss_bytes()
    return res


# ----------------------------------------------------------------------------- writing forecasts
def write_forecasts(fdir, spec, merged, model_label, meta, key):
    """merged[fold][level] = NodeSum with N, FO/FR, NO -> <fdir>/<level>.csv[.gz] in the common format."""
    os.makedirs(fdir, exist_ok=True)
    mn = int(spec["scoring"]["min_households"])
    start = pd.Timestamp(spec["data"]["start"])
    for lv in next(iter(merged.values())).keys():
        parts = []
        hz = np.asarray(ts.level_horizons(spec, lv))
        for f in spec["folds"]:
            ns = merged[f["name"]][lv]
            names = ns.names()
            if not names:
                continue
            N, FO, NO = ns.stack("N"), ns.stack(key), ns.stack("NO")
            test_idx, origins = ts.fold_days(spec, f)
            keep = np.flatnonzero(N >= mn)
            o_grid, h_grid = np.meshgrid(origins, hz, indexing="ij")             # (nO, nh)
            tgt = o_grid + h_grid
            ok = (tgt >= test_idx[0]) & (tgt <= test_idx[-1])
            if not len(keep):
                continue
            oo, hh = o_grid[ok], h_grid[ok]
            fo, no = FO[keep][:, ok], NO[keep][:, ok]                            # (K, rows per node)
            with np.errstate(invalid="ignore", divide="ignore"):
                fc = np.where(no > 0, N[keep][:, None] * fo / no, np.nan)
            K, R = fo.shape
            parts.append(pd.DataFrame({"fold": f["name"], "node": np.repeat(np.asarray(names)[keep], R),
                                       "origin": np.tile((start + pd.to_timedelta(oo, unit="D")).date, K),
                                       "horizon": np.tile(hh, K), "forecast": fc.ravel(),
                                       "forecast_observed_sum": fo.ravel(), "observed": no.ravel().astype(int)}))
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(os.path.join(fdir, ts.level_file(lv)), index=False,
                                                       float_format="%.10g")
    json.dump({**meta, "model": model_label}, open(os.path.join(fdir, "forecast_meta.json"), "w"), indent=2)


# ----------------------------------------------------------------------------- the command
def run_test(args, cfg, git, model_dir, model_config, tracking, scoring):
    spec = ts.load_spec(args.test_set)
    test_dir = os.path.expanduser(args.test_dir or os.path.join("~/work/test_set", spec["version"]))
    man = ts.read_manifest(test_dir)
    ts.check_same_spec(spec, man)
    if man.get("trial"):
        print(f"[test] NOTE: {test_dir} is a trial build on a few districts, not the released test set")
    run_cfg = cfg["run"]
    out_root = os.path.expanduser(args.out_dir or run_cfg["out_dir"])
    workers = args.workers or int(run_cfg["workers"])
    d = spec["data"]
    regions = pd.read_csv(os.path.join(test_dir, "districts.csv"), dtype=str).set_index("district")["region"].to_dict()
    codes = sorted(regions)
    if args.coatos:
        codes = [c for c in codes if c in set(args.coatos.split(","))]
    levels = man["levels"]

    # ---- one fit per fold (cached per model, commit and cutoff)
    arts, fit_seconds = {}, 0.0
    import whole_pipeline_v8 as wp
    for f in spec["folds"]:
        key = f"{args.model}/{git['short']}{'-dirty' if git['dirty'] else ''}/cutoff-{f['cutoff']}"
        art = os.path.join(out_root, "artifacts", key)
        done = os.path.join(art, ".fit_done")
        if args.refit and os.path.exists(art):
            shutil.rmtree(art)
        if os.path.exists(done):
            print(f"[fit] {f['name']}: reusing {art}")
        else:
            os.makedirs(art, exist_ok=True)
            data_end = str(min(pd.Timestamp(f["cutoff"]) + pd.Timedelta(days=HORIZON), pd.Timestamp(d["end"])).date())
            spec_t = TrainSpec(roots=[dict(uri=d["uri"], env_prefix=d.get("env_prefix", ""))], start=d["start"],
                               cutoff=f["cutoff"], data_end=data_end, val_start=f["train_val"][0],
                               val_end=f["train_val"][1], workers=workers, labels=tuple(d["labels"]))
            print(f"[fit] {f['name']}: data {spec_t.start} .. {spec_t.data_end}, origins <= {spec_t.cutoff} -> {art}")
            t0 = time.time()
            load_model(model_dir, model_config).fit(spec_t, art)
            open(done, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
            fit_seconds += time.time() - t0
        arts[f["name"]] = art

    # ---- forecast every fold, district by district
    wcfg = dict(spec=spec, test_dir=test_dir, regions=regions, levels=levels, model_name=args.model,
                model_dir=model_dir, model_config=model_config, artifacts=arts, seed=wp.SEED,
                history_days=int(cfg["data"]["history_days"]), block=int(run_cfg["block"]))
    n_or = sum(len(ts.fold_days(spec, f)[1]) for f in spec["folds"])
    print(f"[test] {args.model} | test set {spec['version']} | {len(spec['folds'])} folds, {n_or} daily origins | "
          f"{len(codes)} districts | {workers} workers | commit {git['short']}{' (dirty)' if git['dirty'] else ''}")
    thr = max(1, (os.cpu_count() or 4) // workers)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(thr)
    print(f"[memory] at start: {ts.memory_line()}", flush=True)
    merged = {f["name"]: {lv: ts.NodeSum() for lv in levels} for f in spec["folds"]}
    gaps = windows = 0
    t1 = time.time()
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        pending = {ex.submit(test_worker, c, wcfg): c for c in codes}
        k = 0
        while pending:
            done_, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
            for fu in done_:
                code = pending.pop(fu)
                try:
                    r = fu.result()
                except Exception as e:                               # noqa: BLE001
                    raise RuntimeError(f"district {code} failed: {type(e).__name__}: {e}") from e
                for fname, acc in r["folds"].items():
                    for lv, e in acc.items():
                        merged[fname][lv].add(e["names"], {x: e[x] for x in ("N", "FO", "FR", "NO")})
                gaps, windows = gaps + r["gaps"], windows + r["windows"]
                print(f"[districts] {code}: {r['seconds']:.0f} s | {ts.memory_line(r.get('peak_bytes'))}", flush=True)
                k += 1
                if k % 10 == 0 or k == len(codes) or k <= 3:
                    el = time.time() - t1
                    print(f"[districts] {k}/{len(codes)}, {el / 60:.1f} min, about {el / k * (len(codes) - k) / 60:.1f} min left",
                          flush=True)
    predict_seconds = time.time() - t1

    # ---- write, score, log
    out = os.path.join(out_root, "runs", f"{time.strftime('%Y%m%d-%H%M%S')}_{args.model}_test")
    meta = dict(test_set_version=spec["version"], test_set_sha256=spec["_sha256"], git_commit=git["commit"],
                dirty=git["dirty"], districts="all" if not args.coatos else args.coatos,
                gaps_filled_with_persistence=gaps, household_forecasts=windows)
    fdir, rdir = os.path.join(out, "forecasts", args.model), os.path.join(out, "forecasts", REFERENCE)
    write_forecasts(fdir, spec, merged, args.model, meta, "FO")
    write_forecasts(rdir, spec, merged, REFERENCE, meta, "FR")
    res = ts.score(fdir, test_dir, spec)
    ref = ts.score(rdir, test_dir, spec)
    text = ts.report(res, spec, reference=ref)
    print("\n" + text)
    ts.save_scores(res, os.path.join(out, "scores", args.model), text)
    ts.save_scores(ref, os.path.join(out, "scores", REFERENCE), ts.report(ref, spec))
    hl = cfg["scoring"]["headline"]
    bl = res["by_level"]
    headline = float(bl[(bl["level"] == hl["level"]) & (bl["score"] == "coverage_adjusted")
                        & bl["horizon"].isin(hl["horizons"])]["sysmape"].mean())
    rbl = ref["by_level"]
    ref_headline = float(rbl[(rbl["level"] == hl["level"]) & (rbl["score"] == "coverage_adjusted")
                             & rbl["horizon"].isin(hl["horizons"])]["sysmape"].mean())

    tags = dict(model_family=model_config.get("family", args.model), model=args.model, git_commit=git["commit"],
                git_branch=git["branch"], dirty=str(git["dirty"]).lower(), mode="test",
                cutoff=",".join(f["cutoff"] for f in spec["folds"]), holdout_period="",
                data_version=man["data_version"], districts=meta["districts"], origins=spec["version"],
                test_set_version=spec["version"], test_set_sha256=spec["_sha256"])
    params = {**{f"model.{k}": v for k, v in model_config.items()}, "test_set": spec["version"],
              "folds": json.dumps([f["name"] for f in spec["folds"]]), "n_origins": n_or, "headline": json.dumps(hl)}
    metrics = {args.model: {**ts.metrics_for_mlflow(res), "headline": headline},
               REFERENCE: {**ts.metrics_for_mlflow(ref), "headline": ref_headline}}
    files = [os.path.join(out, "scores", args.model, f"score_{k}.csv")
             for k in ("by_level", "by_fold", "spec_check", "bias_audit")] + \
            [os.path.join(out, "scores", args.model, "score_report.txt")] + \
            [os.path.join(fdir, ts.level_file(lv)) for lv in ("national", "region") if lv in levels]
    mlcfg = dict(cfg.get("mlflow", {}))
    if args.no_mlflow:
        mlcfg["enabled"] = False
    if not os.getenv("MLFLOW_TRACKING_URI") and not mlcfg.get("uri"):
        mlcfg["uri"] = "sqlite:///" + os.path.join(out_root, "mlflow.db")
    last = spec["folds"][-1]["name"]                                # the newest fit is the one registered
    run_id = tracking.mlflow_log_safe(mlcfg, tags, params, dict(fit_seconds=fit_seconds, predict_seconds=predict_seconds),
                                      metrics, files, args.model, register=args.register, artifacts_dir=arts[last],
                                      model_dir=model_dir, model_config=model_config, headline=headline)
    row = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), model=args.model, commit=git["commit"], mode="test",
               holdout_period=spec["version"], status="done", headline=headline, reference_headline=ref_headline,
               run_dir=out, mlflow_run=run_id or "",
               spec_pass=bool(res["spec_check"]["pass"].all()))
    tracking.append_csv(os.path.join(out_root, "runs.csv"), row)
    tracking.append_csv(os.path.join(out_root, "test_leaderboard.csv"),
                        {**row, **{k: v for k, v in metrics[args.model].items()
                                   if k.startswith(("test/national/sysmape", "test/district/sysmape",
                                                    "test/substation/sysmape", "test/feeder/sysmape"))}})
    print(f"\n[done] {out} | headline ({hl['level']} sysMAPE, h {hl['horizons']}) {headline:.2%} "
          f"(reference {ref_headline:.2%}) | ТЗ thresholds {'met' if row['spec_pass'] else 'NOT met'}")
    return out
