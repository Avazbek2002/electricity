"""evaluate.py: fit a model on the developer zone, forecast every household from the same origins, score every level, log.

  python evaluate.py --model nlinear_x --mode validate
  python evaluate.py --model nlinear_x --mode dev-test --coatos 03202          # quick check on one district
  python evaluate.py --model nlinear_x --mode holdout                          # budgeted; needs a clean git tree
"""
import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import shutil
import time

import pandas as pd
import yaml

from harness import REPO
from harness.contract import HORIZON, TrainSpec
from harness.data import load_env_file
from harness.runner import REFERENCE, district_worker, load_model
from harness import scoring, tracking


def _date(d):
    return pd.Timestamp(d)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="folder name under models/")
    p.add_argument("--mode", required=True, choices=["validate", "dev-test", "holdout"])
    p.add_argument("--config", default=os.path.join(REPO, "configs", "eval.yaml"))
    p.add_argument("--coatos", default=None, help="comma list of districts (quick runs; not allowed for holdout)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--refit", action="store_true", help="fit again even if this commit + cutoff was fitted before")
    p.add_argument("--allow-dirty", action="store_true", help="validate / dev-test with uncommitted changes (tagged dirty)")
    p.add_argument("--register", action="store_true", help="holdout only: register the model in MLflow as 'challenger'")
    p.add_argument("--no-mlflow", action="store_true")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    mode = cfg["modes"][args.mode]
    run_cfg, sc_cfg, data = cfg["run"], cfg["scoring"], cfg["data"]
    out_root = os.path.expanduser(args.out_dir or run_cfg["out_dir"])
    workers = args.workers or int(run_cfg["workers"])

    # ---- guard
    git = tracking.git_info(REPO)
    if git["dirty"] and (args.mode == "holdout" or not args.allow_dirty):
        raise SystemExit(f"[guard] uncommitted changes in {REPO}: commit first"
                         + ("" if args.mode == "holdout" else " (or --allow-dirty for a quick check)"))
    model_dir = args.model if os.path.isdir(args.model) else os.path.join(REPO, "models", args.model)
    args.model = os.path.basename(os.path.normpath(model_dir))
    if not os.path.exists(os.path.join(model_dir, "model.py")):
        raise SystemExit(f"[guard] no models/{args.model}/model.py")
    mcfg_path = os.path.join(model_dir, "config.yaml")
    model_config = yaml.safe_load(open(mcfg_path)) if os.path.exists(mcfg_path) else {}
    runs_csv = os.path.join(out_root, "runs.csv")

    # ---- data roots and origins
    dev = data["dev"]
    roots = [dict(uri=dev["uri"], env_prefix=dev.get("env_prefix", ""), start=data["start"], end=dev["end"])]
    end = dev["end"]
    if args.mode == "holdout":
        v = data["vault"]
        if not v.get("uri") or not v.get("end"):
            raise SystemExit("[holdout] the vault is not configured (data.vault.uri / end in configs/eval.yaml)")
        if args.coatos:
            raise SystemExit("[holdout] runs on every district (no --coatos)")
        load_env_file(v["env_file"])
        roots.append(dict(uri=v["uri"], env_prefix=v.get("env_prefix", "VAULT_"), start=v["start"], end=v["end"]))
        end = v["end"]
        tracking.check_budget(runs_csv, args.model, git["commit"], mode["period"], int(cfg["budget"]["per_commit"]),
                              int(cfg["budget"]["per_month"]))
    d0 = _date(data["start"])
    last = _date(mode["origins"]["last"]) if mode["origins"].get("last") else _date(end) - pd.Timedelta(days=HORIZON)
    last = min(last, _date(end) - pd.Timedelta(days=HORIZON))
    origin_dates = pd.date_range(_date(mode["origins"]["first"]), last, freq=f"{int(mode['origins']['step'])}D")
    if not len(origin_dates):
        raise SystemExit("[config] no origins: check the mode's dates against the data")
    if origin_dates[0] < _date(mode["cutoff"]) + pd.Timedelta(days=HORIZON):
        raise SystemExit("[config] the first origin is less than 60 days after the cutoff: its targets were trained on")
    origins = [int((o - d0).days) for o in origin_dates]

    # ---- fit (cached per model, commit and cutoff)
    key = f"{args.model}/{git['short']}{'-dirty' if git['dirty'] else ''}/cutoff-{mode['cutoff']}"
    art = os.path.join(out_root, "artifacts", key)
    spec = TrainSpec(roots=[dict(uri=dev["uri"], env_prefix=dev.get("env_prefix", ""))], start=data["start"],
                     cutoff=mode["cutoff"],
                     data_end=str(min(_date(mode["cutoff"]) + pd.Timedelta(days=HORIZON), _date(dev["end"])).date()),
                     val_start=mode["train_val"][0], val_end=mode["train_val"][1], workers=workers,
                     labels=tuple(data["labels"]))
    t0 = time.time()
    done_flag = os.path.join(art, ".fit_done")
    if args.refit and os.path.exists(art):
        shutil.rmtree(art)
    if os.path.exists(done_flag):
        print(f"[fit] reusing {art}")
        fit_seconds = 0.0
    else:
        os.makedirs(art, exist_ok=True)
        print(f"[fit] {args.model}: data {spec.start} .. {spec.data_end}, origins <= {spec.cutoff} -> {art}")
        load_model(model_dir, model_config).fit(spec, art)
        open(done_flag, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
        fit_seconds = time.time() - t0

    # ---- predict + score, district by district
    import enriched_io as eio
    fs, root = eio.open_root(dev["uri"], dev.get("env_prefix", ""))
    codes = eio.finished_districts(fs, root, set(args.coatos.split(",")) if args.coatos else None)
    import whole_pipeline_v8 as wp
    wcfg = dict(roots=roots, start=data["start"], end=end, labels=list(data["labels"]), origins=origins,
                horizons=list(sc_cfg["horizons"]), history_days=int(data["history_days"]), block=int(run_cfg["block"]),
                model_name=args.model, model_dir=model_dir, model_config=model_config, artifacts_dir=art, seed=wp.SEED)
    print(f"[run] {args.model} | mode {args.mode} | {len(origins)} origins {origin_dates[0].date()} .. "
          f"{origin_dates[-1].date()} | {len(codes)} districts | {workers} workers | commit {git['short']}"
          f"{' (dirty)' if git['dirty'] else ''}")
    if args.mode == "holdout":                       # count the look before scoring, so a crash still uses it
        tracking.append_csv(runs_csv, dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), model=args.model, commit=git["commit"],
                                           mode="holdout", holdout_period=mode["period"], status="started"))
    thr = max(1, (os.cpu_count() or 4) // workers)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(thr)
    results, t1 = [], time.time()
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        pending = {ex.submit(district_worker, c, wcfg): c for c in codes}
        k = 0
        while pending:
            done, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
            for f in done:
                code = pending.pop(f)
                try:
                    results.append(f.result())
                except Exception as e:                       # noqa: BLE001
                    raise RuntimeError(f"district {code} failed: {type(e).__name__}: {e}") from e
                del f
                k += 1
                if k % 10 == 0 or k == len(codes) or k <= 3:
                    el = time.time() - t1
                    print(f"[districts] {k}/{len(codes)}, {el / 60:.1f} min, about {el / k * (len(codes) - k) / 60:.1f} min left",
                          flush=True)
    predict_seconds = time.time() - t1

    models = [args.model, REFERENCE]
    sc = scoring.score(results, models, list(sc_cfg["horizons"]), int(sc_cfg["min_households"]), sc_cfg["headline"])
    scoring.print_report(sc, models, sc_cfg["headline"])
    out = os.path.join(out_root, "runs", f"{time.strftime('%Y%m%d-%H%M%S')}_{args.model}_{args.mode}")
    os.makedirs(out, exist_ok=True)
    files = {}
    for name in ("summary", "by_size", "mae", "nodes"):
        files[name] = os.path.join(out, f"{name}.csv")
        sc[name].to_csv(files[name], index=False)

    # ---- log
    tags = dict(model_family=model_config.get("family", args.model), model=args.model, git_commit=git["commit"],
                git_branch=git["branch"], dirty=str(git["dirty"]).lower(), mode=args.mode, cutoff=mode["cutoff"],
                holdout_period=mode.get("period", ""), data_version=os.path.basename(dev["uri"].rstrip("/")),
                districts="all" if not args.coatos else args.coatos)
    params = {**{f"model.{k}": v for k, v in model_config.items()},
              "origins": f"{origin_dates[0].date()}..{origin_dates[-1].date()} every {mode['origins']['step']} days",
              "n_origins": len(origins), "headline": json.dumps(sc_cfg["headline"])}
    metrics = {m: scoring.metrics_dict(sc, m) for m in models}
    mlcfg = dict(cfg.get("mlflow", {}))
    if args.no_mlflow:
        mlcfg["enabled"] = False
    if not os.getenv("MLFLOW_TRACKING_URI") and not mlcfg.get("uri"):
        mlcfg["uri"] = "sqlite:///" + os.path.join(out_root, "mlflow.db")
    tracking.save_json(os.path.join(out, "run.json"), dict(tags=tags, params=params, data=sc["data"],
                                                            headline=sc["headline"], metrics=metrics))
    run_id = tracking.mlflow_log_safe(mlcfg, tags, params, dict(fit_seconds=fit_seconds, predict_seconds=predict_seconds),
                                 metrics, list(files.values()) + [os.path.join(out, "run.json")], args.model,
                                 register=args.register, artifacts_dir=art)
    row = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), model=args.model, commit=git["commit"], mode=args.mode,
               holdout_period=mode.get("period", ""), status="done", headline=sc["headline"][args.model],
               reference_headline=sc["headline"][REFERENCE], run_dir=out, mlflow_run=run_id or "")
    tracking.append_csv(runs_csv, row)
    lb = {**row, **{k: v for k, v in metrics[args.model].items() if k.startswith(("feeder/", "substation/", "national/",
                                                                                     "household/"))}}
    tracking.append_csv(os.path.join(out_root, "leaderboard.csv"), lb)
    print(f"\n[done] {out} | headline {sc['headline'][args.model]:.2%} (reference {sc['headline'][REFERENCE]:.2%}) "
          f"| {(time.time() - t0) / 60:.1f} min")
