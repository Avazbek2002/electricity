"""Git guard, vault look budget, and run logging (MLflow when available, plus local CSVs that are always written)."""
import json
import os
import subprocess
import time

import pandas as pd


def git_info(repo):
    def run(*a):
        return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True).stdout.strip()
    commit = run("rev-parse", "HEAD") or "unknown"
    dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
    return dict(commit=commit, short=commit[:7], branch=run("rev-parse", "--abbrev-ref", "HEAD"), dirty=dirty)


def read_runs(path):
    path = os.path.expanduser(path)
    return pd.read_csv(path, dtype=str) if os.path.exists(path) else pd.DataFrame()


def check_budget(runs_csv, model, commit, period, per_commit, per_month, mlflow_cfg=None):
    """Raise SystemExit if this holdout run would go over budget. Counts finished and started holdout runs."""
    r = read_runs(runs_csv)
    if len(r):
        h = r[(r["mode"] == "holdout") & (r["holdout_period"] == period) & (r["status"] == "started")]
        same = h[(h["model"] == model) & (h["commit"] == commit)]
        if len(same) >= per_commit:
            raise SystemExit(f"[budget] {model} @ {commit[:7]} already looked at holdout period {period} "
                             f"({len(same)} time(s)); commit a change first")
        month = time.strftime("%Y-%m")
        n_month = int(h["time"].str.startswith(month).sum())
        if n_month >= per_month:
            raise SystemExit(f"[budget] {n_month} holdout runs already this month (limit {per_month})")
        print(f"[budget] holdout looks this month: {n_month} of {per_month}; this commit: {len(same)} of {per_commit}")
    else:
        print(f"[budget] no holdout looks logged yet (limits: {per_commit} per commit, {per_month} per month)")


def append_csv(path, row):
    """Add one row; new columns are added to the file (earlier rows get blanks)."""
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path, dtype=str), df.astype(str)], ignore_index=True)
    df.to_csv(path, index=False)


def mlflow_log(cfg, tags, params, fit_info, metrics_by_model, artifacts, model_name, register=False, artifacts_dir=None):
    """Parent run '<model> @ <commit>' with child runs 'fit' and '<mode>'. Returns the mode run id, or None if MLflow
    is off or not installed."""
    if not cfg.get("enabled", True):
        return None
    try:
        import mlflow
    except ImportError:
        print("[mlflow] not installed: logged to the local CSVs only")
        return None
    uri = os.getenv("MLFLOW_TRACKING_URI") or cfg.get("uri")
    if uri:
        mlflow.set_tracking_uri(os.path.expanduser(uri) if uri.startswith("~") else uri)
    mlflow.set_experiment(cfg.get("experiment", "electricity/eval"))
    with mlflow.start_run(run_name=f"{model_name} @ {tags['git_commit'][:7]}", tags=tags):
        with mlflow.start_run(run_name="fit", nested=True, tags={**tags, "step": "fit"}):
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in fit_info.items() if isinstance(v, (int, float))})
            if artifacts_dir and os.path.isdir(artifacts_dir):
                mlflow.log_artifacts(artifacts_dir, artifact_path="model")
            fit_run = mlflow.active_run().info.run_id
        with mlflow.start_run(run_name=tags["mode"], nested=True, tags={**tags, "step": "score"}) as run:
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
            for m, md in metrics_by_model.items():                 # the reference model's metrics under ref/...
                mlflow.log_metrics({(k if m == model_name else f"ref/{k}"): v for k, v in md.items()})
            for a in artifacts:
                if os.path.exists(a):
                    mlflow.log_artifact(a)
            run_id = run.info.run_id
    if register and tags["mode"] == "holdout":
        try:
            name = cfg.get("registered_model", "electricity_household_forecast")
            client = mlflow.MlflowClient()
            try:
                client.create_registered_model(name)
            except Exception:                                        # noqa: BLE001  (already exists)
                pass
            src = client.get_run(fit_run).info.artifact_uri.rstrip("/") + "/model"
            mv = client.create_model_version(name, source=src, run_id=fit_run, tags={"holdout_run": run_id, **tags})
            client.set_registered_model_alias(name, "challenger", mv.version)
            print(f"[mlflow] registered {name} version {mv.version} as 'challenger' (promote with the champion gate)")
        except Exception as e:                                       # noqa: BLE001
            print(f"[mlflow] WARNING: registration failed ({type(e).__name__}: {str(e)[:200]})")
    print(f"[mlflow] logged to {mlflow.get_tracking_uri()} experiment {cfg.get('experiment', 'electricity/eval')}")
    return run_id


def mlflow_log_safe(*a, **k):
    """mlflow_log, but a tracking failure only warns: the scores are already saved locally."""
    try:
        return mlflow_log(*a, **k)
    except Exception as e:                                           # noqa: BLE001
        print(f"[mlflow] WARNING: not logged ({type(e).__name__}: {str(e)[:200]}); scores are in the run folder")
        return None


def save_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
