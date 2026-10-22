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


def mlflow_log(cfg, tags, params, fit_info, metrics_by_model, artifacts, model_name, register=False, artifacts_dir=None,
               model_dir=None, model_config=None, headline=None, curves=None, artifact_dirs=None):
    """Parent run '<model> @ <commit>' with child runs 'fit' (params, fitted files, and with register the loadable
    pyfunc) and '<mode>' (scores, tables). With register: a new registry version + aliases (harness/registry.py).
    Returns the mode run id, or None if MLflow is off or unreachable."""
    if not cfg.get("enabled", True):
        return None
    from mlflow_utils import Tracker
    from harness import registry
    uri = os.getenv("MLFLOW_TRACKING_URI") or cfg.get("uri")
    tr = Tracker(cfg.get("experiment", "electricity/eval"), uri=uri, enabled=True)
    if not tr.enabled:
        return None
    model_uri = fit_run = None
    sys_tags = {k: v for k, v in tr.system_tags().items() if k not in tags}
    with tr.run(f"{model_name} @ {tags['git_commit'][:7]}", tags={**tags, **sys_tags}):
        with tr.run("fit", nested=True, tags={**tags, "step": "fit"}) as fit_run:
            tr.params(params)
            tr.metrics(fit_info)
            if artifacts_dir and os.path.isdir(artifacts_dir):
                tr.artifacts_dir(artifacts_dir, subdir="model", everything=True, max_total_mb=5000)
            if register and model_dir:
                model_uri = registry.log_pyfunc(tr, model_name, model_dir, artifacts_dir, model_config or {},
                                                evidence={"tags": tags, "headline": headline})
        with tr.run(tags["mode"], nested=True, tags={**tags, "step": "score"}) as run_id:
            tr.params(params)
            for m, md in metrics_by_model.items():                 # the reference model's metrics under ref/...
                tr.metrics({(k if m == model_name else f"ref/{k}"): v for k, v in md.items()})
            for a in artifacts:
                tr.artifact(a)
            for key, pts in (curves or {}).items():                # per-horizon curves: step = horizon
                tr.series(key, [(None, h, v) for h, v in pts])
            for sub, folder in (artifact_dirs or {}).items():       # charts (figures/, windows/)
                tr.artifacts_dir(folder, subdir=sub)
    if register:
        if model_uri:
            registry.register_eval(tr, model_name, model_uri, fit_run, run_id, tags,
                                   metrics_by_model.get(model_name, {}), headline)
        else:
            print("[registry] not registered: the model was not logged as a loadable pyfunc (see the warning above)")
    print(f"[mlflow] logged to experiment {cfg.get('experiment', 'electricity/eval')} (run {run_id})")
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
