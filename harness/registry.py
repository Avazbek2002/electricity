"""Model registry for every model the harness evaluates.

One registered model per model folder:  electricity_household_<model>   (e.g. electricity_household_gam)
Each version = the fitted model as a loadable MLflow pyfunc, tagged with the evaluation that produced it.

Aliases (set automatically when a version is registered):
    best_validate / best_test       the version with the lowest headline (mean national sysMAPE over h 1, 7, 14, 30,
                                    60) among national runs on the same evaluation set and data version
    best_validate_sample / ...      the same among runs on the guide's 5-district sample
    challenger                      the latest version scored on the holdout (the vault)
    champion                        set only by `promote`; the version you would deploy

Production copy: `promote --to-production` also copies the version into electricity_household_forecast (the name the
serving side loads) with alias `champion`.

    python -m harness.registry list
    python -m harness.registry list --model gam
    python -m harness.registry promote --model gam --version 3            # must hold `challenger` (or --force)
    python -m harness.registry promote --model gam --version 3 --to-production

Loading a registered version anywhere (it brings its own code):
    m = mlflow.pyfunc.load_model("models:/electricity_household_gam@champion")
    forecasts = m.predict(panel)            # a harness Panel, or a dict with the Panel's fields -> (n, 60) kWh
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys

import numpy as np

from harness import CORE, REPO

PREFIX = "electricity_household_"
PRODUCTION = "electricity_household_forecast"
SERVING_PACKAGES = ["numpy", "pandas", "pyarrow", "pyyaml", "scikit-learn", "joblib", "interpret-core", "catboost",
                    "xgboost", "scipy"]


def registered_name(model):
    return PREFIX + model.replace("-", "_")


def _alias_mode(mode):
    return "best_" + mode.replace("-", "_")


def _pkg_versions(names):
    out = []
    for n in names:
        try:
            out.append(f"{n}=={importlib.metadata.version(n)}")
        except importlib.metadata.PackageNotFoundError:
            pass
    return out


# ----------------------------------------------------------------------------- the pyfunc wrapper
try:
    import mlflow.pyfunc as _pyfunc
    _Base = _pyfunc.PythonModel
except ImportError:                                                     # mlflow not installed: wrapper unavailable
    _Base = object


class HarnessModel(_Base):
    """Serves a harness model: loads models/<name>/model.py from the logged code and the fitted files, then calls
    Model.predict(panel)."""

    def load_context(self, context):
        code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # <logged model>/code (or the repo)
        for p in (os.path.join(code, "core"), code):
            if p not in sys.path:
                sys.path.insert(0, p)
        info = json.load(open(context.artifacts["info"]))
        mdir = context.artifacts["model_code"]
        if mdir not in sys.path:
            sys.path.insert(0, mdir)
        spec = importlib.util.spec_from_file_location(f"model_{info['model']}", os.path.join(mdir, "model.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.info = info
        self.model = mod.Model(info.get("model_config") or {})
        self.model.load(context.artifacts["fitted"])

    def predict(self, context, model_input, params=None):
        from harness.contract import HORIZON, Panel
        panels = model_input if isinstance(model_input, (list, tuple)) else [model_input]
        out = []
        for p in panels:
            if isinstance(p, dict):
                p = Panel(**p)
            f = np.asarray(self.model.predict(p), dtype="float64")
            if f.shape != (p.n, HORIZON):
                raise ValueError(f"predict returned {f.shape}, expected {(p.n, HORIZON)}")
            out.append(f)
        return out[0] if len(out) == 1 else out


def log_pyfunc(tr, model, model_dir, artifacts_dir, model_config, evidence):
    """Log the fitted model as a pyfunc in the active run; returns the model URI (or None). The code of harness/,
    core/ and the model folder travels with it, so the version loads without this repository."""
    if not tr.enabled:
        return None
    import mlflow.pyfunc
    import tempfile
    d = tempfile.mkdtemp(prefix="harness_pyfunc_")
    info_path = os.path.join(d, "harness_model.json")
    json.dump({"model": model, "model_config": model_config, **evidence}, open(info_path, "w"), indent=2, default=str)
    reqs = ["mlflow==" + importlib.metadata.version("mlflow" if _has("mlflow") else "mlflow-skinny")]
    reqs += _pkg_versions(SERVING_PACKAGES)
    kw = dict(python_model=HarnessModel(),
              artifacts={"fitted": os.path.abspath(artifacts_dir), "model_code": os.path.abspath(model_dir),
                         "info": info_path},
              code_paths=[os.path.join(REPO, "harness"), CORE], pip_requirements=reqs)

    def go():
        try:
            return mlflow.pyfunc.log_model(name="pyfunc", **kw)                      # MLflow 3
        except TypeError:
            return mlflow.pyfunc.log_model(artifact_path="pyfunc", **kw)             # MLflow 2
    info = tr._safe("log pyfunc model", go)
    if info is None:
        return None
    uri = info.model_uri
    try:                                                                # it must load, or it is not registered
        mlflow.pyfunc.load_model(uri)
        print(f"[registry] logged {uri} and loaded it back: OK")
    except Exception as e:                                              # noqa: BLE001
        print(f"[registry] WARNING: the logged model does not load ({type(e).__name__}: {str(e)[:300]}): not registered")
        return None
    return uri


def _has(pkg):
    try:
        importlib.metadata.version(pkg)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


# ----------------------------------------------------------------------------- registration + aliases
def comparable_key(tags):
    return "|".join(str(tags.get(k, "")) for k in ("mode", "cutoff", "origins", "data_version", "districts",
                                                    "holdout_period"))


def register_eval(tr, model, model_uri, fit_run_id, eval_run_id, tags, metrics, headline):
    """Register a new version with its evidence and move the aliases. Returns the version (or None)."""
    name = registered_name(model)
    tr.ensure_registered_model(name, description=f"Household 60-day forecast, model folder models/{model}, "
                                                 f"registered by evaluate.py --register.",
                               tags={"model": model, "family": tags.get("model_family", model)})
    keep = ("national/", "district/", "substation/", "feeder/", "household/mae")
    vtags = {**{k: v for k, v in tags.items()}, "fit_run_id": fit_run_id, "eval_run_id": eval_run_id or "",
             "headline": f"{headline:.6f}", "comparable": comparable_key(tags),
             **{k: f"{v:.6f}" for k, v in metrics.items() if k.startswith(keep) and "_median_" not in k}}
    desc = (f"{model} @ {tags.get('git_commit', '')[:7]} | {tags.get('mode')} cutoff {tags.get('cutoff')} | "
            f"headline {headline:.2%} | origins {tags.get('origins', '')}")
    v = tr.register(model_uri, name, tags=vtags, description=desc)
    if v is None:
        return None
    mode = tags.get("mode", "")
    sample = tags.get("scope") == "sample"                # the guide's 5-district sample competes on its own alias
    if tags.get("districts", "all") != "all" and not sample:
        print(f"[registry] {name} v{v}: quick run on districts {tags['districts']}: registered, no alias moved")
    elif mode == "holdout":
        tr.set_alias(name, "challenger", v)
        print(f"[registry] {name} v{v} -> alias 'challenger' (promote with: python -m harness.registry promote "
              f"--model {model} --version {v})")
    else:
        alias = _alias_mode(mode) + ("_sample" if sample else "")
        cur, _ = tr.version_by_alias(name, alias)
        move, why = True, "first version on this mode"
        if cur is not None:
            mv = tr.version_info(name, cur)
            ct = dict(mv.tags) if mv is not None else {}
            if ct.get("comparable") != vtags["comparable"]:
                why = f"v{cur} was scored on other dates / data"
            elif float(ct.get("headline", "inf")) <= headline + 1e-6:          # ties keep the older version
                move, why = False, f"v{cur} is better or equal ({float(ct['headline']):.2%} vs {headline:.2%})"
            else:
                why = f"better than v{cur} ({headline:.2%} vs {float(ct['headline']):.2%})"
        if move:
            tr.set_alias(name, alias, v)
        print(f"[registry] {name} v{v}: alias '{alias}' {'moved here' if move else 'unchanged'} ({why})")
    return v


# ----------------------------------------------------------------------------- command line
def _tracker():
    from mlflow_utils import Tracker
    tr = Tracker("electricity/eval", enabled=True)
    if not tr.enabled:
        raise SystemExit("MLflow not reachable: set MLFLOW_TRACKING_URI (and the credentials)")
    return tr


def cmd_list(args):
    tr = _tracker()
    flt = f"name = '{registered_name(args.model)}'" if args.model else f"name LIKE '{PREFIX}%'"
    models = tr.client.search_registered_models(filter_string=flt)
    if not models:
        print("no registered harness models yet (run evaluate.py ... --register)")
    for rm in models:
        aliases = {}
        for a, ver in (rm.aliases or {}).items():
            aliases.setdefault(int(ver), []).append(a)
        print(f"\n{rm.name}")
        vs = tr.client.search_model_versions(f"name = '{rm.name}'")
        for mv in sorted(vs, key=lambda m: int(m.version)):
            t = mv.tags
            print(f"  v{mv.version:<3} {t.get('mode', ''):9} cutoff {t.get('cutoff', ''):10} headline "
                  f"{float(t.get('headline', 'nan')):7.2%}  commit {t.get('git_commit', '')[:7]}  "
                  f"{','.join(aliases.get(int(mv.version), []))}")
    prod = tr.version_by_alias(PRODUCTION, "champion")[0]
    print(f"\n{PRODUCTION}: champion = {'v' + str(prod) if prod else 'none'}")


def cmd_promote(args):
    tr = _tracker()
    name = registered_name(args.model)
    mv = tr.version_info(name, args.version)
    if mv is None:
        raise SystemExit(f"{name} v{args.version} does not exist")
    holder, _ = tr.version_by_alias(name, "challenger")
    if holder != int(args.version) and not args.force:
        raise SystemExit(f"v{args.version} is not the 'challenger' (scored on the holdout); use --force to promote "
                         f"a version that was only scored on {mv.tags.get('mode')}")
    tr.set_alias(name, "champion", args.version)
    print(f"{name} v{args.version} is now 'champion'")
    if args.to_production:
        def go():
            tr.ensure_registered_model(PRODUCTION)
            return tr.client.copy_model_version(f"models:/{name}/{args.version}", PRODUCTION)
        new = tr._safe("copy to production", go)
        if new is not None:
            tr.set_version_tags(PRODUCTION, new.version, {**mv.tags, "source": f"{name}/{args.version}"})
            tr.set_alias(PRODUCTION, "champion", new.version)
            print(f"{PRODUCTION} v{new.version} (copy of {name} v{args.version}) is now 'champion'")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("list")
    a.add_argument("--model", default=None)
    b = sub.add_parser("promote")
    b.add_argument("--model", required=True)
    b.add_argument("--version", required=True, type=int)
    b.add_argument("--force", action="store_true")
    b.add_argument("--to-production", action="store_true", help=f"also copy into {PRODUCTION} as 'champion'")
    args = p.parse_args(argv)
    {"list": cmd_list, "promote": cmd_promote}[args.cmd](args)


if __name__ == "__main__":
    main()
