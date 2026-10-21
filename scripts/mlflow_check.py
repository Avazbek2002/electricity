#!/usr/bin/env python
"""
Check an MLflow server end to end before using it: credentials, tracking, artifact upload/download, model logging,
registry (version + alias), loading a registered model back. Cleans up after itself (use --keep to look at it).

    source ~/work/mlflow_env.sh            # MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
    python scripts/mlflow_check.py

Credentials are read from the environment only and never printed.
"""
import argparse
import os
import sys
import tempfile
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
import mlflow_utils as mu  # noqa: E402

HINTS = {
    "401": "wrong or missing credentials: check MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD",
    "403": "the user has no permission for this action (ask the server admin for edit / manage rights)",
    "404": "wrong URL or path prefix: the URI must be the address the MLflow UI is served at",
    "artifact": "artifact storage problem on the server: it needs --serve-artifacts with a writable "
                "--artifacts-destination (S3/MinIO or a persistent volume), or the client needs direct access to the "
                "artifact store the experiment points at",
    "SSL": "TLS problem: set MLFLOW_TRACKING_SERVER_CERT_PATH to the CA bundle, or fix the server certificate",
}


def hint(e):
    s = f"{type(e).__name__}: {e}"
    for k, v in HINTS.items():
        if k in s:
            return v
    return ""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--uri", default=None, help="default: MLFLOW_TRACKING_URI")
    p.add_argument("--keep", action="store_true", help="keep the test run and registered model")
    a = p.parse_args()
    uri = a.uri or os.getenv("MLFLOW_TRACKING_URI")
    print("environment:", ", ".join(f"{v}={'set' if os.getenv(v) else 'MISSING'}" for v in
                                    ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")))
    if not uri:
        raise SystemExit("set MLFLOW_TRACKING_URI first")
    import mlflow
    print(f"mlflow client {mlflow.__version__} | server {mu._mask(mu._clean_uri(uri))}")
    results = []

    def step(name, fn):
        t0 = time.time()
        try:
            out = fn()
            results.append((name, True))
            print(f"  OK    {name} ({time.time() - t0:.1f} s){'  ' + str(out) if out else ''}")
            return out
        except Exception as e:                                          # noqa: BLE001
            results.append((name, False))
            print(f"  FAIL  {name}: {mu._mask(f'{type(e).__name__}: {e}')[:400]}")
            if hint(e):
                print(f"        -> {hint(e)}")
            if os.getenv("MLFLOW_CHECK_TRACE"):
                traceback.print_exc()
            return None

    def server():
        import requests
        base = mu._clean_uri(uri)
        auth = (os.environ["MLFLOW_TRACKING_USERNAME"], os.environ["MLFLOW_TRACKING_PASSWORD"]) \
            if os.getenv("MLFLOW_TRACKING_USERNAME") else None
        h = requests.get(base + "/health", auth=auth, timeout=20)
        v = requests.get(base + "/version", auth=auth, timeout=20)
        r = requests.get(base + "/api/2.0/mlflow/experiments/search?max_results=1", auth=auth, timeout=20)
        if r.status_code != 200:
            raise RuntimeError(f"API answered {r.status_code}")
        return f"health {h.status_code}, server version {v.text.strip()[:20] if v.ok else v.status_code}, API 200"
    step("server answers the REST API with these credentials", server)

    tr = mu.Tracker("electricity/_smoke_test", uri=uri, enabled=True)
    if not tr.enabled:
        raise SystemExit("cannot use the server (see the warning above)")
    exp = tr.client.get_experiment(tr.experiment_id)
    print(f"  experiment artifact location: {exp.artifact_location}")
    state = {}

    def run():
        with tr.run("smoke test", tags={"stage": "smoke"}) as rid:
            tr.params({"p": 1})
            tr.metrics({"m": 0.5})
            state["rid"] = rid
        r = tr.client.get_run(state["rid"])
        assert r.data.metrics.get("m") == 0.5 and r.data.params.get("p") == "1"
        return f"run {state['rid'][:8]}"
    step("create a run, log params / metrics / tags", run)

    def artifact():
        d = tempfile.mkdtemp()
        path = os.path.join(d, "hello.txt")
        open(path, "w").write("smoke test artifact\n")
        tr.client.log_artifact(state["rid"], path, "check")
        back = mlflow.artifacts.download_artifacts(run_id=state["rid"], artifact_path="check/hello.txt",
                                                   dst_path=tempfile.mkdtemp())
        assert open(back).read() == "smoke test artifact\n"
    step("upload an artifact and download it back", artifact)

    def model():
        import mlflow.pyfunc

        class Echo(mlflow.pyfunc.PythonModel):
            def predict(self, context, model_input, params=None):
                return model_input
        with mlflow.start_run(run_id=state["rid"]):
            try:
                info = mlflow.pyfunc.log_model(name="model", python_model=Echo())
            except TypeError:
                info = mlflow.pyfunc.log_model(artifact_path="model", python_model=Echo())
        state["uri"] = info.model_uri
        return info.model_uri
    step("log a pyfunc model", model)

    name = "electricity_smoke_test"

    def register():
        v = tr.register(state["uri"], name, tags={"smoke": "1"})
        if v is None:
            raise RuntimeError("registration failed (see the warning above)")
        tr.client.set_registered_model_alias(name, "smoke", str(v))
        assert tr.version_by_alias(name, "smoke")[0] == v
        state["v"] = v
        return f"{name} v{v} with alias 'smoke'"
    step("register a version and set an alias", register)

    def load():
        import mlflow.pyfunc
        import pandas as pd
        m = mlflow.pyfunc.load_model(f"models:/{name}@smoke")
        out = m.predict(pd.DataFrame({"x": [1, 2]}))
        assert list(out["x"]) == [1, 2]
    step("load the registered model by alias and predict", load)

    if not a.keep:
        def cleanup():
            try:
                tr.client.delete_registered_model(name)
            except Exception:                                         # noqa: BLE001
                pass
            if state.get("rid"):
                tr.client.delete_run(state["rid"])
        step("clean up (registered model deleted, run moved to trash)", cleanup)

    bad = [n for n, ok in results if not ok]
    print("\nALL CHECKS PASSED: the server is ready" if not bad else f"\n{len(bad)} check(s) failed: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
