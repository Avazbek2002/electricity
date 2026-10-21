"""mlflow_utils: the customer-data guard, a local tracking round-trip, registry aliases, and fail-safe behaviour."""
import os
import sys

import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "core"))
mlflow = pytest.importorskip("mlflow")
import mlflow_utils as mu  # noqa: E402


def test_guard_refuses_customer_tables_and_drops_small_nodes(tmp_path):
    a = tmp_path / "cust.csv"
    pd.DataFrame({"customer_key": [1], "kwh": [2.0]}).to_csv(a, index=False)
    ok, _, note = mu.table_guard(str(a))
    assert not ok and "customer_key" in note
    b = tmp_path / "nodes.csv"
    pd.DataFrame({"node": ["x", "y"], "households": [3, 40]}).to_csv(b, index=False)
    ok, path, note = mu.table_guard(str(b))
    assert ok and "left out" in note and pd.read_csv(path)["node"].tolist() == ["y"]


def test_round_trip_and_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "not-a-real-secret")
    tr = mu.Tracker("test/exp", uri=f"sqlite:///{tmp_path}/m.db")
    assert tr.enabled
    with tr.run("parent", tags={"note": "pw not-a-real-secret"}) as rid:
        tr.params({"a": 1})
        tr.metrics({"m": 0.5, "nan": float("nan"), "text": "x"})
        tr.curve("loss", [3, 2, 1])
        with tr.run("child", nested=True) as cid:
            tr.metrics({"c": 1})
        tr.dataframe(pd.DataFrame({"x": [1]}), "evaluation/t.csv")
    r = tr.get(rid)
    assert r.data.metrics == {"m": 0.5, "loss": 1.0} and r.data.tags["note"] == "pw ***"
    assert tr.get(cid).data.tags["mlflow.parentRunId"] == rid
    assert tr.download_artifact(rid, "evaluation/t.csv")
    assert os.path.isdir(tmp_path / "mlartifacts")                  # local artifacts sit next to the database
    with tr.run("fit") as fid:
        d = tmp_path / "model"
        d.mkdir()
        (d / "w.npz").write_text("x")
        tr.artifacts_dir(str(d), "model", everything=True)
    v = tr.register(f"runs:/{fid}/model", "test_model", tags={"k": "v"})
    assert v == 1
    tr.set_alias("test_model", "champion", v)
    assert tr.version_by_alias("test_model", "champion") == (1, fid)
    assert tr.version_by_alias("test_model", "nope") == (None, None)


def test_unreachable_server_turns_tracking_off():
    tr = mu.Tracker("x", uri="http://127.0.0.1:9")
    assert not tr.enabled
    with tr.run("n") as rid:                                        # every call is a no-op
        tr.metrics({"a": 1})
        tr.artifact("/does/not/exist")
    assert rid is None


def test_credentials_leave_the_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    assert mu._clean_uri("https://bob:p%40ss@host/x/") == "https://host/x"
    assert os.environ["MLFLOW_TRACKING_USERNAME"] == "bob" and os.environ["MLFLOW_TRACKING_PASSWORD"] == "p@ss"
    assert mu._mask("https://bob:pw@host") == "https://***@host"
