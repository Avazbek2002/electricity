"""Harness checks on a synthetic enriched dataset.

  EVAL_TEST_CONFIG=/path/to/eval_synthetic.yaml python -m pytest tests/ -q
(eval_synthetic.yaml = configs/eval.yaml with data.dev.uri pointing at a local synthetic copy; skipped without it.)
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.getenv("EVAL_TEST_CONFIG")


@pytest.mark.skipif(not CFG, reason="set EVAL_TEST_CONFIG to a config pointing at synthetic data")
def test_probe_sees_only_the_past():
    env = {**os.environ, "AWS_ACCESS_KEY_ID": "dummy", "AWS_SECRET_ACCESS_KEY": "dummy"}
    r = subprocess.run([sys.executable, os.path.join(REPO, "evaluate.py"), "--model", os.path.join(REPO, "tests", "probe_model"),
                        "--mode", "dev-test", "--config", CFG, "--allow-dirty", "--no-mlflow", "--workers", "1"],
                       capture_output=True, text=True, env=env, cwd=REPO)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "headline" in r.stdout


def test_contract_imports():
    sys.path.insert(0, REPO)
    import harness  # noqa: F401
    from harness.contract import BaseModel, Panel, TrainSpec  # noqa: F401
    import whole_pipeline_v8, enriched_io, household_nlinear, node_syswape  # noqa: F401,E401


def test_credentials_come_back_for_the_next_district():
    """Worker processes are reused: keys hidden for one district's predict must be back for the next district's read."""
    sys.path.insert(0, REPO)
    import harness  # noqa: F401
    from harness.data import restore_credentials, scrub_credentials
    os.environ["AWS_ACCESS_KEY_ID"], os.environ["VAULT_S3_ENDPOINT"] = "a", "b"
    scrub_credentials()
    assert "AWS_ACCESS_KEY_ID" not in os.environ and "VAULT_S3_ENDPOINT" not in os.environ
    restore_credentials()
    assert os.environ["AWS_ACCESS_KEY_ID"] == "a" and os.environ["VAULT_S3_ENDPOINT"] == "b"
