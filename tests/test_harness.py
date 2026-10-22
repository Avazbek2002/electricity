"""Harness checks on a synthetic enriched dataset.

  EVAL_TEST_CONFIG=eval_synthetic.yaml TEST_SET_SPEC=test_set_synthetic.yaml TEST_SET_DIR=<built dir> \
      python -m pytest tests/ -q
(the synthetic files point the data uri at a local synthetic copy; the probe test is skipped without them.)
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG, SPEC, TDIR = os.getenv("EVAL_TEST_CONFIG"), os.getenv("TEST_SET_SPEC"), os.getenv("TEST_SET_DIR")


@pytest.mark.skipif(not (CFG and SPEC and TDIR), reason="set EVAL_TEST_CONFIG, TEST_SET_SPEC, TEST_SET_DIR (synthetic)")
def test_probe_sees_only_the_past_and_the_substitution_test_passes(tmp_path):
    """The probe asserts inside predict: history ends on o - 1, no S3 keys, at most one forecast issue day, no weather
    of the issue date or later in the climate object. The run must also pass the substitution test."""
    env = {**os.environ, "AWS_ACCESS_KEY_ID": "dummy", "AWS_SECRET_ACCESS_KEY": "dummy", "MLFLOW_DISABLE": "1"}
    r = subprocess.run([sys.executable, os.path.join(REPO, "evaluate.py"), "--model", os.path.join(REPO, "tests", "probe_model"),
                        "--mode", "test", "--config", CFG, "--test-set", SPEC, "--test-dir", TDIR, "--allow-dirty",
                        "--no-mlflow", "--workers", "1", "--no-network", "--out-dir", str(tmp_path)],
                       capture_output=True, text=True, env=env, cwd=REPO)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "[leakage] PASSED" in r.stdout and "headline" in r.stdout


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
