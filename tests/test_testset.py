"""The fixed test set: spec rules, scoring maths and refusal of incomplete submissions, on a tiny hand-made test set."""
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from harness import testset as ts  # noqa: E402

SPEC = os.path.join(REPO, "configs", "test_set.yaml")


def test_released_spec_is_consistent():
    s = ts.load_spec(SPEC)
    assert [len(ts.fold_days(s, f)[0]) for f in s["folds"]] == [45] * 4
    assert sum(len(ts.fold_days(s, f)[0]) for f in s["folds"]) >= 180      # ТЗ: at least 180 test days


def test_spec_rejects_a_leaky_cutoff(tmp_path):
    s = yaml.safe_load(open(SPEC))
    s["folds"][0]["cutoff"] = s["folds"][0]["first_origin"]                # training targets would reach the test
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(s))
    with pytest.raises(AssertionError):
        ts.load_spec(str(p))


def _tiny(tmp_path, observed_sum=True):
    """One fold of 3 days, national level only, horizons 1..2: actuals + a forecast submission."""
    s = yaml.safe_load(open(SPEC))
    f = s["folds"][0]
    f.update(test_start="2025-07-01", test_end="2025-07-03", first_origin="2025-06-29", last_origin="2025-07-02",
             cutoff="2025-04-30", train_val=["2025-01-01", "2025-04-30"])
    s["folds"] = [f]
    s["horizons"] = 2
    s["levels"] = {"national": {"horizons": "all", "min_coverage": 0.8}}
    s["scoring"]["spec_thresholds"] = {1: 0.08, 2: 0.11}
    s["scoring"]["control_horizons"] = [1, 2]
    s["holidays"] = {}
    sp = tmp_path / "spec.yaml"
    sp.write_text(yaml.safe_dump(s))
    spec = ts.load_spec(str(sp))
    td = tmp_path / "ts"
    (td / "actuals").mkdir(parents=True)
    dates = pd.date_range("2025-07-01", "2025-07-03")
    a = pd.DataFrame({"fold": f["name"], "node": "national", "date": dates.date, "households": 100,
                      "observed": [100, 80, 50], "actual_observed_sum": [1000.0, 800.0, 500.0]})
    a["coverage"] = a["observed"] / a["households"]
    a["actual"] = a["households"] * a["actual_observed_sum"] / a["observed"]       # 1000 each day
    a["scored"] = a["coverage"] >= 0.8                                              # day 3 (50%) not scored
    a.to_csv(td / "actuals" / "national.csv", index=False)
    json.dump({"test_set_sha256": spec["_sha256"], "test_set_version": spec["version"], "data_version": "x"},
              open(td / "manifest.json", "w"))
    fd = tmp_path / "fc"
    fd.mkdir()
    rows = []
    for d, n_obs in zip(dates[:2], (100, 80)):
        for h in (1, 2):
            fo = 1.1 * 10 * n_obs                                   # every household forecast 11 kWh, actual 10
            rows.append(dict(fold=f["name"], node="national", origin=(d - pd.Timedelta(days=h)).date(), horizon=h,
                             forecast=100 * fo / n_obs, **({"forecast_observed_sum": fo, "observed": n_obs}
                                                           if observed_sum else {})))
    pd.DataFrame(rows).to_csv(fd / "national.csv", index=False)
    json.dump({"model": "tiny", "test_set_sha256": spec["_sha256"]}, open(fd / "forecast_meta.json", "w"))
    return spec, str(td), str(fd)


def test_scores_are_the_ratio_of_sums(tmp_path):
    spec, td, fd = _tiny(tmp_path)
    r = ts.score(fd, td, spec)
    bl = r["by_level"]
    for name in ("coverage_adjusted", "matched"):
        v = bl[(bl["score"] == name) & (bl["horizon"] == 1)]
        assert np.isclose(v["sysmape"].iloc[0], 0.10) and np.isclose(v["bias"].iloc[0], 0.10)
        assert v["node_days"].iloc[0] == 2                          # the 50 %-coverage day is not scored
    assert r["spec_check"]["pass"].tolist() == [False, True]       # 10 %: over the 8 % limit at h=1, under 11 % at h=2


def test_incomplete_or_misaligned_submissions_are_refused(tmp_path):
    spec, td, fd = _tiny(tmp_path)
    f = pd.read_csv(os.path.join(fd, "national.csv"))
    f.iloc[1:].to_csv(os.path.join(fd, "national.csv"), index=False)
    with pytest.raises(SystemExit, match="missing"):
        ts.score(fd, td, spec)
    f.assign(observed=f["observed"] + 1).to_csv(os.path.join(fd, "national.csv"), index=False)
    with pytest.raises(SystemExit, match="different set of observed households"):
        ts.score(fd, td, spec)
    pd.concat([f, f.head(1)]).to_csv(os.path.join(fd, "national.csv"), index=False)
    with pytest.raises(SystemExit, match="duplicated"):
        ts.score(fd, td, spec)


def test_a_changed_spec_is_refused(tmp_path):
    spec, td, fd = _tiny(tmp_path)
    other = copy.deepcopy(spec)
    other["_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="changed since the test set was built"):
        ts.score(fd, td, other)


def test_template_lists_exactly_the_scored_rows(tmp_path):
    spec, td, fd = _tiny(tmp_path, observed_sum=False)
    n = ts.write_template(td, spec, str(tmp_path / "tpl"))
    assert n == {"national": 4}                                     # 2 scored days x 2 horizons
