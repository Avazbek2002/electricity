"""The standard evaluation (ML guide part 4): spec rules, T2, B0, the common panel and the 98 % national rule."""
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


def test_released_spec_follows_the_guide():
    s = ts.load_spec(SPEC)
    assert ts.issue_dates(s, "validation")[[0, -1]].strftime("%Y-%m-%d").tolist() == ["2024-01-01", "2024-12-31"]
    assert ts.issue_dates(s, "test")[[0, -1]].strftime("%Y-%m-%d").tolist() == ["2025-01-01", "2025-12-31"]
    assert s["history_gap_days"] == 1
    blocks = ts.refit_blocks(s, ["validation", "test"])
    assert [str(r.date()) for r, _ in blocks] == ["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01",
                                                  "2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01"]
    assert sum(len(b) for _, b in blocks) == 366 + 365                 # every issue date served once
    cutoff, data_end = ts.train_window(s, "2025-04-01")
    assert str(data_end.date()) == "2025-03-31"                         # targets d <= R - 1
    assert cutoff + pd.Timedelta(days=60) == data_end                   # every training horizon ends by R - 1
    assert max(pd.Timestamp(x) for x in s["refits"]) < pd.Timestamp("2026-01-01")   # the 2026 reserve is not touched


def test_spec_rejects_validation_overlapping_the_test(tmp_path):
    s = yaml.safe_load(open(SPEC))
    s["splits"]["validation"]["target_end"] = "2025-02-01"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(s))
    with pytest.raises(AssertionError):
        ts.load_spec(str(p))


def _counts(n=40, start="2024-01-01"):
    d = pd.date_range(start, periods=n)
    return pd.DataFrame({"date": d, "node": "x", "n_rows": 1000.0, "n_obs": 700.0, "n_est": 100.0, "n_lg": 50.0,
                         "kwh_obs": 7000.0, "kwh_est": 1000.0, "kwh_lg": 600.0, "n_right": 0.0, "crisis": 0.0,
                         "interruption": 0.0})


def test_t2_is_scaled_by_the_known_share_and_flags_follow_the_guide():
    spec = ts.load_spec(SPEC)
    c = _counts()
    c.loc[1, ["n_obs", "n_est", "kwh_obs", "kwh_est"]] = [60.0, 20.0, 600.0, 200.0]   # 80 known: < 100 -> artefact
    c.loc[2, ["n_obs", "kwh_obs"]] = [300.0, 3000.0]                                   # coverage 0.4 -> flagged
    c.loc[3, "n_right"] = 5                                                            # open interval -> artefact
    c.loc[4, "crisis"] = 1
    t = ts.target_series(c, spec)
    assert t.loc[0, "T2"] == pytest.approx(1000 * 8000 / 800)          # n_rows x kWh / known
    assert t.loc[0, "T3"] == pytest.approx(1000 * 8600 / 850)
    assert np.isnan(t.loc[1, "T2"]) and np.isnan(t.loc[3, "T2"])
    assert t.loc[2, "flag"] and "low_coverage" in t.loc[2, "flag_reason"] and np.isfinite(t.loc[2, "T2"])
    assert t.loc[4, "flag"] and t.loc[4, "flag_reason"] == "crisis"
    assert not t.loc[0, "flag"]


def test_b0_is_level_times_last_year_ratio():
    spec = ts.load_spec(SPEC)
    d = pd.date_range("2022-01-01", "2025-12-31")
    y = 100 + 50 * (d.dayofyear >= 182)                                 # a step every July
    tg = pd.DataFrame({"node": "x", "date": d, "y": y.astype(float), "flag": False, "artefact": False})
    b = ts.b0_forecasts(tg, spec, issue=pd.DatetimeIndex(["2024-06-15"]))
    f = b.set_index("horizon")["forecast"]
    assert f[1] == pytest.approx(100)                                   # level 100, last year flat around the target
    assert f[30] == pytest.approx(150)                                  # last year: 150 / 100 around d - 364
    assert (b["b0src"] == 0).all()


def _tiny(tmp_path, n_dist=5, drop=None):
    """5 districts, one issue date per split, horizons 1..2; every district A = 100, forecast = 110."""
    s = yaml.safe_load(open(SPEC))
    s["horizons"] = 2
    s["control_horizons"] = [1, 2]
    s["scoring"]["spec_thresholds"] = {1: 0.08, 2: 0.11}
    s["splits"]["validation"].update(issue_start="2024-06-01", issue_end="2024-06-01")
    s["splits"]["test"].update(issue_start="2025-06-01", issue_end="2025-06-01")
    sp = tmp_path / "spec.yaml"
    sp.write_text(yaml.safe_dump(s))
    spec = ts.load_spec(str(sp))
    td = tmp_path / "ts"
    (td / "targets").mkdir(parents=True)
    nodes = [f"d{i}" for i in range(n_dist)]
    days = pd.date_range("2022-01-01", "2025-12-31")
    rows = [pd.DataFrame({"date": days, "node": n, "T2": 100.0, "T3": 100.0, "flag": False, "artefact": False})
            for n in nodes]
    t = pd.concat(rows, ignore_index=True)
    for (n, day) in (drop or []):                                       # flagged district-days
        t.loc[(t["node"] == n) & (t["date"] == day), "flag"] = True
    t.to_csv(td / "targets" / "district.csv.gz", index=False)
    pd.DataFrame({"district": nodes, "region": "r1"}).to_csv(td / "districts.csv", index=False)
    ts.standing_calendar(days).to_csv(td / "calendar.csv", index=False)
    json.dump({"test_set_sha256": spec["_sha256"], "test_set_version": spec["version"], "data_version": "x"},
              open(td / "manifest.json", "w"))
    fd = tmp_path / "fc"
    fd.mkdir()
    f = []
    for o in ("2024-06-01", "2025-06-01"):
        for n in nodes:
            for h in (1, 2):
                f.append(dict(node=n, origin=o, horizon=h, forecast=110.0))
    pd.DataFrame(f).to_csv(fd / "district.csv", index=False)
    json.dump({"model": "tiny", "test_set_sha256": spec["_sha256"]}, open(fd / "forecast_meta.json", "w"))
    return spec, str(td), str(fd)


def test_national_is_the_sum_of_districts_and_scores_are_ratios_of_sums(tmp_path):
    spec, td, fd = _tiny(tmp_path)
    r = ts.score(fd, td, spec, reference=False)
    t = r["splits"]["test"]["scores"]
    v = t[(t["model"] == "tiny") & (t["horizon"] == 1)].iloc[0]
    assert v["sysmape"] == pytest.approx(0.10) and v["bias"] == pytest.approx(0.10) and v["days"] == 1
    assert r["splits"]["test"]["panel"]["national"]["y"].iloc[0] == pytest.approx(500)
    assert r["spec_check"]["passed"].tolist() == [False, True]          # 10 %: over 8 % at h1, under 11 % at h2


def test_national_day_needs_98_percent_of_districts(tmp_path):
    # 5 districts: int(0.98 * 5) = 4 -> one flagged district keeps the day, two drop it
    spec, td, fd = _tiny(tmp_path, drop=[("d0", "2025-06-02")])
    r = ts.score(fd, td, spec, reference=False)
    nat = r["splits"]["test"]["panel"]["national"]
    assert nat[(nat["horizon"] == 1)]["y"].iloc[0] == pytest.approx(400)
    (tmp_path / "b").mkdir()
    spec, td, fd = _tiny(tmp_path / "b", drop=[("d0", "2025-06-02"), ("d1", "2025-06-02")])
    r = ts.score(fd, td, spec, reference=False)
    nat = r["splits"]["test"]["panel"]["national"]
    assert (nat["horizon"] == 1).sum() == 0 and (nat["horizon"] == 2).sum() == 1


def test_a_row_missing_for_one_model_is_dropped_for_all(tmp_path):
    spec, td, fd = _tiny(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    f = pd.read_csv(os.path.join(fd, "district.csv"))
    f[~((f["node"] == "d0") & (f["origin"] == "2025-06-01") & (f["horizon"] == 1))].to_csv(other / "district.csv", index=False)
    r = ts.score(fd, td, spec, reference=False, others={"other": str(other)})
    p = r["splits"]["test"]["panel"]
    assert r["splits"]["test"]["missing"]["other"] == 1
    d = p["district"]
    assert not ((d["node"] == "d0") & (d["horizon"] == 1)).any()        # dropped for the main model too
    assert p["national"][p["national"]["horizon"] == 1]["y"].iloc[0] == pytest.approx(400)


def test_duplicates_and_a_changed_spec_are_refused(tmp_path):
    spec, td, fd = _tiny(tmp_path)
    f = pd.read_csv(os.path.join(fd, "district.csv"))
    pd.concat([f, f.head(1)]).to_csv(os.path.join(fd, "district.csv"), index=False)
    with pytest.raises(SystemExit, match="duplicated"):
        ts.score(fd, td, spec, reference=False)
    spec2 = dict(spec, _sha256="0" * 64)
    with pytest.raises(SystemExit, match="changed since"):
        ts.score(fd, td, spec2, reference=False)


def test_standing_calendar_is_known_in_advance():
    c = ts.standing_calendar(pd.date_range("2025-01-01", "2025-12-31")).set_index("date")
    assert c.loc["2025-03-21", "hname"] == "navruz" and c.loc["2025-01-01", "hname"] == "newyear"
    assert c.loc["2025-03-10", "xoff"] == 1                             # Women's Day on Saturday -> Monday off
    assert c["ram"].sum() in (29, 30)


def test_template_lists_every_issue_date_and_horizon(tmp_path):
    spec, td, fd = _tiny(tmp_path)
    n = ts.write_template(spec, str(tmp_path / "tpl"), ["d0", "d1"])
    assert n == 2 * (2 + 2)                                             # 2 districts x (1 issue date x 2 h) x 2 splits


def test_curves_and_charts(tmp_path):
    spec, td, fd = _tiny(tmp_path)
    r = ts.score(fd, td, spec)
    c = ts.curves_for_mlflow(r)
    assert c["test/sysWAPE"] == [(1, pytest.approx(0.10)), (2, pytest.approx(0.10))]
    assert c["test/MAE"][0][1] == pytest.approx(10.0)                  # |110 - 100| per district-day
    assert "test/B0/WAPE" in c
    from harness import plots
    assert os.path.exists(plots.metrics_by_horizon(r, str(tmp_path / "out")))
