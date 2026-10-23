"""Real-forecast path (core/reforecast.py, core/wx_windows.py, core/boost_wx_harness.py) on synthetic data."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "core")):
    if p not in sys.path:
        sys.path.insert(0, p)

import reforecast as rf  # noqa: E402

INIT = pd.Timestamp("2025-01-09")


def _long(init=INIT, points=("3203", "10240"), kelvin=True, valid_time=False, minmax=True):
    """6-hourly long table: temperature = 10 * point index + local day offset (+ 0.5 * step within the day)."""
    rows = []
    for pi, pt in enumerate(points):
        for step in range(0, 385, 6):
            t = init + pd.Timedelta(hours=step)
            local_day = (t + pd.Timedelta(hours=5)).floor("D")
            k = (local_day - init).days
            base = 10.0 * pi + k + 0.5 * ((t.hour // 6) - 1.5)          # mean over the day's 4 steps = 10 pi + k
            rec = {"point": pt, ("valid_time" if valid_time else "step"): t if valid_time else step}
            off = 273.15 if kelvin else 0.0
            rows.append({**rec, "variable": "2t", "value": base + off})
            if minmax:
                rows.append({**rec, "variable": "mn2t6", "value": base - 2 + off})
                rows.append({**rec, "variable": "mx2t6", "value": base + 3 + off})
            rows.append({**rec, "variable": "10u", "value": 1.0})
    return pd.DataFrame(rows)


def _w(**kw):
    return rf.wx_config(dict(wx=kw))


def test_local_days_leads_kelvin_and_codes():
    a, info = rf.to_daily(_long(), INIT, _w(lead_days=14))
    assert sorted(a["lead_day"].unique()) == list(range(1, 15))           # lead 0 (= day o) dropped, h1..h14 kept
    assert set(a["coato_id"]) == {"03203", "10240"}                       # zero-padded like the enriched codes
    assert info["steps_per_day"] == 4 and info["tmin"] == "mn2t6"
    r = a[(a["coato_id"] == "10240")].set_index("lead_day")
    np.testing.assert_allclose(r["t_fc"].values, 10 + np.arange(1, 15), atol=1e-9)   # deg C, local-day mean
    assert (r["t_fc_min"] <= r["t_fc"]).all() and (r["t_fc_max"] >= r["t_fc"]).all()
    np.testing.assert_allclose(r["t_fc_min"].values, 10 + np.arange(1, 15) - 2 - 0.75, atol=1e-9)
    assert (a["issue_date"] == INIT).all()


def test_valid_time_column_and_minmax_fallback():
    a, info = rf.to_daily(_long(valid_time=True, kelvin=False, minmax=False), INIT, _w(lead_days=10))
    assert a["lead_day"].max() == 10
    r = a[a["coato_id"] == "03203"].set_index("lead_day")
    np.testing.assert_allclose(r["t_fc"].values, np.arange(1, 11), atol=1e-9)
    np.testing.assert_allclose((r["t_fc_max"] - r["t_fc_min"]).values, 1.5, atol=1e-9)   # range of the 4 steps
    assert info["tmin"].startswith("min of")


def test_reserve_is_refused_and_sources_checked():
    with pytest.raises(SystemExit):
        _w(sets=["wx-reserve2026-20261022t045357z"])
    with pytest.raises(SystemExit):
        _w(source="gfs")


def test_training_and_serving_see_the_same_forecast():
    """A window with origin o - 1 looks the forecast up at v8 leads 2..15 in training (to_window_convention) and at
    predict time (harness grid shifted by grid_at_last_day): the same numbers, lead 1 (day o) empty in both."""
    import whole_pipeline_v8 as wp
    from harness.weather import grid_at_last_day, reforecast_grid
    import boost_wx_harness as bw
    old = {k: getattr(wp, k) for k in ("FORECAST_DAYS", "FC_BLOCKS", "FC_TAIL_DAYS")}
    try:
        for k, v in bw.v8_forecast_settings(_w(lead_days=14)).items():
            setattr(wp, k, v)
        K = wp.FORECAST_DAYS
        assert K == 15 and wp.FC_BLOCKS[0][0] == 2 and wp.FC_BLOCKS[-1][1] == 15
        a, _ = rf.to_daily(_long(), INIT, _w(lead_days=14))
        days = pd.date_range("2024-01-01", "2025-06-30")
        wdf = pd.DataFrame({"coato_id": "03203", "d": days, "t": 5.0, "tmin": 0.0, "tmax": 10.0})
        clim = wp.build_climate(wdf)
        pdf = pd.DataFrame({"coato_code": ["03203"], "origin_date": [INIT - pd.Timedelta(days=1)]})
        a8 = rf.to_window_convention(a, K)
        train = wp.lookup_forecast_archive(pdf, clim, wp.build_forecast_grid(a8, clim))[0]
        g, gm = reforecast_grid("03203", a, clim, INIT, wp)
        panel = SimpleNamespace(origin=INIT, pipeline=dict(clim=clim, fc_grid=g, fc_grid_mm=gm))
        gs, gms = grid_at_last_day(panel)
        serve = wp.lookup_forecast_archive(pdf, clim, gs)[0]
        assert np.isnan(train[0]) and np.isnan(serve[0])
        np.testing.assert_allclose(train[1:], np.arange(1, 15), atol=1e-9)
        np.testing.assert_allclose(serve, train, equal_nan=True)
        tmin_t = wp.lookup_forecast_archive(pdf, clim, wp.build_forecast_grid(a8, clim, col="t_fc_min"))[0]
        np.testing.assert_allclose(wp.lookup_forecast_archive(pdf, clim, gms["min"])[0], tmin_t, equal_nan=True)
        g0, _ = reforecast_grid("03203", a, clim, INIT + pd.Timedelta(days=1), wp)   # no forecast issued that day
        assert np.isnan(g0).all()
    finally:
        for k, v in old.items():
            setattr(wp, k, v)


def test_fixed_origins_plus_stride_without_duplicates():
    import wx_windows as ww
    d0 = pd.Timestamp("2022-01-01")
    crc = np.array([0, 7, 29], dtype=np.uint64)
    fix = [pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-16")]
    o = ww.origin_grid(crc, d0, T=1200, L=90, H=60, S=30, dates=fix, stride=False)
    assert o.shape == (3, 2) and (o == [(f - d0).days for f in fix]).all()
    o2 = ww.origin_grid(crc, d0, T=1200, L=90, H=60, S=30, dates=fix, stride=True)
    for row in o2:
        v = row[row >= 0]
        assert len(v) == len(set(v.tolist()))
        assert {(f - d0).days for f in fix} <= set(v.tolist())
    fixed_day = 89 + 7 + 30 * 25                                          # a stride day of customer crc = 7
    o3 = ww.origin_grid(crc[1:2], d0, 1200, 90, 60, 30, [d0 + pd.Timedelta(days=fixed_day)], True)
    v = o3[0][o3[0] >= 0]
    assert (v == fixed_day).sum() == 1


def test_fixed_origins_context_restores_v8():
    import whole_pipeline_v8 as wp
    import wx_windows as ww
    before = wp._enriched_district_worker
    with ww.fixed_origins(wp, [INIT], stride=True):
        assert wp._enriched_district_worker is ww.district_worker
        assert os.environ[ww.ENV_ORIGINS] == "2025-01-09" and os.environ[ww.ENV_STRIDE] == "1"
    assert wp._enriched_district_worker is before and ww.ENV_ORIGINS not in os.environ


def test_issue_dates_are_filtered_to_reforecast_dates(monkeypatch, tmp_path):
    from harness import test_mode
    monkeypatch.setattr(rf, "issue_dates", lambda w: [pd.Timestamp("2025-01-09"), pd.Timestamp("2025-01-17")])
    blocks = [(pd.Timestamp("2025-01-01"), pd.date_range("2025-01-01", "2025-03-31")),
              (pd.Timestamp("2025-04-01"), pd.date_range("2025-04-01", "2025-06-30"))]
    out, path = test_mode._reforecast_blocks(blocks, "none", {}, str(tmp_path))
    assert path is None and len(out) == 1 and list(out[0][1]) == [pd.Timestamp("2025-01-09"), pd.Timestamp("2025-01-17")]
    with pytest.raises(SystemExit):                                       # trained on aifs, run with fcn3
        test_mode._reforecast_blocks(blocks, "fcn3", {"wx": {"source": "aifs"}}, str(tmp_path))
