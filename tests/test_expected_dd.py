"""Expected degree days (EXPECTED_DEGREE_DAYS, Fix W): climatological mean DD >= DD of the normal, used only on target
days without a real forecast, causal (past years only), off by default."""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "core")):
    if p not in sys.path:
        sys.path.insert(0, p)

import whole_pipeline_v8 as wp  # noqa: E402


def _weather(seed=0):
    rng = np.random.default_rng(seed)
    d = pd.date_range("2022-01-01", "2025-12-31", freq="D")
    doy = d.dayofyear.values
    t = 15 - 14 * np.cos(2 * np.pi * (doy - 15) / 365.25) + rng.normal(0, 3.5, len(d))
    return pd.DataFrame({"coato_id": "03203", "d": d, "t": t, "tmin": t - 6, "tmax": t + 7})


def test_expected_dd_is_at_least_dd_of_normal():
    clim = wp.build_climate(_weather())
    pdf = pd.DataFrame({"coato_code": "03203", "origin_date": pd.to_datetime(["2025-05-15", "2025-08-25", "2025-01-10"])})
    edd = wp.lookup_edd(pdf, clim, "all")
    _, nrm, _ = wp.lookup_weather(pdf, clim, "all")
    hn, cn = wp.degree_days(nrm[:, wp.LOOKBACK:])
    _, nmin, _, nmax = wp.lookup_minmax(pdf, clim, "all")
    hmn, cmx = wp.degree_days_minmax(nmin[:, wp.LOOKBACK:], nmax[:, wp.LOOKBACK:])
    for e, n in ((edd["h"], hn), (edd["c"], cn), (edd["hmin"], hmn), (edd["cmax"], cmx)):
        assert e.shape == (3, wp.HORIZON)
        assert np.all(e >= n - 1e-9)                      # Jensen: mean of a convex function >= function of the mean
    # shoulder season (the 15 days after a mid-May origin): cooling from Tmax is clearly under-stated by DD(normal)
    assert edd["cmax"][0, :15].sum() > 1.05 * cmx[0, :15].sum()
    # mid-winter: almost every day is below the heating base, so the two nearly agree
    assert abs(edd["h"][2, :20].sum() / hn[2, :20].sum() - 1) < 0.1


def test_jensen_term_only_without_forecast():
    edd = np.full((2, 5), 3.0)
    dd_n = np.array([[1.0, 1, 1, 1, 1], [3.5, 3, np.nan, 2, 2]])
    no_fc = np.array([[False, False, True, True, True]] * 2)
    j = wp._jensen(edd, dd_n, no_fc)
    assert np.allclose(j, [[0, 0, 2, 2, 2], [0, 0, 0, 1, 1]])   # 0 on forecast days, clipped >= 0, NaN -> 0


def test_past_years_only():
    """The expected DD of 2025 days from a climate built on data to 2024 must not change when 2025 weather changes."""
    w = _weather()
    pdf = pd.DataFrame({"coato_code": "03203", "origin_date": pd.to_datetime(["2025-03-01"])})
    a = wp.lookup_edd(pdf, wp.build_climate(w[w["d"] < "2025-01-01"].assign()), "all")
    w2 = w.copy()
    w2.loc[w2["d"] >= "2025-01-01", ["t", "tmin", "tmax"]] += 10
    b = wp.lookup_edd(pdf, wp.build_climate(w2[w2["d"] < "2025-01-01"]), "all")
    for k in a:
        assert np.allclose(a[k], b[k], equal_nan=True)


def test_off_by_default():
    assert wp.EXPECTED_DEGREE_DAYS is False
