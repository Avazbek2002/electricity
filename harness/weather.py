"""District weather context shared by the harness and by models that train on the same data: climate normals and the
simulated 10-day forecast issued on every origin day (same seed rule as household_nlinear.py / node_syswape.py)."""
import zlib

import numpy as np
import pandas as pd


def district_weather(code, wt, dates, seed, wp):
    """wt (3, T) observed t_mean / t_min / t_max, dates (T,) -> (clim, grid, grid_mm). grid[coato, issue day, lead]."""
    from household_nlinear import _quiet
    L, H, T = wp.LOOKBACK, wp.HORIZON, len(dates)
    have = np.isfinite(wt).any(axis=0)
    wdf = pd.DataFrame({"coato_id": str(code), "d": dates[have], "t": wt[0, have], "tmin": wt[1, have], "tmax": wt[2, have]})
    od = np.arange(L - 1, T - H)
    with _quiet():
        clim = wp.build_climate(wdf)
        u_fc = pd.DataFrame({"coato_code": str(code), "origin_date": dates[0] + pd.to_timedelta(od, unit="D")})
        s = int(seed) + zlib.crc32(str(code).encode()) % 1_000_003
        t_obs, _, _ = wp.lookup_weather(u_fc, clim, "all")
        t_fc = wp.simulate_forecast(u_fc, t_obs[:, L:], seed=s)
        K = wp.FORECAST_DAYS
        fc = pd.DataFrame({"coato_id": str(code), "issue_date": np.repeat(u_fc["origin_date"].to_numpy(), K),
                           "lead_day": np.tile(np.arange(1, K + 1), len(u_fc)), "t_fc": t_fc.reshape(-1)})
        fc["issue_date"] = pd.to_datetime(fc["issue_date"])
        grid = wp.build_forecast_grid(fc, clim)
        tmin_obs, _, tmax_obs, _ = wp.lookup_minmax(u_fc, clim, "all")
        tmin_fc, tmax_fc = wp.simulate_forecast_minmax(u_fc, t_fc, tmin_obs[:, L:], tmax_obs[:, L:], seed=s)
        fc["t_fc_min"], fc["t_fc_max"] = tmin_fc.reshape(-1), tmax_fc.reshape(-1)
        grid_mm = dict(min=wp.build_forecast_grid(fc, clim, col="t_fc_min"), max=wp.build_forecast_grid(fc, clim, col="t_fc_max"))
    return clim, grid, grid_mm


def issue_index(clim, day):
    """Index of calendar day `day` in the forecast grid's issue-day axis."""
    return int((np.datetime64(pd.Timestamp(day).date(), "D") - np.datetime64(clim.start, "D")).astype(int))


def climate_asof(code, wt, dates, end, wp, pad=None):
    """Climate object from the observed weather of days < end (index): on issue date o use end = o, i.e. data up to
    o - 1 only. Its normals use past years only, and the days from `end` on are present but empty (NaN) up to
    end + pad, so lookups of the issue and target days work and find no observation (ML guide part 5.3)."""
    from household_nlinear import _quiet
    pad = wp.HORIZON + 2 if pad is None else pad
    have = np.isfinite(wt[:, :end]).any(axis=0)
    idx = np.flatnonzero(have)
    fut = pd.date_range(dates[end - 1] + pd.Timedelta(days=1), periods=pad, freq="D")
    nan = np.full(len(fut), np.nan)
    wdf = pd.DataFrame({"coato_id": str(code), "d": np.r_[dates[idx].values, fut.values],
                        "t": np.r_[wt[0, idx], nan], "tmin": np.r_[wt[1, idx], nan], "tmax": np.r_[wt[2, idx], nan]})
    wdf["d"] = pd.to_datetime(wdf["d"])
    with _quiet():
        return wp.build_climate(wdf)


def no_forecast(n_days, K):
    """The forecast grid of a run without target-day weather: no forecast issued (models fall back to normals)."""
    g = np.full((1, n_days, K), np.nan)
    return g, dict(min=g.copy(), max=g.copy())


def grid_at_last_day(panel):
    """For models whose window origin is the last history day (o - 1): the forecast issued on o, moved to issue day
    o - 1 with leads shifted by one (window lead k + 1 = forecast lead k; window lead 1 = day o has no forecast).
    -> (grid, grid_mm) for whole_pipeline_v8.lookup_forecast_archive with origin_date = o - 1."""
    pp = panel.pipeline
    clim, g, gm = pp["clim"], pp["fc_grid"], pp.get("fc_grid_mm") or {}
    gi = issue_index(clim, panel.origin)

    def shift(a):
        out = np.full_like(a, np.nan)
        if 1 <= gi < a.shape[1]:
            out[:, gi - 1, 1:] = a[:, gi, :-1]
        return out
    return shift(g), {k: shift(v) for k, v in gm.items()}
