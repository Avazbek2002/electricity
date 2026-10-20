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
