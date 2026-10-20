"""One district in a worker process: build as-of panels per origin, run the model, add forecasts and actuals per node."""
import importlib.util
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(os.path.dirname(HERE), "core"), os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from harness.contract import HORIZON, Panel                                 # noqa: E402
from harness.data import MISSING, read_district, restore_credentials, scrub_credentials  # noqa: E402

REFERENCE = "persistence30_ref"                     # scored next to every model, on the same households and days


def load_model(model_dir, config):
    """models/<name>/model.py -> an instance of its Model class."""
    path = os.path.join(model_dir, "model.py")
    spec = importlib.util.spec_from_file_location(f"model_{os.path.basename(model_dir)}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Model(config)


def persistence30(kwh):
    with np.errstate(invalid="ignore"), _quiet_warn():
        p = np.nanmean(kwh[:, -30:], axis=1)
        p = np.where(np.isfinite(p), p, np.nanmean(kwh[:, -90:], axis=1))
    return np.nan_to_num(p, nan=0.0)


class _quiet_warn:
    def __enter__(self):
        import warnings
        self.c = warnings.catch_warnings()
        self.c.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self.c.__exit__(*a)


def _groups(network, code):
    out = {}
    for j, lvl in enumerate(("transformer", "feeder", "substation")):
        v = network[:, j].copy()
        v[v < 0] = MISSING
        u, g = np.unique(v, return_inverse=True)
        out[lvl] = (g, [("unassigned:" + str(code)) if x == MISSING else str(int(x)) for x in u])
    out["district"] = (np.zeros(len(network), np.int64), [str(code)])
    return out


def district_worker(code, cfg):
    t0 = time.time()
    import whole_pipeline_v8 as wp
    from household_nlinear import _quiet
    L, H = wp.LOOKBACK, HORIZON
    restore_credentials()                                           # this worker may have hidden them for an earlier district
    D = read_district(code, cfg["roots"], cfg["start"], cfg["end"], cfg["labels"])
    res = dict(code=str(code), customers=int(len(D["keys"])), windows=0, gaps=0, seconds=0.0)
    if not len(D["keys"]):
        return res
    M, wt, dates = D["M"], D["wt"], D["dates"]
    T = M.shape[1]
    d0 = dates[0]
    origins = np.asarray(cfg["origins"], dtype=np.int64)           # day indices from cfg["start"]
    origins = origins[(origins >= L - 1) & (origins + H <= T - 1)]
    hz = np.asarray(cfg["horizons"]) - 1
    nO = len(cfg["origins"])
    oi_of = {int(o): i for i, o in enumerate(cfg["origins"])}

    # weather: climate normals and the simulated 10-day forecast, exactly as household_nlinear / node_syswape build them
    have = np.isfinite(wt).any(axis=0)
    wdf = pd.DataFrame({"coato_id": str(code), "d": dates[have], "t": wt[0, have], "tmin": wt[1, have], "tmax": wt[2, have]})
    od = np.arange(L - 1, T - H)
    with _quiet():
        clim = wp.build_climate(wdf)
        u_fc = pd.DataFrame({"coato_code": str(code), "origin_date": d0 + pd.to_timedelta(od, unit="D")})
        seed = int(cfg["seed"]) + zlib.crc32(str(code).encode()) % 1_000_003
        t_obs, _, _ = wp.lookup_weather(u_fc, clim, "all")
        t_fc = wp.simulate_forecast(u_fc, t_obs[:, L:], seed=seed)
        K = wp.FORECAST_DAYS
        fc = pd.DataFrame({"coato_id": str(code), "issue_date": np.repeat(u_fc["origin_date"].to_numpy(), K),
                           "lead_day": np.tile(np.arange(1, K + 1), len(u_fc)), "t_fc": t_fc.reshape(-1)})
        fc["issue_date"] = pd.to_datetime(fc["issue_date"])
        grid = wp.build_forecast_grid(fc, clim)
        tmin_obs, _, tmax_obs, _ = wp.lookup_minmax(u_fc, clim, "all")
        tmin_fc, tmax_fc = wp.simulate_forecast_minmax(u_fc, t_fc, tmin_obs[:, L:], tmax_obs[:, L:], seed=seed)
        fc["t_fc_min"], fc["t_fc_max"] = tmin_fc.reshape(-1), tmax_fc.reshape(-1)
        grid_mm = dict(min=wp.build_forecast_grid(fc, clim, col="t_fc_min"), max=wp.build_forecast_grid(fc, clim, col="t_fc_max"))
    del fc, t_obs, t_fc

    groups = _groups(D["network"], code)
    models = [cfg["model_name"], REFERENCE]
    acc = {lvl: dict(A=np.zeros((len(nm), nO, len(hz))), N=np.zeros((len(nm), nO, len(hz))),
                     P={m: np.zeros((len(nm), nO, len(hz))) for m in models}) for lvl, (g, nm) in groups.items()}
    mae = {m: np.zeros((2, len(hz))) for m in models}              # sum |P - A|, count
    scored = np.zeros(len(D["keys"]), bool)

    scrub_credentials()                                             # from here on nothing can read S3
    model = load_model(cfg["model_dir"], cfg["model_config"])
    model.load(cfg["artifacts_dir"])

    cnt = np.zeros((len(D["keys"]), T + 1), dtype=np.int32)
    np.cumsum(np.isfinite(M), axis=1, out=cnt[:, 1:])
    W = int(cfg["history_days"])
    blk = int(cfg["block"])
    for o in origins:
        n_in = cnt[:, o + 1] - cnt[:, o - L + 1]
        n_rec = cnt[:, o + 1] - cnt[:, o - wp.RECENT_WINDOW + 1]
        elig = np.flatnonzero((n_in >= wp.MIN_INPUT_DAYS) & (n_rec >= wp.MIN_RECENT_DAYS))
        if not len(elig):
            continue
        lo = max(0, o - W + 1)
        gi = int((np.datetime64(dates[o].date(), "D") - np.datetime64(clim.start, "D")).astype(int))
        g1 = np.full_like(grid, np.nan)
        g1[:, gi] = grid[:, gi]                                     # only the forecast issued on this origin
        gm = {k: np.full_like(v, np.nan) for k, v in grid_mm.items()}
        for k in gm:
            gm[k][:, gi] = grid_mm[k][:, gi]
        oi = oi_of[int(o)]
        for a in range(0, len(elig), blk):
            r = elig[a:a + blk]
            panel = Panel(coato=str(code), origin=dates[o], keys=D["keys"][r], kwh=M[r, lo:o + 1].copy(),
                          dates=dates[lo:o + 1], weather=wt[:, lo:o + 1].copy(), housing=D["housing"][r].copy(),
                          network=D["network"][r].copy(),
                          pipeline=dict(clim=clim, fc_grid=g1, fc_grid_mm=gm, seed=int(cfg["seed"])))
            P = np.asarray(model.predict(panel), dtype=np.float64)
            if P.shape != (len(r), H):
                raise RuntimeError(f"{cfg['model_name']}.predict returned shape {P.shape}, expected {(len(r), H)}")
            ref = persistence30(panel.kwh)
            bad = ~np.isfinite(P) | (P < 0)
            res["gaps"] += int(bad.sum())
            P = np.where(bad, ref[:, None], P)
            A = M[r, o + 1:o + 1 + H][:, hz].astype(np.float64)    # actuals: never shown to the model
            obs = np.isfinite(A)
            A0 = np.where(obs, A, 0.0)
            preds = {cfg["model_name"]: P[:, hz], REFERENCE: np.repeat(ref[:, None], len(hz), 1)}
            for m, Pm in preds.items():
                mae[m][0] += np.where(obs, np.abs(Pm - A0), 0.0).sum(0)
                mae[m][1] += obs.sum(0)
            for lvl, (g, nm) in groups.items():
                G = len(nm)
                flat = ((g[r][:, None] * nO + oi) * len(hz) + np.arange(len(hz))[None, :]).ravel()
                size = G * nO * len(hz)
                ac = acc[lvl]
                ac["A"] += np.bincount(flat, weights=A0.ravel(), minlength=size).reshape(G, nO, len(hz))
                ac["N"] += np.bincount(flat, weights=obs.ravel().astype(float), minlength=size).reshape(G, nO, len(hz))
                for m, Pm in preds.items():
                    ac["P"][m] += np.bincount(flat, weights=np.where(obs, Pm, 0.0).ravel(), minlength=size).reshape(G, nO, len(hz))
            res["windows"] += int(len(r))
            scored[r] = True
    for lvl, (g, nm) in groups.items():
        acc[lvl]["names"] = nm
        acc[lvl]["households"] = np.bincount(g, minlength=len(nm))
        acc[lvl]["scored"] = np.bincount(g[scored], minlength=len(nm))
    res.update(acc=acc, mae=mae, seconds=time.time() - t0)
    return res
