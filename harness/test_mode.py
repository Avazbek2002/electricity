"""evaluate.py --mode validate / test: the standard split and evaluation (configs/test_set.yaml, ML guide part 4).

  - One fit per quarterly refit date R (cached per model, commit and R); training targets d <= R - 1.
  - Per district (one worker reads it once): for every issue date o of the split, the households with fresh history
    (data up to o - 1) get the model's forecast for o + 1 .. o + 60; the district forecast is
        (customers in the district on o - 1) x (mean forecast of the households with fresh history)
    (guide part 3: customers without fresh history get the district mean per customer).
  - Climate is rebuilt for every issue date from weather up to o - 1; the standard run has no target-day weather.
  - A substitution test (guide part 5) runs first: inputs after a day T are randomised; forecasts issued up to T + 1
    must not change.
  - The district forecasts are written in the common format and scored by the common scorer against T2, with B0 on
    the same rows. Supplementary: bottom-up matched sums at substation / feeder / transformer level and household
    errors (observed target days of the forecast households).
  - Scaling (configs/eval.yaml run.scaling): the district forecast is multiplied by
        c(o) = mean T2 per customer over the last `days` days up to o - 1 / the fresh households' observed mean there
    (fresh households use more than the average customer, ~ +10 %; bottom_up_oracle.py). The unscaled forecast is
    written as <model>_unscaled and scored next to it.
"""
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import shutil
import time

import numpy as np
import pandas as pd

from harness import testset as ts
from harness.contract import HORIZON, Panel, TrainSpec
from harness.data import read_district, restore_credentials, scrub_credentials
from harness.runner import load_model, persistence30
from harness.weather import REFORECAST, climate_asof, district_weather, issue_index, no_forecast, reforecast_grid

NETWORK_LEVELS = ("substation", "feeder", "transformer")
NODE_FILES = ("substation", "feeder")          # per-node files; every level is also summarised per district and day


# ----------------------------------------------------------------------------- worker: one district, every issue date
def _perturb(D, T_idx, seed=12345):
    """Randomise every input after day T (consumption, presence, network, weather): the substitution test."""
    rng = np.random.default_rng(seed)
    M = D["M"]
    after = slice(T_idx + 1, None)
    shape = M[:, after].shape
    M[:, after] = np.where(rng.random(shape) < 0.3, np.nan, rng.gamma(2.0, 4.0, shape)).astype(M.dtype)
    D["wt"][:, after] = rng.normal(10, 15, D["wt"][:, after].shape)
    late = D["last"] > T_idx
    D["last"][late] = T_idx + 1 + rng.integers(0, 30, int(late.sum()))
    return D


def test_worker(code, cfg):
    t0 = time.time()
    import whole_pipeline_v8 as wp
    spec = cfg["spec"]
    d = spec["data"]
    restore_credentials()                                           # this worker may have hidden them for an earlier district
    D = read_district(code, [dict(uri=d["uri"], env_prefix=d.get("env_prefix", ""), start=d["start"], end=d["end"])],
                      d["start"], d["end"], list(cfg["labels"]))
    res = dict(code=str(code), seconds=0.0, gaps=0, windows=0, notes=[])
    if not len(D["keys"]):
        return res
    simulated = cfg["weather"] == "simulated"
    if simulated:                  # the simulated forecast archive is an outside input known on its issue date: it is
        # drawn from the unperturbed weather, so the substitution test checks the household data and observed weather
        _, grid_all, grid_mm_all = district_weather(code, D["wt"].copy(), D["dates"], cfg["seed"], wp)
    refc = cfg["weather"] in REFORECAST
    if refc:                       # the real AIFS / FCN3 forecasts issued on each date (an outside input, not perturbed)
        arch = pd.read_parquet(cfg["reforecast_path"])
        arch = arch[arch["coato_id"].astype(str) == str(code)]
        arch["issue_date"] = pd.to_datetime(arch["issue_date"])
    if cfg.get("perturb_after") is not None:
        D = _perturb(D, ts.day_index(spec, cfg["perturb_after"]))
    M, wt, dates = D["M"], D["wt"], D["dates"]
    T = M.shape[1]
    L, H = wp.LOOKBACK, HORIZON
    gap = int(spec["history_gap_days"])
    groups = ts.groups(D["network"], cfg["network_levels"]) if cfg["network_levels"] else {}
    ctrl = np.asarray(spec["control_horizons"]) - 1
    sc = cfg.get("scaling") or {}
    pc = cfg.get("per_customer", {}).get(str(code))                  # T2 per customer per day (test set), or None
    sdays = int(sc.get("days", 28))
    lo_c, hi_c = (sc.get("clip") or [0.5, 2.0])
    scrub_credentials()                                             # from here on nothing can read S3

    cnt = np.zeros((len(D["keys"]), T + 1), dtype=np.int32)
    np.cumsum(np.isfinite(M), axis=1, out=cnt[:, 1:])
    W, blk = int(cfg["history_days"]), int(cfg["block"])
    issues = [o for _, os_ in cfg["blocks"] for o in os_]
    nI = len(issues)
    F = np.full((nI, H), np.nan)                                     # district forecast per issue date and horizon
    F_raw = np.full((nI, H), np.nan)                                 # unscaled (present x mean forecast)
    scale = np.full(nI, np.nan)
    HH = np.zeros((nI, len(ctrl), 4))                                # household: sum|P-A|, sum A, sum(P-A), n observed
    n_present = np.zeros(nI, int)
    n_fresh = np.zeros(nI, int)
    net = {lv: dict(names=nm, FO=np.zeros((len(nm), nI, len(ctrl))), AO=np.zeros((len(nm), nI, len(ctrl))),
                    NO=np.zeros((len(nm), nI, len(ctrl)))) for lv, (g, nm) in groups.items()}
    k = 0
    for R, block in cfg["blocks"]:
        model = load_model(cfg["model_dir"], cfg["model_config"])
        model.load(cfg["artifacts"][str(R.date())])
        for o_day in block:
            o = ts.day_index(spec, o_day)
            if o - gap < L or o + H > T - 1:
                k += 1
                continue
            last = o - gap                                           # the last day of known data
            n_in = cnt[:, last + 1] - cnt[:, last + 1 - L]
            n_rec = cnt[:, last + 1] - cnt[:, last + 1 - wp.RECENT_WINDOW]
            fresh = np.flatnonzero((n_in >= wp.MIN_INPUT_DAYS) & (n_rec >= wp.MIN_RECENT_DAYS))
            present = int(((D["first"] <= last) & (D["last"] >= last)).sum())
            n_present[k], n_fresh[k] = present, len(fresh)
            if not len(fresh):
                k += 1
                continue
            clim = climate_asof(code, wt, dates, last + 1, wp)       # weather of days <= o - 1 only
            if simulated:                                            # flagged preliminary run: the forecast issued on o
                gi = issue_index(clim, dates[o])
                g1 = np.full_like(grid_all, np.nan)
                gm = {kk: np.full_like(v, np.nan) for kk, v in grid_mm_all.items()}
                if 0 <= gi < grid_all.shape[1]:
                    g1[:, gi] = grid_all[:, gi]
                    for kk in gm:
                        gm[kk][:, gi] = grid_mm_all[kk][:, gi]
            elif refc:                                               # the real forecast issued on o (harness leads)
                g1, gm = reforecast_grid(code, arch, clim, dates[o], wp)
            else:
                g1, gm = no_forecast(int(clim.n_days), wp.FORECAST_DAYS)
            lo = max(0, last + 1 - W)
            tot = np.zeros(H)
            for a in range(0, len(fresh), blk):
                r = fresh[a:a + blk]
                panel = Panel(coato=str(code), origin=dates[o], keys=D["keys"][r], kwh=M[r, lo:last + 1].copy(),
                              dates=dates[lo:last + 1], weather=wt[:, lo:last + 1].copy(), housing=D["housing"][r].copy(),
                              network=D["network"][r].copy(),
                              pipeline=dict(clim=clim, fc_grid=g1, fc_grid_mm=gm, seed=int(cfg["seed"]),
                                            weather_mode=cfg["weather"]))
                P = np.asarray(model.predict(panel), dtype=np.float64)
                if P.shape != (len(r), H):
                    raise RuntimeError(f"{cfg['model_name']}.predict returned {P.shape}, expected {(len(r), H)}")
                ref = persistence30(panel.kwh)
                bad = ~np.isfinite(P) | (P < 0)
                res["gaps"] += int(bad.sum())
                P = np.where(bad, ref[:, None], P)
                tot += P.sum(0)
                res["windows"] += len(r)
                Ah = M[r][:, o + 1:o + 1 + H][:, ctrl].astype(np.float64)   # household level, observed days only
                ob_h = np.isfinite(Ah)
                e = np.where(ob_h, P[:, ctrl] - np.nan_to_num(Ah), 0.0)
                HH[k, :, 0] += np.abs(e).sum(0)
                HH[k, :, 1] += np.where(ob_h, Ah, 0.0).sum(0)
                HH[k, :, 2] += e.sum(0)
                HH[k, :, 3] += ob_h.sum(0)
                if groups:                                           # supplementary: matched, observed households only
                    A = M[r][:, o + 1:o + 1 + H][:, ctrl].astype(np.float64)
                    ob = np.isfinite(A)
                    for lv, (g, nm) in groups.items():
                        gb, K = g[r], len(nm)
                        net[lv]["FO"][:, k, :] += ts.bincount2(gb, np.where(ob, P[:, ctrl], 0.0), K)
                        net[lv]["AO"][:, k, :] += ts.bincount2(gb, np.where(ob, A, 0.0), K)
                        net[lv]["NO"][:, k, :] += ts.bincount2(gb, ob.astype(np.float64), K)
            F_raw[k] = present * tot / len(fresh)
            c = 1.0
            if sc.get("method", "ratio") == "ratio" and pc is not None:
                w0 = max(0, last + 1 - sdays)
                win = M[fresh, w0:last + 1]
                n_obs = np.isfinite(win).sum()
                fm = float(np.nansum(win) / n_obs) if n_obs else np.nan
                pm = float(np.nanmean(pc[w0:last + 1])) if np.isfinite(pc[w0:last + 1]).any() else np.nan
                if np.isfinite(fm) and np.isfinite(pm) and fm > 0:
                    c = float(np.clip(pm / fm, lo_c, hi_c))
            scale[k] = c
            F[k] = F_raw[k] * c
            k += 1
        for n in getattr(model, "notes", []) or []:
            if n not in res["notes"]:
                res["notes"].append(n)
    # every network level summarised per issue date and control horizon (node-days with >= min households observed);
    # per-node arrays are kept only for NODE_FILES (the transformer level would be too large to return)
    mh = int(spec["scoring"]["min_households_network"])
    net_sum = {}
    for lv, e in net.items():
        ok = e["NO"] >= mh
        net_sum[lv] = np.stack([np.where(ok, np.abs(e["FO"] - e["AO"]), 0).sum(0), np.where(ok, e["AO"], 0).sum(0),
                                np.where(ok, e["FO"] - e["AO"], 0).sum(0), ok.sum(0)], axis=-1)   # (nI, nctrl, 4)
    net = {lv: e for lv, e in net.items() if lv in NODE_FILES}
    res.update(tminmax=np.asarray(wt[1:3], dtype=np.float32), dates0=str(dates[0].date()),
               F=F.astype(np.float32), F_raw=F_raw.astype(np.float32), scale=scale, HH=HH, net_sum=net_sum,
               n_present=n_present, n_fresh=n_fresh, issues=issues, net=net,
               seconds=time.time() - t0, peak_bytes=ts.peak_rss_bytes())
    return res


def per_customer_series(test_dir, spec, codes):
    """{district: T2 per customer per day (day index of the data start), NaN on artefact days} for the scaling."""
    t = pd.read_csv(os.path.join(test_dir, "targets", "district.csv.gz"), dtype={"node": str}, parse_dates=["date"],
                    usecols=["node", "date", "n_obs", "n_est", "kwh_obs", "kwh_est", "artefact"])
    t = t[t["node"].isin(set(codes))]
    T = ts.day_index(spec, spec["data"]["end"]) + 1
    out = {}
    for node, g in t.groupby("node"):
        nc = g["n_obs"].fillna(0) + g["n_est"].fillna(0)
        v = (g["kwh_obs"].fillna(0) + g["kwh_est"].fillna(0)) / nc.where(nc > 0)
        v = v.where(~g["artefact"].astype(str).str.lower().isin(("true", "1")))
        a = np.full(T, np.nan)
        i = ((g["date"] - pd.Timestamp(spec["data"]["start"])).dt.days).to_numpy()
        ok = (i >= 0) & (i < T)
        a[i[ok]] = v.to_numpy()[ok]
        out[str(node)] = a
    return out


# ----------------------------------------------------------------------------- writing
def district_frame(results, key="F"):
    parts = []
    H = np.arange(1, HORIZON + 1)
    for r in results:
        if key not in r:
            continue
        nI = len(r["issues"])
        parts.append(pd.DataFrame({"node": r["code"], "origin": np.repeat(pd.DatetimeIndex(r["issues"]).date, HORIZON),
                                   "horizon": np.tile(H, nI), "forecast": r[key].ravel()}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["node", "origin", "horizon", "forecast"])


def write_levels(fdir, spec, results):
    """levels.csv.gz: per district, issue date and control horizon, the error sums of the household level (observed
    target days of the forecast households) and of every network level (node-days with >= min households observed):
    abs_err = sum |F - A|, actual = sum A, err = sum (F - A), n = household-days or node-days. District sums only."""
    ctrl = list(spec["control_horizons"])
    rows = []
    for r in results:
        if "HH" not in r:
            continue
        nI = len(r["issues"])
        base = dict(node=r["code"], origin=np.repeat(pd.DatetimeIndex(r["issues"]).date, len(ctrl)),
                    horizon=np.tile(ctrl, nI))
        for lv, A in [("household", r["HH"])] + list(r.get("net_sum", {}).items()):
            rows.append(pd.DataFrame({**base, "level": lv, "abs_err": A[..., 0].ravel(), "actual": A[..., 1].ravel(),
                                      "err": A[..., 2].ravel(), "n": A[..., 3].ravel().astype(int)}))
    if rows:
        df = pd.concat(rows, ignore_index=True)
        df[df["n"] > 0].to_csv(os.path.join(fdir, "levels.csv.gz"), index=False, float_format="%.6f")


def write_network(fdir, spec, results):
    ctrl = list(spec["control_horizons"])
    for lv in NODE_FILES:
        acc = {}
        for r in results:
            e = r.get("net", {}).get(lv)
            if not e:
                continue
            for i, nm in enumerate(e["names"]):
                cur = acc.get(nm)
                parts = (e["FO"][i], e["AO"][i], e["NO"][i])
                acc[nm] = parts if cur is None else tuple(a + b for a, b in zip(cur, parts))
            issues = r["issues"]
        if not acc:
            continue
        rows = []
        for nm, (fo, ao, no) in acc.items():
            rows.append(pd.DataFrame({"node": nm, "origin": np.repeat(pd.DatetimeIndex(issues).date, len(ctrl)),
                                      "horizon": np.tile(ctrl, len(issues)), "forecast_observed_sum": fo.ravel(),
                                      "actual_observed_sum": ao.ravel(), "observed": no.ravel().astype(int)}))
        df = pd.concat(rows, ignore_index=True)
        df = df[df["observed"] >= int(spec["scoring"]["min_households_network"])]
        os.makedirs(os.path.join(fdir, "network"), exist_ok=True)
        df.to_csv(os.path.join(fdir, "network", f"{lv}.csv.gz"), index=False, float_format="%.6f")


# ----------------------------------------------------------------------------- the substitution test
def substitution_test(wcfg, code, spec):
    """Guide part 5: randomise all inputs after day T; forecasts issued up to T + 1 must be identical."""
    R, block = wcfg["blocks"][0]
    T_day = pd.Timestamp(block[min(5, len(block) - 1)])
    sel = [o for o in block if T_day - pd.Timedelta(days=1) <= o <= T_day + pd.Timedelta(days=3)]
    sel = sorted(set(sel) | set([o for o in block if o > T_day + pd.Timedelta(days=1)][:1]))   # sparse issue dates
    c = dict(wcfg, blocks=[(R, sel)], network_levels=())
    a = test_worker(code, dict(c, perturb_after=None))
    b = test_worker(code, dict(c, perturb_after=str(T_day.date())))
    keep = np.array([o <= T_day + pd.Timedelta(days=1) for o in sel])
    diff = np.abs(a["F"].astype(float) - b["F"].astype(float))
    same = float(np.nanmax(diff[keep])) if keep.any() else 0.0
    later = float(np.nanmax(diff[~keep])) if (~keep).any() else np.nan
    passed = bool(same == 0.0 and np.isfinite(a["F"][keep]).all())
    detail = (f"district {code}, T = {T_day.date()}: max change of forecasts issued <= T+1: {same:.3g}; "
              f"issued >= T+2: {later:.3g}" + ("" if later > 0 else " (later forecasts did not react: check the model)"))
    return dict(passed=passed, detail=detail, district=code, T=str(T_day.date()))


def _reforecast_blocks(blocks, weather, model_config, out_root):
    """Keep the issue dates that have an AIFS / FCN3 forecast (guide part 2: weather and no-weather models are compared
    on the same dates). With --weather aifs|fcn3|ens the archive is built (cached) here, in the main process, and handed
    to the workers as a local file. -> (blocks, archive path or None)"""
    import reforecast as rf
    w = rf.wx_config(model_config)
    if weather in REFORECAST:
        if model_config.get("wx") and w["source"] != weather:
            raise SystemExit(f"[weather] this model was trained with wx.source = {w['source']}: run --weather {w['source']}")
        w["source"] = weather
        arch = rf.load_archive(w)
        dates = set(pd.to_datetime(arch["issue_date"].unique()))
        sig = rf._cache_path(w, weather, "eval").rsplit("_", 1)[-1]
        path = os.path.join(os.path.expanduser(w["cache_dir"]), f"eval_{weather}_{sig}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        arch.to_parquet(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)
    else:
        dates, path = set(rf.issue_dates(w)), None
    out = [(R, pd.DatetimeIndex([o for o in b if o in dates])) for R, b in blocks]
    out = [(R, b) for R, b in out if len(b)]
    n0, n1 = sum(len(b) for _, b in blocks), sum(len(b) for _, b in out)
    print(f"[weather] {weather}: scoring the {n1} of {n0} issue dates that have a {w['source']} forecast "
          f"(sets {', '.join(w['sets'])})")
    if not out:
        raise SystemExit("[weather] no issue date of this split has a reforecast")
    return out, path


# ----------------------------------------------------------------------------- the command
def run_test(args, cfg, git, model_dir, model_config, tracking, scoring=None):
    """evaluate.py --mode validate / test. The W&B run opens before the first fit (training is logged live) and is
    closed with a failure state if anything stops the run."""
    holder = {}
    try:
        return _run_test(args, cfg, git, model_dir, model_config, tracking, holder)
    except BaseException:
        r = holder.get("run")
        if r is not None:
            try:
                r.finish(exit_code=1)
            except Exception:                                        # noqa: BLE001
                pass
        raise


def _run_test(args, cfg, git, model_dir, model_config, tracking, holder):
    from harness import wandb_log
    spec = ts.load_spec(args.test_set)
    scope = "sample" if args.sample else ("trial" if args.coatos else "national")
    default_dir = os.path.join("~/work/test_set", spec["version"] + {"sample": "-sample", "trial": "-trial"}.get(scope, ""))
    test_dir = os.path.expanduser(args.test_dir or default_dir)
    if not os.path.exists(os.path.join(test_dir, "manifest.json")) and scope == "sample":
        test_dir = os.path.expanduser(args.test_dir or os.path.join("~/work/test_set", spec["version"]))
    man = ts.read_manifest(test_dir)
    ts.check_same_spec(spec, man)
    run_cfg = cfg["run"]
    out_root = os.path.expanduser(args.out_dir or run_cfg["out_dir"])
    workers = args.workers or int(run_cfg["workers"])
    d = spec["data"]
    regions = pd.read_csv(os.path.join(test_dir, "districts.csv"), dtype=str).set_index("district")["region"].to_dict()
    codes = sorted(regions)
    if args.sample:
        codes = [c for c in spec["sample_districts"] if c in regions]
    elif args.coatos:
        codes = [c for c in codes if c in set(args.coatos.split(","))]
    if not codes:
        raise SystemExit("no districts to run (check --coatos / --sample against the built test set)")
    splits = ["validation"] if args.mode == "validate" else ["validation", "test"]
    if args.mode == "test":
        runs = tracking.read_runs(os.path.join(out_root, "runs.csv"))
        seen = len(runs) and ((runs["model"] == args.model) & (runs["commit"] == git["commit"]) &
                              (runs["mode"] == "validate")).any()
        if not seen:
            print("[test] NOTE: no validation run of this model and commit is recorded; the guide chooses models on "
                  "validation and looks at the test only for the chosen variants")
    blocks = ts.refit_blocks(spec, splits)
    weather = args.weather or spec.get("weather", "none")
    reforecast_path = None
    if weather in REFORECAST or getattr(args, "issue_dates", "all") == "reforecast":
        blocks, reforecast_path = _reforecast_blocks(blocks, weather, model_config, out_root)
    notes = [] if weather == "none" else [f"weather = {weather}: preliminary (not the standard no-weather run)"]
    wcfg = dict(cfg.get("wandb") or {})
    if getattr(args, "no_wandb", False):
        wcfg["enabled"] = False
    sc = run_cfg.get("scaling") or {"method": "ratio"}
    wrun = holder["run"] = wandb_log.start(
        wcfg, name=f"{args.model} | {args.mode} | {scope} | {git['short']}", group=f"{spec['version']} {scope} {weather}",
        job_type=args.mode, tags=[args.model, args.mode, scope, f"weather-{weather}", model_config.get("family", "")],
        config=dict(model=args.model, family=model_config.get("family", args.model), model_config=model_config,
                    test_set=spec["version"], test_set_sha256=spec["_sha256"][:12], data_version=man["data_version"],
                    scope=scope, districts="all" if scope == "national" else ",".join(codes), weather=weather,
                    scaling=sc, commit=git["commit"], dirty=git["dirty"], splits=splits,
                    refits=[str(R.date()) for R, _ in blocks], workers=workers))

    # ---- one fit per refit date
    arts, fit_seconds = {}, 0.0
    import whole_pipeline_v8 as wp
    for i_fit, (R, _) in enumerate(blocks):
        cutoff, data_end = ts.train_window(spec, R)
        t_fit, reused = time.time(), False
        key = f"{args.model}/{git['short']}{'-dirty' if git['dirty'] else ''}/refit-{R.date()}"
        art = os.path.join(out_root, "artifacts", key)
        done = os.path.join(art, ".fit_done")
        if args.refit and os.path.exists(art):
            shutil.rmtree(art)
        if os.path.exists(done):
            print(f"[fit] refit {R.date()}: reusing {art}")
            reused = True
        else:
            os.makedirs(art, exist_ok=True)
            spec_t = TrainSpec(roots=[dict(uri=d["uri"], env_prefix=d.get("env_prefix", ""))], start=d["start"],
                               cutoff=str(cutoff.date()), data_end=str(data_end.date()),
                               val_start=str((cutoff - pd.Timedelta(days=84)).date()), val_end=str(cutoff.date()),
                               workers=workers, labels=tuple(cfg["data"]["labels"]))
            print(f"[fit] refit {R.date()}: targets <= {data_end.date()}, issue dates <= {cutoff.date()} -> {art}")
            t0 = time.time()
            load_model(model_dir, model_config).fit(spec_t, art)
            open(done, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
            fit_seconds += time.time() - t0
        arts[str(R.date())] = art
        wandb_log.log_fit(wrun, i_fit, R.date(), art, time.time() - t_fit, reused)

    wcfg = dict(spec=spec, blocks=blocks, labels=cfg["data"]["labels"], model_name=args.model, model_dir=model_dir,
                model_config=model_config, artifacts=arts, seed=wp.SEED, weather=weather, reforecast_path=reforecast_path,
                history_days=int(cfg["data"]["history_days"]), block=int(run_cfg["block"]),
                network_levels=() if args.no_network else NETWORK_LEVELS,
                scaling=run_cfg.get("scaling") or {"method": "ratio"},
                per_customer=per_customer_series(test_dir, spec, codes))
    thr = max(1, (os.cpu_count() or 4) // workers)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(thr)

    # ---- substitution test (leakage), on the first district
    sub = None
    if not args.no_substitution_test:
        print(f"[leakage] substitution test on district {codes[0]} ...", flush=True)
        with cf.ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
            sub = ex.submit(substitution_test, wcfg, codes[0], spec).result()
        print(f"[leakage] {'PASSED' if sub['passed'] else 'FAILED'}: {sub['detail']}", flush=True)
        if not sub["passed"] and not args.allow_leak:
            raise SystemExit("[leakage] the model's forecasts depend on data after the issue date - 1: fix it "
                             "(or --allow-leak to run anyway, recorded as FAILED)")

    n_issue = sum(len(b) for _, b in blocks)
    print(f"[run] {args.model} | {spec['version']} {'+'.join(splits)} | {len(blocks)} refits, {n_issue} issue dates | "
          f"{len(codes)} districts ({scope}) | weather {weather} | {workers} workers | commit {git['short']}"
          f"{' (dirty)' if git['dirty'] else ''}")
    print(f"[memory] at start: {ts.memory_line()}", flush=True)
    results, t1 = [], time.time()
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        pending = {ex.submit(test_worker, c, wcfg): c for c in codes}
        k = 0
        while pending:
            done_, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
            for fu in done_:
                code = pending.pop(fu)
                try:
                    r = fu.result()
                except Exception as e:                               # noqa: BLE001
                    raise RuntimeError(f"district {code} failed: {type(e).__name__}: {e}") from e
                results.append(r)
                for n in r.get("notes", []):
                    if n not in notes:
                        notes.append(n)
                k += 1
                print(f"[districts] {k}/{len(codes)} {code}: {r['seconds']:.0f} s | {ts.memory_line(r.get('peak_bytes'))}",
                      flush=True)
    predict_seconds = time.time() - t1

    # ---- write, score, log
    out = os.path.join(out_root, "runs", f"{time.strftime('%Y%m%d-%H%M%S')}_{args.model}_{args.mode}")
    fdir = os.path.join(out, "forecasts", args.model)
    os.makedirs(fdir, exist_ok=True)
    district_frame(results).to_csv(os.path.join(fdir, "district.csv.gz"), index=False, float_format="%.6f")
    write_network(fdir, spec, results)
    write_levels(fdir, spec, results)
    sc = run_cfg.get("scaling") or {"method": "ratio"}
    others = {}
    if sc.get("method", "ratio") != "none":                         # the unscaled forecast, scored next to it
        udir = os.path.join(out, "forecasts", f"{args.model}_unscaled")
        os.makedirs(udir, exist_ok=True)
        district_frame(results, "F_raw").to_csv(os.path.join(udir, "district.csv.gz"), index=False, float_format="%.6f")
        json.dump(dict(model=f"{args.model}_unscaled", test_set_sha256=spec["_sha256"], target="T2"),
                  open(os.path.join(udir, "forecast_meta.json"), "w"))
        others[f"{args.model}_unscaled"] = udir
        allc = np.concatenate([r["scale"] for r in results if "scale" in r])
        allc = allc[np.isfinite(allc)]
        if len(allc):
            print(f"[scaling] district factor c(o) = T2 per customer / fresh households' mean over the last "
                  f"{sc.get('days', 28)} days: median {np.median(allc):.3f} (5-95 %: {np.percentile(allc, 5):.3f} .. "
                  f"{np.percentile(allc, 95):.3f})")
    meta = dict(model=args.model, test_set_version=spec["version"], test_set_sha256=spec["_sha256"], target="T2",
                weather=weather, level="client (bottom-up)", git_commit=git["commit"], dirty=git["dirty"], scope=scope,
                districts=codes if scope != "national" else "all", splits=splits, substitution_test=sub, notes=notes,
                household_forecasts=sum(r["windows"] for r in results),
                gaps_filled_with_persistence=sum(r["gaps"] for r in results), scaling=sc)
    meta["test_dir"] = test_dir
    json.dump(meta, open(os.path.join(fdir, "forecast_meta.json"), "w"), indent=2, default=str)
    res = ts.score(fdir, test_dir, spec, others=others)
    text = ts.report(res, spec)
    print("\n" + text)
    ts.save_scores(res, os.path.join(out, "scores"), text)
    weather_daily = wandb_log.weather_frame(results)
    if weather_daily is not None:
        weather_daily.to_csv(os.path.join(out, "scores", "weather_daily.csv"), index=False, float_format="%.2f")
    charts = {}
    try:
        from harness import plots
        plots.metrics_by_horizon(res, os.path.join(out, "scores"))
        plots.levels(res, os.path.join(out, "scores"))
        n_win = len(plots.window_samples(res, os.path.join(out, "scores")))
        charts = {k: os.path.join(out, "scores", k) for k in ("figures", "windows")}
        print(f"[charts] {os.path.join(out, 'scores')}: metrics by horizon, {n_win} window charts")
    except Exception as e:                                          # noqa: BLE001  charts never stop a run
        print(f"[charts] WARNING: not drawn ({type(e).__name__}: {e})")
    split_h = "validation" if args.mode == "validate" else "test"
    headline = ts.headline(res, split_h)
    tags = dict(model_family=model_config.get("family", args.model), model=args.model, git_commit=git["commit"],
                git_branch=git["branch"], dirty=str(git["dirty"]).lower(), mode=args.mode,
                cutoff=",".join(str(R.date()) for R, _ in blocks), holdout_period="", data_version=man["data_version"],
                districts="all" if scope == "national" else ",".join(codes), origins=spec["version"],
                test_set_version=spec["version"], test_set_sha256=spec["_sha256"], scope=scope, weather=weather,
                target="T2", substitution_test="skipped" if sub is None else ("passed" if sub["passed"] else "FAILED"))
    params = {**{f"model.{k}": v for k, v in model_config.items()}, "test_set": spec["version"], "splits": "+".join(splits),
              "n_issue_dates": n_issue, "refits": len(blocks), "weather": weather}
    metrics = {args.model: {**ts.metrics_for_mlflow(res), "headline": headline}}
    files = [os.path.join(out, "scores", f) for f in ("scores.csv", "spec_check.csv", "bias_audit.csv", "report.txt",
                                                      "national_daily.csv", "levels.csv")
             if os.path.exists(os.path.join(out, "scores", f))]
    # ---- experiment tracking: Weights & Biases (the run opened before the fits)
    wandb_url = wandb_log.log_run(
        wcfg, name=f"{args.model} | {args.mode} | {scope} | {git['short']}", group=f"{spec['version']} {scope} {weather}",
        job_type=args.mode, tags=[args.model, args.mode, scope, f"weather-{weather}", model_config.get("family", "")],
        config=dict(model=args.model, family=model_config.get("family", args.model), model_config=model_config,
                    test_set=spec["version"], test_set_sha256=spec["_sha256"][:12], data_version=man["data_version"],
                    scope=scope, districts=tags["districts"], weather=weather, scaling=sc, commit=git["commit"],
                    dirty=git["dirty"], refits=[str(R.date()) for R, _ in blocks], issue_dates=n_issue, splits=splits,
                    substitution_test=tags["substitution_test"], fit_seconds=round(fit_seconds),
                    predict_seconds=round(predict_seconds), run_dir=out),
        res=res, weather=weather_daily, out_dir=os.path.join(out, "scores"),
        extra_summary={"headline": headline, "substitution_test": tags["substitution_test"]},
        files=files, run=wrun)
    holder["run"] = None                                             # finished by log_run

    # ---- model registry: MLflow (only with --register; tracking is in W&B)
    mlcfg = dict(cfg.get("mlflow", {}))
    if args.no_mlflow or not args.register:
        mlcfg["enabled"] = False
    if mlcfg.get("enabled", True) and not os.getenv("MLFLOW_TRACKING_URI") and not mlcfg.get("uri"):
        mlcfg["uri"] = "sqlite:///" + os.path.join(out_root, "mlflow.db")
        print("\n" + "!" * 100 + "\n[mlflow] WARNING: MLFLOW_TRACKING_URI is not set -> this run goes to the LOCAL store "
              f"{mlcfg['uri']}, not the team server.\n          source ~/work/mlflow_env.sh before running.\n" + "!" * 100 + "\n")
    lastR = str(blocks[-1][0].date())                                # the newest fit is the one registered
    run_id = tracking.mlflow_log_safe(mlcfg, tags, params, dict(fit_seconds=fit_seconds, predict_seconds=predict_seconds),
                                      metrics, files, args.model, register=args.register, artifacts_dir=arts[lastR],
                                      model_dir=model_dir, model_config=model_config, headline=headline,
                                      curves=ts.curves_for_mlflow(res), artifact_dirs=charts)
    t = res["splits"][split_h]["scores"]
    row = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), model=args.model, commit=git["commit"], mode=args.mode,
               holdout_period=spec["version"], status="done", scope=scope, weather=weather, headline=headline,
               reference_headline=float(t[(t["model"] == "B0") & t["horizon"].isin(res["ctrl"])]["sysmape"].mean()),
               substitution_test=tags["substitution_test"], run_dir=out, mlflow_run=run_id or "", wandb_run=wandb_url or "",
               spec_pass=bool(res["spec_check"]["passed"].all()) if args.mode == "test" else "")
    tracking.append_csv(os.path.join(out_root, "runs.csv"), row)
    tracking.append_csv(os.path.join(out_root, "test_leaderboard.csv"),
                        {**row, **{k: v for k, v in metrics[args.model].items() if "/national/sysmape" in k}})
    print(f"\n[done] {out} | {split_h} headline (national sysMAPE, mean of h {res['ctrl']}) {headline:.2%} "
          f"(B0 {row['reference_headline']:.2%})")
    return out
