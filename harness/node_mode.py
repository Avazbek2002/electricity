"""evaluate.py --test-set configs/test_set_substations_2025.yaml: the substation / feeder test for any household model.

  python evaluate.py --model tide_agg --mode validate --test-set configs/test_set_substations_2025.yaml   # dev nodes
  python evaluate.py --model tide_agg --mode test     --test-set configs/test_set_substations_2025.yaml   # test, once

  - Fits: one per quarterly refit R of the spec (targets d <= R - 1), cached per model, commit and R in the same place
    as the district runs (artifacts/<model>/<commit>/refit-<R>): a model fitted for the district test is reused.
  - Per district of the role's substations (one worker reads it once): every issue date o of 2025, the households
    under the role's substations and feeders with fresh history (data up to o - 1) get the model's forecast for
    o + 1 .. o + 60; per node
        F_node(o, h) = (households in span on o - 1) x (mean forecast of the node's fresh households) x c_node(o)
    c_node(o) = node T2 per household over the last `days` days up to o - 1 / the fresh households' observed mean there
    (configs/eval.yaml run.scaling, as the district harness). The unscaled forecast is scored next to it.
  - Node membership is the household's network ids as read by harness.data.read_district (first known); the node
    targets use the row-level ids of each day (moved households <= 10 % by the selection rules).
  - Substitution test first (inputs after a day T randomised; node forecasts issued up to T + 1 must not change).
  - Scores: harness/nodeset.py against node T2 and the node B0, per node, median and pooled, by size band.
"""
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import shutil
import time

import numpy as np
import pandas as pd

from harness import nodeset as ns
from harness import testset as ts
from harness.contract import HORIZON, Panel, TrainSpec
from harness.data import read_district, restore_credentials, scrub_credentials
from harness.runner import load_model, persistence30
from harness.weather import climate_asof, no_forecast


def refit_blocks(spec, role):
    days = ns.issue_dates(spec, role)
    R = pd.to_datetime(spec["refits"])
    out = []
    for i, r in enumerate(R):
        nxt = R[i + 1] if i + 1 < len(R) else pd.Timestamp.max
        sel = days[(days >= r) & (days < nxt)]
        if len(sel):
            out.append((r, sel))
    return out


def _perturb(D, T_idx, seed=12345):
    rng = np.random.default_rng(seed)
    after = slice(T_idx + 1, None)
    shape = D["M"][:, after].shape
    D["M"][:, after] = np.where(rng.random(shape) < 0.3, np.nan, rng.gamma(2.0, 4.0, shape)).astype(D["M"].dtype)
    D["wt"][:, after] = rng.normal(10, 15, D["wt"][:, after].shape)
    late = D["last"] > T_idx
    D["last"][late] = T_idx + 1 + rng.integers(0, 30, int(late.sum()))
    return D


def node_worker(code, cfg):
    """One district -> per node of the role: F (K, nI, H) scaled, F_raw unscaled, scale, fresh counts."""
    t0 = time.time()
    import whole_pipeline_v8 as wp
    spec = cfg["spec"]
    d = cfg["data"]
    restore_credentials()
    D = read_district(code, [dict(uri=d["uri"], env_prefix=d.get("env_prefix", ""), start=d["start"], end=d["end"])],
                      d["start"], d["end"], list(cfg["labels"]))
    names = cfg["nodes"][code]                                     # [(node, level, node_id)]
    K = len(names)
    issues = [o for _, b in cfg["blocks"] for o in b]
    nI, H = len(issues), HORIZON
    res = dict(code=str(code), names=[n for n, _, _ in names], issues=issues, seconds=0.0, gaps=0, windows=0, notes=[])
    if not len(D["keys"]) or not K:
        return res
    if cfg.get("perturb_after") is not None:
        D = _perturb(D, ts.day_index(spec, cfg["perturb_after"]))
    M, wt, dates = D["M"], D["wt"], D["dates"]
    T = M.shape[1]
    # household -> substation node / feeder node (-1 = none of the role's nodes)
    sub = {nid: k for k, (_, lv, nid) in enumerate(names) if lv == "substation"}
    fed = {nid: k for k, (_, lv, nid) in enumerate(names) if lv == "feeder"}
    gs = np.array([sub.get(int(v), -1) for v in D["network"][:, 2]], dtype=np.int64)
    gf = np.array([fed.get(int(v), -1) for v in D["network"][:, 1]], dtype=np.int64)
    sel = np.flatnonzero((gs >= 0) | (gf >= 0))
    L, gap = wp.LOOKBACK, int(spec["history_gap_days"])
    sc = cfg.get("scaling") or {}
    sdays = int(sc.get("days", 28))
    lo_c, hi_c = (sc.get("clip") or [0.5, 2.0])
    pcs, nrs = cfg["per_customer"], cfg["n_rows"]
    scrub_credentials()

    cnt = np.zeros((len(sel), T + 1), dtype=np.int32)
    np.cumsum(np.isfinite(M[sel]), axis=1, out=cnt[:, 1:])
    W, blk = int(cfg["history_days"]), int(cfg["block"])
    F = np.full((K, nI, H), np.nan)
    F_raw = np.full((K, nI, H), np.nan)
    scale = np.full((K, nI), np.nan)
    n_fresh = np.zeros((K, nI), int)
    k = 0
    for R, block in cfg["blocks"]:
        model = load_model(cfg["model_dir"], cfg["model_config"])
        model.load(cfg["artifacts"][str(R.date())])
        for o_day in block:
            o = ts.day_index(spec, o_day)
            if o - gap < L or o + H > T - 1:
                k += 1
                continue
            last = o - gap
            n_in = cnt[:, last + 1] - cnt[:, last + 1 - L]
            n_rec = cnt[:, last + 1] - cnt[:, last + 1 - wp.RECENT_WINDOW]
            fresh = sel[(n_in >= wp.MIN_INPUT_DAYS) & (n_rec >= wp.MIN_RECENT_DAYS)]
            if not len(fresh):
                k += 1
                continue
            clim = climate_asof(code, wt, dates, last + 1, wp)
            g1, gm = no_forecast(int(clim.n_days), wp.FORECAST_DAYS)
            lo = max(0, last + 1 - W)
            tot = np.zeros((K, H))
            num = np.zeros(K)
            for a in range(0, len(fresh), blk):
                r = fresh[a:a + blk]
                panel = Panel(coato=str(code), origin=dates[o], keys=D["keys"][r], kwh=M[r, lo:last + 1].copy(),
                              dates=dates[lo:last + 1], weather=wt[:, lo:last + 1].copy(), housing=D["housing"][r].copy(),
                              network=D["network"][r].copy(),
                              pipeline=dict(clim=clim, fc_grid=g1, fc_grid_mm=gm, seed=int(cfg["seed"]), weather_mode="none"))
                P = np.asarray(model.predict(panel), dtype=np.float64)
                if P.shape != (len(r), H):
                    raise RuntimeError(f"{cfg['model_name']}.predict returned {P.shape}, expected {(len(r), H)}")
                ref = persistence30(panel.kwh)
                bad = ~np.isfinite(P) | (P < 0)
                res["gaps"] += int(bad.sum())
                P = np.where(bad, ref[:, None], P)
                res["windows"] += len(r)
                for g in (gs[r], gf[r]):
                    tot += ts.bincount2(g, P, K)
                    num += np.bincount(g[g >= 0], minlength=K)
            for j, (node, lv, nid) in enumerate(names):
                if num[j] == 0:
                    continue
                n_fresh[j, k] = int(num[j])
                members = fresh[(gs[fresh] == j) | (gf[fresh] == j)]
                present = nrs.get(node, np.full(T, np.nan))[last] if node in nrs else np.nan
                if not np.isfinite(present) or present <= 0:          # the node's households in span on o - 1
                    present = float((((D["first"] <= last) & (D["last"] >= last)) &
                                     ((gs == j) | (gf == j))).sum())
                F_raw[j, k] = present * tot[j] / num[j]
                c = 1.0
                pc = pcs.get(node)
                if sc.get("method", "ratio") == "ratio" and pc is not None:
                    w0 = max(0, last + 1 - sdays)
                    win = M[members, w0:last + 1]
                    n_obs = np.isfinite(win).sum()
                    fm = float(np.nansum(win) / n_obs) if n_obs else np.nan
                    seg = pc[w0:last + 1]
                    pm = float(np.nanmean(seg)) if np.isfinite(seg).any() else np.nan
                    if np.isfinite(fm) and np.isfinite(pm) and fm > 0:
                        c = float(np.clip(pm / fm, lo_c, hi_c))
                scale[j, k] = c
                F[j, k] = F_raw[j, k] * c
            k += 1
        for n in getattr(model, "notes", []) or []:
            if n not in res["notes"]:
                res["notes"].append(n)
    res.update(F=F.astype(np.float32), F_raw=F_raw.astype(np.float32), scale=scale, n_fresh=n_fresh,
               seconds=time.time() - t0, peak_bytes=ts.peak_rss_bytes())
    return res


def node_frame(results, key="F"):
    parts = []
    hz = np.arange(1, HORIZON + 1)
    for r in results:
        if key not in r:
            continue
        nI = len(r["issues"])
        for j, node in enumerate(r["names"]):
            parts.append(pd.DataFrame({"node": node, "origin": np.repeat(pd.DatetimeIndex(r["issues"]).date, HORIZON),
                                       "horizon": np.tile(hz, nI), "forecast": r[key][j].ravel()}))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["node", "origin", "horizon", "forecast"])
    return df[np.isfinite(df["forecast"].astype(float))]


def substitution_test(wcfg, code, spec):
    R, block = wcfg["blocks"][0]
    T_day = pd.Timestamp(block[min(5, len(block) - 1)])
    sel = [o for o in block if T_day - pd.Timedelta(days=1) <= o <= T_day + pd.Timedelta(days=3)]
    c = dict(wcfg, blocks=[(R, pd.DatetimeIndex(sel))])
    a = node_worker(code, dict(c, perturb_after=None))
    b = node_worker(code, dict(c, perturb_after=str(T_day.date())))
    keep = np.array([o <= T_day + pd.Timedelta(days=1) for o in sel])
    if "F" not in a:
        return dict(passed=False, detail=f"district {code}: no forecasts", district=code, T=str(T_day.date()))
    diff = np.abs(a["F"].astype(float) - b["F"].astype(float))
    same = float(np.nanmax(diff[:, keep])) if keep.any() and np.isfinite(diff[:, keep]).any() else 0.0
    later = float(np.nanmax(diff[:, ~keep])) if (~keep).any() and np.isfinite(diff[:, ~keep]).any() else np.nan
    finite = np.isfinite(a["F"][:, keep]).any()
    passed = bool(same == 0.0 and finite)
    detail = (f"district {code}, T = {T_day.date()}: max change of node forecasts issued <= T+1: {same:.3g}; "
              f"issued >= T+2: {later:.3g}")
    return dict(passed=passed, detail=detail, district=code, T=str(T_day.date()))


def run_node(args, cfg, git, model_dir, model_config, tracking):
    spec = ns.load_spec(args.test_set)
    role = "dev" if args.mode == "validate" else "test"
    test_dir = os.path.expanduser(args.test_dir or os.path.join("~/work/test_set", spec["version"]))
    man = ns.read_manifest(test_dir)
    ns.check_same_spec(spec, man)
    run_cfg = cfg["run"]
    out_root = os.path.expanduser(args.out_dir or run_cfg["out_dir"])
    workers = args.workers or int(run_cfg["workers"])
    d = dict(spec["data"])
    nodes = ns.load_nodes(test_dir)
    rn = nodes[nodes["role"] == role]
    codes = sorted(rn["coato"].unique())
    if args.coatos:
        codes = [c for c in codes if c in set(args.coatos.split(","))]
    if not codes:
        raise SystemExit("no districts to run (check --coatos against the node set)")
    if role == "test":
        print("[test] the test substations are scored once, for the variants chosen on the dev substations")
    targets = ns.load_targets(test_dir)
    pcs, nrs = ns.per_customer(targets[targets["node"].isin(set(rn["node"]))], spec)
    blocks = refit_blocks(spec, role)

    # ---- one fit per refit date (shared with the district runs of the same model and commit)
    arts, fit_seconds = {}, 0.0
    import whole_pipeline_v8 as wp
    for R, _ in blocks:
        cutoff, data_end = ts.train_window(spec, R)
        key = f"{args.model}/{git['short']}{'-dirty' if git['dirty'] else ''}/refit-{R.date()}"
        art = os.path.join(out_root, "artifacts", key)
        done = os.path.join(art, ".fit_done")
        if args.refit and os.path.exists(art):
            shutil.rmtree(art)
        if os.path.exists(done):
            print(f"[fit] refit {R.date()}: reusing {art}")
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

    node_lists = {c: [(r.node, r.level, int(r.node_id)) for r in rn[rn["coato"] == c].itertuples()] for c in codes}
    wcfg = dict(spec=spec, data=d, blocks=blocks, labels=cfg["data"]["labels"], model_name=args.model,
                model_dir=model_dir, model_config=model_config, artifacts=arts, seed=wp.SEED,
                history_days=int(cfg["data"]["history_days"]), block=int(run_cfg["block"]),
                scaling=run_cfg.get("scaling") or {"method": "ratio"}, nodes=node_lists, per_customer=pcs, n_rows=nrs)
    thr = max(1, (os.cpu_count() or 4) // workers)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(thr)

    sub = None
    if not args.no_substitution_test:
        print(f"[leakage] substitution test on district {codes[0]} ...", flush=True)
        with cf.ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
            sub = ex.submit(substitution_test, wcfg, codes[0], spec).result()
        print(f"[leakage] {'PASSED' if sub['passed'] else 'FAILED'}: {sub['detail']}", flush=True)
        if not sub["passed"] and not args.allow_leak:
            raise SystemExit("[leakage] the model's forecasts depend on data after the issue date - 1")

    n_issue = sum(len(b) for _, b in blocks)
    print(f"[run] {args.model} | {spec['version']} {role} | {len(blocks)} refits, {n_issue} issue dates | "
          f"{len(codes)} districts, {int((rn['level'] == 'substation').sum())} substations, "
          f"{int((rn['level'] == 'feeder').sum())} feeders | {workers} workers | commit {git['short']}"
          f"{' (dirty)' if git['dirty'] else ''}", flush=True)
    results, notes, t1 = [], [], time.time()
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        pending = {ex.submit(node_worker, c, wcfg): c for c in codes}
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
                notes += [n for n in r.get("notes", []) if n not in notes]
                k += 1
                print(f"[districts] {k}/{len(codes)} {code}: {r['seconds']:.0f} s, {r['windows']:,} household "
                      f"forecasts | {ts.memory_line(r.get('peak_bytes'))}", flush=True)
    predict_seconds = time.time() - t1

    # ---- write, score
    out = os.path.join(out_root, "runs", f"{time.strftime('%Y%m%d-%H%M%S')}_{args.model}_nodes-{role}")
    fdir = os.path.join(out, "forecasts", args.model)
    udir = os.path.join(out, "forecasts", f"{args.model}_unscaled")
    for p_ in (fdir, udir):
        os.makedirs(p_, exist_ok=True)
    fm, fu_ = node_frame(results, "F"), node_frame(results, "F_raw")
    fm.to_csv(os.path.join(fdir, "node.csv.gz"), index=False, float_format="%.6f")
    fu_.to_csv(os.path.join(udir, "node.csv.gz"), index=False, float_format="%.6f")
    allc = np.concatenate([r["scale"].ravel() for r in results if "scale" in r]) if results else np.array([])
    allc = allc[np.isfinite(allc)]
    if len(allc):
        print(f"[scaling] node factor c(o): median {np.median(allc):.3f} (5-95 %: {np.percentile(allc, 5):.3f} .. "
              f"{np.percentile(allc, 95):.3f})")
    b0 = ns.b0_forecasts(targets, spec, nodes=set(rn["node"]), issue=ns.issue_dates(spec, role))
    meta = dict(model=args.model, test_set_version=spec["version"], test_set_sha256=spec["_sha256"], target="T2_node",
                role=role, git_commit=git["commit"], dirty=git["dirty"], districts=codes, substitution_test=sub,
                notes=notes, household_forecasts=sum(r["windows"] for r in results),
                gaps_filled_with_persistence=sum(r["gaps"] for r in results), test_dir=test_dir,
                scaling=wcfg["scaling"])
    json.dump(meta, open(os.path.join(fdir, "forecast_meta.json"), "w"), indent=2, default=str)
    fm["origin"], fu_["origin"] = pd.to_datetime(fm["origin"]), pd.to_datetime(fu_["origin"])
    models = {args.model: fm, f"{args.model}_unscaled": fu_, ns.REFERENCE: b0[["node", "origin", "horizon", "forecast"]]}
    res = ns.score(models, targets, spec, role, nodes)
    text = ns.report(res, list(models))
    print("\n" + text)
    ns.save(res, os.path.join(out, "scores"), text)
    head = res["headline"]
    row = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), model=args.model, commit=git["commit"], mode=args.mode,
               test_set=spec["version"], role=role, status="done",
               substation=head.get(f"{args.model}/substation"), feeder=head.get(f"{args.model}/feeder"),
               b0_substation=head.get("B0/substation"), b0_feeder=head.get("B0/feeder"),
               substitution_test="skipped" if sub is None else ("passed" if sub["passed"] else "FAILED"),
               fit_seconds=round(fit_seconds), predict_seconds=round(predict_seconds), run_dir=out)
    tracking.append_csv(os.path.join(out_root, "node_leaderboard.csv"), row)
    _wandb(cfg, args, spec, role, git, model_config, res, row)
    fmt = lambda v: "n/a" if v is None or not np.isfinite(v) else f"{v:.2%}"      # noqa: E731
    print(f"\n[done] {out} | {role} median node sysMAPE (mean of h {res['ctrl']}): substation "
          f"{fmt(row['substation'])} (B0 {fmt(row['b0_substation'])}), feeder {fmt(row['feeder'])} "
          f"(B0 {fmt(row['b0_feeder'])})")
    return out


def _wandb(cfg, args, spec, role, git, model_config, res, row):
    """A small W&B run: per-horizon median / pooled node sysMAPE per level (node aggregates only)."""
    from harness import wandb_log
    wcfg = dict(cfg.get("wandb") or {})
    if getattr(args, "no_wandb", False):
        wcfg["enabled"] = False
    run = wandb_log.start(wcfg, name=f"{args.model} | nodes-{role} | {git['short']}", group=f"{spec['version']} {role}",
                          job_type=f"nodes-{role}", config=dict(model=args.model, model_config=model_config,
                                                                 test_set=spec["version"], commit=git["commit"]),
                          tags=[args.model, f"nodes-{role}", model_config.get("family", "")])
    if run is None:
        return
    try:
        s = res["summary"]
        for h in sorted(s["horizon"].unique()):
            q = s[s["horizon"] == h]
            log = {"horizon": int(h)}
            for r in q.itertuples():
                log[f"nodes/{r.level}/{r.model}/median"] = r.median
                log[f"nodes/{r.level}/{r.model}/pooled"] = r.pooled
                log[f"nodes/{r.level}/{r.model}/bias"] = r.bias
            run.log(log)
        run.summary.update({k: v for k, v in row.items() if isinstance(v, (int, float, str))})
        run.finish()
    except Exception as e:                                             # noqa: BLE001
        print(f"[wandb] WARNING: {type(e).__name__}: {e}")
