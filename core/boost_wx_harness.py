"""Baseline + CatBoost / XGBoost (whole_pipeline_v8) trained and served with ACTUAL weather forecasts.

Same model as core/boost_harness.py (v8 features, thermal baseline, residual boosters, one shared fit per refit for
catboost_wx / xgboost_wx), with the simulated forecast replaced by the AIFS / FCN3 reforecasts (core/reforecast.py):

  - forecast days: h1 .. h14 from the forecast issued on o (wx.lead_days); after day 14 the climate normal plus the
    last days' anomaly fading out (v8's blend_forecast). In v8's window convention (origin = o - 1) that is
    FORECAST_DAYS = 15 with lead 1 (= day o, not a target) empty, at training and at predict time alike;
  - training windows (wx.origins):
      reforecast          only windows whose origin is an issue date - 1 (every one has the real forecast)
      reforecast+stride   those plus v8's usual staggered windows, which have no forecast (normals only), so 2022-2023
                          seasons stay in training; the trees see whether a forecast is present (fc_* features NaN)
    The reforecasts start in 2024, so the standard split's first refits (2024-01, 2024-04) have no or few forecast
    windows: use configs/test_set_2025.yaml, or reforecast+stride;
  - the forecast a training window gets is the one issued on its issue date (known then), never the observed weather;
  - evaluation: evaluate.py --weather aifs | fcn3 | ens (matching wx.source), on the issue dates that have a forecast.

Saved per refit as boost_harness does (catboost_model.cbm, xgboost_model.json, boost.json), plus `wx` in boost.json.
"""
import hashlib
import json
import os
import shutil
import time

import numpy as np
import pandas as pd

import boost_harness as bh
import reforecast as rf

SHARED_KEYS = bh.SHARED_KEYS + ("wx", "gpu")


def config_hash(cfg):
    c = {k: cfg.get(k) for k in SHARED_KEYS if k != "workers"}
    c["wx"] = {k: v for k, v in rf.wx_config(cfg).items() if k not in ("cache_dir", "min_windows")}
    return hashlib.sha1(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:10]


def shared_dir(artifacts_dir, cfg):
    """<out_root>/artifacts/_boost_wx/<commit>/<config hash>/refit-<R>."""
    art = os.path.normpath(artifacts_dir)
    refit, commit = os.path.basename(art), os.path.basename(os.path.dirname(art))
    root = os.path.dirname(os.path.dirname(os.path.dirname(art)))
    return os.path.join(root, "_boost_wx", commit, config_hash(cfg), refit)


def v8_forecast_settings(w):
    """FORECAST_DAYS / FC_BLOCKS for lead_days forecast days in v8's window convention (lead 1 = day o: empty)."""
    K = int(w["lead_days"]) + 1
    edges = [b for b in (4, 8) if b < K]
    blocks, a = [], 2
    for b in edges + [K]:
        blocks.append((a, b))
        a = b + 1
    return dict(FORECAST_DAYS=K, FC_BLOCKS=blocks, FC_TAIL_DAYS=min(3, K - 1))


def gpu_settings(cfg, wp_cfg):
    """`gpu: [catboost, xgboost]` in the model config -> v8 parameter overrides that train those boosters on GPU 0.
    Training only: prediction stays on CPU (the scoring workers would otherwise share one GPU)."""
    g = [str(x).lower() for x in (cfg.get("gpu") or [])]
    bad = set(g) - {"catboost", "xgboost"}
    if bad:
        raise SystemExit(f"gpu: only catboost and/or xgboost, not {sorted(bad)}")
    out = {}
    if "catboost" in g:
        out["CATBOOST_PARAMS"] = {**(wp_cfg.get("CATBOOST_PARAMS") or {}), "task_type": "GPU", "devices": "0"}
    if "xgboost" in g:
        out["XGB_PARAMS"] = {**(wp_cfg.get("XGB_PARAMS") or {}), "device": "cuda"}
    return out


def train(spec, artifacts_dir, cfg):
    """One v8 fit with real forecasts for refit spec (shared by catboost_wx / xgboost_wx)."""
    sdir = shared_dir(artifacts_dir, cfg)
    if all(os.path.exists(os.path.join(sdir, f)) for f in bh.FILES):
        print(f"[boost-wx] reusing the shared fit {sdir}")
    else:
        w = rf.wx_config(cfg)
        wp = bh._wp()
        fs_set = v8_forecast_settings(w)
        wp_cfg = cfg.get("wp") or {}
        c = dict(cfg, forecast="archive", wp={**wp_cfg, **fs_set, **gpu_settings(cfg, wp_cfg)})
        if cfg.get("gpu"):
            print(f"[boost-wx] training on GPU: {cfg.get('gpu')} (non-deterministic; prediction on CPU)")
        s = bh._settings(wp, spec, c)
        t0 = time.time()
        arch = rf.load_archive(w)                                         # harness convention: issue o, lead k = o + k
        cutoff = pd.Timestamp(spec.cutoff)
        inits = sorted(pd.to_datetime(arch["issue_date"].unique()))
        origins = [d - pd.Timedelta(days=1) for d in inits if d - pd.Timedelta(days=1) <= cutoff]
        stride = w["origins"] == "reforecast+stride"
        if w["origins"] not in ("reforecast", "reforecast+stride"):
            raise SystemExit(f"wx.origins must be reforecast or reforecast+stride, not {w['origins']!r}")
        who = "all customers" if s["ENRICHED_BUCKETS"] is None else f"customer buckets {s['ENRICHED_BUCKETS']}"
        print(f"[boost-wx] v8 fit: cutoff {spec.cutoff}, data to {spec.data_end}, validation ({spec.val_start}, "
              f"{spec.val_end}] | {who} | train sample {s['ENRICHED_TRAIN_SAMPLE']} | forecast {w['source']} "
              f"({len(w['sets'])} sets), h1..h{w['lead_days']} | {len(origins)} forecast issue dates <= cutoff"
              f"{' + the staggered windows (no forecast)' if stride else ''}")
        if not origins and not stride:
            raise SystemExit(f"[boost-wx] no reforecast issue date before the cutoff {spec.cutoff}: the reforecasts "
                             f"start {inits[0].date() if inits else '-'}; use the 2025 split or wx.origins: reforecast+stride")
        import wx_windows
        with wx_windows.fixed_origins(wp, origins, stride=stride):
            pdf, kin, ktg, daily_t = wp.extract_windows_enriched()
        clim = wp.build_climate(daily_t)
        share, missing = rf.coverage(arch, clim.coatos)
        print(f"[boost-wx] archive covers {share:.0%} of the {len(clim.coatos)} districts read"
              + (f"; missing e.g. {missing}" if missing else ""))
        if share < 0.5:
            raise SystemExit("[boost-wx] the forecast points do not match the district codes: set wx.point_map "
                             "(csv: point, coato_code); see python core/reforecast.py --inspect")
        a8 = rf.to_window_convention(arch, fs_set["FORECAST_DAYS"])
        key = pd.Index(a8["coato_id"].astype(str) + "|" + a8["issue_date"].dt.strftime("%Y-%m-%d")).unique()
        has_fc = (pdf["coato_code"].astype(str) + "|" + pdf["origin_date"].dt.strftime("%Y-%m-%d")).isin(key)
        n_fc = int(has_fc.sum())
        print(f"[boost-wx] {len(pdf):,} training windows, {n_fc:,} ({n_fc / max(len(pdf), 1):.0%}) with the real "
              f"forecast issued on their issue date")
        if n_fc < int(w["min_windows"]):
            raise SystemExit(f"[boost-wx] only {n_fc:,} training windows have a forecast (< wx.min_windows = "
                             f"{w['min_windows']}): too few for this refit")
        fc_grid = wp.build_forecast_grid(a8, clim)
        fc_grid_mm = dict(min=wp.build_forecast_grid(a8, clim, col="t_fc_min"),
                          max=wp.build_forecast_grid(a8, clim, col="t_fc_max"))
        del arch
        fold = wp.fit_and_evaluate(pdf, kin, ktg, ktg.copy(), clim, fc_grid, fc_grid_mm, spec.cutoff, spec.val_start,
                                   label=f"refit {os.path.basename(sdir)} (forecast {w['source']})", val_end=spec.val_end,
                                   train_only=True, fallback_iters=bh._earlier_iters(sdir))
        del pdf, kin, ktg, fc_grid, fc_grid_mm
        os.makedirs(sdir, exist_ok=True)
        fold.cb_model.save_model(os.path.join(sdir, "catboost_model.cbm"))
        fold.xgb_model.save_model(os.path.join(sdir, "xgboost_model.json"))
        km = fold.meter_segment.attrs.get("kmeans")
        info = dict(
            feature_cols=fold.feature_cols, cat_features=list(wp.CAT_FEATURES), xgb_categories=fold.xgb_categories,
            segment_priors={k: list(v) for k, v in fold.segment_priors.items()}, pooled_prior=list(fold.pooled_prior),
            segment_priors_mm={k: list(v) for k, v in (fold.segment_priors_mm or {}).items()},
            pooled_prior_mm=list(fold.pooled_prior_mm) if fold.pooled_prior_mm[0] is not None else None,
            kmeans=km, segment_min_days=int(wp.SEGMENT_MIN_COVERAGE_DAYS),
            segment_start=str(wp.ENRICHED_START if wp.SEGMENT_USE_OLD else
                              (pd.Timestamp(wp.REGIME_CUTOFF) - pd.Timedelta(days=wp.LOOKBACK - 1)).date()),
            catboost_best_iteration=fold.cb_best_iter, catboost_final_iterations=fold.cb_final_iters,
            xgboost_best_iteration=fold.xgb_best_iter, xgboost_final_iterations=fold.xgb_final_iters,
            curves=bh._curves(fold), counts=fold.counts, forecast="archive",
            wx=dict(source=w["source"], sets=list(w["sets"]), lead_days=int(w["lead_days"]), origins=w["origins"],
                    issue_dates=len(origins), windows_with_forecast=n_fc, root=w["root"]),
            wp_overrides=c["wp"], settings={k: v for k, v in s.items() if k != "ENRICHED_ROOT"},
            cutoff=spec.cutoff, data_end=spec.data_end, val=[spec.val_start, spec.val_end],
            seconds=round(time.time() - t0, 1),
            top_features={"catboost": fold.cb_imp.sort_values("importance", ascending=False).head(25).values.tolist(),
                          "xgboost": fold.xgb_imp.sort_values("importance", ascending=False).head(25).values.tolist()})
        json.dump(info, open(os.path.join(sdir, "boost.json"), "w"), indent=2, default=str)
        print(f"[boost-wx] fitted in {(time.time() - t0) / 60:.1f} min: train {fold.counts['train']:,} windows "
              f"(fit {fold.counts['fit']:,}, validation {fold.counts['val']:,}), {fold.counts['features']} features | "
              f"trees catboost {fold.cb_final_iters}, xgboost {fold.xgb_final_iters} -> {sdir}")
    os.makedirs(artifacts_dir, exist_ok=True)
    for f in bh.FILES:
        shutil.copy(os.path.join(sdir, f), artifacts_dir)


class Predictor(bh.Predictor):
    """boost_harness.Predictor for a model trained on real forecasts: the panel must carry the forecast issued on o
    from the same source (evaluate.py --weather <wx.source>). A model trained on GPU predicts on CPU: the harness
    scores districts in parallel worker processes, which must not compete for one GPU."""

    def __init__(self, artifacts_dir, variant):
        super().__init__(artifacts_dir, variant)
        if variant == "xgboost":
            try:
                self.model.set_params(device="cpu")
            except Exception:                                            # noqa: BLE001  older xgboost: no device param
                pass

    def predict(self, panel, horizon):
        wp, L = self.wp, self.wp.LOOKBACK
        n = panel.n
        pp = panel.pipeline
        src = self.info["wx"]["source"]
        if pp.get("weather_mode") != src:
            raise SystemExit(f"this model was trained with the {src} forecast: run it with --weather {src}")
        last = pd.Timestamp(panel.dates[-1])
        gap = int((pd.Timestamp(panel.origin) - last).days)
        kin = panel.kwh[:, -L:].astype("float64")
        keys = panel.keys.astype(str)
        seg = self._segments(panel)
        ht = pd.Series(panel.housing).astype("Int64").astype(str).replace("<NA>", np.nan).to_numpy()
        pdf = pd.DataFrame({"meter_id": keys, "consumer_id": keys, "origin_date": last, "coato_code": str(panel.coato),
                            "housing_type_id": ht})
        bh_, bc_ = self._priors(seg.values, "")
        bhm, bcm = self._priors(seg.values, "_mm")
        from harness.weather import grid_at_last_day
        g, gm = grid_at_last_day(panel)
        kw = dict(beta_h_prior=bh_, beta_c_prior=bc_, meter_segment=seg[~seg.index.duplicated()],
                  beta_h_prior_mm=bhm, beta_c_prior_mm=bcm, forecast_mode="archive", fc_grid=g, fc_grid_mm=gm)
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            meta, feats, base, _ = wp.build_features(pdf, kin, np.full((n, wp.HORIZON), np.nan), pp["clim"], **kw)
        cols = self.info["feature_cols"]
        X = pd.concat([feats, meta[self.info["cat_features"]]], axis=1)
        missing = [c for c in cols if c not in X]
        if missing:
            raise RuntimeError(f"features missing at predict: {missing[:5]}")
        X = X[cols]
        if self.variant == "xgboost":
            X = wp.to_xgb_frame(X, self.info["xgb_categories"])
        P = np.maximum(base + self.model.predict(X), 0.0)
        if gap > 0:                                                      # o .. o+59 -> o+1 .. o+60
            P = np.concatenate([P[:, gap:], np.repeat(P[:, -1:], gap, axis=1)], axis=1)
        return P[:, :horizon]
