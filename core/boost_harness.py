"""Baseline + CatBoost and Baseline + XGBoost (whole_pipeline_v8) behind the evaluation harness.

Training is v8's own (fit_and_evaluate, train_only=True): customers' windows from the enriched dataset (history ends on
the window's origin, targets the next 60 days), behaviour segments and segment beta priors from the training period,
v8's features and thermal baseline, residual target = actual - baseline, early stopping on the inner validation block,
final refit on all training windows. Both boosters come from ONE fit per refit date, stored once and shared by the
model folders catboost / xgboost (same configuration apart from `variant`).

Forecast for issue date o (harness contract: history up to o - 1):
  - the window's origin is o - 1, so v8's 60 days are o .. o + 59; the harness needs o + 1 .. o + 60: the output is
    shifted by one day (exact for h1 .. h59, h60 repeats day 60) -- the same as nlinear_x;
  - a household's segment is assigned from its own history up to o - 1 with the fitted clustering (no customer list
    is stored with the model);
  - forecast: "none" (the standard run: normals only) or "simulated" (the harness's simulated forecast issued on o;
    shifted to the window's leads, so lead 1 = day o has no forecast).

Saved per refit (no customer keys): catboost_model.cbm, xgboost_model.json, boost.json (features, priors, clustering,
categories, tree counts, the pipeline settings used).
"""
import hashlib
import json
import os
import shutil
import time

import numpy as np
import pandas as pd

FILES = ("catboost_model.cbm", "xgboost_model.json", "boost.json")
SHARED_KEYS = ("forecast", "buckets", "train_sample", "max_window_gb", "workers", "wp", "fixed_iters")  # what makes two fits equal


def _wp():
    import whole_pipeline_v8 as wp
    return wp


def config_hash(cfg):
    c = {k: cfg.get(k) for k in SHARED_KEYS if k != "workers"}
    return hashlib.sha1(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:10]


def shared_dir(artifacts_dir, cfg):
    """<out_root>/artifacts/_boost/<commit>/<config hash>/refit-<R> for <out_root>/artifacts/<model>/<commit>/refit-<R>."""
    art = os.path.normpath(artifacts_dir)
    refit, commit = os.path.basename(art), os.path.basename(os.path.dirname(art))
    root = os.path.dirname(os.path.dirname(os.path.dirname(art)))
    return os.path.join(root, "_boost", commit, config_hash(cfg), refit)


def fallback_iters(sdir, cfg):
    """Tree counts used when a refit cannot early-stop (no fit or no validation windows): the latest earlier refit's,
    else the config's `fixed_iters: [catboost, xgboost]` (e.g. INCLUDE_OLD none on a first refit), else None."""
    earlier = _earlier_iters(sdir)
    if earlier:
        return earlier
    f = cfg.get("fixed_iters")
    return (int(f[0]), int(f[1])) if f else None


def _earlier_iters(sdir):
    """(catboost, xgboost) final tree counts of the latest earlier refit in the same shared folder, or None."""
    here, root = os.path.basename(sdir), os.path.dirname(sdir)
    if not os.path.isdir(root):
        return None
    older = sorted(d for d in os.listdir(root) if d.startswith("refit-") and d < here
                   and os.path.exists(os.path.join(root, d, "boost.json")))
    if not older:
        return None
    j = json.load(open(os.path.join(root, older[-1], "boost.json")))
    return j["catboost_final_iterations"], j["xgboost_final_iterations"]


def _settings(wp, spec, cfg):
    """v8 module settings for this fit (the whole_pipeline_v8 globals its functions read)."""
    buckets = cfg.get("buckets", [0, 1, 2, 3, 4])
    s = dict(WINDOW_SOURCE="enriched", ENRICHED_ROOT=spec.roots[0]["uri"], ENRICHED_START=spec.start,
             ENRICHED_END=spec.data_end, ENRICHED_LABELS=list(spec.labels),
             ENRICHED_BUCKETS=None if buckets in (None, "all") else [int(b) for b in buckets],
             ENRICHED_TRAIN_SAMPLE=cfg.get("train_sample"), ENRICHED_TEST_SAMPLE=0.0,   # no test windows are read
             ENRICHED_WORKERS=int(cfg.get("workers") or spec.workers), ENRICHED_CACHE_DIR=None,
             MAX_WINDOW_ARRAY_GB=float(cfg.get("max_window_gb", 40)), TRAIN_CUTOFF=spec.cutoff,
             VAL_START=spec.val_start, VAL_END=spec.val_end, BACKTEST_CUTOFFS=[], GLITCH_REPAIR=False,
             FORECAST_MODE=cfg.get("forecast", "none"), PRECOMPUTED_SEGMENT_FEATURES=None,
             **(cfg.get("wp") or {}))
    for k, v in s.items():
        if not hasattr(wp, k):
            raise SystemExit(f"whole_pipeline_v8 has no setting {k}")
        _set(wp, k, v)
    return s


def _set(wp, k, v):
    """setattr, except a dict setting (CATBOOST_PARAMS, XGB_PARAMS) is updated key by key: the config names only the
    keys it changes. Merged into v8's defaults each time, so a key dropped from the config goes back to the default."""
    if isinstance(v, dict):
        base = _V8_DICT_DEFAULTS.setdefault(k, dict(getattr(wp, k)))
        v = {**base, **v}
    setattr(wp, k, v)


_V8_DICT_DEFAULTS = {}


def _curves(fold):
    """Early-stopping learning curves: {"catboost/learn": [...], "catboost/validation": [...], "xgboost/validation": ...}."""
    out = {}
    for name, ev in (("catboost", fold.cb_curves or {}), ("xgboost", fold.xgb_curves or {})):
        for part, d in ev.items():
            for metric, v in (d or {}).items():
                tag = "validation" if part in ("validation", "validation_0") else part
                out[f"{name}/{tag}_{metric}"] = [float(x) for x in v]
    return out


def train(spec, artifacts_dir, cfg):
    """One v8 fit for refit spec (shared by catboost / xgboost); copies the files into artifacts_dir."""
    sdir = shared_dir(artifacts_dir, cfg)
    if all(os.path.exists(os.path.join(sdir, f)) for f in FILES):
        print(f"[boost] reusing the shared fit {sdir}")
    else:
        wp = _wp()
        s = _settings(wp, spec, cfg)
        t0 = time.time()
        who = "all customers" if s["ENRICHED_BUCKETS"] is None else f"customer buckets {s['ENRICHED_BUCKETS']}"
        print(f"[boost] v8 fit: cutoff {spec.cutoff}, data to {spec.data_end}, validation ({spec.val_start}, "
              f"{spec.val_end}] | {who} | train sample {s['ENRICHED_TRAIN_SAMPLE']} | forecast {s['FORECAST_MODE']}")
        pdf, kin, ktg, daily_t = wp.extract_windows_enriched()
        clim = wp.build_climate(daily_t)
        fold = wp.fit_and_evaluate(pdf, kin, ktg, ktg.copy(), clim, None, None, spec.cutoff, spec.val_start,
                                   label=f"refit {os.path.basename(sdir)}", val_end=spec.val_end, train_only=True,
                                   fallback_iters=fallback_iters(sdir, cfg))
        del pdf, kin, ktg
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
            curves=_curves(fold),
            counts=fold.counts, forecast=s["FORECAST_MODE"], wp_overrides=cfg.get("wp") or {},
            settings={k: v for k, v in s.items() if k != "ENRICHED_ROOT"},
            cutoff=spec.cutoff, data_end=spec.data_end, val=[spec.val_start, spec.val_end],
            seconds=round(time.time() - t0, 1),
            top_features={"catboost": fold.cb_imp.sort_values("importance", ascending=False).head(25).values.tolist(),
                          "xgboost": fold.xgb_imp.sort_values("importance", ascending=False).head(25).values.tolist()})
        json.dump(info, open(os.path.join(sdir, "boost.json"), "w"), indent=2, default=str)
        print(f"[boost] fitted in {(time.time() - t0) / 60:.1f} min: train {fold.counts['train']:,} windows "
              f"(fit {fold.counts['fit']:,}, validation {fold.counts['val']:,}), {fold.counts['features']} features | "
              f"trees catboost {fold.cb_final_iters}, xgboost {fold.xgb_final_iters} -> {sdir}")
    os.makedirs(artifacts_dir, exist_ok=True)
    for f in FILES:
        shutil.copy(os.path.join(sdir, f), artifacts_dir)


class Predictor:
    """Loads one refit and forecasts harness Panels with the chosen booster."""

    def __init__(self, artifacts_dir, variant):
        wp = _wp()
        self.wp, self.variant = wp, variant
        self.info = json.load(open(os.path.join(artifacts_dir, "boost.json")))
        for k, v in (self.info.get("wp_overrides") or {}).items():      # the fit's pipeline settings (config `wp`)
            _set(wp, k, v)
        if variant == "catboost":
            from catboost import CatBoostRegressor
            self.model = CatBoostRegressor()
            self.model.load_model(os.path.join(artifacts_dir, "catboost_model.cbm"))
        elif variant == "xgboost":
            import xgboost as xgb
            self.model = xgb.XGBRegressor()
            self.model.load_model(os.path.join(artifacts_dir, "xgboost_model.json"))
            self.model.set_params(device="cpu")                          # trained on GPU or not: predict on CPU
                                                                         # (many scoring workers, one GPU)
        else:
            raise ValueError(f"unknown variant {variant!r}")
        self.segment_of = {}                                             # customer key -> segment (this process only)

    # ---- segments from a household's own history (as v8's full-history segment features)
    def _segments(self, panel):
        wp, km = self.wp, self.info.get("kmeans")
        keys = panel.keys.astype(str)
        todo = [i for i, k in enumerate(keys) if k not in self.segment_of]
        if todo:
            if km is None:
                for i in todo:
                    self.segment_of[keys[i]] = "unclassified"
            else:
                dates = pd.DatetimeIndex(panel.dates)
                s0 = int(np.searchsorted(dates.values, np.datetime64(self.info["segment_start"])))
                months = dates.month.values[s0:]
                t = np.asarray(panel.weather[0, s0:], dtype="float64")
                mean, scale, C = np.asarray(km["mean"]), np.asarray(km["scale"]), np.asarray(km["centers"])
                for i in todo:
                    y = np.asarray(panel.kwh[i, s0:], dtype="float64")
                    seg = "unclassified"
                    if np.isfinite(y).sum() >= self.info["segment_min_days"]:
                        yc, _r, masked = wp.clean_daily_series(y)
                        yc = np.where(masked, np.nan, yc)
                        if np.isfinite(yc).sum() >= self.info["segment_min_days"]:
                            ts_, r2 = wp._weekly_thermal_fit(yc, t)
                            f = np.array([np.log1p(max(float(np.nanmedian(yc)), 0.0)), ts_, r2,
                                          wp._seasonality_strength(yc, months)], dtype="float64")
                            if np.isfinite(f).all():
                                seg = km["names"][int(np.argmin((((f - mean) / scale - C) ** 2).sum(1)))]
                    self.segment_of[keys[i]] = seg
        return pd.Series([self.segment_of[k] for k in keys], index=keys, name="segment")

    def _priors(self, seg, which):
        sp = self.info["segment_priors" + which]
        pooled = self.info["pooled_prior" + which]
        if pooled is None:
            return None, None
        ph = np.array([sp.get(s, pooled)[0] for s in seg], dtype=float)
        pc = np.array([sp.get(s, pooled)[1] for s in seg], dtype=float)
        return ph, pc

    def predict(self, panel, horizon):
        wp, L = self.wp, self.wp.LOOKBACK
        n = panel.n
        pp = panel.pipeline
        mode = self.info["forecast"]
        if mode == "simulated" and pp.get("weather_mode") != "simulated":
            raise SystemExit("this model was trained with a simulated weather forecast: run it with --weather simulated")
        last = pd.Timestamp(panel.dates[-1])                             # history end = o - 1 = the window's origin
        gap = int((pd.Timestamp(panel.origin) - last).days)
        kin = panel.kwh[:, -L:].astype("float64")
        keys = panel.keys.astype(str)
        seg = self._segments(panel)
        ht = pd.Series(panel.housing).astype("Int64").astype(str).replace("<NA>", np.nan).to_numpy()
        pdf = pd.DataFrame({"meter_id": keys, "consumer_id": keys, "origin_date": last, "coato_code": str(panel.coato),
                            "housing_type_id": ht})
        bh, bc = self._priors(seg.values, "")
        bhm, bcm = self._priors(seg.values, "_mm")
        kw = dict(beta_h_prior=bh, beta_c_prior=bc, meter_segment=seg[~seg.index.duplicated()],
                  beta_h_prior_mm=bhm, beta_c_prior_mm=bcm)
        if mode == "simulated":
            from harness.weather import grid_at_last_day
            g, gm = grid_at_last_day(panel)
            kw.update(forecast_mode="archive", fc_grid=g, fc_grid_mm=gm)
        else:
            kw.update(forecast_mode="none")
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            meta, feats, base, _ = wp.build_features(pdf, kin, np.full((n, wp.HORIZON), np.nan), pp["clim"], **kw)
        cols = self.info["feature_cols"]
        X = pd.concat([feats, meta[self.info["cat_features"]]], axis=1)
        missing = [c for c in cols if c not in X]
        if missing:
            raise RuntimeError(f"features missing at predict: {missing[:5]} (trained with forecast={mode})")
        X = X[cols]
        if self.variant == "xgboost":
            X = wp.to_xgb_frame(X, self.info["xgb_categories"])
        P = np.maximum(base + self.model.predict(X), 0.0)
        if gap > 0:                                                      # o .. o+59 -> o+1 .. o+60
            P = np.concatenate([P[:, gap:], np.repeat(P[:, -1:], gap, axis=1)], axis=1)
        return P[:, :horizon]
