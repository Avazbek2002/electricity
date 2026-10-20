"""GAM (Explainable Boosting Machine, interpret) behind the harness interface. Inputs: gam_features.py."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from harness.contract import HORIZON, BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import gam_features as gf                                                     # noqa: E402


class Model(BaseModel):
    name = "gam"

    # ------------------------------------------------------------------ fit
    def fit(self, spec, artifacts_dir):
        import concurrent.futures as cf
        import multiprocessing as mp
        import joblib
        import enriched_io as eio
        import whole_pipeline_v8 as wp
        from interpret.glassbox import ExplainableBoostingRegressor
        from harness import CORE, REPO
        c = self.config
        fs, root = eio.open_root(spec.roots[0]["uri"], spec.roots[0].get("env_prefix", ""))
        codes = eio.finished_districts(fs, root, set(map(str, c["fit_coatos"])) if c.get("fit_coatos") else None)
        cfg = dict(paths=[CORE, REPO, HERE], roots=[dict(r, start=spec.start, end=spec.data_end) for r in spec.roots],
                   start=spec.start, data_end=spec.data_end, labels=list(spec.labels), cutoff=spec.cutoff,
                   household_share=float(c.get("household_share", 0.005)), origin_step=int(c.get("origin_step", 28)),
                   history_days=int(c.get("history_days", 730)), min_level=float(c.get("min_level", 0.2)), seed=wp.SEED)
        print(f"[gam] building training rows: {len(codes)} districts, {cfg['household_share']:.2%} of households, an origin "
              f"every {cfg['origin_step']} days up to {spec.cutoff}, horizons {gf.KNOTS}")
        t0, parts = time.time(), []
        with cf.ProcessPoolExecutor(max_workers=spec.workers, mp_context=mp.get_context("spawn")) as ex:
            pending = {ex.submit(gf.district_rows, code, cfg): code for code in codes}
            k = 0
            while pending:
                done, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
                for f in done:
                    pending.pop(f)
                    res = f.result()
                    if res.get("X") is not None:
                        parts.append(res["X"])
                    k += 1
                    if k % 20 == 0 or k == len(codes):
                        print(f"[gam] rows from {k}/{len(codes)} districts: {sum(len(p) for p in parts):,}; "
                              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        X = pd.concat(parts, ignore_index=True)
        cap = int(c.get("max_rows", 4_000_000))
        if len(X) > cap:
            X = X.sample(cap, random_state=0).reset_index(drop=True)
        y, w = X.pop("y").to_numpy(), X.pop("w").to_numpy()
        X = X[gf.FEATURES]
        inter = [tuple(p) for p in c.get("interactions", [])] or int(c.get("auto_interactions", 10))
        ebm = ExplainableBoostingRegressor(
            feature_names=gf.FEATURES, feature_types=["nominal" if f in gf.CATEGORICAL else "continuous" for f in gf.FEATURES],
            interactions=inter, outer_bags=int(c.get("outer_bags", 4)), max_bins=int(c.get("max_bins", 256)),
            learning_rate=float(c.get("learning_rate", 0.015)), max_rounds=int(c.get("max_rounds", 5000)),
            validation_size=0.15, early_stopping_rounds=100, n_jobs=int(c.get("n_jobs", spec.workers)), random_state=0)
        t1 = time.time()
        print(f"[gam] fitting EBM on {len(X):,} rows, {len(gf.FEATURES)} inputs, interactions {inter}")
        import warnings
        with warnings.catch_warnings():                                  # NaN inputs are handled natively by EBM
            warnings.filterwarnings("ignore", message="Missing values detected")
            ebm.fit(X, y, sample_weight=w)
        joblib.dump(ebm, os.path.join(artifacts_dir, "ebm.joblib"))
        imp = pd.DataFrame({"term": ebm.term_names_, "importance": ebm.term_importances()}).sort_values("importance",
                                                                                                       ascending=False)
        imp.to_csv(os.path.join(artifacts_dir, "term_importance.csv"), index=False)
        json.dump(dict(rows=len(X), fit_minutes=(time.time() - t1) / 60, rows_minutes=(t1 - t0) / 60, config=c,
                       knots=gf.KNOTS, features=gf.FEATURES), open(os.path.join(artifacts_dir, "gam_info.json"), "w"), indent=2)
        print(f"[gam] fitted in {(time.time() - t1) / 60:.1f} min; top terms:\n" + imp.head(12).to_string(index=False))

    # ------------------------------------------------------------------ predict
    def load(self, artifacts_dir):
        import joblib
        import whole_pipeline_v8 as wp
        self.ebm = joblib.load(os.path.join(artifacts_dir, "ebm.joblib"))
        self.wp = wp
        k = np.asarray(gf.KNOTS, dtype=float)
        h = np.arange(1, HORIZON + 1, dtype=float)
        self.ri = np.clip(np.searchsorted(k, h), 1, len(k) - 1)               # right knot of each target day
        self.li = self.ri - 1
        self.wr = (h - k[self.li]) / (k[self.ri] - k[self.li])

    def predict(self, panel):
        wp, n = self.wp, panel.n
        hh = gf.household_block(panel.kwh, wp)
        path, past30 = gf.origin_weather(panel.coato, panel.origin, panel.pipeline["clim"], panel.pipeline["fc_grid"], wp)
        X = gf.rows(hh, path, past30, panel.origin, panel.housing, gf.KNOTS, wp)[gf.FEATURES]
        f = np.clip(self.ebm.predict(X).reshape(n, len(gf.KNOTS)), 0, 5)
        lev = hh["level"]
        ok = np.isfinite(lev) & (lev > float(self.config.get("min_level", 0.2)))
        Pk = np.where(ok, lev, 0.0)[:, None] * f
        P = Pk[:, self.li] * (1 - self.wr) + Pk[:, self.ri] * self.wr        # linear between the knot horizons
        return np.where(ok[:, None], P, np.nan_to_num(lev)[:, None])
