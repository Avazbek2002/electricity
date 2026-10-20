"""nlinear_x (or another NLinear variant) behind the harness interface."""
import json
import os
import shutil

import numpy as np
import pandas as pd

from harness.contract import HORIZON, BaseModel


def _same(a, b):
    return str(pd.Timestamp(a).date()) == str(pd.Timestamp(b).date())


class Model(BaseModel):
    name = "nlinear_x"

    @property
    def variant(self):
        return self.config.get("variant", "nlinear_x")

    def fit(self, spec, artifacts_dir):
        for d in self.config.get("pretrained") or []:
            d = os.path.expanduser(d)
            info_p, w_p = os.path.join(d, "nlinear_info.json"), os.path.join(d, f"weights_{self.variant}.npz")
            if os.path.exists(info_p) and os.path.exists(w_p):
                c = json.load(open(info_p))["config"]
                if _same(c["cutoff"], spec.cutoff) and _same(c["val_start"], spec.val_start) and _same(c["val_end"], spec.val_end):
                    shutil.copy(info_p, artifacts_dir)
                    shutil.copy(w_p, artifacts_dir)
                    print(f"[nlinear_x] reusing the pretrained run {d} (same cutoff and validation block)")
                    return
        import household_nlinear as hn
        if len(spec.roots) != 1 or spec.roots[0].get("env_prefix"):
            raise SystemExit("nlinear_x trains on the developer zone only (one root, normal key)")
        argv = ["--out-dir", artifacts_dir, "--root", spec.roots[0]["uri"], "--cutoff", spec.cutoff,
                "--val-start", spec.val_start, "--val-end", spec.val_end, "--end", spec.data_end,
                "--variants", self.variant, "--workers", str(spec.workers), "--test-sample", "0",
                "--cache-dir", os.path.join(artifacts_dir, "cache")] + [str(x) for x in self.config.get("train_args") or []]
        hn.main(argv)
        shutil.rmtree(os.path.join(artifacts_dir, "cache"), ignore_errors=True)   # holds customer keys: not kept

    def load(self, artifacts_dir):
        import whole_pipeline_v8 as wp
        info = json.load(open(os.path.join(artifacts_dir, "nlinear_info.json")))
        self.cfg = info["config"]
        for k, v in self.cfg.get("wp", {}).items():                 # the training run's pipeline settings
            setattr(wp, k, v)
        self.wp = wp
        self.names = info["features"]["exogenous"]
        n_hist = int(info["features"]["history"])
        p_all = n_hist + len(self.names) + 1
        self.cols = list(range(p_all)) if self.variant.endswith("_x") else list(range(n_hist)) + [p_all - 1]
        self.W = np.load(os.path.join(artifacts_dir, f"weights_{self.variant}.npz"))["W"].astype("float64")

    def predict(self, panel):
        import household_nlinear as hn
        wp, L = self.wp, self.wp.LOOKBACK
        n = panel.n
        kin = panel.kwh[:, -L:].astype("float64")
        ht = pd.Series(panel.housing).astype("Int64").astype(str).replace("<NA>", np.nan).to_numpy()
        pdf = pd.DataFrame({"meter_id": panel.keys.astype(str), "consumer_id": np.nan, "origin_date": panel.origin,
                            "coato_code": panel.coato, "housing_type_id": ht})
        pp = panel.pipeline
        with hn._quiet():
            _, feats, base, _ = wp.build_features_chunked(
                pdf, kin, np.full((n, HORIZON), np.nan), pp["clim"], fc_grid=pp["fc_grid"], fc_grid_mm=pp["fc_grid_mm"],
                chunk=self.cfg.get("chunk", 50000), forecast_mode="archive", normal_mode="loyo",
                beta_h_prior=self.cfg["prior_h"], beta_c_prior=self.cfg["prior_c"],
                beta_h_prior_mm=self.cfg["prior_h_mm"], beta_c_prior_mm=self.cfg["prior_c_mm"], meter_segment=None)
        o = np.full(n, panel.kwh.shape[1] - 1)
        ly = hn.year_ago(panel.kwh, np.arange(n), o, wp.BLOCKS)
        lev = hn.level_of(kin, self.cfg["norm_days"])
        ok = np.isfinite(lev) & (lev > self.cfg["min_level"])
        X, names = hn.design(wp, feats, ly, kin, np.where(ok, lev, 1.0), L)
        if names != self.names:
            raise RuntimeError("the inputs differ from the trained model's")
        base = np.asarray(base, dtype="float64")
        P = hn.to_kwh(self.variant, X[:, self.cols] @ self.W, np.where(ok, lev, 0.0), base)
        fallback = np.nan_to_num(lev)[:, None]
        P = np.where(ok[:, None], P, base if self.variant.startswith("thermal") else fallback)
        return np.where(np.isfinite(P), P, fallback)
