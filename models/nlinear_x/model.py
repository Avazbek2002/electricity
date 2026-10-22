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
        prev = self._earlier_fit(artifacts_dir)
        if prev:                                     # used only if this refit has no validation windows (2024 label hole)
            argv += ["--fallback-penalty", prev]
        hn.main(argv)
        shutil.rmtree(os.path.join(artifacts_dir, "cache"), ignore_errors=True)   # holds customer keys: not kept

    @staticmethod
    def _earlier_fit(artifacts_dir):
        """nlinear_info.json of the latest earlier refit of the same run (sibling folder refit-<date>), or None."""
        here = os.path.basename(os.path.normpath(artifacts_dir))
        root = os.path.dirname(os.path.normpath(artifacts_dir))
        if not here.startswith("refit-") or not os.path.isdir(root):
            return None
        older = sorted(d for d in os.listdir(root) if d.startswith("refit-") and d < here
                       and os.path.exists(os.path.join(root, d, "nlinear_info.json")))
        return os.path.join(root, older[-1], "nlinear_info.json") if older else None

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
        last = pd.Timestamp(panel.dates[-1])                    # history end = origin - 1 (the harness contract)
        gap = int((pd.Timestamp(panel.origin) - last).days)
        pdf = pd.DataFrame({"meter_id": panel.keys.astype(str), "consumer_id": np.nan, "origin_date": last,
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
        P = np.where(np.isfinite(P), P, fallback)
        if gap > 0:                       # step k of the model = day last + k; the harness wants origin + h = last + gap + h
            self.notes = [f"nlinear_x: trained with history ending on its origin; forecasts shifted by the {gap}-day "
                          f"history gap (h1..h{HORIZON - gap} exact, the last {gap} horizon(s) repeat step {HORIZON})"]
            P = np.concatenate([P[:, gap:], np.repeat(P[:, -1:], gap, axis=1)], axis=1)
        return P
