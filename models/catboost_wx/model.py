"""Baseline + Catboost (whole_pipeline_v8) trained and served with the real AIFS / FCN3 forecasts; see core/boost_wx_harness.py."""
from harness.contract import HORIZON, BaseModel


class Model(BaseModel):
    name = "catboost_wx"

    @property
    def variant(self):
        return self.config.get("variant", "catboost")

    def fit(self, spec, artifacts_dir):
        import boost_wx_harness as bw
        if len(spec.roots) != 1 or spec.roots[0].get("env_prefix"):
            raise SystemExit("the v8 boosters train on the developer zone only (one root, normal key)")
        bw.train(spec, artifacts_dir, self.config)

    def load(self, artifacts_dir):
        import boost_wx_harness as bw
        self.p = bw.Predictor(artifacts_dir, self.variant)
        w = self.p.info["wx"]
        self.notes = [f"v8 {self.variant} + real forecast {w['source']} h1..h{w['lead_days']} (issue o, 00 UTC; then the "
                      f"normal with a fading anomaly); training windows: {w['origins']}, {w['windows_with_forecast']:,} "
                      f"with a forecast; output shifted one day (h60 repeats day 60)"]

    def predict(self, panel):
        return self.p.predict(panel, HORIZON)
