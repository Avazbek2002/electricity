"""Baseline + CatBoost (whole_pipeline_v8) behind the harness interface; see core/boost_harness.py."""
from harness.contract import HORIZON, BaseModel


class Model(BaseModel):
    name = "catboost"

    @property
    def variant(self):
        return self.config.get("variant", "catboost")

    def fit(self, spec, artifacts_dir):
        import boost_harness as bh
        if len(spec.roots) != 1 or spec.roots[0].get("env_prefix"):
            raise SystemExit("the v8 boosters train on the developer zone only (one root, normal key)")
        bh.train(spec, artifacts_dir, self.config)

    def load(self, artifacts_dir):
        import boost_harness as bh
        self.p = bh.Predictor(artifacts_dir, self.variant)
        f = self.p.info["forecast"]
        self.notes = [f"v8 {self.variant}: history ends on o - 1 = the window's origin; output shifted one day (h60 "
                      f"repeats day 60); weather forecast: {f}" + ("" if f == "none" else " (lead 1 = day o: none)")]

    def predict(self, panel):
        return self.p.predict(panel, HORIZON)
