"""TiDE household model trained on daily labels only (the ablation of tide_agg): core/tide.py, core/tide_data.py."""
from harness.contract import BaseModel


class Model(BaseModel):
    name = "tide_daily"

    def fit(self, spec, artifacts_dir):
        import tide
        tide.fit(spec, artifacts_dir, self.config)

    def load(self, artifacts_dir):
        import tide
        self._p = tide.Predictor(artifacts_dir, self.config)

    def predict(self, panel):
        return self._p.predict(panel)
