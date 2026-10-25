"""TiDE household model trained on daily and reading-interval (aggregate) labels: core/tide.py, core/tide_data.py."""
from harness.contract import BaseModel


class Model(BaseModel):
    name = "tide_agg"

    def fit(self, spec, artifacts_dir):
        import tide
        tide.fit(spec, artifacts_dir, self.config)

    def load(self, artifacts_dir):
        import tide
        self._p = tide.Predictor(artifacts_dir, self.config)

    def predict(self, panel):
        return self._p.predict(panel)
