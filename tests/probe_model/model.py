"""Test model: checks what the harness hands to predict, then forecasts persistence."""
import os

import numpy as np

from harness.contract import HORIZON, BaseModel


class Model(BaseModel):
    def predict(self, panel):
        assert panel.dates[-1] == panel.origin, "history must end on the origin"
        assert panel.kwh.shape == (panel.n, len(panel.dates))
        assert not [k for k in os.environ if k.endswith(("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"))], "S3 keys visible"
        g = panel.pipeline["fc_grid"]
        issued = np.flatnonzero(np.isfinite(g).any(axis=(0, 2)))
        assert len(issued) <= 1, f"forecasts from {len(issued)} issue days visible"
        lv = np.nanmean(panel.kwh[:, -30:], axis=1)
        return np.repeat(lv[:, None], HORIZON, axis=1)
