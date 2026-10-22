"""Test model: checks what the harness hands to predict, then forecasts persistence."""
import os

import numpy as np

from harness.contract import HORIZON, BaseModel


class Model(BaseModel):
    def predict(self, panel):
        import pandas as pd
        assert panel.dates[-1] == panel.origin - pd.Timedelta(days=1), "history must end on origin - 1"
        clim = panel.pipeline.get("clim")
        if clim is not None:                         # observed weather in the climate object: only before the issue date
            seen = np.flatnonzero(np.isfinite(clim.W[0]))
            end = pd.Timestamp(clim.start) + pd.Timedelta(days=int(seen[-1]))
            assert end < panel.origin, f"climate object has weather of {end.date()}, issue date {panel.origin.date()}"
        assert panel.kwh.shape == (panel.n, len(panel.dates))
        assert not [k for k in os.environ if k.endswith(("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"))], "S3 keys visible"
        g = panel.pipeline["fc_grid"]
        issued = np.flatnonzero(np.isfinite(g).any(axis=(0, 2)))
        assert len(issued) <= 1, f"forecasts from {len(issued)} issue days visible"
        lv = np.nanmean(panel.kwh[:, -30:], axis=1)
        return np.repeat(lv[:, None], HORIZON, axis=1)
