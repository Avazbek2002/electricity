"""30-day persistence: every one of the next 60 days = the mean of the last 30 observed days (baseline)."""
import warnings

import numpy as np

from harness.contract import HORIZON, BaseModel


class Model(BaseModel):
    name = "persistence30"

    def predict(self, panel):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            level = np.nanmean(panel.kwh[:, -30:], axis=1)
        return np.repeat(level[:, None], HORIZON, axis=1)          # NaN where no observed day: the harness fills it
