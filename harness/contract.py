"""The interface every model implements (models/<name>/model.py defines `class Model(BaseModel)`).

The harness calls, in this order:
    model = Model(config)                     config = models/<name>/config.yaml as a dict
    model.fit(spec, artifacts_dir)            once per evaluation, in the main process (may run its own workers)
    model.load(artifacts_dir)                 once per worker process
    model.predict(panel) -> (n, 60) kWh       once per district x issue date x block of households, in workers
                                              forecasts for origin + 1 .. origin + 60; history ends at origin - 1

A model sees the data only through `spec` (fit: targets up to data_end = refit date - 1) and `panel` (predict: one issue
date `origin`; data known up to origin - 1, as in the ML guide part 4). Inside predict the S3 keys have been removed
from the environment, so it cannot read anything else.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HORIZON = 60


@dataclass
class TrainSpec:
    """What fit may use: every day from `start` to `cutoff` (origins <= cutoff - 60 days have all their targets)."""
    roots: list            # [{"uri", "env_prefix"}] readable for training (the developer zone only)
    start: str             # first day of data
    cutoff: str            # last origin allowed in training; targets may run to cutoff + 60 days only if <= data_end
    data_end: str          # last day fit may read (cutoff + 60 days, capped by the developer zone)
    val_start: str         # suggested inner validation block (origins), inside the training period
    val_end: str
    workers: int = 4
    labels: tuple = ("complete_observed",)


@dataclass
class Panel:
    """One district, one issue date `origin`, a block of households with fresh history at that date.

    kwh        (n, W) float32, observed daily kWh (NaN = missing); the last column is origin - 1 (history gap 1 day)
    dates      (W,) the calendar days of the columns (dates[-1] == origin - 1 day)
    weather    (3, W) district t_mean, t_min, t_max, observed, up to origin - 1
    housing    (n,) housing_type_id (float, NaN unknown)
    network    (n, 3) transformer_id, feeder_id, substation_id (int64, negative = unknown)
    pipeline   objects for whole_pipeline_v8-based models: clim (built from weather up to origin - 1), fc_grid /
               fc_grid_mm (the forecast issued on origin; all NaN in the standard no-weather run), weather_mode.
    The forecast returned is for origin + 1 .. origin + 60 (column h - 1 = day origin + h).
    """
    coato: str
    origin: pd.Timestamp
    keys: np.ndarray
    kwh: np.ndarray
    dates: pd.DatetimeIndex
    weather: np.ndarray
    housing: np.ndarray
    network: np.ndarray
    pipeline: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.keys)

    @property
    def last_day(self):
        """The last day of history (origin - 1 day)."""
        return self.dates[-1]


class BaseModel:
    name = "base"

    def __init__(self, config: dict):
        self.config = dict(config or {})

    def fit(self, spec: TrainSpec, artifacts_dir: str) -> None:
        """Train on spec's data and save whatever predict needs into artifacts_dir. Default: nothing to train."""

    def load(self, artifacts_dir: str) -> None:
        """Read what fit saved (called in every worker before predict)."""

    def predict(self, panel: Panel) -> np.ndarray:
        """-> (panel.n, 60) kWh for the 60 days after panel.origin. NaN / negative values are replaced by 30-day
        persistence and counted as gaps."""
        raise NotImplementedError
