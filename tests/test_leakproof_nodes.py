"""Leak-proof node pieces: B0's level window skips the estimated tail (o - 2, o - 1); the expanding blend weights of a
month only see target days before that month."""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from harness import nodeset as ns  # noqa: E402
import node_blend as nb  # noqa: E402

SPEC = dict(horizons=60, history_gap_days=1, data=dict(start="2023-01-01"), target=dict(min_coverage=0.7))


def _targets(seed=0):
    rng = np.random.default_rng(seed)
    d = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    y = 1000 + 300 * np.cos(2 * np.pi * d.dayofyear / 365.25) + rng.normal(0, 30, len(d))
    return pd.DataFrame({"date": d, "node": "S:1", "T2": y, "T3": y, "artefact": False, "coverage3": 0.9})


def test_b0_ignores_the_estimated_tail():
    o = pd.Timestamp("2025-06-10")
    t = _targets()
    a = ns.b0_forecasts(t, SPEC, issue=pd.DatetimeIndex([o]))
    t2 = t.copy()
    tail = t2["date"].isin([o - pd.Timedelta(days=1), o - pd.Timedelta(days=2)])
    t2.loc[tail, ["T2", "T3"]] *= 5                              # whatever the late estimated readings say
    b = ns.b0_forecasts(t2, SPEC, issue=pd.DatetimeIndex([o]))
    assert np.allclose(a["forecast"], b["forecast"])
    t3 = t.copy()
    t3.loc[t3["date"] == o - pd.Timedelta(days=3), ["T2", "T3"]] *= 5   # o - 3 is complete on o - 1: it is used
    c = ns.b0_forecasts(t3, SPEC, issue=pd.DatetimeIndex([o]))
    assert not np.allclose(a["forecast"], c["forecast"])
    d = ns.b0_forecasts(t2, SPEC, issue=pd.DatetimeIndex([o]), safe_lag=0)   # the old window does see them
    assert not np.allclose(a["forecast"], d["forecast"])


def _rows(seed=1):
    rng = np.random.default_rng(seed)
    o = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    r = pd.DataFrame([(n, oo, h) for n in ("a", "b") for oo in o for h in (1, 7, 30, 60)], columns=["node", "origin", "horizon"])
    r["date"] = r["origin"] + pd.to_timedelta(r["horizon"], unit="D")
    r["y"] = 100 + rng.normal(0, 5, len(r))
    r["m1"] = r["y"] + rng.normal(0, 3, len(r))
    r["m2"] = r["y"] + rng.normal(4, 3, len(r))
    return r


def test_expanding_weights_use_only_earlier_targets():
    r = _rows()
    W = nb.simplex(2)
    _, wa = nb.expanding(r, ["m1", "m2"], W, "M")
    r2 = r.copy()
    late = r2["date"] >= "2025-07-01"
    r2.loc[late, "m2"] = r2.loc[late, "y"]                       # m2 perfect from July: invisible before July
    _, wb = nb.expanding(r2, ["m1", "m2"], W, "M")
    for k in ("2025-01", "2025-03", "2025-06", "2025-07"):
        assert wa[k] == wb[k], k
    assert wa["2025-09"] != wb["2025-09"]
    assert wa["2025-01"]["1-7"] == [0.5, 0.5]                    # nothing known yet: equal weights
