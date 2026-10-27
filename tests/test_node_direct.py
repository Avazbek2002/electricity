"""Direct node model (scripts/node_direct.py): leak-free features (substitution test), the o - 2 / o - 1 tail from the
observed part only, and a forecast close to a clean synthetic series."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("lightgbm")
import node_direct as nd  # noqa: E402

SPEC = dict(data=dict(start="2023-01-01", end="2025-12-31"), refits=["2025-01-01"])


def _data(seed=0):
    rng = np.random.default_rng(seed)
    d = pd.date_range(SPEC["data"]["start"], SPEC["data"]["end"], freq="D")
    rows, obs, wx = [], [], []
    temp = 15 - 14 * np.cos(2 * np.pi * (d.dayofyear - 15) / 365.25) + rng.normal(0, 3, len(d))
    for k, node in enumerate(("S:1:1", "F:1:10")):
        hh = 500 + 100 * k + np.arange(len(d)) * 0.05
        y = hh * (6 + 0.15 * np.maximum(18 - temp, 0) + 0.4 * (d.dayofweek == 6)) * rng.lognormal(0, 0.03, len(d))
        rows.append(pd.DataFrame({"date": d, "node": node, "n_rows": hh, "T2": y, "T3": y, "coverage3": 0.95,
                                  "flag": False, "flag_reason": "", "artefact": False}))
        obs.append(pd.DataFrame({"date": d, "node": node, "T2obs": y * rng.lognormal(0, 0.02, len(d))}))
    wx = pd.DataFrame({"coato": "1", "date": d, "t_mean": temp})
    nodes = pd.DataFrame({"node": ["S:1:1", "F:1:10"], "level": ["substation", "feeder"], "coato": ["1", "1"],
                          "role": ["dev", "dev"]})
    return pd.concat(rows, ignore_index=True), nodes, pd.concat(obs, ignore_index=True), wx


def test_substitution_and_fit():
    t, nodes, obs, wx = _data()
    issue = pd.date_range("2025-01-01", "2025-03-31", freq="D")
    r = nd.substitution_test(t, nodes, obs, wx, SPEC, pd.Timestamp("2025-01-01"), issue, pd.Timestamp("2025-01-10"),
                             n_trees=30)
    assert r["passed"], r["detail"]


def test_tail_uses_observed_part_only():
    t, nodes, obs, wx = _data()
    o = pd.Timestamp("2025-02-10")
    P = nd.build_panel(t, nodes, obs, wx, SPEC, [o])
    t2 = t.copy()
    tail = t2["date"].isin([o - pd.Timedelta(days=1), o - pd.Timedelta(days=2)])
    t2.loc[tail, "T2"] *= 3                                  # the full T2 of o-2, o-1 is not known on o
    P2 = nd.build_panel(t2, nodes, obs, wx, SPEC, [o])
    cols = [c for c in nd.FEATURES if c != "y"] + ["m28", "B1"]
    assert np.allclose(P[cols].to_numpy(float), P2[cols].to_numpy(float), equal_nan=True)
    o2 = obs.copy()
    o2.loc[o2["date"] == o - pd.Timedelta(days=1), "T2obs"] *= 3   # the observed part is
    P3 = nd.build_panel(t, nodes, o2, wx, SPEC, [o])
    assert not np.allclose(P["lastv"], P3["lastv"])


def test_forecast_tracks_a_clean_series():
    t, nodes, obs, wx = _data()
    issue = pd.date_range("2025-01-01", "2025-03-31", freq="D")
    origins = list(pd.date_range("2023-01-01", "2024-12-31", freq="3D")) + list(issue)
    P = nd.build_panel(t, nodes, obs, wx, SPEC, origins)
    F = nd.run_refits(P, ["2025-01-01"], issue, n_trees=100, log=lambda *_: None)
    m = F.merge(P[["node", "origin", "h", "y", "m28", "wd"]].rename(columns={"h": "horizon"}),
                on=["node", "origin", "horizon"])
    m = m[np.isfinite(m["y"]) & (m["horizon"] <= 14)]
    actual = m["y"] * m["m28"]
    assert (m["forecast"] - actual).abs().sum() / actual.sum() < 0.08
    sun = (m["wd"] == 6).to_numpy()                          # the Sunday load the seasonal B1 smooths away

    def sunday_gap(f):                                       # Sunday bias minus the other days' bias
        e = f - actual
        return abs(e[sun].sum() / actual[sun].sum() - e[~sun].sum() / actual[~sun].sum())
    assert sunday_gap(m["forecast"]) < sunday_gap(m["B1"])
