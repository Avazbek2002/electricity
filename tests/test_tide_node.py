"""TiDE with aggregate labels (core/tide_data.py, core/tide.py) and the substation / feeder test (harness/nodeset.py,
harness/node_mode.py, scripts/test_set/build_node_test_set.py) on synthetic districts in the enriched layout.

Synthetic regimes: before 2024-08-01 meters are read every 2-5 days in whole kWh (4-5 day intervals -> allocated long
gaps, 2-3 day -> estimated runs, a few 1-day readings); from 2024-08-01 mostly daily readings.
"""
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO)
import harness  # noqa: E402,F401
import tide_data as td  # noqa: E402
from harness import nodeset as ns  # noqa: E402

D0, D1 = "2023-06-01", "2026-03-05"
CUT = pd.Timestamp("2024-08-01")
CODES = ("91001", "91002")
try:
    import torch
except ImportError:
    torch = None


def make_district(root, code, seed, n_sub=2, n_feed=2, per_feed=30):
    """-> also returns the true interval sums {(customer, last day): kWh} for checks."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(D0, D1)
    T = len(dates)
    doy = dates.dayofyear.to_numpy()
    tmean = 15 - 13 * np.cos(2 * np.pi * (doy - 15) / 365.25) + rng.normal(0, 2, T)
    season = 1 + 0.5 * np.clip((12 - tmean) / 12, 0, None) + 0.3 * np.clip((tmean - 26) / 6, 0, None)
    dow = np.where(dates.dayofweek >= 5, 1.1, 1.0)
    rows, truth = [], {}
    hh = 0
    base = int(code) * 1000
    for s in range(n_sub):
        sid = base + s
        for f in range(n_feed):
            fid = sid * 10 + f
            for _ in range(per_feed):
                key = f"k{code}_{hh}"
                lvl = rng.lognormal(np.log(5), 0.4)
                old_bias = 1.1                                     # old winters read higher
                y = lvl * season * dow * rng.lognormal(0, 0.15, T)
                y = np.where((dates < CUT) & (dates.month.isin([12, 1, 2])), y * old_bias, y)
                d = 0
                while d < T:
                    old = dates[d] < CUT
                    ln = int(rng.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.1, 0.35, 0.35] if old else
                                        [0.9, 0.03, 0.03, 0.02, 0.02]))
                    ln = min(ln, T - d)
                    E = float(y[d:d + ln].sum())
                    if old:
                        E = float(np.round(E))
                    if rng.random() < 0.02 and not old:            # a missing reading
                        st_ = ["unlabelled"] * ln
                        kw = [None] * ln
                        lg = [None] * ln
                    elif ln == 1:
                        st_, kw, lg = ["complete_observed"], [E], [None]
                    elif ln <= 3:
                        st_, kw, lg = ["complete_with_estimated"] * ln, [E / ln] * ln, [None] * ln
                    else:
                        st_, kw, lg = ["unlabelled"] * ln, [None] * ln, [E / ln] * ln
                    if ln >= 2 and st_[0] != "unlabelled" or lg[0] is not None:
                        truth[(key, d + ln - 1)] = E
                    for j in range(ln):
                        dd = d + j
                        isl = lg[j] is not None
                        rows.append((key, dates[dd].date(), st_[j], kw[j], lg[j], isl,
                                     dates[d].date() if isl else None, (dates[d] + pd.Timedelta(days=ln)).date() if isl else None,
                                     1 if isl else 0, "allocated" if isl else None, 1.0 if hh % 3 else 2.0,
                                     float(tmean[dd]), float(tmean[dd] - 6), float(tmean[dd] + 6),
                                     sid * 100 + f * 10 + hh % 3, fid, sid, False, 0, dd == T - 1, False, False,
                                     "R9", "Region 9", f"District {code}", "house__neither"))
                    d += ln
                hh += 1
    cols = ["customer_key", "date", "label_state", "kwh", "kwh_long_gap_allocated", "is_long_gap_allocated",
            "long_gap_interval_start", "long_gap_interval_end", "long_gap_interval_count", "long_gap_allocation_status",
            "housing_type_id", "t_mean", "t_min", "t_max", "transformer_id", "feeder_id", "substation_id",
            "placeholder_transformer", "customer_bucket", "is_right_edge", "crisis_context", "data_interruption",
            "region_code", "region_name", "admin_location_name", "operational_segment"]
    df = pd.DataFrame(rows, columns=cols)
    df["customer_bucket"] = df["customer_bucket"].astype("int32")
    b = os.path.join(root, "household_daily_enriched", f"coato={code}")
    for per, g in df.groupby(pd.to_datetime(df["date"]).dt.strftime("%Y-%m")):
        o = os.path.join(b, "household_daily_enriched", "data", f"period={per}")
        os.makedirs(o, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(g, preserve_index=False), os.path.join(o, "part-0.parquet"))
    open(os.path.join(b, "_SUCCESS"), "w").close()
    return truth


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("enriched"))
    truth = {c: make_district(root, c, i) for i, c in enumerate(CODES)}
    return root, truth


def _cfg(root, **kw):
    c = dict(uri=root, start=D0, end="2025-12-31", buckets=[0], lookback=90, stride=7, val_stride=7, min_input_days=60,
             min_recent_days=20, min_target_days=20, coarse_inputs="old", interval_labels=True,
             regime_idx=(CUT - pd.Timestamp(D0)).days, val_households=0.2, exclude_ranges=[], seed=1,
             train_hi=(pd.Timestamp("2025-12-31") - pd.Timestamp(D0)).days - 1,
             val_lo=(pd.Timestamp("2025-09-01") - pd.Timestamp(D0)).days, regime_flag=False, min_scale=0.05)
    c.update(kw)
    return c


def test_intervals_are_exact_and_inputs_do_not_cross_the_issue_date(data):
    root, truth = data
    cfg = _cfg(root)
    raw = td.read_raw(CODES[0], cfg)
    st, stats = td.build_intervals(raw)
    assert stats["lg_intervals"] > 1000 and stats["est_runs"] > 300
    keys = raw["keys"]
    T = raw["obs"].shape[1]
    r, c = np.nonzero(st["ilen"])
    for i, j in list(zip(r, c))[:3000]:                   # every reconstructed interval sum is the true reading
        ln = int(st["ilen"][i, j])
        assert st["kind"][i, j - ln + 1:j + 1].tolist() == [td.KIND_INT] * ln
        assert st["coff"][i, j - ln + 1] == ln - 1 and st["coff"][i, j] == 0
        tr = sum(truth[CODES[0]].get((keys[i], d), 0.0) for d in range(j - ln + 1, j + 1))
        assert abs(float(st["isum"][i, j]) - tr) < 1e-3     # (an estimated run may merge adjacent 2-3 day intervals)
        assert abs(np.nansum(st["val"][i, j - ln + 1:j + 1]) - st["isum"][i, j]) < 1e-3
    # no labelled day is in two channels; intervals only on unlabelled / estimated days
    assert not ((st["kind"] == td.KIND_INT) & np.isfinite(raw["obs"])).any()
    # windows: a straddling interval is never an input; target intervals lie wholly inside o + 1 .. o + 60
    S = dict(st, house=np.zeros(len(keys), np.int64), dist=np.ones(len(keys), np.int64))
    S = {k: np.concatenate([v, np.zeros((v.shape[0], 62), v.dtype) if v.dtype != np.float32 else
                            np.full((v.shape[0], 62), np.nan, np.float32)], 1) if v.ndim == 2 else v for k, v in S.items()}
    S["cov"] = np.zeros((2, T + 62, td.N_COV), np.float32)
    rng = np.random.default_rng(0)
    h, o = td.candidate_origins(st, cfg, 0, cfg["train_hi"], rng)
    assert len(h) > 500 and (o < cfg["regime_idx"]).sum() > 100
    b = td.gather(S, h, o, cfg)
    L = 90
    for k in rng.choice(len(h), 300, replace=False):
        i, oo = h[k], o[k]
        for t in range(L):
            d = oo - L + t
            if b["m"][k, t]:
                assert st["kind"][i, d] == td.KIND_OBS or d + st["coff"][i, d] <= oo - 1
        for t in np.flatnonzero(b["im"][k]):
            assert int(b["il"][k, t]) <= t + 1
            assert abs(b["isum"][k, t] * b["s"][k] - st["isum"][i, oo + 1 + t]) < 1e-3
    new = o >= cfg["regime_idx"]
    assert (b["c"][new] == 0).all()                       # no coarse input after the cutoff (as the harness serves)
    assert b["c"][~new].sum() > 0 and b["im"][~new].sum() > b["mt"][~new].sum()   # old windows: mostly interval labels


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_gather_numpy_equals_torch_and_loss_uses_intervals(data):
    import tide
    root, _ = data
    cfg = _cfg(root)
    out = td.district_worker(CODES[1], cfg)
    st = out["store"]
    n, T = st["val"].shape
    S = {k: np.concatenate([v, np.zeros((n, 62), v.dtype) if v.dtype != np.float32 else np.full((n, 62), np.nan,
                                                                                                    np.float32)], 1)
         for k, v in st.items()}
    S.update(house=out["house"], dist=np.ones(n, np.int64), cov=np.random.default_rng(1).normal(
        size=(2, T + 62, td.N_COV)).astype(np.float32))
    h, o = out["train"]
    a = td.gather(S, h[:400], o[:400], cfg)
    St = {k: torch.from_numpy(v) for k, v in S.items()}
    b = td.gather(St, torch.from_numpy(h[:400]), torch.from_numpy(o[:400]), cfg, torch)
    for k in a:
        x, y = np.asarray(a[k], dtype=float), b[k].numpy().astype(float)
        assert np.allclose(x, y, equal_nan=True, atol=1e-6), (k, np.argwhere(~np.isclose(x, y, equal_nan=True, atol=1e-6))[:3], x[~np.isclose(x, y, equal_nan=True, atol=1e-6)][:3], y[~np.isclose(x, y, equal_nan=True, atol=1e-6)][:3])
    mc = tide.merged({})["model"]
    net = tide.TiDE(90, 60, td.N_COV, td.N_STAT, 1, dict(mc, hidden=16, temporal_hidden=8, cov_hidden=8))
    pred = net(b)
    tc = tide.merged({})["train"]
    loss, cnt = tide.window_loss(pred, b, tc, 1.0)
    assert cnt["interval"] > 0 and torch.isfinite(loss)
    # the interval rows move the loss: a prediction exact on every interval sum but off on the days in between
    perfect = b["yt"].clone()
    for i in range(perfect.shape[0]):
        for t in np.flatnonzero(b["im"][i].numpy()):
            ln = int(b["il"][i, t])
            perfect[i, t - ln + 1:t + 1] = b["isum"][i, t] / ln
    l0, _ = tide.window_loss(perfect, b, tc, 1.0)
    shifted = perfect.clone()
    m = b["im"].bool()
    shifted[m] += 1.0
    l1, _ = tide.window_loss(shifted, b, tc, 1.0)
    assert float(l0) < 1e-6 < float(l1)


def _node_spec(path, root):
    def subs(code, role):
        base = int(code) * 1000
        return [dict(coato_code=code, substation_id=base + s, region="R9", district=f"District {code}",
                     stratum="large" if s == 0 else "mid", households=60,
                     feeders=[(base + s) * 10 + f for f in range(2)], feeders_excluded_small=0)
                for s in range(2)][:1 if role == "dev" else 2]
    spec = dict(version="ss-synthetic", data=dict(uri=root, start=D0, end=D1),
                target=dict(name="T2_node", min_coverage=0.6, levels=["substation", "feeder"]),
                history_gap_days=1,
                splits=dict(dev=dict(issue_start="2025-01-01", issue_end="2025-12-31"),
                            test=dict(issue_start="2025-01-01", issue_end="2025-12-31")),
                refits=["2025-01-01", "2025-07-01"], feeders=dict(min_households=20), horizons=60,
                control_horizons=[1, 7, 14, 30, 60], reference_model="B0_node", weather="none",
                test_substations=subs(CODES[0], "test"), dev_substations=subs(CODES[1], "dev"))
    yaml.safe_dump(spec, open(path, "w"), sort_keys=False)


def test_node_targets_and_b0(data, tmp_path):
    root, _ = data
    sp = str(tmp_path / "spec.yaml")
    _node_spec(sp, root)
    spec_mod = importlib.util.spec_from_file_location("bnts", os.path.join(REPO, "scripts", "test_set",
                                                                          "build_node_test_set.py"))
    b = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(b)
    out = b.main(["--spec", sp, "--out-root", str(tmp_path / "ts"), "--workers", "1"])
    spec = ns.load_spec(sp)
    t = ns.load_targets(out)
    nodes = ns.load_nodes(out)
    assert set(nodes["level"]) == {"substation", "feeder"} and len(nodes) == 3 + 6
    s = t[(t["node"] == f"S:{CODES[0]}:{int(CODES[0]) * 1000}") & (t["date"].dt.year == 2025)]
    assert (~s["flag"]).mean() > 0.9 and np.isclose(s["n_rows"].median(), 60)
    f = t[(t["node"] == f"F:{CODES[0]}:{int(CODES[0]) * 10000}") & (t["date"].dt.year == 2025)]
    assert np.isclose(f["n_rows"].median(), 30)
    old = t[(t["node"] == s["node"].iloc[0]) & (t["date"] < "2024-07-01") & (t["date"] > "2024-01-01")]
    assert old["flag"].mean() > 0.9 and np.isfinite(old["T3"]).mean() > 0.9   # T2 flagged, T3 there: the B0 ratio
    b0 = ns.b0_forecasts(t, spec, nodes=set(nodes["node"]), issue=ns.issue_dates(spec, "test"))
    assert np.isfinite(b0["forecast"]).mean() > 0.95
    assert (b0[b0["origin"] < "2025-06-01"]["b0src"] == 0).mean() > 0.8    # last-year ratio from T3
    res = ns.score({"B0": b0[["node", "origin", "horizon", "forecast"]],
                    "B0x": b0[["node", "origin", "horizon", "forecast"]].assign(forecast=lambda d: d["forecast"] * 1.1)},
                   t, spec, "test", nodes)
    q = res["summary"].set_index(["model", "level", "horizon"])
    assert q.loc[("B0x", "substation", 1), "beats_b0"] <= 0.5
    assert "SUBSTATION" in ns.report(res, ["B0", "B0x"])


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_tide_end_to_end_on_nodes(data, tmp_path):
    root, _ = data
    sp = str(tmp_path / "spec.yaml")
    _node_spec(sp, root)
    subprocess.run([sys.executable, os.path.join(REPO, "scripts", "test_set", "build_node_test_set.py"), "--spec", sp,
                    "--out-root", str(tmp_path / "ts"), "--workers", "1"], check=True, cwd=REPO, capture_output=True)
    mdir = tmp_path / "tide_tiny"
    mdir.mkdir()
    open(mdir / "model.py", "w").write(open(os.path.join(REPO, "models", "tide_agg", "model.py")).read())
    cfg = yaml.safe_load(open(os.path.join(REPO, "models", "tide_agg", "config.yaml")))
    cfg.update(buckets=[0], val_households=0.2, exclude_ranges=[])
    cfg["model"].update(hidden=32, temporal_hidden=16, cov_hidden=16)
    cfg["train"].update(epochs=2, windows_per_epoch=3000, batch=256, val_windows=2000, device="cpu", patience=2,
                       min_val_windows=100)
    cfg["predict"].update(threads=1)
    yaml.safe_dump(cfg, open(mdir / "config.yaml", "w"))
    ev = yaml.safe_load(open(os.path.join(REPO, "configs", "eval.yaml")))
    ev["run"].update(out_dir=str(tmp_path / "eval"), workers=1, block=1000)
    ev["wandb"]["enabled"] = False
    ev["mlflow"]["enabled"] = False
    ev_p = tmp_path / "eval.yaml"
    yaml.safe_dump(ev, open(ev_p, "w"))
    env = {**os.environ, "WANDB_MODE": "disabled", "MLFLOW_DISABLE": "1"}
    r = subprocess.run([sys.executable, os.path.join(REPO, "evaluate.py"), "--model", str(mdir), "--mode", "validate",
                        "--test-set", sp, "--test-dir", str(tmp_path / "ts" / "ss-synthetic"), "--config", str(ev_p),
                        "--allow-dirty", "--no-mlflow", "--no-wandb", "--workers", "1"],
                       capture_output=True, text=True, cwd=REPO, env=env)
    log = r.stdout + r.stderr
    assert r.returncode == 0, log[-4000:]
    assert "[leakage] PASSED" in log and "estimated runs" in log and "SUBSTATION" in log and "FEEDER" in log, log[-3000:]
    run = sorted((tmp_path / "eval" / "runs").iterdir())[-1]
    s = pd.read_csv(run / "scores" / "dev" / "summary.csv")
    m = s[(s["model"] == "tide_tiny") & (s["level"] == "substation")]
    assert len(m) == 60 and np.isfinite(m["median"]).all()
    assert m["median"].mean() < 0.6, s.to_string()           # a sane forecast after two tiny epochs
    assert len(list((run / "scores" / "dev" / "windows").glob("*.png"))) >= 3     # substation window charts
    # test mode: the dev AND the test substations in one run, the fits reused
    r = subprocess.run([sys.executable, os.path.join(REPO, "evaluate.py"), "--model", str(mdir), "--mode", "test",
                        "--test-set", sp, "--test-dir", str(tmp_path / "ts" / "ss-synthetic"), "--config", str(ev_p),
                        "--allow-dirty", "--no-mlflow", "--no-wandb", "--workers", "1", "--no-substitution-test"],
                       capture_output=True, text=True, cwd=REPO, env=env)
    log = r.stdout + r.stderr
    assert r.returncode == 0, log[-4000:]
    assert "reusing" in log and "NODE SCORES (dev)" in log and "NODE SCORES (test)" in log, log[-3000:]
    run2 = sorted((tmp_path / "eval" / "runs").iterdir())[-1]
    assert run2.name.endswith("_nodes-dev-test")
    for role in ("dev", "test"):
        assert (run2 / "scores" / role / "summary.csv").exists() and list((run2 / "scores" / role / "windows").glob("*.png"))
    cmp = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "compare_node_runs.py"), str(run2), "--role",
                          "test", "--spec", sp], capture_output=True, text=True, cwd=REPO)
    assert cmp.returncode == 0 and "NODE SCORES (test)" in cmp.stdout, cmp.stdout[-2000:] + cmp.stderr[-2000:]
