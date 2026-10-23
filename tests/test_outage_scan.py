"""outage_scan.py on a synthetic district with planted outages (local files in the enriched layout)."""
import importlib.util
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
spec = importlib.util.spec_from_file_location("outage_scan", os.path.join(REPO, "scripts", "analysis", "outage_scan.py"))
osc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(osc)

CODE = "99001"
D0, D1 = "2024-08-01", "2024-12-31"


def make_district(root, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(D0, D1)
    T = len(dates)
    weather = 1 + 0.25 * np.sin(np.arange(T) / 20) + 0.05 * rng.standard_normal(T)     # shared by everyone
    rows = []
    hh = 0
    paths = []
    for s in range(3):
        for f in range(3):
            for t in range(4):
                for _ in range(12):
                    paths.append((hh, 1000 + s * 100 + f * 10 + t, 100 + s * 10 + f, 10 + s))
                    hh += 1
    P = pd.DataFrame(paths, columns=["c", "t", "f", "s"])
    level = rng.lognormal(np.log(6), 0.4, len(P))
    K = level[:, None] * weather[None, :] * rng.lognormal(0, 0.15, (len(P), T))
    d = {x: dates.get_loc(pd.Timestamp(x)) for x in ("2024-10-09", "2024-10-10", "2024-11-08", "2024-11-28",
                                                     "2024-10-29", "2024-10-30", "2024-10-31")}
    K[(P["t"] == 1102).to_numpy(), d["2024-10-09"]:d["2024-10-10"] + 1] *= 0.3       # transformer, 2 days
    K[(P["f"] == 101).to_numpy(), d["2024-11-08"]] *= 0.35                            # feeder, 1 day
    K[(P["s"] == 11).to_numpy(), d["2024-11-28"]] *= 0.4                              # substation, 1 day
    silent = (P["t"] == 1221).to_numpy()                                              # transformer silent 3 days
    state = np.where(rng.random((len(P), T)) < 0.9, "complete_observed", "complete_with_estimated")
    for i in range(len(P)):
        for j in range(T):
            if silent[i] and d["2024-10-29"] <= j <= d["2024-10-31"]:
                continue
            rows.append((f"c{i:04d}", dates[j].date(), int(P.t[i]), int(P.f[i]), int(P.s[i]), False, state[i, j],
                         float(np.round(K[i, j], 3))))
    df = pd.DataFrame(rows, columns=["customer_key", "date", "transformer_id", "feeder_id", "substation_id",
                                     "placeholder_transformer", "label_state", "kwh"])
    base = os.path.join(root, "household_daily_enriched", f"coato={CODE}")
    for per, g in df.groupby(pd.to_datetime(df["date"]).dt.strftime("%Y-%m")):
        out = os.path.join(base, "household_daily_enriched", "data", f"period={per}")
        os.makedirs(out, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(g, preserve_index=False), os.path.join(out, "part-0.parquet"))
    open(os.path.join(base, "_SUCCESS"), "w").close()


def test_planted_outages_found_at_their_level(tmp_path):
    make_district(str(tmp_path / "root"))
    out = osc.main(["--out-dir", str(tmp_path / "out"), "--root", str(tmp_path / "root"), "--districts", CODE,
                    "--start", D0, "--end", D1, "--workers", "1", "--inject-transformers", "16",
                    "--inject-feeders", "8"])
    ev = pd.read_csv(os.path.join(out, "events.csv"))
    sup = ev[ev["type"] == "supply_dip"]

    def found(level, node, start):
        return ((sup["level"] == level) & (sup["node_id"] == node) & (sup["start"] == start)).any()
    assert found("transformer", 1102, "2024-10-09"), sup.to_string()
    assert sup[(sup["node_id"] == 1102)]["days"].max() == 2
    assert found("feeder", 101, "2024-11-08"), sup.to_string()
    assert found("substation", 11, "2024-11-28"), sup.to_string()
    # covered children are not reported again
    assert not ((sup["level"] == "transformer") & (sup["start"] == "2024-11-08") & (sup["feeder_id"] == 101)).any()
    sil = ev[(ev["type"] == "silent") & (ev["node_id"] == 1221)]
    assert len(sil) and sil["start"].iloc[0] == "2024-10-29"
    # the planted events are almost all there is: few false events in noise
    assert len(sup) <= 8, sup.to_string()
    ctl = pd.read_csv(os.path.join(out, "control.csv"))
    real = ctl[(ctl["run"] == "real") & (ctl["type"] == "supply_dip")]["events"].sum()
    shuf = ctl[(ctl["run"] != "real") & (ctl["type"] == "supply_dip")]["events"].sum()
    assert real > shuf
    inj = pd.read_csv(os.path.join(out, "injection.csv"))
    assert len(inj) >= 16
    deep = inj[inj["depth"] == 0.6]
    assert deep["detected_same_level"].mean() >= 0.8, inj.to_string()
    nd = pd.read_parquet(os.path.join(out, "node_day.parquet"))
    assert (nd["households_observed"] >= 10).all()
    assert os.path.exists(os.path.join(out, "figures", "top_events.png"))


def test_attribution_top_down():
    cfg = dict(osc.DEFAULTS)
    T = 3
    t_par = np.array([0, 0, 1, 1])          # 4 transformers on 2 feeders
    f_par = np.array([0, 1])                # feeder 0 -> substation 0, feeder 1 -> substation 1
    el = dict(transformer=np.ones((4, T), bool), feeder=np.ones((2, T), bool), substation=np.ones((2, T), bool))
    fl = {k: np.zeros_like(v) for k, v in el.items()}
    fl["transformer"][[0, 1], 0] = True     # day 0: both transformers of feeder 0 -> feeder 0 (and substation 0)
    fl["feeder"][0, 0] = True
    fl["substation"][0, 0] = True
    fl["transformer"][2, 1] = True          # day 1: one transformer alone -> transformer 2
    out = osc.attribute(fl, el, t_par, f_par, cfg)
    assert out["substation"][0][0, 0]       # substation 0 holds only feeder 0: the event goes to the top
    assert not out["feeder"][0][0, 0] and not out["transformer"][0][:, 0].any()
    assert out["transformer"][0][2, 1] and not out["transformer"][0][3, 1]
