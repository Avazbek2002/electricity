"""outage_scan.py on synthetic districts with planted events (local files in the enriched layout)."""
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

CODE, PLACEHOLDER = "99001", "99002"
D0, D1 = "2024-08-01", "2024-12-31"
HOLIDAY = "2024-12-20"


def _write(root, code, df):
    base = os.path.join(root, "household_daily_enriched", f"coato={code}")
    for per, g in df.groupby(pd.to_datetime(df["date"]).dt.strftime("%Y-%m")):
        out = os.path.join(base, "household_daily_enriched", "data", f"period={per}")
        os.makedirs(out, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(g, preserve_index=False), os.path.join(out, "part-0.parquet"))
    open(os.path.join(base, "_SUCCESS"), "w").close()


def make_district(root, seed=1, code=CODE, placeholder=False):
    """3 substations x 3 feeders x 4 transformers x 12 households, Aug-Dec 2024, with planted events:
    transformer 1102 -70 % on Oct 9-10, feeder 101 -65 % on Nov 8, substation 11 -60 % on Nov 28, transformer 1221
    silent Oct 29-31, feeder 121 half its households missing and the rest -50 % on Oct 16 (partial silent),
    transformer 1023 -50 % for 10 days from Sep 20 (level shift), and on the holiday Dec 20 one transformer per feeder
    -45 % (clustered)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(D0, D1)
    T = len(dates)
    weather = 1 + 0.25 * np.sin(np.arange(T) / 20) + 0.05 * rng.standard_normal(T)     # shared by everyone
    paths = []
    hh = 0
    for s in range(3):
        for f in range(3):
            for t in range(4):
                for _ in range(12):
                    paths.append((hh, 1000 + s * 100 + f * 10 + t, 100 + s * 10 + f, 10 + s))
                    hh += 1
    P = pd.DataFrame(paths, columns=["c", "t", "f", "s"])
    level = rng.lognormal(np.log(6), 0.4, len(P))
    K = level[:, None] * weather[None, :] * rng.lognormal(0, 0.15, (len(P), T))
    d = lambda x: dates.get_loc(pd.Timestamp(x))                                      # noqa: E731
    K[(P["t"] == 1102).to_numpy(), d("2024-10-09"):d("2024-10-10") + 1] *= 0.3
    K[(P["f"] == 101).to_numpy(), d("2024-11-08")] *= 0.35
    K[(P["s"] == 11).to_numpy(), d("2024-11-28")] *= 0.4
    K[(P["f"] == 121).to_numpy(), d("2024-10-16")] *= 0.5
    K[(P["t"] == 1023).to_numpy(), d("2024-09-20"):d("2024-09-29") + 1] *= 0.5
    K[(P["t"] % 10 == 0).to_numpy(), d(HOLIDAY)] *= 0.55                              # transformer 0 of every feeder
    silent = (P["t"] == 1221).to_numpy()
    part = ((P["f"] == 121) & (P["c"] % 2 == 0)).to_numpy()
    state = np.where(rng.random((len(P), T)) < 0.9, "complete_observed", "complete_with_estimated")
    rows = []
    for i in range(len(P)):
        for j in range(T):
            if silent[i] and d("2024-10-29") <= j <= d("2024-10-31"):
                continue
            if part[i] and j == d("2024-10-16"):
                continue
            hol = dates[j] == pd.Timestamp(HOLIDAY)
            rows.append((f"c{i:04d}", dates[j].date(), int(P.t[i]), int(P.f[i]), int(P.s[i]), bool(placeholder),
                         state[i, j], float(np.round(K[i, j], 3)), bool(hol), "Test holiday" if hol else None,
                         False))
    df = pd.DataFrame(rows, columns=["customer_key", "date", "transformer_id", "feeder_id", "substation_id",
                                     "placeholder_transformer", "label_state", "kwh", "is_public_holiday",
                                     "holiday_name", "is_ramadan"])
    _write(root, code, df)


def test_planted_events_found_and_classified(tmp_path):
    root = str(tmp_path / "root")
    make_district(root)
    make_district(root, seed=2, code=PLACEHOLDER, placeholder=True)
    out = osc.main(["--out-dir", str(tmp_path / "out"), "--root", root, "--districts", f"{CODE},{PLACEHOLDER}",
                    "--start", D0, "--end", D1, "--workers", "1", "--inject-transformers", "16",
                    "--inject-feeders", "8"])
    ev = pd.read_csv(os.path.join(out, "events.csv"), dtype={"coato_code": str})
    assert set(ev["coato_code"]) == {CODE}                                  # the placeholder district is skipped
    info = pd.read_csv(os.path.join(out, "districts.csv"), dtype={"coato_code": str}).set_index("coato_code")
    assert "placeholder" in info.loc[PLACEHOLDER, "skipped"] and pd.isna(info.loc[CODE, "skipped"])

    def get(typ, level, node, start):
        return ev[(ev["type"] == typ) & (ev["level"] == level) & (ev["node_id"] == node) & (ev["start"] == start)]
    t = get("supply_dip", "transformer", 1102, "2024-10-09")
    assert len(t) and t["days"].iloc[0] == 2 and t["tier"].iloc[0] == "deep", ev.to_string()
    assert len(get("supply_dip", "feeder", 101, "2024-11-08")), ev.to_string()
    s = get("supply_dip", "substation", 11, "2024-11-28")                    # one substation is not a cluster
    assert len(s) and s["tier"].iloc[0] == "deep", ev.to_string()
    assert not ((ev["level"] == "transformer") & (ev["start"] == "2024-11-08") & (ev["feeder_id"] == 101)).any()
    assert len(get("silent", "transformer", 1221, "2024-10-29"))
    ps = get("partial_silent", "feeder", 121, "2024-10-16")
    assert len(ps) and pd.isna(ps["lost_kwh_est"].iloc[0])                   # no kWh lost for missing readings
    assert not ((ev["type"] == "supply_dip") & (ev["start"] == "2024-10-16")).any(), ev.to_string()
    ls = ev[(ev["type"] == "level_shift") & (ev["node_id"] == 1023)]
    assert len(ls) and ls["days"].iloc[0] >= 8, ev.to_string()
    cl = ev[(ev["type"] == "clustered_dip") & (ev["start"] == HOLIDAY)]
    assert len(cl) >= 5 and (cl["holiday"] == "Test holiday").all(), ev.to_string()
    assert not ((ev["type"] == "supply_dip") & (ev["start"] == HOLIDAY)).any()
    assert ev.loc[ev["type"].isin(["silent", "partial_silent"]), "lost_kwh_est"].isna().all()
    sup = ev[ev["type"] == "supply_dip"]
    assert len(sup) <= 8, sup.to_string()
    assert (ev["lost_kwh_est"].dropna() <= 2.0 * ev["lost_kwh_observed"].dropna() + 1e-6).all()
    ctl = pd.read_csv(os.path.join(out, "control.csv"))
    assert {"tier", "type", "level"} <= set(ctl.columns)
    real = ctl[(ctl["run"] == "real") & (ctl["type"] == "supply_dip")]["events"].sum()
    shuf = ctl[(ctl["run"] != "real") & (ctl["type"] == "supply_dip")]["events"].sum()
    assert real > shuf
    dd = pd.read_csv(os.path.join(out, "district_days.csv"))
    assert dd.loc[dd["date"] == HOLIDAY, "clustered"].all() and not dd.loc[dd["date"] == "2024-11-28", "clustered"].any()
    inj = pd.read_csv(os.path.join(out, "injection.csv"))
    assert len(inj) >= 16
    assert inj[inj["depth"] == 0.6]["detected_same_level"].mean() >= 0.8, inj.to_string()
    nd = pd.read_parquet(os.path.join(out, "node_day.parquet"))
    assert (nd["households_observed"] >= 10).all()
    rep = open(os.path.join(out, "outage_report.txt")).read()
    assert "SKIPPED 99002" in rep and "usable" in rep
    for f in ("events_by_month", "lift", "injection_recall", "top_events"):
        assert os.path.exists(os.path.join(out, "figures", f + ".png")), f


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
