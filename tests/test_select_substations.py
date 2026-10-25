"""select_substations.py on synthetic districts (local files in the enriched layout)."""
import importlib.util
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
spec = importlib.util.spec_from_file_location("select_substations",
                                              os.path.join(REPO, "scripts", "test_set", "select_substations.py"))
sel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sel)

D0, D1 = "2024-08-01", "2025-12-31"
REGIONS = {"90001": "R1", "90002": "R1", "90003": "R2", "90004": "R2", "90005": "R3", "90006": "R3", "90007": "R4",
           "90008": "R4"}


def make(root, code, region, seed):
    """4 substations: S1 60 households, S2 40, S3 15 (too small), S4 40 on placeholder transformers; in 90008 S2 has
    poor 2025 coverage."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(D0, D1)
    rows = []
    c = 0
    base = int(code) * 100
    for s, (n, ph) in enumerate([(60, False), (40, False), (15, False), (40, True)]):
        sid = base + s
        for i in range(n):
            t = sid * 10 + i % 3
            f = sid * 10 + 5 + i % 2
            seg = "apartment__gas_only" if i % 4 == 0 else "house__neither"
            lvl = rng.lognormal(np.log(6), 0.3)
            poor = code == "90008" and s == 1
            for j, d in enumerate(dates):
                st = "complete_observed" if rng.random() < (0.4 if poor and d.year == 2025 else 0.92) else "unlabelled"
                rows.append((f"k{code}_{c}", d.date(), t, f, sid, ph, st,
                             float(lvl * rng.lognormal(0, 0.1)) if st != "unlabelled" else None,
                             region, f"Region {region}", f"District {code}", seg))
            c += 1
    df = pd.DataFrame(rows, columns=["customer_key", "date", "transformer_id", "feeder_id", "substation_id",
                                     "placeholder_transformer", "label_state", "kwh", "region_code", "region_name",
                                     "admin_location_name", "operational_segment"])
    b = os.path.join(root, "household_daily_enriched", f"coato={code}")
    for per, g in df.groupby(pd.to_datetime(df["date"]).dt.strftime("%Y-%m")):
        o = os.path.join(b, "household_daily_enriched", "data", f"period={per}")
        os.makedirs(o, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(g, preserve_index=False), os.path.join(o, "part-0.parquet"))
    open(os.path.join(b, "_SUCCESS"), "w").close()


def test_selection(tmp_path):
    root = str(tmp_path / "root")
    for k, (code, reg) in enumerate(REGIONS.items()):
        make(root, code, reg, k)
    spec_out = str(tmp_path / "spec.yaml")
    out = sel.main(["--out-dir", str(tmp_path / "out"), "--root", root, "--ref-root", "none", "--start", D0,
                    "--end", D1, "--workers", "1", "--min-households", "30", "--test-districts", "5",
                    "--per-district", "2", "--dev", "3", "--exclude-test", "90001", "--spec-out", spec_out])
    sub = pd.read_csv(os.path.join(out, "substations_all.csv"), dtype={"coato_code": str})
    assert len(sub) == 32
    why = sub.set_index(["coato_code", "substation_id"])["reason"].fillna("")
    assert why[("90003", 9000302)].startswith("< 30 households")
    assert why[("90003", 9000303)].startswith("path share")
    assert why[("90008", 9000801)].startswith("2025 coverage")
    assert sub["eligible"].sum() == 8 * 2 - 1
    s = pd.read_csv(os.path.join(out, "selection.csv"), dtype={"coato_code": str})
    test, dev = s[s["role"] == "test"], s[s["role"] == "dev"]
    assert "90001" not in set(test["coato_code"])                          # excluded from test
    assert "90008" not in set(test["coato_code"])                          # only one eligible substation there
    assert len(test) == 2 * test["coato_code"].nunique() and test["coato_code"].nunique() == 5
    assert set(test["region_code"]) == {"R1", "R2", "R3", "R4"}            # every region with a candidate
    assert set(test["stratum"]) == {"large", "mid"}
    assert len(dev) == 3 and "90001" in set(dev["coato_code"])            # the excluded district gives dev first
    assert not set(dev["coato_code"]) & set(test["coato_code"])
    sp = yaml.safe_load(open(spec_out))
    assert len(sp["test_substations"]) == len(test) and len(sp["dev_substations"]) == 3
    assert all(len(x["feeders"]) == 0 and x["feeders_excluded_small"] == 2 for x in sp["test_substations"])  # < 50
    assert sp["selection"]["selection_csv_sha256"]
    # deterministic with the seed
    out2 = sel.main(["--out-dir", str(tmp_path / "out2"), "--root", root, "--ref-root", "none", "--start", D0,
                     "--end", D1, "--workers", "1", "--min-households", "30", "--test-districts", "5",
                     "--per-district", "2", "--dev", "3", "--exclude-test", "90001", "--spec-out",
                     str(tmp_path / "spec2.yaml")])
    assert open(os.path.join(out, "selection.csv")).read() == open(os.path.join(out2, "selection.csv")).read()
    rep = open(os.path.join(out, "selection_report.txt")).read()
    assert "ELIGIBILITY FUNNEL" in rep and "BY REGION" in rep
    assert os.path.exists(os.path.join(out, "figures", "distributions.png"))
