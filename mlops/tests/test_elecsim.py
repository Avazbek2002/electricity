"""The simulator: readings conserve the true energy, arrive after they are taken, and the clock releases them in order.
The database tests need a PostgreSQL (env SOURCE_TEST_DSN, an empty database the test may wipe); skipped otherwise."""
import copy
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from elecsim import clock, generate as g  # noqa: E402


@pytest.fixture(scope="module")
def small():
    cfg = g.load_config()
    cfg = copy.deepcopy(cfg)
    cfg.update(start="2024-01-01", end="2025-03-31", seed=7)
    cfg["network"].update(districts=["90101"], substations_per_district=2, feeders_per_substation=2,
                          households_per_feeder=[20, 30])
    cfg["events"]["remaps"] = 1
    return cfg, g.generate(cfg)


def test_registers_conserve_true_energy(small):
    cfg, d = small
    R = d["readings"].drop_duplicates(["meter_id", "reading_date"]).sort_values(["meter_id", "reading_date"])
    M = d["meters"].set_index("meter_id")
    R["kwh"] = R.groupby("meter_id")["reading_a_plus"].diff() * R["meter_id"].map(M["coef"]) / 100 / 1000
    total_intervals = R["kwh"].sum()
    # truth over the same spans: every meter from its first to its last reading
    days = pd.date_range(cfg["start"], cfg["end"])
    H = d["households"]
    rng = np.random.default_rng(int(cfg["seed"]))                       # regenerate the truth matrix the same way
    feeders, _ = g.network(cfg, rng)
    wx = g.weather(days, cfg["network"]["districts"], cfg, rng)
    cal = g.calendar_factor(days, cfg)
    H2 = g.households(cfg, feeders, days, rng)
    g.remaps(cfg, H2, feeders, days, rng)
    Y, _ = g.truth(H2, wx, cal, days, cfg, rng)
    pos = {c: i for i, c in enumerate(H2["consumer_id"])}
    true = 0.0
    for mid, q in R.groupby("meter_id"):
        a, b = (q["reading_date"].iloc[0] - days[0]).days, (q["reading_date"].iloc[-1] - days[0]).days
        true += Y[pos[M.loc[mid, "consumer_id"]], a + 1:b + 1].sum()
    assert len(H) == len(H2)
    assert abs(total_intervals / true - 1) < 0.002            # whole-kWh rounding before the regime change only


def test_arrivals_and_schedule(small):
    cfg, d = small
    R = d["readings"]
    assert (R["arrival_date"] >= R["reading_date"]).all()
    assert R["seq"].is_monotonic_increasing and R["arrival_date"].is_monotonic_increasing
    ask = R[R["reading_source_type_id"] == g.ASKUE]
    lag = (ask["arrival_date"] - ask["reading_date"]).dt.days
    assert 0.8 < (lag == 1).mean() < 0.95                      # most automatic readings arrive the next day
    old = R[R["reading_date"] < cfg["households"]["regime_change"]]
    coef = old["meter_id"].map(d["meters"].set_index("meter_id")["coef"])
    unit = 1000 * 100 / coef
    assert np.allclose(old["reading_a_plus"] / unit, np.round(old["reading_a_plus"] / unit))   # whole kWh then
    assert len(d["remaps"]) > 0 and d["truth"]["node"].str.startswith("S:").any()


@pytest.mark.skipif(not os.environ.get("SOURCE_TEST_DSN"), reason="needs SOURCE_TEST_DSN (a PostgreSQL to wipe)")
def test_clock_releases_in_arrival_order(small):
    import psycopg
    cfg, d = small
    dsn = os.environ["SOURCE_TEST_DSN"]
    g.load(dsn, d, cfg)
    clock.advance(dsn, "2025-01-31", log=lambda *_: None)
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT max(created_at)::date, max(reading_date), count(*) FROM household_reading.reading_histories")
        mc, mr, n = cur.fetchone()
        cur.execute("SELECT min(id), max(id), count(DISTINCT id) FROM household_reading.reading_histories")
        lo, hi, k = cur.fetchone()
        cur.execute("SELECT count(*) FROM household_reading.reading_histories r JOIN household_consumer_new.consumers c "
                    "ON c.id = r.consumer_id")
        joined = cur.fetchone()[0]
    assert str(mc) <= "2025-01-31" and str(mr) <= "2025-01-31"
    assert n == (d["readings"]["arrival_date"] <= "2025-01-31").sum()
    assert lo == 1 and hi == n == k                            # dense ids: the incremental watermark
    assert joined == n                                         # every reading's customer is in the snapshot
    first = clock.advance(dsn, "2025-02-01", log=lambda *_: None)["readings"]
    again = clock.advance(dsn, "2025-02-01", log=lambda *_: None)["readings"]
    assert first > 0 and again == 0                            # idempotent
