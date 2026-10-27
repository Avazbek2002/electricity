"""The simulated clock: release everything that ARRIVES up to a date into the operator tables.

    python -m elecsim.clock --to 2025-01-15          # one or many days forward (never backwards)
    python -m elecsim.clock --status

Readings get increasing ids in arrival order (the pipeline's incremental watermark) and created_at = arrival date
06:00 + a few minutes. The current customer snapshot is updated in place (upsert by id), monthly histories appear on the
first day of their month, weather observations the morning after, forecast runs on the morning of their issue date.
"""
import argparse
import os

import pandas as pd

MOVE_READINGS = """
WITH moved AS (
    DELETE FROM sim.pending_readings WHERE arrival_date <= %(to)s RETURNING *
), numbered AS (
    SELECT m.*, row_number() OVER (ORDER BY arrival_date, seq) AS rn FROM moved m
)
INSERT INTO household_reading.reading_histories
    (id, consumer_id, meter_id, coato_code, reading_date, reading_a_plus, cycle_a_plus, reading_source_type_id,
     meter_reading_status_id, created_at)
SELECT %(next_id)s + rn - 1, consumer_id, meter_id, coato_code, reading_date, reading_a_plus, cycle_a_plus,
       reading_source_type_id, meter_reading_status_id,
       arrival_date + time '06:00' + (rn %% 120) * interval '1 minute'
FROM numbered
"""

UPSERT_CONSUMERS = """
WITH due AS (
    DELETE FROM sim.pending_consumers WHERE effective_date <= %(to)s RETURNING *
), latest AS (
    SELECT DISTINCT ON (id) * FROM due ORDER BY id, effective_date DESC, updated_at DESC
)
INSERT INTO household_consumer_new.consumers
    (id, coato_code, transformer_id, additional_substation_id, tariff_id, meter_id, accounting_status_id,
     consumer_status_id, consumer_status_update_at, housing_type_id, housing_category_id, gaz_connection,
     central_heating, residents_amount, application_date, contraction_date, updated_at)
SELECT id, coato_code, transformer_id, additional_substation_id, tariff_id, meter_id, accounting_status_id,
       consumer_status_id, consumer_status_update_at, housing_type_id, housing_category_id, gaz_connection,
       central_heating, residents_amount, application_date, contraction_date, updated_at
FROM latest
ON CONFLICT (id) DO UPDATE SET
    transformer_id = EXCLUDED.transformer_id, additional_substation_id = EXCLUDED.additional_substation_id,
    meter_id = EXCLUDED.meter_id, consumer_status_id = EXCLUDED.consumer_status_id,
    consumer_status_update_at = EXCLUDED.consumer_status_update_at, updated_at = EXCLUDED.updated_at
"""

MOVE_SIMPLE = [
    ("household_billing.consumer_histories", "sim.pending_histories", "period <= %(to)s"),
    ("weather.observed_daily", "sim.pending_weather_obs", "arrived_at < %(to)s::date + 1"),
    ("weather.forecast_daily", "sim.pending_weather_fc", "arrived_at < %(to)s::date + 1"),
]


def status(dsn):
    import psycopg
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT sim_date, next_reading_id FROM sim.clock")
        d, nid = cur.fetchone()
        cur.execute("SELECT count(*), max(reading_date), max(created_at) FROM household_reading.reading_histories")
        n, md, mc = cur.fetchone()
        cur.execute("SELECT count(*) FROM sim.pending_readings")
        p = cur.fetchone()[0]
    return dict(sim_date=d, readings=n, last_reading_date=md, last_created_at=mc, pending=p, next_reading_id=nid)


def advance(dsn, to, log=print):
    """Move the clock forward to `to` (inclusive). Idempotent: a date at or before the clock is a no-op."""
    import psycopg
    to = pd.Timestamp(to).date()
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT sim_date, next_reading_id FROM sim.clock FOR UPDATE")
        cur_date, next_id = cur.fetchone()
        if to <= cur_date:
            log(f"[clock] already at {cur_date}; nothing to do for {to}")
            return dict(sim_date=cur_date, readings=0)
        cur.execute(MOVE_READINGS, dict(to=to, next_id=next_id))
        n = cur.rowcount
        cur.execute(UPSERT_CONSUMERS, dict(to=to))
        nc = cur.rowcount
        moved = {}
        for dst, src, cond in MOVE_SIMPLE:
            cur.execute(f"WITH m AS (DELETE FROM {src} WHERE {cond} RETURNING *) INSERT INTO {dst} SELECT * FROM m "
                        f"ON CONFLICT DO NOTHING", dict(to=to))
            moved[dst] = cur.rowcount
        cur.execute("UPDATE sim.clock SET sim_date = %s, next_reading_id = %s, updated_at = now()", (to, next_id + n))
        conn.commit()
    log(f"[clock] {cur_date} -> {to}: {n:,} readings, {nc:,} customer snapshot changes, "
        + ", ".join(f"{k.split('.')[-1]} {v:,}" for k, v in moved.items()))
    return dict(sim_date=to, readings=n, consumers=nc, **moved)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dsn", default=os.environ.get("SOURCE_DSN"))
    p.add_argument("--to", default=None, help="advance the clock to this date (inclusive)")
    p.add_argument("--status", action="store_true")
    a = p.parse_args(argv)
    if not a.dsn:
        raise SystemExit("no DSN: --dsn or SOURCE_DSN")
    if a.to:
        advance(a.dsn, a.to)
    print(status(a.dsn))


if __name__ == "__main__":
    main()
