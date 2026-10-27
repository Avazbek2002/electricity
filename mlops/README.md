# MLOps pipeline: substation / feeder forecasts from the operator's daily readings

The operator's PostgreSQL (household readings, customers, monthly customer state) is not reachable yet, so phase 1
simulates it in the same schema (`sql/001_source_schema.sql`, from ANALYTICAL_DATASETS_MASTER_GUIDE.md section 5).
The pipeline only ever reads the operator tables; switching to the real database is a change of `SOURCE_DSN`
(plus the tables marked ASSUMED in the schema, which must be checked against the real one).

| Phase | What | Status |
|---|---|---|
| 1 | Source simulator: schema, synthetic households and readings, simulated clock | **done** |
| 2 | Local stack: MinIO (S3), MLflow, Airflow next to the source database | next |
| 3 | `daily` DAG, data half: ingest -> validate -> intervals -> household-days -> node aggregates -> as-of snapshot | |
| 4 | `train` DAG: `node_direct` on the snapshots -> MLflow registry | |
| 5 | `daily` DAG, forecast half: predict -> publish (fallbacks: no-weather model -> B1 -> B0) | |
| 6 | `monitor` DAG: score matured forecasts, drift, alerts -> trigger `train` | |
| 7 | GitLab CI/CD: tests, image build, DAG deploy, approved model promotion | |

Orchestration: Airflow runs everything triggered by time or data (daily, monitor, train, backfills); GitLab CI/CD
runs everything triggered by a change in this repository (tests, image, DAG deploy, approved promotion). Every step is a
CLI module (`python -m ... --date D`), so the same command runs in CI tests, in Airflow and by hand.

## Phase 1: the simulated source

`elecsim/generate.py` builds, from `elecsim/config.yaml`:
- a network: districts -> substations -> feeders -> transformers (`household_billing.network_feeders`, ASSUMED table);
- households: base load, gas vs electric heating (heating / cooling sensitivity), apartments, ~3 %/year new
  connections, disconnections, blocks re-mapped to another substation (the 2023-24 jumps of the real data);
- the true daily kWh: temperature (degree days), weekday (Sunday +5 %, Monday -5 %), holidays, the 3 days before Eid
  al-Fitr, the 25 Jan 2022 blackout, household noise;
- meters and readings: manual readings every 25-35 days, migration to ASKUE (daily, 93 % success, outages of 3-60
  days), meter replacements, coefficients 100 / 200, whole-kWh registers before 2024-08, duplicates, bad statuses;
- arrival dates: ASKUE readings mostly the next day, some days or weeks late; manual readings 0-7 days late;
- weather: observed daily mean / min / max per district, AIFS- and FCN3-like forecast runs (lead 0-15, error growing
  with lead, per-season bias) on odd issue dates.

Everything goes into `sim.pending_*` with its arrival date; `elecsim/clock.py` releases each day's arrivals into the
operator tables (readings get increasing ids = the pipeline's watermark; the customer snapshot is updated in place).
`sim.truth_node_daily` keeps the true kWh per substation / feeder for diagnostics; the pipeline never reads `sim.*`.

Default size: 2 districts, 10 substations, 30 feeders, ~3,250 households, ~2.6 M readings 2022-01 .. 2026-06
(about 1.5 minutes to generate and load).

```bash
cp .env.example .env                                    # set the password
docker compose up -d source-db
docker compose --profile init run --rm sim-init         # generate, load, release history up to SIM_START
docker compose --profile tools run --rm sim elecsim.clock --to 2025-01-05
docker compose --profile tools run --rm sim elecsim.clock --status
```

Without Docker: `SOURCE_DSN=postgresql://... python -m elecsim.generate --clock-to 2024-12-31`.

Tests: `python -m pytest mlops/tests` (energy conservation, arrival order, whole-kWh regime; the clock test needs
`SOURCE_TEST_DSN`, an empty PostgreSQL database it may wipe).
