# The fixed test set (`ts-2026-10-22-v1`)

Every model, from any developer, is scored on the same test days, households, actuals and code.
Definition: [`configs/test_set.yaml`](../configs/test_set.yaml). Implementation: [`harness/testset.py`](../harness/testset.py).

## Design

| Choice | Value | Why |
|---|---|---|
| Test days | 180 = four seasonal blocks of 45 days | ТЗ 4.2.4: at least 180 days; every season, both peaks |
| Folds | one per block, each with its own fit | ТЗ walk-forward rule; about quarterly retraining |
| Origins | every day; each test day scored at every horizon 1..60 | ТЗ sums over all days of the test sample |
| Households | all eligible; fixed panel per fold (eligible at the fold's first origin) | same households for every model |
| Node actual | N x mean kWh of the panel households observed that day | comparable for bottom-up and direct models |
| Scored node-days | N >= 10 households and >= 80 % of the panel observed | disclosure rule; reliable totals |
| Levels | national, region, district (all horizons); substation, feeder, transformer (h 1, 7, 14, 30, 60) | file size |

| Fold | Test days | Training cutoff | Inner validation | Daily origins | Covers |
|---|---|---|---|---|---|
| summer | 2025-07-01 .. 08-14 | 2025-03-03 | 2024-09-01 .. 2025-03-03 | 2025-05-02 .. 08-13 | cooling peak |
| autumn | 2025-10-01 .. 11-14 | 2025-06-03 | 2024-12-02 .. 2025-06-03 | 2025-08-02 .. 11-13 | heating start, Teachers' Day |
| winter | 2025-12-20 .. 2026-02-02 | 2025-08-22 | 2025-02-20 .. 2025-08-22 | 2025-10-21 .. 2026-02-01 | heating peak, New Year |
| spring | 2026-03-01 .. 04-14 | 2025-11-01 | 2025-05-02 .. 2025-11-01 | 2025-12-31 .. 2026-04-13 | Women's Day, Eid al-Fitr, Navruz, heating end |

Training data of a fold ends at `cutoff + 60 = first_origin`, so no training target is a test day of that fold.

## Two scores

For one node and target day: N panel households, n of them observed, P forecasts, A actuals.

| Score | Forecast | Actual | For |
|---|---|---|---|
| **coverage-adjusted** (headline) | N x mean P over the n observed | N x mean A over the n observed | every model |
| **matched** | sum P over the n observed | sum A over the n observed | bottom-up models |

Per day the two differ only by the factor N / n (1 .. 1.25 with the 80 % rule).

## Reported

- ТЗ 4.2.4 check: national coverage-adjusted sysMAPE at h = 1, 7, 14, 30, 60 against 8 / 11 / 15 / 18 / 20 %.
- sysMAPE, bias, median and p90 node sysMAPE per level and horizon, per fold, both scores.
- Bias audit (ТЗ): season (fold), day type (workday / weekend / holiday) and region, each at most 1.5 x the overall value.
- Relative change against the 30-day persistence reference on the same rows.

## How to use it

```bash
# once: build the panels, actuals and node history (~ one pass over the data)
python scripts/test_set/build_test_set.py --workers 16

# a harness model: 4 fits, daily forecasts, forecast CSVs, scores, MLflow (alias best_test with --register)
python evaluate.py --model gam --mode test --workers 16 --register

# any other model (e.g. a direct feeder or substation model)
python scripts/test_set/score_test.py --template ~/work/my_forecasts        # the exact rows to fill in
#   fill `forecast`; train each fold's model only on data up to its first_origin;
#   a forecast for origin t may use history/<fold>/<level>.csv.gz rows with date <= t only
python scripts/test_set/score_test.py --forecasts ~/work/my_forecasts \
    --reference ~/work/eval/runs/<run>/forecasts/persistence30_ref
```

Files in `~/work/test_set/<version>/`: `manifest.json` (hash of the spec, counts, file hashes), `districts.csv`,
`panel/` (customer keys: stays in the analytical environment), `actuals/<level>.csv[.gz]`,
`history/<fold>/<level>.csv.gz`.

## Rules

- A released version is never edited or rebuilt: change it by copying the spec to a new `version`.
- The scorer refuses forecasts made for a different spec hash, missing or duplicated rows, non-finite values, and
  bottom-up forecasts summed over a different set of observed households than the actuals.
- Limitations are printed in every report: network mapping is last-known and retrospective; feeder / substation cover
  households with a complete network path only (~86 %); blocks 2-4 overlap the dev-test window seen during
  development, so this is the internal benchmark, not the customer-approved acceptance sample; no tariff-group slice.
- The holiday dates in the spec must be checked against the official 2025-2026 decrees.
