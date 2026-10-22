# The standard split and evaluation (`ts-2026-10-22-v2`)

Implements the mandatory part of `ML_EXPERIMENTS_GUIDE_RU.md` (2026-10-22, part 4) and its target (part 2), so results
of every developer and model are comparable. Definition: [`configs/test_set.yaml`](../configs/test_set.yaml).
Implementation: [`harness/testset.py`](../harness/testset.py) (shared by the builder, the harness and the scorer).
Replaces v1 (seasonal blocks, panel actuals), which never left trial and did not match the guide.

## The rules

| | Value | Guide |
|---|---|---|
| Issue date / horizon | issue date o, target d = o + h, h = 1..60 | part 1 |
| History | data known up to **o - 1** | part 4 |
| Validation | issue dates 2024-01-01 .. 2024-12-31, targets d <= 2024-12-31 | part 4 |
| Test | issue dates 2025-01-01 .. 2025-12-31, every target | part 4 |
| Reserve | 2026: not touched | part 4 |
| Refits | 2024-01-01, 04-01, 07-01, 10-01, 2025-01-01, 04-01, 07-01, 10-01; model fitted at R trains on targets **d <= R - 1** and forecasts the issue dates [R, next R) | part 4 |
| Target | district **T2** = n_rows x (kWh observed + estimated) / (n observed + estimated) | part 2 |
| Days not used | < 100 known client-days, open intervals at the end (`is_right_edge`), no data -> no target; crisis (Jan-Feb 2023), data interruption (Aug 2024), coverage < 50 % -> flagged: not a training target, not evaluated | parts 2, 4 |
| National / region | sum of district values for an (o, h) with >= 98 % of the run's districts clean and forecast | part 4 |
| Same days | a district row missing for any compared model is dropped for all (reported) | parts 4, 5 |
| Reference | **B0** = 28-day level at o - 1 x last year's seasonal ratio (else two years back, else 1), always on the same rows | part 6 |
| Weather | standard run: no target-day weather; climatology from data before the issue date | parts 2, 5 |
| Metric | sysMAPE_h = sum abs(P - A) / sum A and bias = sum (P - A) / sum A at h 1, 7, 14, 30, 60 | parts 1, 4 |
| Leakage check | substitution test: randomise every input after T; forecasts issued <= T + 1 must not change | part 5 |
| Sample | 03203, 10240, 22202, 26266, 33220 (sum of 5 districts) for experiments; national runs for the winners | part 2 |

Also reported: the ТЗ 4.2.4 thresholds on the test national sum (h1 8 %, h7 11 %, h14 15 %, h30 18 %, h60 20 %) and
the bias audit (season, day type from the standing calendar, regions; each <= 1.5 x national). Supplementary, not
part of the standard: bottom-up matched sysMAPE at substation and feeder level (harness runs only).

## Client level (bottom-up) in the harness

For every issue date: the households with fresh history (>= 60 observed of the last 90 days, >= 20 of the last 30,
all up to o - 1) are forecast by the model; the district forecast is
`(customers in the district on o - 1) x (mean forecast of the households with fresh history)` -- the guide's
"customers without fresh history get the district mean per customer". The climate object is rebuilt per issue date
from weather up to o - 1. Existing models: persistence as is; GAM trains with the same one-day gap; NLinear (trained
with history ending on its origin) is shifted by one day -- exact for h1..h59, h60 repeats step 60 (noted in the run).

## How to use it

```bash
# once per version: the district target series, flags, calendar (minutes; district aggregates only)
python scripts/test_set/build_test_set.py --workers 16            # every district
python scripts/test_set/build_test_set.py --sample --workers 5    # the guide's 5 districts

# harness models: choose on validation, then the test for the chosen variants
python evaluate.py --model gam --mode validate --sample --workers 5
python evaluate.py --model gam --mode validate --workers 16 --register         # national
python evaluate.py --model gam --mode test --workers 16 --register             # validation + test rows

# any other model (district level, or client level summed to districts)
python scripts/test_set/score_test.py --template ~/work/my_forecasts --sample  # rows to fill: node, origin, horizon
python scripts/test_set/score_test.py --forecasts ~/work/my_forecasts --compare gam=<run>/forecasts/gam
```

A district-level model may use `targets/district.csv.gz` rows with date <= o - 1 as its history. The report follows
the guide's part 7 table: model and B0 on the same days, validation and test, sysMAPE (bias), day counts, and the
substitution test result.

## Files

`~/work/test_set/<version>[-sample]/`: `manifest.json` (spec hash, data version, per-split clean shares, file
hashes), `districts.csv` (district -> region), `targets/district.csv.gz` (district x day counts, T2, T3, flags),
`calendar.csv` (standing calendar). A run folder: `forecasts/<model>/district.csv.gz` (+ `network/`), `scores/`
(scores.csv, spec_check.csv, bias_audit.csv, national_daily.csv, report.txt).

## Rules for changing it

A released version is never edited or rebuilt: copy the spec to a new `version`. The scorer refuses forecasts made for
a different spec hash and duplicated rows. Report other splits (refit frequency, windows) separately, next to the
standard result, as the guide asks.
