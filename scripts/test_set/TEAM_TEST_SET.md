# Team test set: household electricity forecasting

This is one frozen evaluation set for the whole team. Every model is scored on the same district targets, issue dates,
horizons and rules, against the same B0, by the same scorer. Results from different people can then go into one table.
It implements the mandatory part of `ML_EXPERIMENTS_GUIDE_RU.md` (2026-10-22): the target in part 2 and the split and
evaluation in part 4.

Location: `s3://phase1/electricity/_staging/test-sets/<release>/`. The exact release name, spec hashes and counts are
in `release.json`.

## What is in it

| Path | Content |
|---|---|
| `standard/` | The guide's split: validation = issue dates 2024 (targets up to 2024-12-31), test = issue dates 2025. **Use this for the official report.** |
| `s25/` | Supplementary split: validation = issue dates 2025-01..06 (targets up to 2025-06-30), test = 2025-07..12. Both halves lie in the new data regime (decimal readings from 2024-08), so they are more representative of production. Report it next to the standard result, never instead of it. |
| `<set>/spec.yaml` | The definition of the set: dates, refits, target rules, thresholds. Do not edit it: the scorer checks its hash. |
| `<set>/targets/district.csv.gz` | One row per district and day, 2022-01-01 .. 2026-06-29. Columns: customer counts by day state, kWh sums, **T2** (the target), T3 (experimental), `coverage`, `flag`, `flag_reason`. |
| `<set>/districts.csv` | District → region, customer count, clean days. |
| `<set>/calendar.csv` | The standing calendar, known in advance: holidays, days off, Ramadan. |
| `reference/B0_<set>.parquet` | B0 for every district, issue date and horizon of the set. |
| `weather/issue_dates.csv` | The issue dates that have an AIFS / FCN3 reforecast (the 1st, 9th, 17th and 25th of each month), and their split in each set. |
| `scorer/score.py` | The common scorer. It needs `numpy`, `pandas`, `pyyaml` and `pyarrow`. |
| `SHA256SUMS`, `release.json` | Checksums and the release record. |

Only district-day aggregates are included. A district-day with fewer than 10 customers has its kWh blanked and is
flagged.

## The rules (all sets)

| Rule | Value |
|---|---|
| Forecast | Issued on day **o** for the target days d = o + h, **h = 1..60**. |
| History | On issue date o you may use data up to **o − 1** only: consumption, weather, flags, everything. |
| Refits | A model is fitted at each refit date R (`refits` in `spec.yaml`: quarterly). It trains only on pairs whose **target day d ≤ R − 1**: filter by target day, not issue date. It forecasts the issue dates [R, next R). |
| Target | District **T2** = n_rows × (kWh observed + estimated) / (n observed + estimated). |
| Days not scored | Flagged days are not scored and are not training targets. Flags: crisis Jan–Feb 2023, data interruption Aug 2024, coverage < 50 %, < 100 known client-days, open intervals at the end of history, no data. |
| National / region | The sum over districts for one (o, h), kept when ≥ 98 % of the submission's districts are present and clean. |
| Same rows | A district row missing for any compared model is dropped for all of them, including B0, and the report lists how many. |
| Metric | sysMAPE_h = Σ\|P − A\| / ΣA and bias = Σ(P − A) / ΣA at h = 1, 7, 14, 30, 60 (every h in `scores.csv`). |
| Requirements (ТЗ) | National **test**: h1 ≤ 8 %, h7 ≤ 11 %, h14 ≤ 15 %, h30 ≤ 18 %, h60 ≤ 20 %. Each region and each season / day-type group must stay ≤ 1.5 × the national error. |
| Model choice | Choose on **validation**. Look at the test only for the variants you chose. |
| Calendar | Use `calendar.csv` (known in advance), not the realised holiday dates. |

**Leakage check (required, guide part 5).** Randomise every input after a day T (consumption, flags, weather) and rerun.
Forecasts issued on or before T + 1 must be bit-identical; changes may start from o = T + 2. A result that looks too
good is a leak until proven otherwise (for example, h1 below 3–4 % on the 5-district sum).

## The weather track

The standard run uses **no target-day weather**: only weather up to o − 1, and climatology built from data before o.

To use weather forecasts:
- Use only the AIFS / FCN3 forecast issued on o:
  `s3://data-weather/reforecast/<set>/data/model=<aifs|fcn3>/init_date=<o>/part-0.parquet`.
  The relevant sets are `wx-val2024-20261021t120926z` and `wx-test2025-20261021t120926z`. **Never read `wx-reserve2026-*`.**
- After the forecast ends (16 days), use climatology. The actual weather of the target day is allowed only as an
  oracle ceiling, clearly labelled.
- The forecasts exist only on `weather/issue_dates.csv`. Score with `--track weather`, which keeps only those issue
  dates, and compare against your no-weather model on the same track.

Reforecast files are in long format: `valid_time_utc`, `lead_hours`, `coato_id` (= the district code), `variable`
(`t2m` in °C; AIFS also has `d2m`, `u10m`, `v10m`, `tp06`, `tcc`), `value`, in 6-hour steps. Tashkent is UTC+5, so
local day D = the 00, 06, 12 and 18 UTC steps of D.

## Quick start

Copy the release to your workspace once, from Jupyter on the platform:

```python
import os, pyarrow.fs as pafs
ep = os.environ["S3_ENDPOINT"]; scheme, host = ep.split("://", 1) if "://" in ep else ("https", ep)
s3 = pafs.S3FileSystem(access_key=os.environ["AWS_ACCESS_KEY_ID"], secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                       endpoint_override=host, scheme=scheme)
REL = "<release>"                                    # e.g. team-testset-2026-10-23-v1
pafs.copy_files(f"phase1/electricity/_staging/test-sets/{REL}", os.path.expanduser(f"~/work/test_set/{REL}"),
                source_filesystem=s3, destination_filesystem=pafs.LocalFileSystem())
```

Check the files (`cd ~/work/test_set/<release> && sha256sum -c SHA256SUMS`), then:

```bash
cd ~/work/test_set/<release>
python scorer/score.py --set standard --template ~/work/my_model --sample   # rows to fill: 5 districts
python scorer/score.py --set standard --template ~/work/my_model            # every district (national)
python scorer/score.py --set s25 --track weather --template ~/work/my_wx     # weather track rows
```

Fill the `forecast` column of `district.csv.gz` (district T2 in kWh). Edit `forecast_meta.json`, keeping
`test_set_version` and `test_set_sha256` as written:

```json
{"model": "lgbm_s1h_v3", "author": "name", "test_set_version": "...", "test_set_sha256": "...",
 "target": "T2", "weather": "none | aifs | fcn3 | ens", "level": "district | client",
 "strategy": "direct | recursive | hybrid", "crisis_2023": "a | b", "commit": "..."}
```

Then score:

```bash
python scorer/score.py --set standard --forecasts ~/work/my_model
python scorer/score.py --set standard --forecasts ~/work/my_model --compare gam=~/work/gam_run   # several models
python scorer/score.py --set s25 --track weather --forecasts ~/work/my_wx --compare base=~/work/my_model
```

The scores go to `<forecasts>/scores_<set>_<track>/`: `report.txt`, `scores.csv` (every horizon, national and district),
`spec_check.csv` (ТЗ), `bias_audit.csv` and `national_daily.csv`.

**Client-level models.** Forecast each customer, then sum to the district and submit at district level. Customers
without fresh history get the district mean per customer (guide part 3). Never submit customer rows.

**District-level models.** You may use `targets/district.csv.gz` rows with `date ≤ o − 1` as history. Remember the flags.

**A sanity check.** Submit B0 itself (`reference/B0_standard.parquet` as `node, origin, horizon, forecast`). It must
score exactly like the B0 the scorer computes.

## Reporting (guide part 7)

Report one entry per experiment:
- name, author, date, commit or notebook;
- level, target, districts (5-district sample or national), set (`standard`, or `s25` next to it) and track;
- crisis-2023 variant (a / b), weather, calendar, strategy, the model and how many configurations you tried;
- the `report.txt` table: model and B0 on the same days, validation and test, sysMAPE (bias), day counts;
- whether the substitution test passed, and 1–3 lines on what you learned.

The test rows only go in the report for the variants chosen on validation.

## Versioning

A published release is never changed or rebuilt. A fix or a new data drop becomes a new release name, and results from
different releases are not mixed in one table. The scorer refuses a submission made for a different `spec.yaml` (hash),
and refuses duplicated rows.
