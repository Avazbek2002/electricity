# electricity

Household electricity forecasting for Uzbekistan (national enriched dataset v1.4): models, analysis scripts and a
personal hidden-test harness that scores every model the same way.

## Layout

```
evaluate.py            one command: fit a model, forecast every household from the same origins, score, log
configs/eval.yaml      data zones, modes (validate / dev-test / holdout), headline metric, budget, MLflow
harness/               the evaluation harness
  contract.py            the Model interface (fit / load / predict) and the Panel predict receives
  data.py                reads one district (developer zone + vault) into a customer x day matrix
  runner.py              per district: as-of panels per origin -> predict -> node sums (worker processes)
  scoring.py             system WAPE per level, household MAE, headline
  tracking.py            git guard, vault look budget, MLflow + local CSVs
  registry.py            model registry: pyfunc wrapper, versions + aliases, list / promote
  testset.py             the fixed test set: definition checks, panels, node sums, scoring, report
  test_mode.py           evaluate.py --mode test (per-fold fits, daily origins, forecast CSVs)
  cli.py                 the evaluate.py command line
models/<name>/         one folder per model: model.py (class Model), config.yaml, NOTES.md
  nlinear_x/             NLinear with weather inputs (trains through core/household_nlinear.py)
  persistence30/         baseline: mean of the last 30 observed days
  persistence7/          baseline: mean of the last 7 observed days
core/                  shared code (imported by everything else)
  whole_pipeline_v8.py   features, weather, CatBoost / XGBoost household pipeline
  enriched_io.py         bounded reads of the enriched dataset
  mlflow_utils.py        experiment tracking for every entry point (Tracker, run_script)
  household_nlinear.py   NLinear training on every customer (also a CLI)
  node_syswape.py        bottom-up system WAPE per transformer / feeder / substation (also a CLI)
  transformer_forecast.py, process_raw.py   transformer-level pipeline, raw-data processing
scripts/analysis/      EDA and checks (hierarchy, feeders, autocorrelation, ...)
scripts/transformer/   transformer-level experiments
scripts/mlflow_check.py  end-to-end check of an MLflow server
scripts/test_set/      build_test_set.py (once), score_test.py (the common scorer)
docs/TEST_SET.md       the fixed test set: design, scores, how to submit
tests/                 harness and tracking tests (a probe model checks what predict can see)
```

## The evaluation harness

```bash
source ~/work/ec_env.sh
python evaluate.py --model nlinear_x --mode validate        # tuning and model choice: unlimited
python evaluate.py --model nlinear_x --mode dev-test        # sanity check on Nov 2025 - Apr 2026 (already seen)
python evaluate.py --model nlinear_x --mode holdout --register   # the honest score: clean git tree, budgeted
python evaluate.py --model persistence30 --mode dev-test --coatos 03202 --allow-dirty   # quick check, one district
```

| Mode | Training cutoff | Origins (every household, same dates) | Looks |
|---|---|---|---|
| `validate` | 2025-01-31 | 2025-04-01 .. 2025-06-30, every 14 days | unlimited |
| `dev-test` | 2025-09-01 | 2025-10-31 .. 2026-04-30, every 14 days | unlimited, logged |
| `holdout` | 2026-04-30 | the vault (next data release), every 7 days | 1 per commit, 2 per month |
| `test` | four seasonal folds | the **fixed test set** (180 days, daily origins): see [docs/TEST_SET.md](docs/TEST_SET.md) | logged |

What a run does:
1. **Guard**: records the git commit; holdout refuses uncommitted changes and checks the look budget (`runs.csv`).
2. **Fit** (main process) on the developer zone up to the cutoff + 60 days; cached per model, commit and cutoff.
3. **Predict** (worker per district): for each origin, the households passing the input rules get a `Panel` with
   history up to the origin and the forecast issued that day. S3 keys are removed from the worker before the model runs.
4. **Score**: forecasts and actuals summed per transformer, feeder, substation, district and the country (system WAPE),
   household MAE; a 30-day persistence reference is scored on the same households and days.
5. **Log**: run folder (`summary.csv`, `by_size.csv`, `mae.csv`, `nodes.csv`, `run.json`), `runs.csv`,
   `leaderboard.csv`, and MLflow.

Headline metric (fixed in `configs/eval.yaml`): feeder system WAPE, pooled, mean of 7 and 30 days.

### The fixed test set

All models are compared on one frozen test set ([docs/TEST_SET.md](docs/TEST_SET.md), `configs/test_set.yaml`): four
45-day seasonal blocks (summer, autumn, winter, spring; one fit each), daily origins, a fixed household panel per
fold, coverage-adjusted node actuals, and one scorer (`scripts/test_set/score_test.py`) that also checks the ТЗ
thresholds and runs the bias audit.
```bash
python scripts/test_set/build_test_set.py --workers 16          # once per test-set version
python evaluate.py --model gam --mode test --workers 16           # harness models
python scripts/test_set/score_test.py --template ~/work/my_fc     # other models: rows to fill, then --forecasts
```

### Adding a model

```python
# models/my_model/model.py
from harness.contract import BaseModel, HORIZON

class Model(BaseModel):
    def fit(self, spec, artifacts_dir):      # spec.roots, spec.cutoff, spec.data_end, spec.val_start/val_end
        ...                                   # train on the developer zone only; save to artifacts_dir
    def load(self, artifacts_dir):
        ...
    def predict(self, panel):                 # panel.kwh (n, days up to origin), panel.weather, panel.pipeline, ...
        return forecasts                      # (panel.n, 60) kWh
```
Add `config.yaml` (`family:` plus hyperparameters) and `NOTES.md`, commit, run `validate`.

### The vault (holdout)

The data owner builds the next enriched release into a separate prefix with its own read key. Put the key in
`~/work/vault_env.sh` (`VAULT_AWS_ACCESS_KEY_ID`, `VAULT_AWS_SECRET_ACCESS_KEY`, `VAULT_S3_ENDPOINT`), never in
`ec_env.sh`, and set `data.vault.uri` and `data.vault.end` in `configs/eval.yaml`. Only `evaluate.py --mode holdout`
loads it.

### MLflow

Every entry point records its runs through one module, `core/mlflow_utils.py`. It never stops a job: if the server
is unreachable it prints one warning and the job runs untracked.

**Setup (once):**
```bash
cp configs/mlflow_env.example.sh ~/work/mlflow_env.sh && chmod 600 ~/work/mlflow_env.sh   # fill in user + password
source ~/work/mlflow_env.sh
python scripts/mlflow_check.py        # credentials, runs, artifact upload, model logging, registry, load-back
```
Credentials only in the environment (`MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`), never in the repo or
the URI; they are masked in every log and tag. Without `MLFLOW_TRACKING_URI`, evaluate.py uses a local SQLite store
(`~/work/eval/mlflow.db`) and the other scripts run untracked. `--no-mlflow` on any script turns it off;
`MLFLOW_DISABLE=1` turns it off everywhere.

| Experiment | Runs | Written by |
|---|---|---|
| `electricity/eval` | parent `<model> @ <commit>`, children `fit` (fitted files, pyfunc) and `<mode>` (scores) | evaluate.py |
| `electricity/training` | CatBoost / XGBoost pipeline; household NLinear training | core/whole_pipeline_v8.py, core/household_nlinear.py |
| `electricity/transformer-forecast` | transformer-level forecasts and NLinear | core/transformer_forecast.py, scripts/transformer/transformer_nlinear.py |
| `electricity/data-prep` | data builds, transformer panels and audits | core/process_raw.py, scripts/transformer/* |
| `electricity/analysis` | EDA and checks, node-level system WAPE | scripts/analysis/*, core/node_syswape.py |
| `electricity/monitoring` | live accuracy of the production model | core/whole_pipeline_v8.py monitor |

- Each script run records the command line (params), `git_commit` / `dirty` / host / conda env (tags), the
  printed log, and the files in its `--out-dir` (tables, figures, reports) as artifacts. Training scripts also log
  their result table as metrics (`<model>/mae_h7`, `<level>/<model>/pooled_h30`, ...).
- evaluate.py metrics: `<level>/syswape_h<h>`, `<level>/syswape_median_h<h>`, `<level>/bias_h<h>`,
  `household/mae_h<h>`, `headline`; the persistence reference under `ref/...`. Per-node results are a CSV artifact.
- **Data rule:** a table whose header has a customer key (`customer_key`, `series_key`, `meter_id`, ...) is never
  uploaded; rows of node tables with fewer than 10 households are dropped from the uploaded copy.
- Leaderboard: in `electricity/eval`, filter `tags.mode = "validate"` and sort by `headline`.

### Model registry

```bash
python evaluate.py --model gam --mode validate --register        # every mode can register
python -m harness.registry list                                   # versions, headlines, aliases
python -m harness.registry promote --model gam --version 3 --to-production
```

| Registered model | Versions | Aliases |
|---|---|---|
| `electricity_household_<model>` (one per models/ folder) | fitted model as a loadable pyfunc (its code travels with it) + evaluation evidence as version tags (`mode`, `cutoff`, `origins`, `headline`, `national/syswape_h<h>`, commit, run ids) | `best_validate` / `best_dev_test`: lowest headline among full runs on the same dates and data (moved automatically; quick `--coatos` runs never take it); `challenger`: latest holdout version; `champion`: set by `promote` only |
| `electricity_household_forecast` | the production line: whole_pipeline_v8 serving model, or a harness version copied by `promote --to-production` | `challenger`, `champion` |

`promote` requires the version to hold `challenger` (scored on the holdout), unless `--force`. Load any version:
`mlflow.pyfunc.load_model("models:/electricity_household_gam@champion").predict(panel)` returns `(n, 60)` kWh.

## Other entry points

```bash
python core/household_nlinear.py --out-dir ~/work/hh_nlinear --workers 4 --cache-dir ~/work/hh_nlinear_cache
python core/node_syswape.py --weights-dir ~/work/hh_nlinear --out-dir ~/work/hh_nodes --workers 4
python core/whole_pipeline_v8.py train
python scripts/analysis/hierarchy_check.py --out-dir ~/work/hierarchy --workers 12
python scripts/analysis/feature_autocorr.py --out-dir ~/work/feature_autocorr --buckets 0 --workers 8
```

## Setup

```bash
pip install -r requirements.txt
EVAL_TEST_CONFIG=<config pointing at a synthetic copy> python -m pytest tests -q
```

`core/whole_pipeline_v8.py` logs its serving model only when `core/electricity_pyfunc.py` is present; that file is not
in the repository yet, so until it is added the pipeline tracks its runs but registers no serving model.

Data rules (from the analytical datasets guide): credentials only from the environment; customer keys stay inside the
authorised environment; only aggregates of at least 10 customers leave it; never read the whole `_staging` parent.
