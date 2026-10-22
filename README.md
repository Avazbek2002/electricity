# electricity

Household electricity forecasting for Uzbekistan (national enriched dataset v1.4): models, analysis scripts and a
personal hidden-test harness that scores every model the same way.

## Layout

```
evaluate.py            one command: fit a model, forecast every household from the same origins, score, log
configs/test_set.yaml  the ML guide's standard split and evaluation (validate / test)
configs/eval.yaml      data zones, the holdout (vault) mode, budget, MLflow
harness/               the evaluation harness
  contract.py            the Model interface (fit / load / predict) and the Panel predict receives
  data.py                reads one district (developer zone + vault) into a customer x day matrix
  runner.py              per district: as-of panels per origin -> predict -> node sums (worker processes)
  scoring.py             system WAPE per level, household MAE, headline
  tracking.py            git guard, vault look budget, MLflow + local CSVs
  registry.py            model registry: pyfunc wrapper, versions + aliases, list / promote
  testset.py             the standard evaluation: T2 target, B0, calendar, common panel, scoring, report
  test_mode.py           evaluate.py --mode validate / test (refits, issue dates, district forecasts, substitution test)
  plots.py               charts of a run: metrics over every horizon, window samples (logged to MLflow)
  cli.py                 the evaluate.py command line
models/<name>/         one folder per model: model.py (class Model), config.yaml, NOTES.md
  catboost/, xgboost/    Baseline + CatBoost / XGBoost from whole_pipeline_v8 (one shared fit; core/boost_harness.py)
  catboost_fc/, xgboost_fc/  the same with the simulated 10-day forecast (--weather simulated, preliminary)
  gam/                   GAM (EBM): every horizon 1..60, no day-of-year input
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
scripts/analysis/      EDA and checks (hierarchy, feeders, autocorrelation, bottom-up oracle, ...)
scripts/transformer/   transformer-level experiments
scripts/mlflow_check.py  end-to-end check of an MLflow server
scripts/test_set/      build_test_set.py (once), score_test.py (the common scorer)
docs/TEST_SET.md       the standard split and evaluation: rules, how to run and submit
tests/                 harness and tracking tests (a probe model checks what predict can see)
```

## The evaluation harness

Every model is scored on the ML guide's standard split ([docs/TEST_SET.md](docs/TEST_SET.md), `configs/test_set.yaml`):
history up to o - 1, validation = issue dates 2024, test = issue dates 2025, quarterly refits (targets <= R - 1),
district target T2 summed to the national total, B0 on the same rows, no target-day weather, substitution test.

```bash
source ~/work/ec_env.sh                                            # S3 + MLflow settings
python scripts/test_set/build_test_set.py --sample --workers 5     # once: district targets (also without --sample)
python evaluate.py --model gam --mode validate --sample            # experiments: the guide's 5 districts
python evaluate.py --model gam --mode validate --workers 16        # national
python evaluate.py --model gam --mode test --workers 16            # chosen variants only: validation + test
python evaluate.py --model gam --mode holdout                      # the vault (next data release); budgeted
python scripts/test_set/score_test.py --template ~/work/my_fc --sample   # other models: rows to fill, then --forecasts
```

What a run does:
1. **Guard**: records the git commit; refuses uncommitted changes unless `--allow-dirty` (holdout: never).
2. **Fit** once per refit date (cached per model, commit and refit): `spec.data_end = R - 1`.
3. **Substitution test** on one district: inputs after T randomised; forecasts issued up to T + 1 must not change.
4. **Predict** (worker per district): for every issue date, the households with fresh history get a `Panel` with
   data up to o - 1; the district forecast = customers on o - 1 x their mean forecast. S3 keys are removed first.
5. **Score** with the common scorer: national and district sysMAPE / bias, B0 on the same days, day counts, the ТЗ
   thresholds, bias audit; supplementary substation / feeder sums.
6. **Log**: run folder (`forecasts/`, `scores/`), `runs.csv`, `test_leaderboard.csv`, MLflow.

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

### Weights & Biases (experiment tracking)

Every `evaluate.py --mode validate|test` run is tracked in W&B (`harness/wandb_log.py`, settings `wandb:` in
`configs/eval.yaml`): sysWAPE, WAPE, MAE, bias over every horizon 1..60 with x = `horizon` (model and B0), every level
(household .. national), window-sample charts (national actual vs model and B0, observed daily max / min temperature
below), score tables, run config (model settings, test set, commit, scope, weather). Runs of different models overlay
in the project workspace; `scripts/compare_models.py --runs <run> <run> ...` puts several models on the same days in
one run (one line per model, window charts with every model).

```bash
pip install wandb plotly && wandb login          # or WANDB_API_KEY in ~/work/ec_env.sh (never in the repo)
python scripts/compare_models.py --runs ~/work/eval/runs/<a> ~/work/eval/runs/<b> --name "sample validation"
```
`--no-wandb` or `WANDB_MODE=disabled` turns it off; `WANDB_MODE=offline` keeps it local (`wandb sync` later);
`WANDB_BASE_URL` points at a self-hosted server. Only district / national aggregates are sent (data rule: aggregates
of >= 10 customers) -- check that the W&B host is approved for them.

### MLflow (model registry)

MLflow is used for the model registry: `evaluate.py ... --register` logs the fitted model as a pyfunc and registers it;
without `--register` an evaluation does not touch MLflow. The other scripts below still record their runs in MLflow.

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
