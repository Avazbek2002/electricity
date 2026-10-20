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
  cli.py                 the evaluate.py command line
models/<name>/         one folder per model: model.py (class Model), config.yaml, NOTES.md
  nlinear_x/             NLinear with weather inputs (trains through core/household_nlinear.py)
  persistence30/         baseline: mean of the last 30 observed days
  persistence7/          baseline: mean of the last 7 observed days
core/                  shared code (imported by everything else)
  whole_pipeline_v8.py   features, weather, CatBoost / XGBoost household pipeline
  enriched_io.py         bounded reads of the enriched dataset
  household_nlinear.py   NLinear training on every customer (also a CLI)
  node_syswape.py        bottom-up system WAPE per transformer / feeder / substation (also a CLI)
  transformer_forecast.py, process_raw.py   transformer-level pipeline, raw-data processing
scripts/analysis/      EDA and checks (hierarchy, feeders, autocorrelation, ...)
scripts/transformer/   transformer-level experiments
tests/                 harness tests (a probe model checks what predict can see)
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

| Experiment | Contents | Written by |
|---|---|---|
| `electricity/eval` | every evaluate.py run: parent run `<model> @ <commit>`, child runs `fit` and `<mode>` | evaluate.py only |
| `electricity/training` | CatBoost / XGBoost pipeline runs | core/whole_pipeline_v8.py |
| `electricity/data-prep` | data builds | core/process_raw.py |
| `electricity/monitoring` | production checks | core/whole_pipeline_v8.py |

- Tags: `model_family`, `model`, `git_commit`, `dirty`, `mode`, `cutoff`, `holdout_period`, `data_version`.
- Metrics: `<level>/syswape_h<h>`, `<level>/syswape_median_h<h>`, `<level>/bias_h<h>`, `household/mae_h<h>`,
  `headline`; the persistence reference under `ref/...`. Per-node results are a CSV artifact, never metrics.
- `--register` (holdout only) registers the fitted model as `electricity_household_forecast` with alias `challenger`;
  promotion to `champion` stays with the gate in core/whole_pipeline_v8.py.
- Tracking server: `MLFLOW_TRACKING_URI`; without it, a local SQLite store in `~/work/eval/mlflow.db`.
- Leaderboard: filter `tags.mode = "validate"` in the `electricity/eval` experiment and sort by `headline`.

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

`core/whole_pipeline_v8.py` and `core/process_raw.py` import `mlflow_utils.py` and `electricity_pyfunc.py` when present;
those two files are not in the repository yet; put them in `core/`.

Data rules (from the analytical datasets guide): credentials only from the environment; customer keys stay inside the
authorised environment; only aggregates of at least 10 customers leave it; never read the whole `_staging` parent.
