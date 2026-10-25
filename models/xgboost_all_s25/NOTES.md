# xgboost_all: the xgboost model trained on every customer

Same features, settings and refits as `models/xgboost`; only the training data changes:

| | xgboost | xgboost_all | xgboost_all_s25 |
|---|---|---|---|
| customers | buckets 0-4 (~5 %) | all | all |
| windows kept | a stable 20 % | all | a stable 25 % |
| share of all trainable windows | ~1 % | 100 % | 25 % |
| CatBoost in the shared fit | yes | no (`TRAIN_CATBOOST: false`) | no |

Split: unchanged (substation spec `configs/test_set_substations_2025.yaml`): fits at 2025-01/04/07/10-01 on targets
before the refit date; scored on the 20 dev (validate) or 40 test (test) substations. Households under those substations
are in training like every other customer, with their history before each refit only.

Memory: about 100x the windows of `xgboost` (it had ~260k at the last refit -> ~25M). Check `free -g` and the
`kin + ktg would need ... GB` line; if it does not fit, run `xgboost_all_s25`.
