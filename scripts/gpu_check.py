#!/usr/bin/env python
"""
Can the v8 boosters train on this GPU, and how much faster is it? Synthetic data shaped like the real fit
(rows x 112 features incl. 3 categorical, 60 outputs, MultiRMSE / multi_output_tree, depth 4, l2 30).

    python scripts/gpu_check.py                    # 200k rows, 100 trees each, CPU vs GPU
    python scripts/gpu_check.py --rows 800000 --trees 200

No customer data is read. Prints per-tree seconds and the speed-up, or the error if a library refuses the GPU.
"""
import argparse
import time

import numpy as np
import pandas as pd


def data(n, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 109)).astype("float32"), columns=[f"f{i}" for i in range(109)])
    for c, k in (("coato_code", 222), ("housing_type_id", 6), ("segment", 3)):
        X[c] = pd.Categorical(rng.integers(0, k, n).astype(str))
    W = rng.normal(size=(109, 60)).astype("float32") * 0.1
    y = X.iloc[:, :109].to_numpy() @ W + rng.normal(size=(n, 60)).astype("float32")
    return X, y


def run_catboost(X, y, trees, gpu):
    from catboost import CatBoostRegressor
    p = dict(iterations=trees, learning_rate=0.15, depth=4, l2_leaf_reg=30, loss_function="MultiRMSE", verbose=0,
             random_seed=42)
    if gpu:
        p.update(task_type="GPU", devices="0")
    m = CatBoostRegressor(cat_features=["coato_code", "housing_type_id", "segment"], **p)
    t = time.time()
    m.fit(X, y)
    return (time.time() - t) / trees, m


def run_xgboost(X, y, trees, gpu):
    import xgboost as xgb
    m = xgb.XGBRegressor(n_estimators=trees, learning_rate=0.15, max_depth=4, reg_lambda=30.0, tree_method="hist",
                         multi_strategy="multi_output_tree", enable_categorical=True, random_state=42,
                         device="cuda" if gpu else "cpu")
    import warnings
    t = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.fit(X, y)
    if gpu and any("GPU" in str(x.message) and ("No visible" in str(x.message) or "changed from GPU" in str(x.message))
                   for x in w):
        raise RuntimeError("XGBoost found no usable GPU and fell back to the CPU")
    return (time.time() - t) / trees, m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=int, default=200_000)
    p.add_argument("--trees", type=int, default=100)
    p.add_argument("--skip-cpu", action="store_true", help="GPU only (the CPU timing is the slow part)")
    a = p.parse_args()
    X, y = data(a.rows)
    print(f"[data] {a.rows:,} rows x {X.shape[1]} features (3 categorical), 60 outputs; {a.trees} trees per run")
    for name, fn in (("catboost", run_catboost), ("xgboost", run_xgboost)):
        res = {}
        for gpu in ([True] if a.skip_cpu else [False, True]):
            dev = "GPU" if gpu else "CPU"
            try:
                sec, m = fn(X, y, a.trees, gpu)
                pred = m.predict(X.iloc[:1000])
                res[dev] = sec
                print(f"[{name}] {dev}: {sec:.3f} s a tree | predict ok, shape {np.shape(pred)}")
            except Exception as e:                                     # noqa: BLE001
                print(f"[{name}] {dev}: FAILED -> {type(e).__name__}: {str(e)[:300]}")
        if "CPU" in res and "GPU" in res:
            print(f"[{name}] speed-up {res['CPU'] / res['GPU']:.1f}x")


if __name__ == "__main__":
    main()
