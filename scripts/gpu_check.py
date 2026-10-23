#!/usr/bin/env python
"""
Check the GPU pod before a long run: is there a GPU, and do our exact booster settings train on it?

    python scripts/gpu_check.py

Trains on random data shaped like ours (60 targets, categorical columns) with v8's CatBoost (MultiRMSE, categorical
features) and XGBoost (hist, multi_output_tree, categorical) settings, once on CPU and once on GPU, and prints the
time per tree. Exit code 0 = both boosters run on the GPU.
"""
import subprocess
import sys
import time

import numpy as np
import pandas as pd


def data(n=60000, p=60, k=60, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    for c in ("coato_code", "housing_type_id", "segment"):
        X[c] = pd.Categorical(rng.integers(0, 30, n).astype(str))
    Y = X[[f"f{i}" for i in range(k)]].to_numpy() * 0.3 + rng.normal(size=(n, k))
    return X, Y


def catboost_run(X, Y, gpu, iters=50):
    from catboost import CatBoostRegressor
    kw = dict(iterations=iters, learning_rate=0.15, depth=4, l2_leaf_reg=30, loss_function="MultiRMSE",
              eval_metric="MultiRMSE", verbose=0, random_seed=42)
    if gpu:
        kw.update(task_type="GPU", devices="0")
    Xc = X.copy()
    cats = ["coato_code", "housing_type_id", "segment"]
    for c in cats:
        Xc[c] = Xc[c].astype(str)
    t0 = time.time()
    CatBoostRegressor(cat_features=cats, **kw).fit(Xc, Y)
    return (time.time() - t0) / iters


def xgboost_run(X, Y, gpu, iters=50):
    import xgboost as xgb
    kw = dict(n_estimators=iters, learning_rate=0.15, max_depth=4, reg_lambda=30.0, tree_method="hist",
              multi_strategy="multi_output_tree", enable_categorical=True, random_state=42)
    if gpu:
        kw["device"] = "cuda"
    t0 = time.time()
    m = xgb.XGBRegressor(**kw).fit(X, Y)
    dt = (time.time() - t0) / iters
    if gpu:                                       # trained on GPU, predicted on CPU as the harness does
        m.set_params(device="cpu")
        m.predict(X.head(100))
    return dt


def main():
    import catboost
    import xgboost
    print(f"xgboost {xgboost.__version__}, catboost {catboost.__version__}")
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True)
    except FileNotFoundError:
        r = None
    if r is None or r.returncode:
        print("no GPU visible (nvidia-smi failed): this pod cannot train on GPU")
        return 1
    print("GPU:", r.stdout.strip())
    X, Y = data()
    ok = True
    for name, fn in (("CatBoost MultiRMSE", catboost_run), ("XGBoost multi_output_tree", xgboost_run)):
        cpu = fn(X, Y, gpu=False)
        try:
            gpu = fn(X, Y, gpu=True)
            print(f"{name:28s} CPU {cpu * 1000:7.1f} ms/tree | GPU {gpu * 1000:7.1f} ms/tree | x{cpu / gpu:5.1f}")
        except Exception as e:                    # noqa: BLE001
            ok = False
            print(f"{name:28s} CPU {cpu * 1000:7.1f} ms/tree | GPU FAILED: {type(e).__name__}: {str(e)[:200]}")
    print("both boosters train on the GPU" if ok else "at least one booster cannot use these settings on the GPU")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
