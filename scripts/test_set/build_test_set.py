#!/usr/bin/env python
"""
Build the fixed test set once (configs/test_set.yaml -> ~/work/test_set/<version>/): the panel of every fold, the
coverage-adjusted actuals of every node and test day, and the node history for direct (node-level) models.

    python scripts/test_set/build_test_set.py --workers 16
    python scripts/test_set/build_test_set.py --workers 16 --with-transformer     # also the transformer level

Run it once per test-set version; the harness and the scorer check the manifest's hash of configs/test_set.yaml.
The panel files hold customer keys: they stay in ~/work (never uploaded). Actuals and history hold node totals of
>= min_households panel households only.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
from harness import testset as ts  # noqa: E402
from harness.data import read_district  # noqa: E402
import enriched_io as eio  # noqa: E402  (harness puts core/ on the path)

HISTORY_LEVELS = ("national", "region", "district", "substation", "feeder")


def district_region(code, uri, env_prefix=""):
    fs, root = eio.open_root(uri, env_prefix)
    try:
        dset = eio.district_daily(fs, root, code)
        if "region_code" not in dset.schema.names:
            return "unknown"
        for fr in dset.get_fragments():
            col = fr.to_table(columns=["region_code"]).column(0).drop_null()
            if len(col):
                return str(col[0].as_py()).strip()
    except (FileNotFoundError, OSError):
        pass
    return "unknown"


def build_district(code, cfg):
    t0 = time.time()
    spec = cfg["spec"]
    d = spec["data"]
    D = read_district(code, [dict(uri=d["uri"], env_prefix=d.get("env_prefix", ""), start=d["start"], end=d["end"])],
                      d["start"], d["end"], d["labels"])
    region = district_region(code, d["uri"], d.get("env_prefix", ""))
    out = dict(code=str(code), region=region, customers=int(len(D["keys"])), folds={}, seconds=0.0)
    if not len(D["keys"]):
        return out
    M = D["M"]
    T = M.shape[1]
    import pyarrow as pa
    import pyarrow.parquet as pq
    for f in spec["folds"]:
        test_idx, origins = ts.fold_days(spec, f)
        r = ts.panel_rows(M, int(origins[0]), spec)
        pdir = os.path.join(cfg["out"], "panel", f["name"])
        os.makedirs(pdir, exist_ok=True)
        pq.write_table(pa.table({"customer_key": pa.array(D["keys"][r].astype(str))}), os.path.join(pdir, f"{code}.parquet"))
        Mp = M[r]
        obs = np.isfinite(Mp)
        X = np.where(obs, Mp, 0.0).astype(np.float64)
        G = ts.groups(D["network"][r], code, region, cfg["levels"])
        lv_out = {}
        for lv, (g, names) in G.items():
            K = len(names)
            N = np.bincount(g[g >= 0], minlength=K).astype(float)
            e = dict(names=names, N=N,
                     n_test=ts.bincount2(g, obs[:, test_idx].astype(float), K),
                     S_test=ts.bincount2(g, X[:, test_idx], K))
            if lv in HISTORY_LEVELS:
                last = int(origins[-1])                            # history up to the fold's last origin
                e["n_hist"] = ts.bincount2(g, obs[:, :last + 1].astype(np.float32), K).astype(np.float32)
                e["S_hist"] = ts.bincount2(g, X[:, :last + 1], K).astype(np.float32)
            lv_out[lv] = e
        out["folds"][f["name"]] = dict(panel=int(len(r)), levels=lv_out)
    out["T"] = T
    out["seconds"] = time.time() - t0
    return out


def write_csv(df, path):
    df.to_csv(path, index=False, float_format="%.10g")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", default=os.path.join(REPO, "configs", "test_set.yaml"))
    p.add_argument("--out-root", default="~/work/test_set")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--coatos", default=None, help="comma list: build on a few districts only (a trial, not the test set)")
    p.add_argument("--with-transformer", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    spec = ts.load_spec(a.spec)
    out = os.path.join(os.path.expanduser(a.out_root), spec["version"] + ("-trial" if a.coatos else ""))
    if os.path.exists(os.path.join(out, "manifest.json")) and not a.overwrite:
        raise SystemExit(f"{out} is already built (manifest.json); a released test set is not rebuilt. "
                         f"Use a new version in the spec, or --overwrite for a trial")
    os.makedirs(out, exist_ok=True)
    levels = ts.levels_of(spec, include_optional=a.with_transformer)
    d = spec["data"]
    fs, root = eio.open_root(d["uri"], d.get("env_prefix", ""))
    codes = eio.finished_districts(fs, root, set(a.coatos.split(",")) if a.coatos else None)
    print(f"[build] {spec['version']} ({spec['_sha256'][:12]}) | {len(codes)} districts | levels {levels} | -> {out}")
    cfg = dict(spec=spec, out=out, levels=levels)
    acc = {f["name"]: {lv: ts.NodeSum() for lv in levels} for f in spec["folds"]}
    dist_rows, t0 = [], time.time()
    with cf.ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(build_district, c, cfg): c for c in codes}
        for k, fu in enumerate(cf.as_completed(futs), 1):
            r = fu.result()
            row = dict(district=r["code"], region=r["region"], customers=r["customers"])
            for fname, fr in r["folds"].items():
                row[f"panel_{fname}"] = fr["panel"]
                for lv, e in fr["levels"].items():
                    acc[fname][lv].add(e["names"], {k2: v for k2, v in e.items() if k2 != "names"})
            dist_rows.append(row)
            if k % 10 == 0 or k == len(codes) or k <= 3:
                el = time.time() - t0
                print(f"[build] {k}/{len(codes)} districts, {el / 60:.1f} min, about {el / k * (len(codes) - k) / 60:.1f} min left",
                      flush=True)
    dist = pd.DataFrame(dist_rows).sort_values("district")
    write_csv(dist, os.path.join(out, "districts.csv"))

    # ---- actuals per level (all folds in one file) and history per fold and level
    mn, cov = int(spec["scoring"]["min_households"]), float(spec["scoring"]["min_coverage"])
    start = pd.Timestamp(d["start"])
    counts = {}
    os.makedirs(os.path.join(out, "actuals"), exist_ok=True)
    for lv in levels:
        parts = []
        for f in spec["folds"]:
            ns = acc[f["name"]][lv]
            names = ns.names()
            if not names:
                continue
            N, n, S = ns.stack("N"), ns.stack("n_test"), ns.stack("S_test")
            keep = N >= mn
            test_idx, _ = ts.fold_days(spec, f)
            dates = start + pd.to_timedelta(test_idx, unit="D")
            K, Dn = int(keep.sum()), len(test_idx)
            nk, Sk = n[keep], S[keep]
            with np.errstate(invalid="ignore", divide="ignore"):
                coverage = nk / N[keep][:, None]
                actual = np.where(nk > 0, N[keep][:, None] * Sk / nk, np.nan)
            parts.append(pd.DataFrame({
                "fold": f["name"], "node": np.repeat(np.asarray(names)[keep], Dn), "date": np.tile(dates.date, K),
                "households": np.repeat(N[keep], Dn).astype(int), "observed": nk.ravel().astype(int),
                "coverage": coverage.ravel(), "actual_observed_sum": Sk.ravel(), "actual": actual.ravel(),
                "scored": ((coverage >= cov) & (nk > 0)).ravel()}))
            if lv in HISTORY_LEVELS:
                hn, hS = ns.stack("n_hist"), ns.stack("S_hist")
                hd = start + pd.to_timedelta(np.arange(hn.shape[1]), unit="D")
                with np.errstate(invalid="ignore", divide="ignore"):
                    ha = np.where(hn[keep] > 0, N[keep][:, None] * hS[keep] / hn[keep], np.nan)
                hdir = os.path.join(out, "history", f["name"])
                os.makedirs(hdir, exist_ok=True)
                Hn = hn.shape[1]
                write_csv(pd.DataFrame({
                    "node": np.repeat(np.asarray(names)[keep], Hn), "date": np.tile(hd.date, K),
                    "households": np.repeat(N[keep], Hn).astype(int), "observed": hn[keep].ravel().astype(int),
                    "actual_observed_sum": hS[keep].ravel(), "actual": ha.ravel()}),
                    os.path.join(hdir, ts.level_file(lv, compressed=True)))
            counts.setdefault(lv, {})[f["name"]] = dict(nodes=int(len(names)), nodes_ge_min=K,
                                                         scored_node_days=int(((coverage >= cov) & (nk > 0)).sum()))
        if parts:
            write_csv(pd.concat(parts, ignore_index=True), os.path.join(out, "actuals", ts.level_file(lv)))

    # ---- manifest
    files = {}
    for base, _, fl in os.walk(out):
        for fn in fl:
            if fn.endswith((".csv", ".csv.gz")):
                pth = os.path.join(base, fn)
                files[os.path.relpath(pth, out)] = ts.sha256_file(pth)
    import subprocess
    commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    man = dict(test_set_version=spec["version"], test_set_sha256=spec["_sha256"], data_uri=d["uri"],
               data_version=os.path.basename(d["uri"].rstrip("/")), built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
               git_commit=commit, trial=bool(a.coatos), districts=len(codes), levels=levels,
               panel_households={f["name"]: int(dist[f"panel_{f['name']}"].sum()) for f in spec["folds"]},
               counts=counts, files=files)
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=2)
    print(f"\n[build] done in {(time.time() - t0) / 60:.1f} min -> {out}")
    print("  panel households per fold: " + ", ".join(f"{k} {v:,}" for k, v in man["panel_households"].items()))
    ndays = {f["name"]: len(ts.fold_days(spec, f)[0]) for f in spec["folds"]}
    for lv, c in counts.items():
        share = sum(v["scored_node_days"] for v in c.values()) / max(1, sum(v["nodes_ge_min"] * ndays[k] for k, v in c.items()))
        print(f"  {lv:12s} nodes >= {mn} households: " + ", ".join(f"{k} {v['nodes_ge_min']}" for k, v in c.items())
              + f" | node-days scored (coverage >= {cov:.0%}): {share:.1%}")
        if lv in ("national", "region", "district") and share < 1:
            print(f"  WARNING: {lv} loses {1 - share:.1%} of its test days to the coverage rule; the ТЗ scores every "
                  f"day -- check the coverage in actuals/{ts.level_file(lv)} before releasing this test set")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(REPO, "core"))
    from mlflow_utils import run_script
    run_script(main, "electricity/test", stage="build_test_set", out_flags=("--no-out-dir",))
