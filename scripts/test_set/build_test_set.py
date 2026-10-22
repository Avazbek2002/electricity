#!/usr/bin/env python
"""
Build the standard evaluation set once (configs/test_set.yaml v2 -> ~/work/test_set/<version>/): the district target
series T2 / T3 with the guide's day flags, the district -> region map, and the standing calendar.

    python scripts/test_set/build_test_set.py --workers 16                  # all districts (national runs)
    python scripts/test_set/build_test_set.py --sample --workers 5          # the guide's 5-district sample only

Only district-day aggregates are written (counts and kWh sums of thousands of customers); no customer keys.
The harness and the scorer check the manifest's hash of configs/test_set.yaml.
"""
import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
from harness import testset as ts  # noqa: E402
import enriched_io as eio  # noqa: E402  (harness puts core/ on the path)

LABEL_OBS, LABEL_EST = "complete_observed", "complete_with_estimated"
WANT = ["date", "label_state", "kwh", "kwh_nano", "is_long_gap_allocated", "kwh_long_gap_allocated", "is_right_edge",
        "crisis_context", "data_interruption", "region_code"]


def district_counts(code, cfg):
    """One district -> daily counts and kWh sums by day state (guide part 2), over the whole data window."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as pads
    t0 = time.time()
    d = cfg["data"]
    fs, root = eio.open_root(d["uri"], d.get("env_prefix", ""))
    dset = eio.district_daily(fs, root, code)
    cols = [c for c in WANT if c in dset.schema.names]
    missing = [c for c in ("label_state", "is_long_gap_allocated", "is_right_edge", "crisis_context", "data_interruption")
               if c not in cols]
    d0, d1 = pd.Timestamp(d["start"]), pd.Timestamp(d["end"])
    T = (d1 - d0).days + 1
    e0 = int((d0 - pd.Timestamp("1970-01-01")).days)
    acc = {k: np.zeros(T) for k in ("n_rows", "n_obs", "n_est", "n_lg", "kwh_obs", "kwh_est", "kwh_lg", "n_right",
                                    "crisis", "interruption")}
    region = None
    flt = (pads.field("date") >= d0.date()) & (pads.field("date") <= d1.date())
    for b in dset.scanner(columns=cols, filter=flt, batch_size=262_144, use_threads=True).to_batches():
        if not b.num_rows:
            continue
        day = pc.cast(pc.cast(b.column("date"), pa.date32()), pa.int32()).to_numpy(zero_copy_only=False).astype(np.int64) - e0
        ok = (day >= 0) & (day < T)
        day = day[ok]

        def col(name, kind=float):
            if name not in cols:
                return np.zeros(len(day), dtype=kind)
            a = b.column(name)
            if kind is bool:
                a = pc.fill_null(a, False)
            v = a.to_numpy(zero_copy_only=False)
            v = v[ok]
            return v.astype(bool) if kind is bool else np.nan_to_num(np.asarray(v, dtype=float))
        if "kwh_nano" in cols:
            kwh = col("kwh_nano") / 1e9
        else:
            kwh = col("kwh")
        lab = b.column("label_state").to_numpy(zero_copy_only=False)[ok] if "label_state" in cols else np.array([], object)
        obs, est = (lab == LABEL_OBS), (lab == LABEL_EST)
        lg = col("is_long_gap_allocated", bool)
        bc = lambda w: np.bincount(day, weights=w, minlength=T)          # noqa: E731
        acc["n_rows"] += bc(np.ones(len(day)))
        acc["n_obs"] += bc(obs.astype(float))
        acc["n_est"] += bc(est.astype(float))
        acc["n_lg"] += bc(lg.astype(float))
        acc["kwh_obs"] += bc(np.where(obs, kwh, 0.0))
        acc["kwh_est"] += bc(np.where(est, kwh, 0.0))
        acc["kwh_lg"] += bc(np.where(lg, col("kwh_long_gap_allocated"), 0.0))
        acc["n_right"] += bc(col("is_right_edge", bool).astype(float))
        acc["crisis"] = np.maximum(acc["crisis"], bc(col("crisis_context", bool).astype(float)))
        acc["interruption"] = np.maximum(acc["interruption"], bc(col("data_interruption", bool).astype(float)))
        if region is None and "region_code" in cols:
            r = b.column("region_code").drop_null()
            if len(r):
                region = str(r[0].as_py()).strip()
    df = pd.DataFrame(acc)
    df.insert(0, "date", pd.date_range(d0, d1, freq="D"))
    df.insert(1, "node", str(code))
    for k, n in (("kwh_obs", "n_obs"), ("kwh_est", "n_est"), ("kwh_lg", "n_lg")):
        df.loc[df[n] == 0, k] = np.nan                              # "nothing of that kind" (guide: kWh NaN)
    df.loc[df["n_rows"] == 0, "n_rows"] = np.nan
    return dict(code=str(code), region=region or "unknown", counts=df, missing=missing, seconds=time.time() - t0,
                peak_bytes=ts.peak_rss_bytes())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", default=os.path.join(REPO, "configs", "test_set.yaml"))
    p.add_argument("--out-root", default="~/work/test_set")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--sample", action="store_true", help="the guide's 5-district sample only (spec sample_districts)")
    p.add_argument("--coatos", default=None, help="comma list: a trial on a few districts (not the test set)")
    p.add_argument("--overwrite", action="store_true", help="trials only: replace an existing build")
    a = p.parse_args()
    spec = ts.load_spec(a.spec)
    want = spec["sample_districts"] if a.sample else (a.coatos.split(",") if a.coatos else None)
    suffix = "-sample" if a.sample else ("-trial" if a.coatos else "")
    out = os.path.join(os.path.expanduser(a.out_root), spec["version"] + suffix)
    if os.path.exists(os.path.join(out, "manifest.json")):
        if not a.overwrite or not suffix:
            raise SystemExit(f"{out} is already built; a released set is not rebuilt (new version in the spec, or "
                             f"--overwrite for a sample / trial)")
    os.makedirs(os.path.join(out, "targets"), exist_ok=True)
    d = spec["data"]
    fs, root = eio.open_root(d["uri"], d.get("env_prefix", ""))
    codes = eio.finished_districts(fs, root, set(want) if want else None)
    if want and set(want) - set(codes):
        raise SystemExit(f"districts not in the data (or unfinished): {sorted(set(want) - set(codes))}")
    print(f"[build] {spec['version']} ({spec['_sha256'][:12]}) | {len(codes)} districts -> {out}")
    print(f"[memory] at start: {ts.memory_line()}", flush=True)
    parts, rows, warned, t0 = [], [], set(), time.time()
    with cf.ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(district_counts, c, dict(data=d)): c for c in codes}
        for k, fu in enumerate(cf.as_completed(futs), 1):
            r = fu.result()
            for m in r["missing"]:
                if m not in warned:
                    print(f"[build] WARNING: column {m} not in the data: treated as absent/false")
                    warned.add(m)
            tgt = ts.target_series(r["counts"], spec)
            parts.append(tgt)
            rows.append(dict(district=r["code"], region=r["region"], max_customers=int(np.nanmax(r["counts"]["n_rows"])),
                             days_clean=int((~tgt["flag"]).sum())))
            if k % 10 == 0 or k == len(codes) or k <= 3:
                el = time.time() - t0
                print(f"[build] {k}/{len(codes)} districts, {el / 60:.1f} min, about {el / k * (len(codes) - k) / 60:.1f} "
                      f"min left | {ts.memory_line(r['peak_bytes'])}", flush=True)
    T = pd.concat(parts, ignore_index=True).sort_values(["node", "date"])
    cols = ["date", "node", "n_rows", "n_obs", "n_est", "n_lg", "kwh_obs", "kwh_est", "kwh_lg", "n_right", "crisis",
            "interruption", "coverage", "T2", "T3", "flag", "flag_reason", "artefact"]
    T[cols].to_csv(os.path.join(out, "targets", "district.csv.gz"), index=False, float_format="%.10g")
    dist = pd.DataFrame(rows).sort_values("district")
    dist.to_csv(os.path.join(out, "districts.csv"), index=False)
    ts.standing_calendar(pd.date_range(d["start"], pd.Timestamp(d["end"]) + pd.Timedelta(days=int(spec["horizons"]))))\
        .to_csv(os.path.join(out, "calendar.csv"), index=False)

    # ---- summary per split: share of clean district-days among the targets of the split
    summ = {}
    for split, s in spec["splits"].items():
        a0 = pd.Timestamp(s["issue_start"]) + pd.Timedelta(days=1)
        a1 = min(pd.Timestamp(s.get("target_end") or "2100-01-01"),
                 pd.Timestamp(s["issue_end"]) + pd.Timedelta(days=int(spec["horizons"])))
        q = T[(T["date"] >= a0) & (T["date"] <= a1)]
        summ[split] = dict(target_days=int(q["date"].nunique()), district_days=int(len(q)),
                           clean_share=float((~q["flag"]).mean()),
                           flag_reasons={k: int(v) for k, v in q.loc[q["flag"], "flag_reason"].value_counts().head(8).items()})
    files = {}
    for base, _, fl in os.walk(out):
        for fn in fl:
            if fn.endswith((".csv", ".csv.gz")):
                pth = os.path.join(base, fn)
                files[os.path.relpath(pth, out)] = ts.sha256_file(pth)
    commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    man = dict(test_set_version=spec["version"], test_set_sha256=spec["_sha256"], data_uri=d["uri"],
               data_version=os.path.basename(d["uri"].rstrip("/")), built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
               git_commit=commit, scope="sample" if a.sample else ("trial" if a.coatos else "national"),
               districts=len(codes), missing_columns=sorted(warned), splits=summ, files=files)
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=2)
    print(f"\n[build] done in {(time.time() - t0) / 60:.1f} min -> {out}")
    for split, s in summ.items():
        print(f"  {split:10s}: {s['target_days']} target days, {s['district_days']:,} district-days, "
              f"{s['clean_share']:.1%} clean | flagged: {s['flag_reasons']}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(REPO, "core"))
    from mlflow_utils import run_script
    run_script(main, "electricity/test", stage="build_test_set", out_flags=("--no-out-dir",))
