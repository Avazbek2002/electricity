#!/usr/bin/env python
"""
Package the shared team test set (both splits + the weather track) and publish it, once, to S3.

    python scripts/test_set/release_team.py --workers 16                 # build what is missing, package, verify
    python scripts/test_set/release_team.py --publish                    # ... and upload (refused if it exists)

Release folder (~/work/test_set/release/<release>/, then s3://phase1/electricity/_staging/test-sets/<release>/):
    README.md                    how to use it (scripts/test_set/TEAM_TEST_SET.md)
    release.json                 versions, spec hashes, data version, counts, commit
    SHA256SUMS                   every file
    standard/  s25/              spec.yaml (exact bytes), manifest.json, districts.csv, calendar.csv,
                                 targets/district.csv.gz (district x day: counts, kWh sums, T2, T3, flags)
    reference/B0_<set>.parquet   B0 for every district, issue date and horizon of the set (node, origin, horizon,
                                 forecast, b0src)
    weather/issue_dates.csv      issue dates with an AIFS / FCN3 reforecast and their split in each set
    scorer/score.py, testset.py  the common scorer (numpy, pandas, pyyaml)

Only district-day aggregates leave the platform (no customer keys); a row whose customer count is under
--min-customers has its kWh columns blanked. A published release is never changed: a fix is a new release name.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "core"))
from harness import testset as ts  # noqa: E402

SETS = {"standard": "configs/test_set.yaml", "s25": "configs/test_set_2025.yaml"}
DEST = "s3://phase1/electricity/_staging/test-sets/"
KWH_COLS = ("kwh_obs", "kwh_est", "kwh_lg", "T2", "T3")


def sha256(path):
    return ts.sha256_file(path)


def ensure_built(spec_path, out_root, workers):
    spec = ts.load_spec(spec_path)
    d = os.path.join(os.path.expanduser(out_root), spec["version"])
    if not os.path.exists(os.path.join(d, "manifest.json")):
        print(f"[release] building {spec['version']} (every district) -> {d}", flush=True)
        r = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "test_set", "build_test_set.py"), "--spec",
                            spec_path, "--out-root", out_root, "--workers", str(workers)], cwd=REPO)
        if r.returncode:
            raise SystemExit(f"[release] build of {spec['version']} failed")
    man = ts.read_manifest(d)
    ts.check_same_spec(spec, man)
    if man.get("scope") != "national":
        raise SystemExit(f"[release] {d} is a {man.get('scope')} build: the team set needs every district")
    return spec, d, man


def copy_set(name, spec, src, dst, min_customers):
    os.makedirs(os.path.join(dst, "targets"), exist_ok=True)
    shutil.copyfile(spec["_path"], os.path.join(dst, "spec.yaml"))            # exact bytes: the hash must match
    for f in ("districts.csv", "calendar.csv"):
        shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))
    t = pd.read_csv(os.path.join(src, "targets", "district.csv.gz"), dtype={"node": str})
    small = t["n_rows"].fillna(0) < min_customers
    for c in KWH_COLS:
        if c in t:
            t.loc[small, c] = np.nan
    if small.any():
        t.loc[small, "flag"] = True
        t.loc[small, "flag_reason"] = t.loc[small, "flag_reason"].fillna("").astype(str).str.cat(
            ["small"] * int(small.sum()), sep="+").str.strip("+")
    assert not ({"customer_key", "meter_id", "consumer_id"} & set(t.columns)), "customer columns in the targets"
    t.to_csv(os.path.join(dst, "targets", "district.csv.gz"), index=False, float_format="%.10g")
    man = json.load(open(os.path.join(src, "manifest.json")))
    man["files"] = {os.path.relpath(os.path.join(b, f), dst): sha256(os.path.join(b, f))
                    for b, _, fl in os.walk(dst) for f in fl if f.endswith((".csv", ".csv.gz"))}
    man["release_note"] = f"rows with fewer than {min_customers} customers blanked: {int(small.sum())}"
    json.dump(man, open(os.path.join(dst, "manifest.json"), "w"), indent=2)
    return t, int(small.sum())


def b0_reference(name, spec, set_dir, out):
    t = ts.load_targets(set_dir, spec, "T2")
    b = ts.b0_forecasts(t, spec)
    b["date"] = b["origin"] + pd.to_timedelta(b["horizon"], unit="D")
    parts = [ts.split_rows(spec, s, b).assign(split=s) for s in spec["splits"]]
    b = pd.concat(parts, ignore_index=True)[["split", "node", "origin", "horizon", "forecast", "b0src"]]
    b["forecast"] = b["forecast"].astype("float32")
    b["b0src"] = b["b0src"].astype("int8")
    b.to_parquet(out, index=False)
    return len(b)


def weather_issue_dates(specs, out, wx=None):
    import reforecast as rf
    w = rf.wx_config({"wx": wx or {}})
    fs, root = rf._open(w["root"])
    have = {m: {d for s in w["sets"] for d, _ in rf.list_inits(fs, root, s, m)} for m in rf.MODELS}
    days = sorted(set().union(*have.values()))
    df = pd.DataFrame({"issue_date": pd.DatetimeIndex(days)})
    for m in rf.MODELS:
        df[m] = df["issue_date"].isin(have[m])
    for name, spec in specs.items():
        col = pd.Series(pd.NA, index=df.index, dtype="object")
        for s in spec["splits"]:
            col[df["issue_date"].isin(ts.issue_dates(spec, s))] = s
        df[f"split_{name}"] = col
    df["sets"] = ",".join(w["sets"])
    df["root"] = w["root"]
    df.to_csv(out, index=False, date_format="%Y-%m-%d")
    return df


def self_check(rel):
    """Score B0 as a submission with the release's own scorer: it must equal the B0 reference on the same rows."""
    tmp = os.path.join(rel, "..", "_selfcheck")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    spec = ts.load_spec(os.path.join(rel, "standard", "spec.yaml"))
    b = pd.read_parquet(os.path.join(rel, "reference", "B0_standard.parquet"))
    b = b[b["node"].isin(spec["sample_districts"])]
    b[["node", "origin", "horizon", "forecast"]].assign(origin=lambda x: x["origin"].dt.date).to_csv(
        os.path.join(tmp, "district.csv.gz"), index=False)
    json.dump({"model": "B0_copy", "test_set_version": spec["version"], "test_set_sha256": spec["_sha256"],
               "target": "T2", "weather": "none", "level": "district"}, open(os.path.join(tmp, "forecast_meta.json"), "w"))
    r = subprocess.run([sys.executable, "-B", os.path.join(rel, "scorer", "score.py"), "--set", "standard",
                        "--forecasts", tmp], capture_output=True, text=True)
    shutil.rmtree(os.path.join(rel, "scorer", "__pycache__"), ignore_errors=True)
    if r.returncode:
        raise SystemExit(f"[release] self-check failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    sc = pd.read_csv(os.path.join(tmp, "scores_standard_all", "scores.csv"))
    piv = sc.pivot_table(index=["split", "horizon"], columns="model", values="sysmape")
    diff = float((piv["B0_copy"] - piv["B0"]).abs().max())
    shutil.rmtree(tmp, ignore_errors=True)
    if not diff < 1e-6:
        raise SystemExit(f"[release] self-check: B0 scored as a submission differs from the reference by {diff}")
    return diff


def upload(rel, dest):
    import enriched_io as eio
    import pyarrow.fs as pafs
    fs, root = eio.open_root(dest)
    if fs.get_file_info(f"{root}/release.json").type != pafs.FileType.NotFound:
        raise SystemExit(f"[publish] {dest} already exists: a published release is never changed (new --release name)")
    files = [os.path.join(b, f) for b, _, fl in os.walk(rel) for f in fl]
    for k, src in enumerate(sorted(files, key=lambda x: x.endswith("release.json"))):   # release.json last
        key = f"{root}/{os.path.relpath(src, rel)}"
        fs.create_dir(os.path.dirname(key), recursive=True)
        with open(src, "rb") as fi, fs.open_output_stream(key, compression=None) as fo:
            shutil.copyfileobj(fi, fo, 8 << 20)
    print(f"[publish] {len(files)} files -> {dest}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--release", default="team-testset-2026-10-23-v1")
    p.add_argument("--out-root", default="~/work/test_set")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--min-customers", type=int, default=10, help="blank kWh of district-days with fewer customers")
    p.add_argument("--no-weather", action="store_true", help="skip the weather track (no access to the reforecasts)")
    p.add_argument("--publish", action="store_true", help=f"upload to {DEST}<release>/ (refused if it exists)")
    p.add_argument("--dest", default=DEST)
    p.add_argument("--spec-standard", default=os.path.join(REPO, SETS["standard"]))
    p.add_argument("--spec-s25", default=os.path.join(REPO, SETS["s25"]))
    p.add_argument("--wx-root", default=None, help="reforecast root (default: core/reforecast.py)")
    p.add_argument("--wx-sets", default=None, help="comma list of reforecast sets (default: core/reforecast.py)")
    a = p.parse_args()
    rel = os.path.join(os.path.expanduser(a.out_root), "release", a.release)
    if os.path.exists(os.path.join(rel, "release.json")):
        print(f"[release] {rel} is already packaged: reusing it")
    else:
        shutil.rmtree(rel, ignore_errors=True)
        os.makedirs(rel)
        specs, info = {}, {}
        for name, path in (("standard", a.spec_standard), ("s25", a.spec_s25)):
            spec, src, man = ensure_built(os.path.abspath(os.path.expanduser(path)), a.out_root, a.workers)
            t, n_small = copy_set(name, spec, src, os.path.join(rel, name), a.min_customers)
            specs[name] = spec
            info[name] = dict(version=spec["version"], spec_sha256=spec["_sha256"], data_uri=man["data_uri"],
                              districts=int(t["node"].nunique()), first_day=str(t["date"].min()),
                              last_day=str(t["date"].max()), blanked_small_rows=n_small,
                              splits={s: dict(v, issue_dates=len(ts.issue_dates(spec, s)))
                                      for s, v in spec["splits"].items()},
                              refits=spec["refits"], sample_districts=spec["sample_districts"])
            print(f"[release] {name}: {spec['version']} ({spec['_sha256'][:12]}), {info[name]['districts']} districts")
        os.makedirs(os.path.join(rel, "reference"))
        for name, spec in specs.items():
            n = b0_reference(name, spec, os.path.join(rel, name), os.path.join(rel, "reference", f"B0_{name}.parquet"))
            info[name]["b0_rows"] = n
            print(f"[release] reference/B0_{name}.parquet: {n:,} rows")
        if not a.no_weather:
            os.makedirs(os.path.join(rel, "weather"))
            wx = {k: v for k, v in (("root", a.wx_root), ("sets", a.wx_sets.split(",") if a.wx_sets else None)) if v}
            w = weather_issue_dates(specs, os.path.join(rel, "weather", "issue_dates.csv"), wx)
            info["weather"] = {name: {s: int(((w[f"split_{name}"] == s) & w["aifs"] & w["fcn3"]).sum())
                                      for s in spec["splits"]} for name, spec in specs.items()}
            print(f"[release] weather/issue_dates.csv: {len(w)} issue dates; per split {info['weather']}")
        os.makedirs(os.path.join(rel, "scorer"))
        shutil.copyfile(os.path.join(REPO, "harness", "testset.py"), os.path.join(rel, "scorer", "testset.py"))
        shutil.copyfile(os.path.join(REPO, "scripts", "test_set", "team_score.py"), os.path.join(rel, "scorer", "score.py"))
        shutil.copyfile(os.path.join(REPO, "scripts", "test_set", "TEAM_TEST_SET.md"), os.path.join(rel, "README.md"))
        open(os.path.join(rel, "scorer", "requirements.txt"), "w").write("numpy\npandas\npyyaml\npyarrow\n")
        diff = self_check(rel)
        print(f"[release] self-check passed: B0 scored as a submission = the reference (max diff {diff:.1e})")
        commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        files = sorted(os.path.relpath(os.path.join(b, f), rel) for b, _, fl in os.walk(rel) for f in fl)
        with open(os.path.join(rel, "SHA256SUMS"), "w") as fh:
            for f in files:
                fh.write(f"{sha256(os.path.join(rel, f))}  {f}\n")
        json.dump(dict(release=a.release, built_at=time.strftime("%Y-%m-%d %H:%M:%S"), git_commit=commit,
                       destination=a.dest.rstrip("/") + "/" + a.release + "/", min_customers=a.min_customers,
                       sets=info, files=len(files) + 1), open(os.path.join(rel, "release.json"), "w"), indent=2)
        print(f"[release] packaged -> {rel}")
    if a.publish:
        upload(rel, a.dest.rstrip("/") + "/" + a.release)


if __name__ == "__main__":
    main()
