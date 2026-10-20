"""
Readers for the national household ENRICHED dataset (v1.4, staging), shared by whole_pipeline_v8.py (household model)
and transformer_prepare_enriched.py (transformer panel). PyArrow only: no Spark, one district at a time (guide rule 10).

Layout (HOUSEHOLD_NATIONAL_DATASETS_GUIDE.md §2):
    <root>/household_daily_enriched/coato=<code>/                      district worker root (+ _SUCCESS when finished)
    <root>/household_daily_enriched/coato=<code>/household_daily_enriched/data/period=YYYY-MM/*.parquet   daily table
Credentials come from the environment (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_ENDPOINT); a local path with the same
layout works too (tests).
"""
import os

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs

ENRICHED_ROOT = "s3://phase1/electricity/_staging/household-enriched-v14-nat-b2c-20261017-8e1414/"
WINDOW_START, WINDOW_END = "2022-01-01", "2026-06-29"              # customer-day window of v1.4 (guide §2)
LABELS = {"observed": ["complete_observed"], "clean": ["complete_observed", "complete_with_estimated"]}


def open_root(uri, env_prefix=""):
    """(filesystem, path without scheme). s3:// or s3a:// -> S3 from the environment; anything else -> local disk.
    env_prefix: read the credentials from <prefix>AWS_ACCESS_KEY_ID, <prefix>AWS_SECRET_ACCESS_KEY, <prefix>S3_ENDPOINT
    (e.g. "VAULT_" for the evaluation harness's holdout key); empty = the usual variables."""
    if uri.startswith(("s3://", "s3a://")):
        names = [env_prefix + v for v in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_ENDPOINT")]
        missing = [v for v in names if not os.getenv(v)]
        if missing:
            raise SystemExit(f"missing environment variables {missing} (source your env file first)")
        ep = os.environ[names[2]]
        scheme, host = ep.split("://", 1) if "://" in ep else ("https", ep)
        fs = pafs.S3FileSystem(access_key=os.environ[names[0]], secret_key=os.environ[names[1]],
                               session_token=os.getenv(env_prefix + "AWS_SESSION_TOKEN"), endpoint_override=host.rstrip("/"),
                               scheme=scheme, region=os.getenv(env_prefix + "AWS_REGION", os.getenv("AWS_REGION", "us-east-1")))
        return fs, uri.split("://", 1)[1].rstrip("/")
    return pafs.LocalFileSystem(), os.path.abspath(os.path.expanduser(uri)).rstrip("/")


def list_districts(fs, root):
    """[(code, finished)] for every coato=<code> worker root under the campaign root."""
    base = f"{root}/household_daily_enriched"
    out = []
    for info in fs.get_file_info(pafs.FileSelector(base)):
        if info.type == pafs.FileType.Directory and info.base_name.startswith("coato="):
            code = info.base_name.split("=", 1)[1]
            done = fs.get_file_info(f"{info.path}/_SUCCESS").type == pafs.FileType.File
            out.append((code, done))
    return sorted(out)


def finished_districts(fs, root, want=None, include_unfinished=False):
    """Codes to read: finished districts (or all), optionally limited to `want`."""
    dist = list_districts(fs, root)
    if not dist:
        raise SystemExit(f"no coato=<code> folders under {root}/household_daily_enriched")
    todo = [c for c, done in dist if (done or include_unfinished) and (want is None or c in want)]
    unfinished = [c for c, done in dist if not done]
    print(f"[enriched] {len(dist)} district folders, {len(dist) - len(unfinished)} finished (_SUCCESS); reading {len(todo)}"
          + (f"; not finished: {unfinished[:10]}{' ...' if len(unfinished) > 10 else ''}" if unfinished else ""))
    if not todo:
        raise SystemExit("no finished district matches the selection")
    return todo


def district_daily(fs, root, code):
    """The district's daily table as a pyarrow dataset; `period` (YYYY-MM) comes from the month folders, as a string."""
    path = f"{root}/household_daily_enriched/coato={code}/household_daily_enriched/data"
    part = ds.partitioning(pa.schema([("period", pa.string())]), flavor="hive")
    return ds.dataset(path, filesystem=fs, format="parquet", partitioning=part)


def period_range(a, b):
    return [p.strftime("%Y-%m") for p in pd.period_range(pd.Timestamp(a), pd.Timestamp(b), freq="M")]


def kept_label(keep):
    """Rows whose clean kwh is a kept label (default: complete_observed only, HH-GAP-001-A1 rule 1)."""
    return ds.field("label_state").isin(keep) & ds.field("kwh").is_valid()
