"""Actual weather forecasts (AIFS / FCN3 reforecasts, ML guide part 2) -> a daily forecast archive per district.

Source (guide, part 2 "Погода"): s3://data-weather/reforecast/<set>/data/model=<aifs|fcn3>/init_date=<date>/part-0.parquet,
long format (`variable`, `value`), 6-hour steps, all 254 weather points, start 00 UTC on the issue date, 384 h (16 days),
initial conditions ERA5. Forecasts exist only on the sets' issue dates (the 1st, 9th, 17th and 25th of each month,
2024 and 2025). The 2026 reserve set is never read.

Archive (the "harness convention"), one row per (district, issue date, lead day):
    coato_id    district code as in the enriched dataset (e.g. "03203")
    issue_date  the issue date o = the forecast's init date (00 UTC run, used for the forecast issued on o)
    lead_day    k = 1 .. lead_days: the Tashkent calendar day o + k (lead 0 = day o itself is dropped: not a target)
    t_fc, t_fc_min, t_fc_max   daily mean / min / max 2 m temperature, deg C

Local days: Tashkent is UTC+5 all year, so local day D runs from 19:00 UTC on D - 1 to 19:00 UTC on D; the 6-hourly
steps 00, 06, 12, 18 UTC of D all fall in local day D (05, 11, 17, 23 local). Daily mean = mean of the day's steps; min /
max = the 6-hourly min / max variables when the file has them (e.g. mn2t6 / mx2t6), else the min / max of the
instantaneous steps (an underestimate of the true range: the models learn it as it is, train and serve alike).

The schema is detected from common names; anything unusual is set in the model config's `wx.columns` / `wx.variables`.
Run `python core/reforecast.py --inspect` on the platform first to see what the files contain.
"""
import argparse
import hashlib
import json
import os

import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MODELS = ("aifs", "fcn3")
SOURCES = ("aifs", "fcn3", "ens")            # ens = mean of AIFS and FCN3 where both exist
DEFAULTS = dict(
    root="s3://data-weather/reforecast/",
    sets=["wx-val2024-20261021t120926z", "wx-test2025-20261021t120926z"],
    source="aifs",
    lead_days=14,                            # forecast days used: h1 .. h14 (then climatology, fading anomaly)
    origins="reforecast+stride",             # training windows: reforecast | reforecast+stride (see boost_wx_harness)
    tz_hours=5,                              # Asia/Tashkent = UTC+5, no daylight saving
    cache_dir="~/work/eval/reforecast",
    columns={},                              # {point:, time:, step:, variable:, value:} when detection fails
    variables={},                            # {t:, tmin:, tmax:} variable names when detection fails
    point_map=None,                          # csv with columns point, coato_code when points are not district codes
    min_windows=20000,                       # refuse to train on fewer windows with a forecast
)
POINT_COLS = ("coato_code", "coato_id", "coato", "point_id", "point", "location_id", "station_id", "station", "id")
TIME_COLS = ("valid_time_utc", "valid_time", "valid_datetime", "valid_date", "time", "datetime", "timestamp")
STEP_COLS = ("step", "step_hours", "lead_hours", "lead_time", "leadtime", "fhour", "forecast_hour", "lead")
VAR_COL, VALUE_COL = "variable", "value"
T_NAMES = ("2t", "t2m", "t2", "tas", "temperature_2m", "2m_temperature", "air_temperature_2m", "temperature")
TMIN_NAMES = ("mn2t6", "mn2t", "mn2t3", "tmin", "t2m_min", "tasmin", "minimum_2m_temperature", "temperature_2m_min")
TMAX_NAMES = ("mx2t6", "mx2t", "mx2t3", "tmax", "t2m_max", "tasmax", "maximum_2m_temperature", "temperature_2m_max")


def wx_config(cfg):
    """The model config's `wx` block over the defaults."""
    w = dict(DEFAULTS)
    w.update((cfg or {}).get("wx") or {})
    if w["source"] not in SOURCES:
        raise SystemExit(f"wx.source must be one of {SOURCES}, not {w['source']!r}")
    bad = [s for s in w["sets"] if "reserve" in s.lower()]
    if bad:
        raise SystemExit(f"the 2026 reserve is never used (guide part 2): remove {bad} from wx.sets")
    return w


# ----------------------------------------------------------------------------- reading
def _open(root):
    import enriched_io as eio
    return eio.open_root(root)


def list_inits(fs, root, set_name, model):
    """Issue (init) dates available for one set and model, from the folder names init_date=<date>."""
    import pyarrow.fs as pafs
    base = f"{root}/{set_name}/data/model={model}"
    out = []
    try:
        infos = fs.get_file_info(pafs.FileSelector(base))
    except (FileNotFoundError, OSError):
        return out
    for info in infos:
        if info.type == pafs.FileType.Directory and info.base_name.startswith("init_date="):
            d = pd.Timestamp(info.base_name.split("=", 1)[1][:10])
            out.append((d, info.path))
    return sorted(out)


def _read_file(fs, path):
    """One init's parquet, without hive partitioning (the column `model` is both in the data and the folder name)."""
    import pyarrow.dataset as pads
    files = [path] if path.endswith(".parquet") else [f"{path}/part-0.parquet"]
    import pyarrow.fs as pafs
    if fs.get_file_info(files[0]).type != pafs.FileType.File:                   # any parquet in the folder
        files = [i.path for i in fs.get_file_info(pafs.FileSelector(path)) if i.path.endswith(".parquet")]
    return pads.dataset(files, filesystem=fs, format="parquet", partitioning=None).to_table().to_pandas()


def _pick(cols, wanted, override=None, what=""):
    if override:
        if override not in cols:
            raise SystemExit(f"[reforecast] column {override!r} ({what}) is not in the file: {sorted(cols)}")
        return override
    low = {c.lower(): c for c in cols}
    for w in wanted:
        if w in low:
            return low[w]
    return None


def _var(names, wanted, override=None):
    if override:
        return override if override in names else None
    low = {str(n).lower(): n for n in names}
    for w in wanted:
        if w in low:
            return low[w]
    return None


def norm_code(x):
    """District codes as in the enriched dataset: digit strings zero-padded to 5 ("3203" -> "03203")."""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(5) if s.isdigit() and len(s) < 5 else s


def to_daily(df, init, w, point_map=None):
    """One init's long table -> daily rows (coato_id, issue_date, lead_day, t_fc, t_fc_min, t_fc_max)."""
    cols = set(df.columns)
    c = w.get("columns") or {}
    pcol = _pick(cols, POINT_COLS, c.get("point"), "point")
    vcol = _pick(cols, (VAR_COL,), c.get("variable"), "variable")
    xcol = _pick(cols, (VALUE_COL,), c.get("value"), "value")
    tcol = _pick(cols, TIME_COLS, c.get("time"), "valid time")
    scol = None if tcol else _pick(cols, STEP_COLS, c.get("step"), "step")
    if not (pcol and vcol and xcol and (tcol or scol)):
        raise SystemExit(f"[reforecast] cannot find point / variable / value / time columns in {sorted(cols)}: set "
                         f"wx.columns in the model config (see core/reforecast.py --inspect)")
    v = w.get("variables") or {}
    names = pd.unique(df[vcol].astype(str))
    vt = _var(names, T_NAMES, v.get("t"))
    if vt is None:
        raise SystemExit(f"[reforecast] no 2 m temperature among the variables {sorted(names)[:30]}: set wx.variables.t")
    vmin, vmax = _var(names, TMIN_NAMES, v.get("tmin")), _var(names, TMAX_NAMES, v.get("tmax"))
    keep = [x for x in (vt, vmin, vmax) if x is not None]
    d = df.loc[df[vcol].astype(str).isin(keep), [pcol, vcol, xcol] + [tcol or scol]].copy()
    d.columns = ["point", "var", "value", "t"]
    d["var"] = d["var"].astype(str)
    if tcol:
        t = pd.to_datetime(d["t"], utc=True).dt.tz_convert(None)
    else:
        s = d["t"]
        hours = (s.dt.total_seconds() / 3600.0) if pd.api.types.is_timedelta64_dtype(s) else pd.to_numeric(s)
        t = pd.Timestamp(init) + pd.to_timedelta(hours, unit="h")
    local = (t + pd.Timedelta(hours=int(w.get("tz_hours", 5)))).dt.floor("D")
    d["lead_day"] = (local - pd.Timestamp(init)).dt.days.astype(int)
    d = d[(d["lead_day"] >= 1) & (d["lead_day"] <= int(w["lead_days"]))]
    d["value"] = pd.to_numeric(d["value"], errors="coerce").astype("float64")
    if np.nanmedian(d.loc[d["var"] == vt, "value"]) > 150:                    # Kelvin -> deg C
        d["value"] -= 273.15
    g = d.groupby(["point", "lead_day", "var"])["value"]
    agg = pd.DataFrame({"mean": g.mean(), "min": g.min(), "max": g.max(), "n": g.count()}).reset_index()
    tt = agg[agg["var"] == vt].set_index(["point", "lead_day"])
    full = tt["n"].median() if len(tt) else 0                                 # steps a full local day has (4 at 6 h)
    out = pd.DataFrame({"t_fc": tt["mean"].where(tt["n"] >= max(1, 0.75 * full)), "t_min_s": tt["min"],
                        "t_max_s": tt["max"]})
    out["t_fc_min"] = (agg[agg["var"] == vmin].set_index(["point", "lead_day"])["min"]
                       if vmin else out["t_min_s"]).reindex(out.index)
    out["t_fc_max"] = (agg[agg["var"] == vmax].set_index(["point", "lead_day"])["max"]
                       if vmax else out["t_max_s"]).reindex(out.index)
    out["t_fc_min"] = np.minimum(out["t_fc_min"], out["t_fc"])                 # min <= mean <= max
    out["t_fc_max"] = np.maximum(out["t_fc_max"], out["t_fc"])
    out = out.drop(columns=["t_min_s", "t_max_s"]).reset_index()
    pts = out["point"].astype(str)
    out["coato_id"] = pts.map(point_map).fillna(pts) if point_map else pts
    out["coato_id"] = out["coato_id"].map(norm_code)
    out["issue_date"] = pd.Timestamp(init)
    return out[["coato_id", "issue_date", "lead_day", "t_fc", "t_fc_min", "t_fc_max"]], dict(
        t=vt, tmin=vmin or f"min of {vt}", tmax=vmax or f"max of {vt}", steps_per_day=float(full))


def _point_map(w):
    p = w.get("point_map")
    if not p:
        return None
    m = pd.read_csv(os.path.expanduser(p), dtype=str)
    return dict(zip(m["point"].astype(str), m["coato_code"].astype(str)))


def _cache_path(w, model, set_name):
    keys = {k: w[k] for k in ("root", "lead_days", "tz_hours", "columns", "variables", "point_map")}
    sig = hashlib.sha1(json.dumps(dict(keys, model=model, set=set_name), sort_keys=True,
                                  default=str).encode()).hexdigest()[:10]
    return os.path.join(os.path.expanduser(w["cache_dir"]), f"{set_name}_{model}_{sig}.parquet")


def load_model_archive(w, model, verbose=True):
    """Daily archive of one forecast model over every set in w['sets'] (cached locally per set)."""
    fs = root = None
    parts = []
    for s in w["sets"]:
        path = _cache_path(w, model, s)
        if os.path.exists(path):
            parts.append(pd.read_parquet(path))
            continue
        if fs is None:
            fs, root = _open(w["root"])
        inits = list_inits(fs, root, s, model)
        if not inits:
            raise SystemExit(f"[reforecast] no init_date folders under {w['root'].rstrip('/')}/{s}/data/model={model}")
        pm, rows, info = _point_map(w), [], None
        for k, (d, p) in enumerate(inits):
            a, info = to_daily(_read_file(fs, p), d, w, pm)
            rows.append(a)
            if verbose and (k == 0 or (k + 1) % 12 == 0 or k + 1 == len(inits)):
                print(f"[reforecast] {s} {model}: {k + 1}/{len(inits)} issue dates read", flush=True)
        a = pd.concat(rows, ignore_index=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        a.to_parquet(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)
        if verbose:
            print(f"[reforecast] {s} {model}: {a['issue_date'].nunique()} issue dates, {a['coato_id'].nunique()} points, "
                  f"leads 1..{int(a['lead_day'].max())} | t = {info['t']}, min = {info['tmin']}, max = {info['tmax']}, "
                  f"{info['steps_per_day']:.0f} steps a day -> {path}")
        parts.append(a)
    return pd.concat(parts, ignore_index=True).drop_duplicates(["coato_id", "issue_date", "lead_day"])


def load_archive(w, verbose=True):
    """The archive of w['source']: aifs, fcn3, or ens (mean of the two on the (district, issue, lead) rows both have)."""
    if w["source"] != "ens":
        a = load_model_archive(w, w["source"], verbose)
    else:
        a1, a2 = (load_model_archive(w, m, verbose) for m in MODELS)
        k = ["coato_id", "issue_date", "lead_day"]
        a = a1.merge(a2, on=k, suffixes=("_1", "_2"))
        for c in ("t_fc", "t_fc_min", "t_fc_max"):
            a[c] = (a[f"{c}_1"] + a[f"{c}_2"]) / 2
        a = a[k + ["t_fc", "t_fc_min", "t_fc_max"]]
    a["issue_date"] = pd.to_datetime(a["issue_date"])
    return a[a["lead_day"].between(1, int(w["lead_days"]))].reset_index(drop=True)


def issue_dates(w):
    """Issue dates with a forecast of w['source'] (from the folder names; ens = dates both models have)."""
    fs, root = _open(w["root"])
    models = MODELS if w["source"] == "ens" else (w["source"],)
    sets = [set(d for s in w["sets"] for d, _ in list_inits(fs, root, s, m)) for m in models]
    return sorted(set.intersection(*sets))


def to_window_convention(a, K):
    """Harness convention (issue o, lead k = day o + k) -> v8's window convention (origin o - 1 = the last history
    day, lead j = day o - 1 + j): issue_date - 1, lead + 1, leads up to K. Lead 1 (= day o) stays empty, exactly as
    harness.weather.grid_at_last_day builds it at predict time, so training and serving see the same inputs."""
    b = a.copy()
    b["issue_date"] = b["issue_date"] - pd.Timedelta(days=1)
    b["lead_day"] = b["lead_day"] + 1
    return b[b["lead_day"] <= K].reset_index(drop=True)


def coverage(a, codes):
    """Share of the given district codes that the archive has (diagnostic for the point -> district mapping)."""
    have = set(a["coato_id"].astype(str))
    codes = [str(c) for c in codes]
    return (sum(c in have for c in codes) / max(len(codes), 1)), sorted(set(codes) - have)[:10]


# ----------------------------------------------------------------------------- inspect
def inspect(w, n_files=1):
    """Print what the files contain: columns, dtypes, variables, points, steps (run this first on the platform)."""
    fs, root = _open(w["root"])
    for s in w["sets"]:
        for m in MODELS:
            inits = list_inits(fs, root, s, m)
            print(f"\n== {s} / {m}: {len(inits)} issue dates"
                  + (f" ({inits[0][0].date()} .. {inits[-1][0].date()})" if inits else ""))
            for d, p in inits[:n_files]:
                df = _read_file(fs, p)
                print(f"   {p}: {len(df):,} rows")
                print("   columns:", {c: str(t) for c, t in df.dtypes.items()})
                for c in df.columns:
                    if df[c].dtype == object or c.lower() in ("variable", "model"):
                        u = pd.unique(df[c].astype(str))
                        print(f"   {c}: {len(u)} values, e.g. {list(u[:12])}")
                print(df.head(3).to_string())
                try:
                    a, info = to_daily(df, d, w, _point_map(w))
                    print(f"   -> daily: {len(a):,} rows, {a['coato_id'].nunique()} points, leads "
                          f"{a['lead_day'].min()}..{a['lead_day'].max()}, t = {info['t']}, min = {info['tmin']}, "
                          f"max = {info['tmax']}, {info['steps_per_day']:.0f} steps a day; sample:")
                    print(a.head(3).to_string())
                except SystemExit as e:
                    print(f"   -> daily: FAILED: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AIFS / FCN3 reforecasts: inspect the files or build the daily archive")
    p.add_argument("--inspect", action="store_true", help="print columns, variables and a daily sample")
    p.add_argument("--build", choices=SOURCES, help="build (and cache) the daily archive of this source")
    p.add_argument("--root", default=DEFAULTS["root"])
    p.add_argument("--sets", default=",".join(DEFAULTS["sets"]))
    p.add_argument("--lead-days", type=int, default=DEFAULTS["lead_days"])
    p.add_argument("--point-map", default=None)
    a = p.parse_args()
    w = wx_config(dict(wx=dict(root=a.root, sets=[s for s in a.sets.split(",") if s], lead_days=a.lead_days,
                               point_map=a.point_map, source=a.build or "aifs")))
    if a.inspect:
        inspect(w)
    if a.build:
        arc = load_archive(w)
        print(f"[reforecast] {a.build}: {len(arc):,} rows, {arc['issue_date'].nunique()} issue dates, "
              f"{arc['coato_id'].nunique()} districts")
        print(arc.groupby("lead_day")[["t_fc", "t_fc_min", "t_fc_max"]].mean().round(2).to_string())
