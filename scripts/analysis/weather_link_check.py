#!/usr/bin/env python
"""
Does the weather table's coato_id mean the same district as the consumption data's coato code?

Two checks:
  1. Keys:   the code sets on both sides, their format (int vs zero-padded string), and what the reference
             tables say each code is (responsible electricity branch vs geographic COATO location).
  2. Values: the enriched dataset carries its own daily t_mean / t_min / t_max per district. For every district,
             compare that series with EVERY weather code's daily series. If the link is right, the district's own
             weather code is the closest one (MAE near 0) at lag 0. A different best code = swapped or wrong link;
             a best lag of +-1 day = a date / time-zone cut problem.

Reads are bounded: the weather master is streamed in batches (3 columns); for consumption only the first file(s)
of each month folder are opened, with the 4 weather columns (no customer columns are read or written).
Outputs are district-level only.

    python scripts/analysis/weather_link_check.py --out-dir ~/work/weather_link --workers 12
    python scripts/analysis/weather_link_check.py --out-dir ~/work/weather_link --coatos 03202 03203 --tz Asia/Tashkent
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq

HERE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core"))
sys.path.insert(0, HERE)
import enriched_io as eio  # noqa: E402

WEATHER = ("s3://phase1/electricity/analytics/reference/coato_hourly_weather/"
           "coato_hourly_weather_2022_2026_master.parquet")
REF = "s3://phase1/electricity/analytics/reference"
MIN_OVERLAP = 200            # days a district and a weather code must share to be compared
NEAR = 0.3                   # deg C: own code counts as "as good as the best" within this margin


# ---------------------------------------------------------------- helpers
def norm(code):
    """'3202', 3202, '3202.0', '03202' -> '03202' (digits zero-padded to 5); anything else stripped as is."""
    s = str(code).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(5) if s.isdigit() else s


def name_key(s):
    """Lower-case Latin letters only, minus common suffixes, for a rough name comparison."""
    import re
    s = re.sub(r"[ʻʼ'`‘’\-]", "", str(s).lower())
    for w in ("tumani", "tuman", "shahri", "shahar", "tumanі", "res", "eltarmoq", "elektr", "filiali", "korxonasi"):
        s = s.replace(w, " ")
    return " ".join(sorted(re.sub(r"[^a-z ]", " ", s).split()))


def name_agree(a, b):
    ka, kb = name_key(a), name_key(b)
    if not ka or not kb:
        return np.nan
    ta, tb = set(ka.split()), set(kb.split())
    return bool(ta & tb) or ka in kb or kb in ka


def pick(names, options, what):
    for o in options:
        if o in names:
            return o
    raise SystemExit(f"[weather] no {what} column; columns are {names}. Pass --{what}-col.")


# ---------------------------------------------------------------- weather side
def show_dir(fs, path, depth=1):
    """List one or two levels of a folder (names and sizes), no recursion beyond that."""
    for info in fs.get_file_info(pafs.FileSelector(path, allow_not_found=True)):
        size = f"{info.size / 1e6:,.1f} MB" if info.type == pafs.FileType.File else "<dir>"
        print(f"    {info.path}   {size}")
        if depth > 1 and info.type == pafs.FileType.Directory:
            show_dir(fs, info.path, depth - 1)


def weather_files(fs, path):
    """The weather source as a list of parquet files: one file, or every *.parquet under a folder.
    If the path is missing, show what its parent folders hold and stop."""
    info = fs.get_file_info(path)
    if info.type == pafs.FileType.File:
        return [path]
    if info.type == pafs.FileType.Directory:
        files = sorted(i.path for i in fs.get_file_info(pafs.FileSelector(path, recursive=True))
                       if i.type == pafs.FileType.File and i.path.endswith(".parquet"))
        if files:
            return files
        print(f"[weather] {path} is a folder with no .parquet files")
    else:
        print(f"[weather] not found: {path}")
    parent = path.rsplit("/", 1)[0]
    for p in (parent, parent.rsplit("/", 1)[0]):
        if fs.get_file_info(p).type == pafs.FileType.Directory:
            print(f"[weather] contents of {p}/ :")
            show_dir(fs, p, depth=2 if p == parent else 1)
            break
    raise SystemExit("[weather] pass the right file or folder with --weather s3://...")


def read_weather(uri, tz, cols):
    fs, path = eio.open_root(uri)
    files = weather_files(fs, path)
    names = pq.ParquetFile(fs.open_input_file(files[0])).schema_arrow.names
    c = cols["coato"] or pick(names, ("coato_id", "coato_code", "coato"), "coato")
    t = cols["time"] or pick(names, ("timestamp", "time", "datetime", "date"), "time")
    v = cols["temp"] or pick(names, ("temperature_2m", "t2m", "temperature", "temp"), "temp")
    raw_type = str(pq.ParquetFile(fs.open_input_file(files[0])).schema_arrow.field(c).type)
    print(f"[weather] {len(files)} file(s) under {path}; columns {names}; using {c} ({raw_type}), {t}, {v}")
    nm = "district_name" if "district_name" in names else None
    rcols = [c, t, v] + ([nm] if nm else [])
    parts, raw_codes, wnames = [], set(), {}
    batches = (b for f in files
               for b in pq.ParquetFile(fs.open_input_file(f)).iter_batches(batch_size=1_000_000, columns=rcols))
    for b in batches:
        df = b.to_pandas()
        raw_codes.update(df[c].astype(str).unique())
        if nm:
            for code, name in df[[c, nm]].drop_duplicates(c).itertuples(index=False):
                wnames.setdefault(norm(code), name)
        ts = pd.to_datetime(df[t])
        if tz:
            ts = (ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts).dt.tz_convert(tz).dt.tz_localize(None)
        elif ts.dt.tz is not None:
            ts = ts.dt.tz_localize(None)
        df = pd.DataFrame({"code": df[c].astype(str).values, "d": ts.dt.normalize().values, "v": df[v].values})
        parts.append(df.groupby(["code", "d"])["v"].agg(["sum", "count", "min", "max"]))
    g = pd.concat(parts).groupby(level=[0, 1]).agg({"sum": "sum", "count": "sum", "min": "min", "max": "max"})
    g["t"] = g["sum"] / g["count"]
    g = g.reset_index().rename(columns={"count": "hours", "min": "tmin", "max": "tmax"})
    return g[["code", "d", "t", "tmin", "tmax", "hours"]], raw_type, raw_codes, wnames


# ---------------------------------------------------------------- consumption side (one worker per district)
def district_weather(code, root):
    fs, r = eio.open_root(root)
    dset = eio.district_daily(fs, r, code)
    names = dset.schema.names
    want = [x for x in ("date", "t_mean", "t_min", "t_max", "coato_code") if x in names]
    by_period = {}
    for fr in dset.get_fragments():
        p = fr.path.split("period=")[1].split("/")[0] if "period=" in fr.path else "?"
        by_period.setdefault(p, []).append(fr)
    rows, conflicts, col_codes = [], 0, set()
    for p, frs in sorted(by_period.items()):
        need = pd.Period(p, "M").days_in_month if p != "?" else 31
        got = []
        for fr in frs[:3]:                       # first file usually holds every day of the month
            df = fr.to_table(columns=want).to_pandas()
            if "coato_code" in df:
                col_codes.update(df["coato_code"].dropna().astype(str).unique())
            df = df.dropna(subset=["t_mean"])
            conflicts += int((df.groupby("date")["t_mean"].nunique() > 1).sum())   # same day, different weather
            got.append(df.drop_duplicates("date"))
            if pd.concat(got)["date"].nunique() >= need:
                break
        if got:
            rows.append(pd.concat(got).drop_duplicates("date"))
    if not rows:
        return code, None, conflicts, sorted(col_codes)
    df = pd.concat(rows)
    df["d"] = pd.to_datetime(df["date"]).dt.normalize()
    return code, df[["d", "t_mean", "t_min", "t_max"]].drop_duplicates("d"), conflicts, sorted(col_codes)


# ---------------------------------------------------------------- reference tables
def read_ref(ref, name, cols):
    try:
        fs, path = eio.open_root(f"{ref}/{name}/data/data.parquet")
        t = pq.read_table(fs.open_input_file(path))
        return t.select([c for c in cols if c in t.schema.names]).to_pandas()
    except (Exception, SystemExit) as e:         # reference roots are optional context
        print(f"[ref] {name}: not read ({type(e).__name__}: {e})")
        return None


# ---------------------------------------------------------------- comparison
def compare(wd, cons, out):
    dates = pd.DatetimeIndex(sorted(set(wd["d"]) | set(pd.concat([v for v in cons.values()])["d"])))
    di = {d: i for i, d in enumerate(dates)}
    wcodes = sorted(wd["code"].unique())
    W = {k: np.full((len(wcodes), len(dates)), np.nan) for k in ("t", "tmin", "tmax")}
    wi = {c: i for i, c in enumerate(wcodes)}
    rr, cc = wd["code"].map(wi).values, wd["d"].map(di).values
    for k in W:
        W[k][rr, cc] = wd[k].values
    nat = np.nanmean(W["t"], axis=0)             # national daily mean -> anomalies for the correlation
    wnorm = {}
    for c in wcodes:
        wnorm.setdefault(norm(c), c)
    rows = []
    for code, df in sorted(cons.items()):
        E = {k: np.full(len(dates), np.nan) for k in ("t_mean", "t_min", "t_max")}
        idx = df["d"].map(di).values
        for k in E:
            E[k][idx] = df[k].values
        diff = np.abs(W["t"] - E["t_mean"])
        n = np.sum(~np.isnan(diff), axis=1)
        mae = np.where(n >= MIN_OVERLAP, np.nanmean(np.where(np.isnan(diff), np.nan, diff), axis=1), np.inf)
        order = np.argsort(mae)
        best = order[0]
        own_raw = code if code in wi else None
        own = own_raw or wnorm.get(norm(code))
        link = "exact" if own_raw else ("after_normalising" if own else "none")
        rec = {"coato": code, "consumption_days": int(np.sum(~np.isnan(E["t_mean"]))), "weather_code": own,
               "link": link, "best_code": wcodes[best], "best_mae": round(float(mae[best]), 3)}
        if own is not None:
            j = wi[own]
            rec.update(overlap_days=int(n[j]), own_mae=round(float(mae[j]), 3),
                       own_rank=int(np.where(order == j)[0][0]) + 1)
            a, b = W["t"][j] - nat, E["t_mean"] - nat
            ok = ~np.isnan(a) & ~np.isnan(b)
            rec["own_anom_corr"] = round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 3) if ok.sum() > 30 else np.nan
            lags = {}
            for lag in (-1, 0, 1):                # consumption day d vs weather day d + lag
                w = np.roll(W["t"][j], -lag)
                if lag:
                    w[-1 if lag > 0 else 0] = np.nan       # no wrap-around
                dd = np.abs(w - E["t_mean"])
                lags[lag] = np.nanmean(dd) if np.sum(~np.isnan(dd)) >= MIN_OVERLAP else np.inf
            rec["best_lag"] = min(lags, key=lags.get)
            rec["own_mae_lag_m1"], rec["own_mae_lag_p1"] = round(float(lags[-1]), 3), round(float(lags[1]), 3)
            for k_e, k_w in (("t_min", "tmin"), ("t_max", "tmax")):
                dd = np.abs(W[k_w][j] - E[k_e])
                rec[f"own_{k_e}_mae"] = round(float(np.nanmean(dd)), 3) if np.sum(~np.isnan(dd)) else np.nan
            if not np.isfinite(mae[j]):
                rec["verdict"] = "too_little_overlap"
            elif rec["own_rank"] == 1 or mae[j] <= mae[best] + NEAR:
                rec["verdict"] = "ok" if rec["best_lag"] == 0 else "ok_but_shifted"
            else:
                rec["verdict"] = "SUSPECT"
        else:
            rec["verdict"] = "no_weather_row"
        rows.append(rec)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "weather_link_values.csv"), index=False)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=eio.ENRICHED_ROOT)
    ap.add_argument("--weather", default=WEATHER)
    ap.add_argument("--ref", default=REF, help="reference analytical root (branch / location lookups)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--coatos", nargs="*", default=None, help="limit to these districts (default: all finished)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tz", default=None, help="IANA zone if weather timestamps are UTC (e.g. Asia/Tashkent)")
    ap.add_argument("--coato-col"), ap.add_argument("--time-col"), ap.add_argument("--temp-col")
    a = ap.parse_args()
    out = os.path.expanduser(a.out_dir)
    os.makedirs(out, exist_ok=True)

    wd, raw_type, raw_codes, wnames = read_weather(a.weather, a.tz, {"coato": a.coato_col, "time": a.time_col, "temp": a.temp_col})
    fs, r = eio.open_root(a.root)
    codes = eio.finished_districts(fs, r, set(a.coatos) if a.coatos else None)
    cons, conflicts, colcodes = {}, {}, {}
    with ProcessPoolExecutor(a.workers) as ex:
        for code, df, nconf, cc in ex.map(district_weather, codes, [a.root] * len(codes)):
            if df is not None and len(df):
                cons[code] = df
            conflicts[code], colcodes[code] = nconf, cc
            print(f"[consumption] {code}: {0 if df is None else len(df)} days, {nconf} same-day conflicts")

    # ---- 1. keys
    wset, cset = set(raw_codes), set(codes)
    wn, cn = {norm(x) for x in wset}, {norm(x) for x in cset}
    br = read_ref(a.ref, "coato_branch_hierarchy", ["coato_code", "name_uz", "level_id", "location_id", "status"])
    loc = read_ref(a.ref, "coato_locations", ["location_id", "coato_code", "name_uz"])
    allc = sorted(wn | cn)
    key = pd.DataFrame({"code": allc})
    key["in_weather"] = key["code"].isin(wn)
    key["in_consumption"] = key["code"].isin(cn)
    if br is not None:
        b = br.assign(code=br["coato_code"].map(norm)).drop_duplicates("code")
        key = key.merge(b[["code", "name_uz", "level_id", "location_id"]].rename(columns={"name_uz": "branch_name"}),
                        on="code", how="left")
        key["is_branch_code"] = key["branch_name"].notna()
    if loc is not None:
        lc = loc.assign(code=loc["coato_code"].map(norm))
        key["is_location_code"] = key["code"].isin(set(lc["code"]))
        if br is not None:   # the geographic COATO of the branch's own location; differs for 41 branches (guide)
            lmap = dict(zip(loc["location_id"], lc["code"]))
            key["branch_location_code"] = key["location_id"].map(lmap)
            key["branch_code_eq_location_code"] = key["branch_location_code"] == key["code"]
    key["weather_name"] = key["code"].map(wnames)
    if "branch_name" in key:
        key["names_agree"] = [name_agree(w, b) if isinstance(w, str) and isinstance(b, str) else np.nan
                              for w, b in zip(key["weather_name"], key["branch_name"])]
    key["column_codes_in_folder"] = key["code"].map(lambda c: ",".join(colcodes.get(c, [])) if c in colcodes else "")
    key.to_csv(os.path.join(out, "weather_link_codes.csv"), index=False)

    # ---- 2. values
    res = compare(wd, cons, out)

    # ---- summary
    L = []
    p = L.append
    p("WEATHER <-> CONSUMPTION COATO LINK CHECK")
    p(f"weather: {a.weather}\nconsumption: {a.root}\n")
    p("1. KEYS")
    p(f"  weather codes: {len(wset)} (column type {raw_type}; e.g. {sorted(wset)[:5]})")
    p(f"  consumption districts: {len(cset)} (folder names; e.g. {sorted(cset)[:5]})")
    p(f"  exact string matches: {len(wset & cset)}   after zero-padding to 5 digits: {len(wn & cn)}")
    p(f"  consumption districts with no weather code: {len(cn - wn)} {sorted(cn - wn)[:20]}")
    p(f"  weather codes with no consumption district: {len(wn - cn)} {sorted(wn - cn)[:20]}")
    if "is_branch_code" in key:
        w_ = key[key.in_weather]
        p(f"  weather codes that are responsible-branch codes: {int(w_.is_branch_code.sum())}/{len(w_)}")
    if "is_location_code" in key:
        w_ = key[key.in_weather]
        p(f"  weather codes that are geographic COATO-location codes: {int(w_.is_location_code.sum())}/{len(w_)}")
    if "branch_code_eq_location_code" in key:
        both = key[key.in_weather & key.in_consumption]
        diff = both[both["branch_code_eq_location_code"] == False]  # noqa: E712
        p(f"  linked districts whose branch code differs from its location's geographic code: {len(diff)}"
          f" {diff['code'].tolist()[:20]}")
    if "names_agree" in key:
        both = key[key.in_weather & key.in_consumption & key["names_agree"].notna()]
        bad = both[both["names_agree"] == False]  # noqa: E712
        p(f"  linked codes whose weather district_name and branch name agree: {int((both['names_agree'] == True).sum())}"  # noqa: E712
          f"/{len(both)}")
        if len(bad):
            p("  names that do NOT agree (check by eye):")
            p(bad[["code", "weather_name", "branch_name"]].head(40).to_string(index=False))
    mism = [c for c, cc in colcodes.items() if cc and any(norm(x) != norm(c) for x in cc)]
    p(f"  districts whose coato_code column disagrees with the folder name: {len(mism)} {mism[:20]}")
    p(f"  districts with different t_mean values on the same day (should be 0): "
      f"{sum(1 for v in conflicts.values() if v)}")
    p("\n2. VALUES (district's own t_mean vs every weather code's daily mean)")
    p("  verdicts: " + ", ".join(f"{k} {v}" for k, v in res["verdict"].value_counts().items()))
    if "own_mae" in res:
        ok = res[np.isfinite(res["own_mae"].astype(float))]
        p(f"  own-code MAE, deg C: median {ok['own_mae'].median():.3f}, 95th pct {ok['own_mae'].quantile(.95):.3f},"
          f" max {ok['own_mae'].max():.3f}")
        p(f"  own code is the single best match: {int((res['own_rank'] == 1).sum())}/{len(ok)}")
        p("  best lag (days): " + ", ".join(f"{int(k)}: {v}" for k, v in res["best_lag"].value_counts().items()))
        for k in ("own_t_min_mae", "own_t_max_mae"):
            if k in res:
                p(f"  {k}: median {res[k].median():.3f}")
        sus = res[res["verdict"] == "SUSPECT"]
        if len(sus):
            p("\n  SUSPECT districts (another weather code fits better than their own):")
            p(sus[["coato", "weather_code", "own_mae", "own_rank", "best_code", "best_mae"]].to_string(index=False))
    p("\nReading it: own MAE ~0 at lag 0 -> same source, link right. Own MAE 0.5-2 but rank 1 -> different source,"
      " link right.\nSUSPECT -> the code points at another district. Best lag +-1 or t_min much worse than t_max ->"
      " day cut / time zone (try --tz).")
    txt = "\n".join(L)
    print("\n" + txt)
    with open(os.path.join(out, "weather_link_checks.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
