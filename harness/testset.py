"""The fixed test set (configs/test_set.yaml): one implementation shared by the builder, the harness and the scorer.

Files of a built test set (<test_dir> = ~/work/test_set/<version>/):
    manifest.json                       version, spec hash, data version, counts, file hashes
    districts.csv                       district -> region
    panel/<fold>/<district>.parquet     the fixed panel (customer_key): stays in the analytical environment
    actuals/<level>.csv[.gz]            fold, node, date, households, observed, coverage, actual_observed_sum, actual, scored
    history/<fold>/<level>.csv.gz       coverage-adjusted daily totals 2022.. (national, region, district) for direct models

Scores per level (configs/test_set.yaml `levels`):
    coverage_adjusted  national / region / district; district = N x mean of the observed panel households,
                       region / national = the sum of their districts (build_from: district). Bottom-up or direct.
    matched            substation / feeder / transformer (bottom-up only): sum of forecasts vs sum of actuals of the
                       same observed panel households. Also reported for bottom-up forecasts at the coarse levels.

Forecast submission (<forecast_dir>/):
    forecast_meta.json                  {"model": ..., "test_set_version": ..., "test_set_sha256": ...}
    <level>.csv[.gz]                    fold, node, origin, horizon, forecast [, forecast_observed_sum, observed]
        forecast               coverage-adjusted node total for target day origin + horizon (national/region/district)
        forecast_observed_sum  bottom-up: sum of the forecasts of the panel households observed on the target day
                               (required for substation / feeder / transformer)
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
import yaml

LEVELS = ("national", "region", "district", "substation", "feeder", "transformer")
NETWORK = ("transformer", "feeder", "substation")          # network columns 0, 1, 2
FINE = ("substation", "feeder", "transformer")              # written as .csv.gz


# ----------------------------------------------------------------------------- the definition
def load_spec(path):
    raw = open(os.path.expanduser(path), "rb").read()
    spec = yaml.safe_load(raw)
    spec["_sha256"] = hashlib.sha256(raw).hexdigest()
    spec["_path"] = os.path.abspath(os.path.expanduser(path))
    check_spec(spec)
    return spec


def check_spec(spec):
    """The fold dates must follow the agreed rules; a mistyped date fails here, not silently in the scores."""
    H = int(spec["horizons"])
    for f in spec["folds"]:
        s, e = pd.Timestamp(f["test_start"]), pd.Timestamp(f["test_end"])
        fo, lo, c = pd.Timestamp(f["first_origin"]), pd.Timestamp(f["last_origin"]), pd.Timestamp(f["cutoff"])
        assert fo == s - pd.Timedelta(days=H), f"{f['name']}: first_origin must be test_start - {H}"
        assert lo == e - pd.Timedelta(days=1), f"{f['name']}: last_origin must be test_end - 1"
        assert c + pd.Timedelta(days=H) <= fo, f"{f['name']}: training targets (cutoff + {H}) must end by the first origin"
        assert pd.Timestamp(f["train_val"][1]) <= c, f"{f['name']}: inner validation must end by the cutoff"
    names = [f["name"] for f in spec["folds"]]
    assert len(set(names)) == len(names), "fold names must be unique"
    ends = sorted((pd.Timestamp(f["test_start"]), pd.Timestamp(f["test_end"])) for f in spec["folds"])
    for (a0, a1), (b0, b1) in zip(ends, ends[1:]):
        assert a1 < b0, "test blocks must not overlap"


def level_horizons(spec, level):
    h = spec["levels"][level]["horizons"]
    return list(range(1, int(spec["horizons"]) + 1)) if h == "all" else [int(x) for x in h]


def levels_of(spec, include_optional=False):
    return [lv for lv in LEVELS if lv in spec["levels"]
            and (include_optional or not spec["levels"][lv].get("optional"))]


def level_score(spec, level):
    return spec["levels"][level].get("score", "coverage_adjusted")


def direct_levels(spec):
    """Levels a direct (node-level) model may submit: those scored coverage-adjusted."""
    return [lv for lv in levels_of(spec) if level_score(spec, lv) == "coverage_adjusted"]


def built_from(spec, level):
    return spec["levels"][level].get("build_from")


def scored_mask(spec, level, N, n):
    """A node-day is scored when the node has >= min_households panel households, >= min_households of them observed,
    and the observed share reaches the level's min_coverage (0 if not set)."""
    mn = int(spec["scoring"]["min_households"])
    cov = float(spec["levels"][level].get("min_coverage", 0.0))
    N = np.asarray(N, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (N >= mn) & (n >= mn) & (n / np.where(N > 0, N, np.nan) >= cov)


def roll_up(df, group_of, by, total_col, sum_cols=("households", "observed")):
    """District rows -> region / national rows: `sum_cols` and `total_col` summed per group; the total is NaN if any
    district's total is NaN. Adds `districts` and `min_district_observed` (when an `observed` column is given)."""
    d = df.assign(group=df["node"].map(group_of))
    d = d[d["group"].notna()]
    g = d.groupby(list(by) + ["group"], sort=False)
    out = g[list(sum_cols)].sum()
    out[total_col] = g[total_col].sum(min_count=1)
    out.loc[d[total_col].isna().groupby([d[c] for c in by] + [d["group"]], sort=False).sum() > 0, total_col] = np.nan
    out["districts"] = g.size()
    if "observed" in d:
        out["min_district_observed"] = g["observed"].min()
    return out.reset_index().rename(columns={"group": "node"})


def level_file(level, compressed=None):
    gz = level in FINE if compressed is None else compressed
    return f"{level}.csv" + (".gz" if gz else "")


def day_index(spec, day):
    return int((pd.Timestamp(day) - pd.Timestamp(spec["data"]["start"])).days)


def fold_days(spec, f):
    """(test day indices, origin day indices) of a fold, relative to data.start."""
    t0, t1 = day_index(spec, f["test_start"]), day_index(spec, f["test_end"])
    o0, o1 = day_index(spec, f["first_origin"]), day_index(spec, f["last_origin"])
    return np.arange(t0, t1 + 1), np.arange(o0, o1 + 1, int(spec.get("origin_step_days", 1)))


def panel_rows(M, origin, spec):
    """Households eligible at `origin` (day index): the fixed panel of a fold."""
    p = spec["panel"]
    L, R = int(p["lookback_days"]), int(p["recent_days"])
    n_in = np.isfinite(M[:, max(0, origin - L + 1):origin + 1]).sum(1)      # only the window: no full-matrix copy
    n_rec = np.isfinite(M[:, max(0, origin - R + 1):origin + 1]).sum(1)
    return np.flatnonzero((n_in >= int(p["min_input_days"])) & (n_rec >= int(p["min_recent_days"])))


def groups(network, code, region, levels):
    """{level: (group index per household (-1 = not in this level), node names)} for one district's households."""
    n = len(network)
    out = {}
    for lv in levels:
        if lv == "national":
            out[lv] = (np.zeros(n, np.int64), ["national"])
        elif lv == "region":
            out[lv] = (np.zeros(n, np.int64), [str(region)])
        elif lv == "district":
            out[lv] = (np.zeros(n, np.int64), [str(code)])
        else:
            v = network[:, NETWORK.index(lv)]
            ok = v >= 0                                            # unknown / placeholder: not in this level
            u, g = np.unique(v[ok], return_inverse=True)
            gi = np.full(n, -1, np.int64)
            gi[ok] = g
            out[lv] = (gi, [str(int(x)) for x in u])
    return out


def indicator(g, K):
    """Sparse (K, n) 0/1 matrix: row k sums the households in group k (g < 0 = in no group). Memory ~ n, not n x days."""
    from scipy import sparse
    m = np.flatnonzero(g >= 0)
    return sparse.csr_matrix((np.ones(len(m)), (g[m], m)), shape=(K, len(g)))


def column_sums(M_rows, cols, ops, chunk=64):
    """For each level: (observed count, observed kWh sum) per node and column of `cols`, read `chunk` days at a time.
    M_rows: (n, T) kWh with NaN = not observed; ops: {level: indicator matrix}. Returns {level: (n (K, C), S (K, C))}."""
    cols = np.asarray(cols)
    out = {lv: (np.zeros((I.shape[0], len(cols))), np.zeros((I.shape[0], len(cols)))) for lv, I in ops.items()}
    for a in range(0, len(cols), chunk):
        c = cols[a:a + chunk]
        X = M_rows[:, c].astype(np.float64)                                 # (n, <= chunk)
        obs = np.isfinite(X)
        X[~obs] = 0.0
        O = obs.astype(np.float64)
        for lv, I in ops.items():
            out[lv][0][:, a:a + len(c)] = I @ O
            out[lv][1][:, a:a + len(c)] = I @ X
    return out


def bincount2(g, values, G):
    """Sum rows of `values` (n, k) into G groups by g (n,), skipping g < 0 -> (G, k)."""
    m = g >= 0
    k = values.shape[1]
    flat = (g[m][:, None] * k + np.arange(k)[None, :]).ravel()
    return np.bincount(flat, weights=values[m].ravel(), minlength=G * k).reshape(G, k)


class NodeSum:
    """Adds per-district node arrays into national node arrays (a node may span districts: substations)."""

    def __init__(self):
        self.idx, self.parts = {}, []

    def add(self, names, arrays):
        """names: list; arrays: dict of (G, ...) arrays, summed per name."""
        for i, nm in enumerate(names):
            j = self.idx.get(nm)
            if j is None:
                self.idx[nm] = len(self.parts)
                self.parts.append({k: np.array(v[i], dtype=np.float64) for k, v in arrays.items()})
            else:
                for k, v in arrays.items():
                    self.parts[j][k] += v[i]

    def names(self):
        return list(self.idx)

    def stack(self, key):
        return np.stack([p[key] for p in self.parts]) if self.parts else np.zeros((0,))


def _read_num(path):
    try:
        v = open(path).read().strip()
        return None if v in ("max", "") else float(v)
    except OSError:
        return None


def memory_line(worker_peak_bytes=None):
    """'worker peak 1.2 GB | pod 5.3 of 16.0 GB, OOM kills 0': this process and the pod's memory (cgroup v2 or v1)."""
    import resource
    parts = []
    if worker_peak_bytes is not None:
        parts.append(f"worker peak {worker_peak_bytes / 1e9:.1f} GB")
    else:
        parts.append(f"this process peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9:.1f} GB")
    cur = _read_num("/sys/fs/cgroup/memory.current") or _read_num("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    lim = _read_num("/sys/fs/cgroup/memory.max") or _read_num("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if cur is not None:
        txt = f"pod {cur / 1e9:.1f}" + (f" of {lim / 1e9:.1f} GB" if lim and lim < 1e15 else " GB (no limit found)")
        try:
            ev = dict(line.split() for line in open("/sys/fs/cgroup/memory.events"))
            txt += f", OOM kills so far {ev.get('oom_kill', '?')}"
        except (OSError, ValueError):
            pass
        parts.append(txt)
    return " | ".join(parts)


def peak_rss_bytes():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def read_manifest(test_dir):
    p = os.path.join(os.path.expanduser(test_dir), "manifest.json")
    if not os.path.exists(p):
        raise SystemExit(f"no manifest.json in {test_dir}: build the test set first (scripts/test_set/build_test_set.py)")
    return json.load(open(p))


def check_same_spec(spec, manifest):
    if manifest["test_set_sha256"] != spec["_sha256"]:
        raise SystemExit(f"configs/test_set.yaml changed since the test set was built "
                         f"({manifest['test_set_sha256'][:12]} built vs {spec['_sha256'][:12]} now): rebuild under a new "
                         f"version, or restore the file")


def day_type(dates, holidays):
    d = pd.to_datetime(pd.Series(dates))
    hol = set(pd.to_datetime(list(holidays or {})).normalize())
    return np.where(d.dt.normalize().isin(hol), "holiday", np.where(d.dt.dayofweek >= 5, "weekend", "workday"))


# ----------------------------------------------------------------------------- scoring (shared by every model)
def _read_level(path):
    for p in (path, path + ".gz", path[:-3] if path.endswith(".gz") else None):
        if p and os.path.exists(p):
            return pd.read_csv(p, dtype={"node": str})
    return None


def load_forecasts(forecast_dir, spec, level):
    df = _read_level(os.path.join(forecast_dir, f"{level}.csv"))
    if df is None:
        return None
    main = "forecast" if level_score(spec, level) == "coverage_adjusted" else "forecast_observed_sum"
    need = {"fold", "node", "origin", "horizon", main}
    if need - set(df.columns):
        raise SystemExit(f"{level}: forecast file is missing columns {sorted(need - set(df.columns))}")
    df["origin"] = pd.to_datetime(df["origin"])
    df["horizon"] = df["horizon"].astype(int)
    return df


def expected_keys(actuals, spec, level):
    """Every (fold, node, origin, horizon) a complete submission must contain: scored node-days x level horizons."""
    a = actuals[actuals["scored"]][["fold", "node", "date"]]
    hz = level_horizons(spec, level)
    e = a.loc[a.index.repeat(len(hz))].copy()
    e["horizon"] = np.tile(hz, len(a))
    e["origin"] = e["date"] - pd.to_timedelta(e["horizon"], unit="D")
    return e


def join_level(forecast_dir, test_dir, spec, level):
    """Forecasts joined to actuals for one level, after strict key checks. None if the level was not submitted."""
    f = load_forecasts(forecast_dir, spec, level)
    if f is None:
        return None
    a = _read_level(os.path.join(test_dir, "actuals", f"{level}.csv"))
    if a is None:
        raise SystemExit(f"{level}: no actuals in {test_dir}/actuals")
    a["date"] = pd.to_datetime(a["date"])
    a["scored"] = a["scored"].astype(str).str.lower().isin(("true", "1"))
    e = expected_keys(a, spec, level)
    key = ["fold", "node", "origin", "horizon"]
    dup = f.duplicated(key)
    if dup.any():
        raise SystemExit(f"{level}: {int(dup.sum())} duplicated forecast rows, e.g.\n{f[dup].head(3)}")
    m = e.merge(f, on=key, how="left", indicator=True)
    miss = m["_merge"] == "left_only"
    if miss.any():
        raise SystemExit(f"{level}: {int(miss.sum())} of {len(m)} required forecasts are missing, e.g.\n"
                         f"{m.loc[miss, key].head(5).to_string(index=False)}")
    main = "forecast" if level_score(spec, level) == "coverage_adjusted" else "forecast_observed_sum"
    bad = ~np.isfinite(m[main].astype(float))
    if bad.any():
        raise SystemExit(f"{level}: {int(bad.sum())} values of {main} are not finite numbers")
    m = m.drop(columns="_merge")
    if "observed" in m:
        m = m.rename(columns={"observed": "observed_forecast"})
    m = m.merge(a, on=["fold", "node", "date"], how="left")
    if "observed_forecast" in m:                                  # bottom-up: same panel, same observed households?
        off = m["observed_forecast"].astype(float) != m["observed"].astype(float)
        if off.any():
            raise SystemExit(f"{level}: on {int(off.sum())} node-days the forecasts were summed over a different set of "
                             f"observed households than the actuals (built on another panel or data version)")
    m["level"] = level
    return m


def write_template(test_dir, spec, out_dir, levels=None):
    """The exact rows a submission must contain, per level (forecast left empty): for models outside the harness."""
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for lv in levels or direct_levels(spec):
        if level_score(spec, lv) != "coverage_adjusted":
            raise SystemExit(f"{lv} is scored bottom-up only (matched): no template for direct forecasts")
        a = _read_level(os.path.join(test_dir, "actuals", f"{lv}.csv"))
        if a is None:
            continue
        a["date"] = pd.to_datetime(a["date"])
        a["scored"] = a["scored"].astype(str).str.lower().isin(("true", "1"))
        e = expected_keys(a, spec, lv)
        e = e[["fold", "node", "origin", "horizon"]].assign(origin=lambda x: x["origin"].dt.date, forecast="")
        path = os.path.join(out_dir, level_file(lv))
        e.to_csv(path, index=False)
        written[lv] = len(e)
    json.dump({"model": "<your model name>", "test_set_version": spec["version"], "test_set_sha256": spec["_sha256"]},
              open(os.path.join(out_dir, "forecast_meta.json"), "w"), indent=2)
    return written


def _stats(g, fc, ac):
    err = g[fc] - g[ac]
    tot = g[ac].sum()
    return pd.Series({"sysmape": np.abs(err).sum() / tot if tot > 0 else np.nan,
                      "bias": err.sum() / tot if tot > 0 else np.nan,
                      "node_days": int(len(g))})


def _node_median(g, fc, ac):
    per = g.groupby("node").apply(lambda x: np.abs(x[fc] - x[ac]).sum() / x[ac].sum() if x[ac].sum() > 0 else np.nan,
                                  include_groups=False)
    return pd.Series({"median_node_sysmape": per.median(), "p90_node_sysmape": per.quantile(0.9), "nodes": int(len(per))})


def score(forecast_dir, test_dir, spec, levels=None):
    """-> dict of DataFrames: by_level, by_fold, spec_check, bias_audit; and the levels that were scored."""
    test_dir, forecast_dir = os.path.expanduser(test_dir), os.path.expanduser(forecast_dir)
    man = read_manifest(test_dir)
    check_same_spec(spec, man)
    meta_p = os.path.join(forecast_dir, "forecast_meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    if meta.get("test_set_sha256") not in (None, spec["_sha256"]):
        raise SystemExit("these forecasts were made for a different test set definition (forecast_meta.json)")
    if not meta:
        print("[score] WARNING: no forecast_meta.json: cannot confirm the forecasts were made for this test set")
    sc = spec["scoring"]
    rows, fold_rows, joined = [], [], {}
    for lv in levels or levels_of(spec, include_optional=True):
        m = join_level(forecast_dir, test_dir, spec, lv)
        if m is None:
            continue
        joined[lv] = m
        primary = level_score(spec, lv)
        if primary == "matched":
            variants = [("matched", "forecast_observed_sum", "actual_observed_sum", None)]
        else:
            matched = "forecast_observed_sum" in m and m["forecast_observed_sum"].notna().all()
            sens = float(sc.get("sensitivity_min_coverage", 0.8))
            variants = [("coverage_adjusted", "forecast", "actual", None)] + \
                       ([("matched", "forecast_observed_sum", "actual_observed_sum", None)] if matched else []) + \
                       [(f"coverage_adjusted_cov{int(round(sens * 100))}", "forecast", "actual", m["coverage"] >= sens)]
        for name, fc, ac, keep in variants:
            mm = m if keep is None else m[keep]
            for h, g in mm.groupby("horizon"):
                rows.append({"level": lv, "score": name, "headline": name == primary, "horizon": int(h),
                             **_stats(g, fc, ac), **_node_median(g, fc, ac)})
                for fo, gf in g.groupby("fold", sort=False):
                    fold_rows.append({"level": lv, "score": name, "fold": fo, "horizon": int(h), **_stats(gf, fc, ac)})
    if not joined:
        raise SystemExit(f"no <level>.csv[.gz] files in {forecast_dir}")
    by_level, by_fold = pd.DataFrame(rows), pd.DataFrame(fold_rows)

    # ТЗ 4.2.4 acceptance check on the primary level
    pl = sc["primary_level"]
    spec_rows = []
    for h, thr in sorted(sc["spec_thresholds"].items(), key=lambda x: int(x[0])):
        r = by_level[(by_level["level"] == pl) & (by_level["score"] == "coverage_adjusted") & (by_level["horizon"] == int(h))]
        v = float(r["sysmape"].iloc[0]) if len(r) else np.nan
        spec_rows.append({"level": pl, "horizon": int(h), "sysmape": v, "threshold": float(thr),
                          "accuracy": 1 - v if np.isfinite(v) else np.nan, "pass": bool(np.isfinite(v) and v <= float(thr))})
    spec_check = pd.DataFrame(spec_rows)

    # bias audit (ТЗ): season = fold, day type, geographic zone = region; each vs the overall value on the same data
    audit = []
    ratio = float(sc["bias_audit_ratio"])
    ctrl = [int(h) for h in sc["control_horizons"]]
    if pl in joined:
        n = joined[pl]
        n = n.assign(day_type=day_type(n["date"], spec.get("holidays")))
        for h in ctrl:
            g = n[n["horizon"] == h]
            overall = _stats(g, "forecast", "actual")["sysmape"]
            for col, kind in (("fold", "season"), ("day_type", "day type")):
                for val, gg in g.groupby(col, sort=False):
                    s = _stats(gg, "forecast", "actual")
                    audit.append({"group": kind, "value": val, "horizon": h, "sysmape": s["sysmape"], "bias": s["bias"],
                                  "overall": overall, "ratio": s["sysmape"] / overall if overall else np.nan,
                                  "node_days": s["node_days"]})
    if "region" in joined and pl in joined:
        r = joined["region"]
        for h in ctrl:
            overall = _stats(joined[pl][joined[pl]["horizon"] == h], "forecast", "actual")["sysmape"]
            for val, gg in r[r["horizon"] == h].groupby("node"):
                s = _stats(gg, "forecast", "actual")
                audit.append({"group": "region", "value": val, "horizon": h, "sysmape": s["sysmape"], "bias": s["bias"],
                              "overall": overall, "ratio": s["sysmape"] / overall if overall else np.nan,
                              "node_days": s["node_days"]})
    bias_audit = pd.DataFrame(audit)
    if len(bias_audit):
        bias_audit["pass"] = bias_audit["ratio"] <= ratio
    return dict(by_level=by_level, by_fold=by_fold, spec_check=spec_check, bias_audit=bias_audit,
                levels=list(joined), meta=meta, manifest=man)


def report(res, spec, reference=None):
    """Printable summary; `reference` = score() of the persistence reference on the same test set."""
    L = []
    p = L.append
    sc = spec["scoring"]
    meta = res["meta"]
    p(f"TEST SET {spec['version']} ({spec['_sha256'][:12]}) | model {meta.get('model', '?')} | "
      f"commit {str(meta.get('git_commit', ''))[:7]}")
    p(f"folds: " + ", ".join(f"{f['name']} {f['test_start']}..{f['test_end']}" for f in spec["folds"]))
    p("")
    ndays = sum((pd.Timestamp(f["test_end"]) - pd.Timestamp(f["test_start"])).days + 1 for f in spec["folds"])
    p(f"ТЗ 4.2.4 acceptance check ({sc['primary_level']}, coverage-adjusted sysMAPE over all {ndays} test days):")
    if sc["primary_level"] not in res["levels"]:
        p(f"  not checked: no {sc['primary_level']} forecasts in this submission")
    else:
        for r in res["spec_check"].to_dict("records"):
            p(f"  h={r['horizon']:<3} sysMAPE {r['sysmape']:6.2%}  limit {r['threshold']:5.1%}  -> "
              f"{'PASS' if r['pass'] else 'FAIL'}")
        allp = bool(res["spec_check"]["pass"].all())
        p(f"  overall: {'ALL THRESHOLDS MET' if allp else 'NOT MET (grounds for rejection under the ТЗ)'}")
    p("")
    bl = res["by_level"]
    ctrl = [int(h) for h in sc["control_horizons"]]
    sens = f"coverage_adjusted_cov{int(round(float(sc.get('sensitivity_min_coverage', 0.8)) * 100))}"
    tables = [("headline sysMAPE (pooled): coverage-adjusted for national / region / district, matched below", bl["headline"]),
              ("bottom-up check: matched sysMAPE at the coverage-adjusted levels", (bl["score"] == "matched") & ~bl["headline"]),
              (f"sensitivity: coverage-adjusted sysMAPE on days with >= {sc.get('sensitivity_min_coverage', 0.8):.0%} "
               f"coverage only", bl["score"] == sens)]
    for title, sel in tables:
        t = bl[sel & bl["horizon"].isin(ctrl)]
        if t.empty:
            continue
        p(title + ":")
        piv = t.pivot(index="level", columns="horizon", values="sysmape").reindex([x for x in LEVELS if x in t["level"].values])
        p(piv.map(lambda x: f"{x:6.2%}").to_string())
        p("")
    if reference is not None:
        rb = reference["by_level"]
        a = bl[bl["headline"] & bl["horizon"].isin(ctrl)].set_index(["level", "horizon"])["sysmape"]
        b = rb[rb["headline"] & rb["horizon"].isin(ctrl)].set_index(["level", "horizon"])["sysmape"]
        rel = (a / b - 1).dropna().unstack("horizon")
        p(f"vs the reference {reference['meta'].get('model', '')} (relative change in sysMAPE; negative = better):")
        p(rel.reindex([x for x in LEVELS if x in rel.index]).map(lambda x: f"{x:+6.1%}").to_string())
        p("")
    ba = res["bias_audit"]
    if len(ba):
        worst = ba.sort_values("ratio", ascending=False).head(8)
        p(f"bias audit (group sysMAPE / overall, limit {sc['bias_audit_ratio']}x): "
          f"{int((~ba['pass']).sum())} of {len(ba)} group-horizons over the limit; worst:")
        p(worst[["group", "value", "horizon", "sysmape", "overall", "ratio", "bias"]].to_string(
            index=False, formatters={"sysmape": "{:.2%}".format, "overall": "{:.2%}".format, "ratio": "{:.2f}".format,
                                     "bias": "{:+.2%}".format}))
        p("")
    p("notes: " + " | ".join(spec.get("notes", [])))
    return "\n".join(L)


def metrics_for_mlflow(res, prefix="test"):
    out = {}
    for r in res["by_level"].itertuples():
        if r.headline:                                            # the level's own score
            out[f"{prefix}/{r.level}/sysmape_h{r.horizon}"] = r.sysmape
            out[f"{prefix}/{r.level}/bias_h{r.horizon}"] = r.bias
        elif r.score == "matched":
            out[f"{prefix}/{r.level}/matched_sysmape_h{r.horizon}"] = r.sysmape
        else:
            out[f"{prefix}/{r.level}/{r.score.replace('coverage_adjusted_', 'sysmape_')}_h{r.horizon}"] = r.sysmape
    for r in res["spec_check"].to_dict("records"):
        out[f"{prefix}/spec_pass_h{r['horizon']}"] = float(r["pass"])
    out[f"{prefix}/spec_pass_all"] = float(res["spec_check"]["pass"].all())
    if len(res["bias_audit"]):
        out[f"{prefix}/bias_audit_fails"] = float((~res["bias_audit"]["pass"]).sum())
    return out


def save_scores(res, out_dir, text):
    os.makedirs(out_dir, exist_ok=True)
    for k in ("by_level", "by_fold", "spec_check", "bias_audit"):
        res[k].to_csv(os.path.join(out_dir, f"score_{k}.csv"), index=False)
    open(os.path.join(out_dir, "score_report.txt"), "w").write(text + "\n")
