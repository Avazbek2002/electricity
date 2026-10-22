"""The standard split and evaluation (configs/test_set.yaml v2 = ML_EXPERIMENTS_GUIDE_RU.md parts 2 and 4).

One implementation shared by the builder, the harness and the scorer.

Built test set (<test_dir> = ~/work/test_set/<version>/):
    manifest.json             version, spec hash, data version, counts, file hashes
    districts.csv             district -> region
    targets/district.csv.gz   date, node (district), n_rows, n_obs, n_est, n_lg, kwh_obs, kwh_est, kwh_lg, n_right,
                              crisis, interruption, coverage, T2, T3, flag, flag_reason, artefact  (district x day, 2022 ..)
    calendar.csv              the standing calendar (known in advance) for every day

Submission (<forecast_dir>/), any model -- bottom-up (client level) or district level:
    forecast_meta.json        {"model", "test_set_version", "test_set_sha256", "target", "weather", ...}
    district.csv[.gz]         node, origin, horizon, forecast       (issue date `origin`, target day origin + horizon)
    [network/<level>.csv.gz   supplementary, bottom-up harness runs only]

Scoring (guide part 4): per split, district rows that are clean (not flagged, target known) and forecast by every
compared model (the model and B0); national = sum over districts for an (o, h) with >= 98 % of the run's districts
present; sysMAPE_h = sum |P - A| / sum A and bias = sum (P - A) / sum A over the issue dates of the split.
"""
import hashlib
import json
import math
import os

import numpy as np
import pandas as pd
import yaml

NETWORK = ("transformer", "feeder", "substation")          # network columns 0, 1, 2 (supplementary levels)


# ----------------------------------------------------------------------------- the definition
def load_spec(path):
    raw = open(os.path.expanduser(path), "rb").read()
    spec = yaml.safe_load(raw)
    spec["_sha256"] = hashlib.sha256(raw).hexdigest()
    spec["_path"] = os.path.abspath(os.path.expanduser(path))
    check_spec(spec)
    return spec


def check_spec(spec):
    """The dates must follow the guide's rules; a mistyped date fails here, not silently in the scores."""
    R = pd.to_datetime(spec["refits"])
    assert (np.diff(R.values).astype("timedelta64[D]").astype(int) > 0).all(), "refits must be increasing"
    for name, s in spec["splits"].items():
        a, b = pd.Timestamp(s["issue_start"]), pd.Timestamp(s["issue_end"])
        assert a <= b, f"{name}: issue_start after issue_end"
        assert R[0] <= a, f"{name}: issue dates before the first refit"
    v, t = spec["splits"]["validation"], spec["splits"]["test"]
    assert pd.Timestamp(v["target_end"]) < pd.Timestamp(t["issue_start"]), "validation targets must end before the test"
    assert pd.Timestamp(t["issue_end"]) + pd.Timedelta(days=int(spec["horizons"])) <= pd.Timestamp(spec["data"]["end"]), \
        "test targets must lie inside the data"
    assert int(spec["history_gap_days"]) >= 1, "history must end before the issue date"


def issue_dates(spec, split):
    s = spec["splits"][split]
    return pd.date_range(s["issue_start"], s["issue_end"], freq="D")


def refit_for(spec, day):
    """The refit date R that serves issue date `day` (the last R <= day)."""
    R = pd.to_datetime(spec["refits"])
    i = int(np.searchsorted(R.values, np.datetime64(pd.Timestamp(day)), side="right")) - 1
    if i < 0:
        raise ValueError(f"{day} is before the first refit")
    return R[i]


def refit_blocks(spec, splits):
    """[(R, issue dates served by R within the requested splits)] in refit order."""
    days = pd.DatetimeIndex(sorted(set().union(*[set(issue_dates(spec, s)) for s in splits])))
    R = pd.to_datetime(spec["refits"])
    out = []
    for i, r in enumerate(R):
        nxt = R[i + 1] if i + 1 < len(R) else pd.Timestamp.max
        sel = days[(days >= r) & (days < nxt)]
        if len(sel):
            out.append((r, sel))
    return out


def train_window(spec, R, horizon=None):
    """(cutoff, data_end) for a model fitted at R: every training pair has its target <= R - 1 (guide part 4).
    cutoff = the last training issue date whose every horizon still ends by R - 1."""
    H = int(horizon or spec["horizons"])
    data_end = pd.Timestamp(R) - pd.Timedelta(days=1)
    return data_end - pd.Timedelta(days=H), data_end


def day_index(spec, day):
    return int((pd.Timestamp(day) - pd.Timestamp(spec["data"]["start"])).days)


# ----------------------------------------------------------------------------- the standing calendar (guide code)
FIXED = {(1, 1): "newyear", (3, 8): "women", (3, 21): "navruz", (5, 9): "remembrance", (9, 1): "independence",
         (10, 1): "teachers", (12, 8): "constitution"}


def hijri(y, m, d):
    """Tabular Islamic calendar -> Gregorian date (computable years ahead)."""
    jd = d + math.ceil(29.5 * (m - 1)) + (y - 1) * 354 + (3 + 11 * y) // 30 + 1948439.5 - 1
    return pd.Timestamp("1858-11-17") + pd.Timedelta(days=round(jd - 2400000.5))


def standing_calendar(idx):
    """Holiday / day-off / Ramadan flags known in advance: fixed dates, tabular Hayit and Ramadan, weekend rule.
    (ML guide part 5.3; same as standing_calendar() in the guide's national_v3.py.)"""
    idx = pd.DatetimeIndex(idx)
    name = pd.Series("none", index=idx)
    ram = pd.Series(0, index=idx)
    for (mo, dy), nm in FIXED.items():
        name[(idx.month == mo) & (idx.day == dy)] = nm
    for hy in range(1440, 1452):
        for dt, nm in ((hijri(hy, 10, 1), "ramazan"), (hijri(hy, 12, 10), "kurban")):
            if dt in name.index:
                name[dt] = nm
        r0, r1 = hijri(hy, 9, 1), hijri(hy, 10, 1) - pd.Timedelta(days=1)
        ram[(idx >= r0) & (idx <= r1)] = 1
    hol = (name != "none").astype(int)
    xoff = pd.Series(0, index=idx)
    for dt in idx[hol.values == 1]:
        if dt.weekday() >= 5:
            nx = dt + pd.Timedelta(days=7 - dt.weekday())
            while nx in hol.index and (hol[nx] or xoff[nx]):
                nx += pd.Timedelta(days=1)
            if nx in xoff.index:
                xoff[nx] = 1
    return pd.DataFrame({"date": idx, "hol": hol.values, "xoff": xoff.values, "ram": ram.values, "hname": name.values})


def day_type(dates, calendar):
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    cal = calendar.set_index("date")
    off = (cal["hol"] + cal["xoff"]).reindex(d).fillna(0).to_numpy() > 0
    return np.where(off, "holiday", np.where(d.dayofweek >= 5, "weekend", "workday"))


def season(dates):
    m = pd.DatetimeIndex(pd.to_datetime(dates)).month
    return np.select([np.isin(m, (12, 1, 2)), np.isin(m, (3, 4, 5)), np.isin(m, (6, 7, 8))],
                     ["winter", "spring", "summer"], "autumn")


# ----------------------------------------------------------------------------- the target (guide part 2)
def target_series(c, spec):
    """District-day counts -> T2, T3, coverage, flags. c: one district, one row per date, with n_rows, n_obs, n_est,
    n_lg, kwh_obs, kwh_est, kwh_lg, n_right, crisis, interruption (kWh NaN where nothing of that kind)."""
    t = spec["target"]
    c = c.copy()
    nc = c["n_obs"].fillna(0) + c["n_est"].fillna(0)
    kwh = c["kwh_obs"].fillna(0) + c["kwh_est"].fillna(0)
    no_data = c["kwh_obs"].isna() & c["kwh_est"].isna()
    few = nc < int(t["min_known_client_days"])
    edge = (c["n_right"].fillna(0) > 0) if t.get("exclude_right_edge", True) else pd.Series(False, index=c.index)
    no_rows = c["n_rows"].isna() | (c["n_rows"] <= 0)
    artefact = no_data | few | edge | no_rows
    with np.errstate(invalid="ignore", divide="ignore"):
        c["T2"] = (c["n_rows"] * kwh / nc.where(nc > 0)).where(~artefact)
        nl = nc + c["n_lg"].fillna(0)
        c["T3"] = (c["n_rows"] * (kwh + c["kwh_lg"].fillna(0)) / nl.where(nl > 0)).where(~artefact)
        c["coverage"] = nc / c["n_rows"].where(c["n_rows"] > 0)
    low = c["coverage"].fillna(0) < float(t["min_coverage"])
    crisis = c["crisis"].fillna(0) > 0
    interr = c["interruption"].fillna(0) > 0
    c["flag"] = artefact | low | crisis | interr
    reason = np.full(len(c), "", dtype=object)
    for m, txt in ((no_data | no_rows, "no_data"), (few, "few_known"), (edge, "right_edge"), (low, "low_coverage"),
                   (crisis, "crisis"), (interr, "data_interruption")):
        reason = np.where(m.to_numpy(), np.where(reason == "", txt, reason + "+" + txt), reason)
    c["flag_reason"] = reason
    c["artefact"] = artefact
    return c


def load_targets(test_dir, spec, target="T2"):
    p = os.path.join(os.path.expanduser(test_dir), "targets", "district.csv.gz")
    if not os.path.exists(p):
        raise SystemExit(f"no targets in {test_dir}: build the test set first (scripts/test_set/build_test_set.py)")
    t = pd.read_csv(p, dtype={"node": str}, parse_dates=["date"])
    if target not in t:
        raise SystemExit(f"target {target} not in the built test set (T2 / T3)")
    t["y"] = t[target]
    t["flag"] = t["flag"].astype(str).str.lower().isin(("true", "1"))
    t["artefact"] = t["artefact"].astype(str).str.lower().isin(("true", "1"))
    return t


# ----------------------------------------------------------------------------- B0 (guide part 6, build() in the code)
def b0_forecasts(targets, spec, nodes=None, issue=None):
    """B0 = m28(o - gap) x last-year seasonal ratio, else two-years-back ratio, else m28 alone, for every district,
    issue date and horizon. Uses the target series only (as the guide's build())."""
    H, gap = int(spec["horizons"]), int(spec["history_gap_days"])
    start = pd.Timestamp(spec["data"]["start"])
    if issue is None:
        issue = pd.DatetimeIndex(sorted(set(issue_dates(spec, "validation")) | set(issue_dates(spec, "test"))))
    oi = ((issue - start).days).to_numpy()
    out = []
    for node, g in targets.groupby("node", sort=True):
        if nodes is not None and node not in nodes:
            continue
        g = g.set_index("date")
        full = pd.date_range(start, g.index.max(), freq="D")
        g = g.reindex(full)
        s = g["y"].where(~g["artefact"].fillna(True).astype(bool))
        s_ly = s.where(~g["flag"].fillna(True).astype(bool))
        m28 = s.rolling(28, min_periods=20).mean().to_numpy()
        c7 = s_ly.rolling(7, center=True, min_periods=5).mean().to_numpy()
        m28ly = s_ly.rolling(28, min_periods=20).mean().to_numpy()
        nd = len(full)
        O = np.repeat(oi, H)
        Hh = np.tile(np.arange(1, H + 1), len(oi))
        D, LK = O + Hh, O - gap

        def gi(a, i):
            return np.where((i >= 0) & (i < nd), a[np.clip(i, 0, nd - 1)], np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            ly = np.clip(gi(c7, D - 364) / gi(m28ly, LK - 364), 0.6, 1.7)
            ly2 = np.clip(gi(c7, D - 728) / gi(m28ly, LK - 728), 0.6, 1.7)
        ratio = np.where(np.isfinite(ly), ly, np.where(np.isfinite(ly2), ly2, 1.0))
        out.append(pd.DataFrame({"node": node, "origin": start + pd.to_timedelta(O, unit="D"), "horizon": Hh,
                                 "forecast": gi(m28, LK) * ratio,
                                 "b0src": np.where(np.isfinite(ly), 0, np.where(np.isfinite(ly2), 1, 2))}))
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------- reading submissions
def _read(path):
    for p in (path, path + ".gz"):
        if os.path.exists(p):
            return pd.read_csv(p, dtype={"node": str})
    return None


def load_forecasts(forecast_dir):
    df = _read(os.path.join(os.path.expanduser(forecast_dir), "district.csv"))
    if df is None:
        raise SystemExit(f"no district.csv[.gz] in {forecast_dir}")
    need = {"node", "origin", "horizon", "forecast"}
    if need - set(df.columns):
        raise SystemExit(f"district forecasts are missing columns {sorted(need - set(df.columns))}")
    df["origin"] = pd.to_datetime(df["origin"])
    df["horizon"] = df["horizon"].astype(int)
    df["node"] = df["node"].astype(str)
    dup = df.duplicated(["node", "origin", "horizon"])
    if dup.any():
        raise SystemExit(f"{int(dup.sum())} duplicated forecast rows (node, origin, horizon), e.g.\n{df[dup].head(3)}")
    return df[["node", "origin", "horizon", "forecast"]]


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


# ----------------------------------------------------------------------------- the common panel and the scores
def split_rows(spec, split, df):
    s = spec["splits"][split]
    m = (df["origin"] >= s["issue_start"]) & (df["origin"] <= s["issue_end"])
    if s.get("target_end"):
        m &= df["date"] <= pd.Timestamp(s["target_end"])
    return df[m]


def common_panel(models, targets, spec, split, regions=None):
    """District rows of one split that are clean and forecast by EVERY model; national and region sums per (o, h)
    with >= min_district_share of the districts present. models: {name: DataFrame node, origin, horizon, forecast}."""
    names = list(models)
    base = models[names[0]].rename(columns={"forecast": names[0]})
    for n in names[1:]:
        base = base.merge(models[n].rename(columns={"forecast": n}), on=["node", "origin", "horizon"], how="outer")
    base["date"] = base["origin"] + pd.to_timedelta(base["horizon"], unit="D")
    base = split_rows(spec, split, base)
    nodes = sorted(models[names[0]]["node"].unique())
    tg = targets[targets["node"].isin(nodes)][["node", "date", "y", "flag"]]
    m = base.merge(tg, on=["node", "date"], how="left")
    expected = m[~m["flag"].fillna(True).astype(bool) & np.isfinite(m["y"])]
    ok = expected[names].apply(np.isfinite).all(axis=1)
    dist = expected[ok]
    missing = {n: int((~np.isfinite(expected[n])).sum()) for n in names}
    share = float(spec["national"]["min_district_share"])
    key = ["origin", "horizon"]
    nat = dist.groupby(key)[["y"] + names].sum()
    cnt = dist.groupby(key).size()
    nat = nat[cnt >= int(share * len(nodes))].reset_index()
    nat["date"] = nat["origin"] + pd.to_timedelta(nat["horizon"], unit="D")
    reg = None
    if regions is not None and len(dist):
        d2 = dist.assign(region=dist["node"].map(regions))
        size = pd.Series(regions).reindex(nodes).value_counts()
        g = d2.groupby(["region"] + key)
        reg = g[["y"] + names].sum()
        rc = g.size()
        need = np.asarray([int(share * size.get(r, 0)) for r in rc.index.get_level_values(0)])
        reg = reg[rc.to_numpy() >= need].reset_index()
        reg["date"] = reg["origin"] + pd.to_timedelta(reg["horizon"], unit="D")
    return dict(district=dist, national=nat, region=reg, nodes=nodes, expected_rows=int(len(expected)),
                missing=missing)


def _sm(q, m):
    return float((q[m] - q["y"]).abs().sum() / q["y"].sum()) if len(q) and q["y"].sum() > 0 else np.nan


def _mae(q, m):
    return float((q[m] - q["y"]).abs().mean()) if len(q) else np.nan


def _bias(q, m):
    return float((q[m] - q["y"]).sum() / q["y"].sum()) if len(q) and q["y"].sum() > 0 else np.nan


def score(forecast_dir, test_dir, spec, target="T2", reference=True, others=None):
    """-> dict: per split national / district scores for the model (and B0, and `others` {name: forecast_dir}) on the
    common panel, day counts, the ТЗ check (test, national), bias audit (test), supplementary network scores."""
    test_dir, forecast_dir = os.path.expanduser(test_dir), os.path.expanduser(forecast_dir)
    man = read_manifest(test_dir)
    check_same_spec(spec, man)
    meta_p = os.path.join(forecast_dir, "forecast_meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    if meta.get("test_set_sha256") not in (None, spec["_sha256"]):
        raise SystemExit("these forecasts were made for a different test set definition (forecast_meta.json)")
    if not meta:
        print("[score] WARNING: no forecast_meta.json: cannot confirm the forecasts were made for this test set")
    if meta.get("target", target) != target:
        print(f"[score] NOTE: the model was trained on {meta.get('target')}, scored here against {target}")
    name = meta.get("model", "model")
    models = {name: load_forecasts(forecast_dir)}
    for on, od in (others or {}).items():
        models[on] = load_forecasts(od)
    targets = load_targets(test_dir, spec, target)
    nodes = set(models[name]["node"])
    unknown = nodes - set(targets["node"])
    if unknown:
        raise SystemExit(f"forecasts for districts not in the test set: {sorted(unknown)[:10]}")
    if reference:
        b0 = b0_forecasts(targets, spec, nodes=nodes)
        models["B0"] = b0[["node", "origin", "horizon", "forecast"]]
    regions = pd.read_csv(os.path.join(test_dir, "districts.csv"), dtype=str).set_index("district")["region"].to_dict()
    cal = pd.read_csv(os.path.join(test_dir, "calendar.csv"), parse_dates=["date"])
    H = list(range(1, int(spec["horizons"]) + 1))
    ctrl = [int(h) for h in spec["control_horizons"]]
    res = dict(meta=meta, manifest=man, model=name, target=target, districts=sorted(nodes), splits={},
               models=list(models), ctrl=ctrl)
    for split in spec["splits"]:
        p = common_panel(models, targets, spec, split, regions)
        nat, dist = p["national"], p["district"]
        rows = []
        for h in H:
            q, qd = nat[nat["horizon"] == h], dist[dist["horizon"] == h]
            for mname in models:
                rows.append(dict(split=split, model=mname, horizon=h, sysmape=_sm(q, mname), bias=_bias(q, mname),
                                 mae=_mae(q, mname), days=int(len(q)), district_sysmape=_sm(qd, mname),
                                 district_mae=_mae(qd, mname), district_rows=int(len(qd))))
        res["splits"][split] = dict(scores=pd.DataFrame(rows), expected_rows=p["expected_rows"], missing=p["missing"],
                                    panel=p)
    # ТЗ 4.2.4 on test, national
    t = res["splits"]["test"]["scores"]
    spec_rows = []
    for h, thr in sorted(spec["scoring"]["spec_thresholds"].items(), key=lambda x: int(x[0])):
        r = t[(t["model"] == name) & (t["horizon"] == int(h))]
        v = float(r["sysmape"].iloc[0]) if len(r) else np.nan
        spec_rows.append(dict(horizon=int(h), sysmape=v, threshold=float(thr), accuracy=1 - v,
                              passed=bool(np.isfinite(v) and v <= float(thr))))
    res["spec_check"] = pd.DataFrame(spec_rows)
    # bias audit on test: season and day type of the target day (national), and regions vs national
    p = res["splits"]["test"]["panel"]
    nat = p["national"].assign(season=lambda x: season(x["date"]), day_type=lambda x: day_type(x["date"], cal))
    audit = []
    for h in ctrl:
        q = nat[nat["horizon"] == h]
        overall = _sm(q, name)
        for col, kind in (("season", "season"), ("day_type", "day type")):
            for val, g in q.groupby(col):
                audit.append(dict(group=kind, value=val, horizon=h, sysmape=_sm(g, name), bias=_bias(g, name),
                                  overall=overall, days=int(len(g))))
        if p["region"] is not None:
            rq = p["region"][p["region"]["horizon"] == h]
            for val, g in rq.groupby("region"):
                audit.append(dict(group="region", value=val, horizon=h, sysmape=_sm(g, name), bias=_bias(g, name),
                                  overall=overall, days=int(len(g))))
    ba = pd.DataFrame(audit)
    if len(ba):
        ba["ratio"] = ba["sysmape"] / ba["overall"]
        ba["passed"] = ba["ratio"] <= float(spec["scoring"]["bias_audit_ratio"])
    res["bias_audit"] = ba
    res["network"] = score_network(forecast_dir, spec)
    return res


def score_network(forecast_dir, spec):
    """Supplementary (not the standard): matched sysMAPE of bottom-up sums at network levels, if the run wrote them."""
    rows = []
    for lv in ("substation", "feeder", "transformer"):
        df = _read(os.path.join(forecast_dir, "network", f"{lv}.csv"))
        if df is None:
            continue
        df["origin"] = pd.to_datetime(df["origin"])
        df["date"] = df["origin"] + pd.to_timedelta(df["horizon"], unit="D")
        df = df[df["observed"] >= int(spec["scoring"]["min_households_network"])]
        for split in spec["splits"]:
            q = split_rows(spec, split, df)
            for h, g in q.groupby("horizon"):
                a = g["actual_observed_sum"].sum()
                rows.append(dict(split=split, level=lv, horizon=int(h), node_days=int(len(g)),
                                 sysmape=float((g["forecast_observed_sum"] - g["actual_observed_sum"]).abs().sum() / a)
                                 if a > 0 else np.nan))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- report, metrics, files
def report(res, spec):
    L = []
    p = L.append
    m, meta = res["model"], res["meta"]
    ctrl = res["ctrl"]
    p(f"STANDARD EVALUATION {spec['version']} ({spec['_sha256'][:12]}) | model {m} | target {res['target']} | "
      f"weather {meta.get('weather', '?')} | {len(res['districts'])} districts | commit {str(meta.get('git_commit', ''))[:7]}")
    sub = meta.get("substitution_test")
    if sub is not None:
        p(f"substitution test (leakage): {'PASSED' if sub.get('passed') else 'FAILED'} -- {sub.get('detail', '')}")
    if meta.get("notes"):
        p("run notes: " + " | ".join(meta["notes"]))
    p("")
    p(f"{'':46}" + "".join(f"{'h' + str(h):>17}" for h in ctrl))

    def line(label, split, model):
        t = res["splits"][split]["scores"]
        cells = []
        for h in ctrl:
            r = t[(t["model"] == model) & (t["horizon"] == h)]
            ok = len(r) and np.isfinite(r["sysmape"].iloc[0])
            cells.append(f"{r['sysmape'].iloc[0]:6.2%} ({r['bias'].iloc[0]:+5.1%})" if ok else "--")
        p(f"{label:46}" + "".join(f"{c:>17}" for c in cells))
    for split in ("validation", "test"):
        line(f"{m}, {split}: sysMAPE (bias)", split, m)
        for o in res["models"]:
            if o != m:
                line(f"{o} on the same days, {split}", split, o)
    days = []
    for h in ctrl:
        v, t = res["splits"]["validation"]["scores"], res["splits"]["test"]["scores"]
        dv = int(v[(v["model"] == m) & (v["horizon"] == h)]["days"].iloc[0])
        dt = int(t[(t["model"] == m) & (t["horizon"] == h)]["days"].iloc[0])
        days.append(f"{dv} / {dt}")
    p(f"{'days in the evaluation (validation / test)':46}" + "".join(f"{c:>17}" for c in days))
    for split in spec["splits"]:
        s = res["splits"][split]
        miss = {k: v for k, v in s["missing"].items() if v}
        if miss:
            p(f"  {split}: clean district rows without a forecast (dropped for every model): "
              + ", ".join(f"{k} {v:,} of {s['expected_rows']:,}" for k, v in miss.items()))
    p("")
    p("ТЗ 4.2.4 (test, national sum, every scored day):")
    if not len(res["splits"]["test"]["panel"]["national"]):
        p("  not checked: no test rows in this run (validation run)")
    else:
        for r in res["spec_check"].to_dict("records"):
            p(f"  h={r['horizon']:<3} sysMAPE {r['sysmape']:6.2%}  limit {r['threshold']:5.1%}  -> "
              f"{'PASS' if r['passed'] else 'FAIL'}")
        p(f"  overall: {'ALL THRESHOLDS MET' if res['spec_check']['passed'].all() else 'NOT MET'}")
    ba = res["bias_audit"]
    if len(ba) and ba["days"].sum():
        p("")
        p(f"bias audit (test; group sysMAPE / national, limit {spec['scoring']['bias_audit_ratio']}x): "
          f"{int((~ba['passed']).sum())} of {len(ba)} group-horizons over the limit; worst:")
        p(ba.sort_values("ratio", ascending=False).head(8)[["group", "value", "horizon", "sysmape", "overall", "ratio", "bias"]]
          .to_string(index=False, formatters={"sysmape": "{:.2%}".format, "overall": "{:.2%}".format,
                                              "ratio": "{:.2f}".format, "bias": "{:+.2%}".format}))
    sp = "test" if len(res["splits"]["test"]["panel"]["national"]) else "validation"
    t = res["splits"][sp]["scores"]
    dd = t[t["horizon"].isin(ctrl)].pivot(index="model", columns="horizon", values="district_sysmape")
    p("")
    p(f"district level (pooled over the district rows of the common panel), {sp}:")
    p(dd.map(lambda x: f"{x:6.2%}").to_string())
    nw = res["network"]
    if len(nw):
        p("")
        p(f"supplementary, not the standard: bottom-up matched sysMAPE at network levels, {sp}:")
        q = nw[(nw["split"] == sp) & nw["horizon"].isin(ctrl)].pivot(index="level", columns="horizon", values="sysmape")
        p(q.map(lambda x: f"{x:6.2%}").to_string())
    p("")
    p("notes: " + " | ".join(spec.get("notes", [])))
    return "\n".join(L)


def headline(res, split):
    """Mean national sysMAPE over the control horizons (leaderboards, registry aliases)."""
    t = res["splits"][split]["scores"]
    return float(t[(t["model"] == res["model"]) & t["horizon"].isin(res["ctrl"])]["sysmape"].mean())


def metrics_for_mlflow(res):
    out = {}
    for split, s in res["splits"].items():
        t = s["scores"]
        for r in t[t["horizon"].isin(res["ctrl"])].itertuples():
            pre = f"{split}/" + ("" if r.model == res["model"] else f"{r.model}/")
            out[f"{pre}national/sysmape_h{r.horizon}"] = r.sysmape
            out[f"{pre}national/bias_h{r.horizon}"] = r.bias
            out[f"{pre}district/sysmape_h{r.horizon}"] = r.district_sysmape
            out[f"{pre}district/mae_h{r.horizon}"] = r.district_mae
            if r.model == res["model"]:
                out[f"{split}/days_h{r.horizon}"] = r.days
        out[f"{split}/headline"] = headline(res, split)
    for r in res["spec_check"].to_dict("records"):
        out[f"test/spec_pass_h{r['horizon']}"] = float(r["passed"])
    out["test/spec_pass_all"] = float(res["spec_check"]["passed"].all())
    if len(res["bias_audit"]):
        out["test/bias_audit_fails"] = float((~res["bias_audit"]["passed"]).sum())
    return out


# MLflow names of the per-horizon metrics (logged with step = horizon, 1..60, so runs overlay in the chart view)
CURVES = {"sysWAPE": "sysmape", "WAPE": "district_sysmape", "MAE": "district_mae", "bias": "bias",
          "national_MAE": "mae"}


def curves_for_mlflow(res):
    """{key: [(step, value)]}: <split>/<metric> for the model, <split>/B0/<metric> for the reference, step = horizon.
    sysWAPE = national (sum of districts) sum|P - A| / sum A; WAPE = the same pooled over district rows; MAE = mean
    |P - A| per district-day (kWh); national_MAE = per national day (kWh); bias = sum(P - A) / sum A (national)."""
    out = {}
    for split, s in res["splits"].items():
        t = s["scores"]
        for mname, g in t.groupby("model"):
            pre = f"{split}/" + ("" if mname == res["model"] else f"{mname}/")
            g = g.sort_values("horizon")
            for key, col in CURVES.items():
                out[pre + key] = [(int(h), float(v)) for h, v in zip(g["horizon"], g[col]) if np.isfinite(v)]
    return out


def save_scores(res, out_dir, text):
    os.makedirs(out_dir, exist_ok=True)
    pd.concat([s["scores"] for s in res["splits"].values()]).to_csv(os.path.join(out_dir, "scores.csv"), index=False)
    res["spec_check"].to_csv(os.path.join(out_dir, "spec_check.csv"), index=False)
    res["bias_audit"].to_csv(os.path.join(out_dir, "bias_audit.csv"), index=False)
    if len(res["network"]):
        res["network"].to_csv(os.path.join(out_dir, "network_supplementary.csv"), index=False)
    nat = [s["panel"]["national"].assign(split=split) for split, s in res["splits"].items()]
    pd.concat(nat).to_csv(os.path.join(out_dir, "national_daily.csv"), index=False, float_format="%.6f")
    open(os.path.join(out_dir, "report.txt"), "w").write(text + "\n")


def write_template(spec, out_dir, districts):
    """The rows a district-level submission contains (forecast left empty): every district x issue date x horizon of
    both splits (validation only up to its target_end)."""
    os.makedirs(out_dir, exist_ok=True)
    parts = []
    H = np.arange(1, int(spec["horizons"]) + 1)
    for split in spec["splits"]:
        o = issue_dates(spec, split)
        g = pd.DataFrame({"origin": np.repeat(o, len(H)), "horizon": np.tile(H, len(o))})
        g["date"] = g["origin"] + pd.to_timedelta(g["horizon"], unit="D")
        parts.append(split_rows(spec, split, g)[["origin", "horizon"]])
    g = pd.concat(parts, ignore_index=True)
    t = pd.concat([g.assign(node=d) for d in districts], ignore_index=True)[["node", "origin", "horizon"]]
    t["origin"] = t["origin"].dt.date
    t["forecast"] = ""
    t.to_csv(os.path.join(out_dir, "district.csv.gz"), index=False)
    json.dump({"model": "<your model name>", "test_set_version": spec["version"], "test_set_sha256": spec["_sha256"],
               "target": "T2", "weather": "none", "level": "district or client"},
              open(os.path.join(out_dir, "forecast_meta.json"), "w"), indent=2)
    return len(t)


# ----------------------------------------------------------------------------- small shared helpers
def groups(network, levels):
    """{level: (group index per household (-1 = unknown), node names)} for network levels of one district."""
    n = len(network)
    out = {}
    for lv in levels:
        v = network[:, NETWORK.index(lv)]
        ok = v >= 0
        u, g = np.unique(v[ok], return_inverse=True)
        gi = np.full(n, -1, np.int64)
        gi[ok] = g
        out[lv] = (gi, [str(int(x)) for x in u])
    return out


def bincount2(g, values, G):
    """Sum rows of `values` (n, k) into G groups by g (n,), skipping g < 0 -> (G, k)."""
    m = g >= 0
    k = values.shape[1]
    flat = (g[m][:, None] * k + np.arange(k)[None, :]).ravel()
    return np.bincount(flat, weights=values[m].ravel(), minlength=G * k).reshape(G, k)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _read_num(path):
    try:
        v = open(path).read().strip()
        return None if v in ("max", "") else float(v)
    except OSError:
        return None


def memory_line(worker_peak_bytes=None):
    """'worker peak 1.2 GB | pod 5.3 of 16.0 GB, OOM kills 0': this process and the pod's memory (cgroup v2 or v1)."""
    import resource
    parts = [f"worker peak {worker_peak_bytes / 1e9:.1f} GB" if worker_peak_bytes is not None else
             f"this process peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9:.1f} GB"]
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
