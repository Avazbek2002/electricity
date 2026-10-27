"""The substation / feeder test (configs/test_set_substations_2025.yaml): node targets, the node B0, node scores.

One implementation shared by the builder (scripts/test_set/build_node_test_set.py), the harness (harness/node_mode.py)
and the scorer.

Nodes
    S:<coato>:<substation_id>   the selected substations (dev 20, test 40)
    F:<coato>:<feeder_id>       their feeders with >= feeders.min_households households (smaller ones count only in
                                the substation)
Built set (<test_dir> = ~/work/test_set/<version>/):
    manifest.json               version, spec hash, data version, counts, file hashes
    nodes.csv                   node, level, role, coato, substation_id, node_id, region, district, stratum, households
    targets/node.csv.gz         date, node, n_rows, n_known, n_lg, coverage, coverage3, T2, T3, flag, flag_reason,
                                artefact  (node x day, data start .. end). Node-day aggregates only: T2 / T3 are NaN
                                when fewer than 10 households are known that day; no kWh sums of fewer customers.
Target (guide part 2, per node): T2 = households in span x (kWh observed + estimated) / (n observed + n estimated);
flags as the district target (few known, right edge, coverage < min_coverage, crisis, interruption).

Node B0 = m28(T2, o - 1) x seasonal ratio of last year. The ratio is taken from T3 (which adds the allocated long-gap
energy): before 2024-08 most readings are 4-5 day intervals, so T2 is flagged (low coverage) on almost every 2024-H1
node-day and a T2 ratio would fall back to 1.0 for the whole first half of 2025.

Scores, per role (dev = model choice, test = once), level and horizon, on the rows that are clean and forecast by every
compared model:
    per node   sysMAPE_h = sum_o |P - A| / sum_o A, bias_h = sum_o (P - A) / sum_o A over the 2025 issue dates
    summary    median over nodes, pooled (sum over nodes and dates), share of nodes better than B0, by size band
    headline   mean over the control horizons of the median per-node sysMAPE, per level
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
import yaml

from harness import testset as ts

LEVELS = ("substation", "feeder")
MIN_KNOWN = 10                     # a node-day needs this many known households (also the disclosure floor)
FEEDER_BANDS = [(50, 150, "50-149"), (150, 400, "150-399"), (400, 10 ** 9, "400+")]
REFERENCE = "B0"
EST_SAFE_LAG = 2   # node T2 includes 2-3 day estimated readings spread evenly: the share on day d comes from a reading taken
                   # up to d + 2, so on issue date o the T2 of o - 2 and o - 1 can carry consumption of o and o + 1 (ML guide
                   # part 2, estimated days in history). Node-target windows (B0 level, scaling c(o)) end on o - 1 - EST_SAFE_LAG


# ----------------------------------------------------------------------------- the definition
def is_node_spec(path):
    try:
        return "test_substations" in (yaml.safe_load(open(os.path.expanduser(path))) or {})
    except (OSError, yaml.YAMLError):
        return False


def load_spec(path):
    """The frozen node spec (not ts.load_spec: its splits are dev / test, both on the 2025 issue dates)."""
    raw = open(os.path.expanduser(path), "rb").read()
    spec = yaml.safe_load(raw)
    spec["_sha256"] = hashlib.sha256(raw).hexdigest()
    spec["_path"] = os.path.abspath(os.path.expanduser(path))
    R = pd.to_datetime(spec["refits"])
    assert (np.diff(R.values).astype("timedelta64[D]").astype(int) > 0).all(), "refits must be increasing"
    for role in ("dev", "test"):
        s = spec["splits"][role]
        assert R[0] <= pd.Timestamp(s["issue_start"]) <= pd.Timestamp(s["issue_end"]), f"{role}: issue dates"
        assert pd.Timestamp(s["issue_end"]) + pd.Timedelta(days=int(spec["horizons"])) <= pd.Timestamp(spec["data"]["end"])
        assert spec[f"{role}_substations"], f"no {role} substations"
    assert int(spec["history_gap_days"]) >= 1
    return spec


def target_rules(spec):
    """The district target rules (ts.target_series) with the node floor of 10 known households."""
    t = spec["target"]
    return dict(target=dict(min_known_client_days=int(t.get("min_known_client_days", MIN_KNOWN)),
                            exclude_right_edge=bool(t.get("exclude_right_edge", True)),
                            min_coverage=float(t["min_coverage"])))


def node_table(spec):
    """Every node of the spec -> DataFrame node, level, role, coato, substation_id, node_id, region, district, stratum,
    households (substations: from the selection; feeders: filled in by the builder)."""
    rows = []
    for role in ("dev", "test"):
        for s in spec[f"{role}_substations"]:
            c, sid = str(s["coato_code"]), int(s["substation_id"])
            base = dict(role=role, coato=c, substation_id=sid, region=s.get("region"), district=s.get("district"),
                        stratum=s.get("stratum"))
            rows.append(dict(node=f"S:{c}:{sid}", level="substation", node_id=sid, households=s.get("households"), **base))
            for f in s.get("feeders") or []:
                rows.append(dict(node=f"F:{c}:{int(f)}", level="feeder", node_id=int(f), households=np.nan, **base))
    df = pd.DataFrame(rows)
    dup = df.duplicated("node")
    if dup.any():
        raise SystemExit(f"[node spec] nodes listed twice: {df.loc[dup, 'node'].tolist()[:5]}")
    return df


def issue_dates(spec, role):
    s = spec["splits"][role]
    return pd.date_range(s["issue_start"], s["issue_end"], freq="D")


# ----------------------------------------------------------------------------- the target
def node_target(counts, spec):
    """counts: one node, one row per date with n_rows, n_obs, n_est, n_lg, kwh_obs, kwh_est, kwh_lg, n_right, crisis,
    interruption -> + T2, T3, coverage, coverage3 and the flags. T3 / coverage3 include the long-gap allocation."""
    c = ts.target_series(counts, target_rules(spec))
    nl = c["n_obs"].fillna(0) + c["n_est"].fillna(0) + c["n_lg"].fillna(0)
    c["coverage3"] = nl / c["n_rows"].where(c["n_rows"] > 0)
    c["n_known"] = c["n_obs"].fillna(0) + c["n_est"].fillna(0)
    # T3 on its own count: before 2024-08 few days are clean (T2 artefact) but most carry allocated long-gap energy
    kwh3 = c["kwh_obs"].fillna(0) + c["kwh_est"].fillna(0) + c["kwh_lg"].fillna(0)
    no3 = c["kwh_obs"].isna() & c["kwh_est"].isna() & c["kwh_lg"].isna()
    edge = c["n_right"].fillna(0) > 0
    bad3 = no3 | (nl < MIN_KNOWN) | edge | c["n_rows"].isna() | (c["n_rows"] <= 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        c["T3"] = (c["n_rows"] * kwh3 / nl.where(nl > 0)).where(~bad3)
    return c


def public_columns(t):
    """The columns the built set keeps (no kWh sums: T2 / T3 are means scaled to the households in span)."""
    return t[["date", "node", "n_rows", "n_known", "n_lg", "coverage", "coverage3", "T2", "T3", "flag", "flag_reason",
              "artefact"]]


def load_targets(test_dir):
    p = os.path.join(os.path.expanduser(test_dir), "targets", "node.csv.gz")
    if not os.path.exists(p):
        raise SystemExit(f"no node targets in {test_dir}: build them first (scripts/test_set/build_node_test_set.py)")
    t = pd.read_csv(p, dtype={"node": str}, parse_dates=["date"])
    for k in ("flag", "artefact"):
        t[k] = t[k].astype(str).str.lower().isin(("true", "1"))
    t["y"] = t["T2"]
    return t


def load_nodes(test_dir):
    return pd.read_csv(os.path.join(os.path.expanduser(test_dir), "nodes.csv"), dtype={"node": str, "coato": str})


def read_manifest(test_dir):
    p = os.path.join(os.path.expanduser(test_dir), "manifest.json")
    if not os.path.exists(p):
        raise SystemExit(f"no manifest.json in {test_dir}: build the node set first (scripts/test_set/build_node_test_set.py)")
    return json.load(open(p))


def check_same_spec(spec, man):
    if man["test_set_sha256"] != spec["_sha256"]:
        raise SystemExit(f"{spec['_path']} changed since the node set was built ({man['test_set_sha256'][:12]} built vs "
                         f"{spec['_sha256'][:12]} now): rebuild under a new version, or restore the file")


def per_customer(targets, spec):
    """{node: T2 per household per day (day index of the data start), NaN on artefact days} for the node scaling c(o)
    and {node: households in span per day} for the node count on o - 1."""
    T = ts.day_index(spec, spec["data"]["end"]) + 1
    pc, nr = {}, {}
    for node, g in targets.groupby("node"):
        i = ((g["date"] - pd.Timestamp(spec["data"]["start"])).dt.days).to_numpy()
        ok = (i >= 0) & (i < T)
        a, b = np.full(T, np.nan), np.full(T, np.nan)
        v = (g["T2"] / g["n_rows"].where(g["n_rows"] > 0)).where(~g["artefact"])
        a[i[ok]] = v.to_numpy()[ok]
        b[i[ok]] = g["n_rows"].to_numpy(float)[ok]
        pc[node], nr[node] = a, b
    return pc, nr


# ----------------------------------------------------------------------------- B0 (node)
def b0_forecasts(targets, spec, nodes=None, issue=None, safe_lag=None):
    """B0 per node: m28 of T2 at o - gap x the last-year seasonal ratio of T3 (7-day centred mean on the target day over
    the 28-day mean at o - gap - 364), else two years back, else 1; ratio clipped to 0.6 .. 1.7. The level window ends
    safe_lag (EST_SAFE_LAG) days earlier, so no estimated reading taken after o - 1 is in it."""
    H, gap = int(spec["horizons"]), int(spec["history_gap_days"])
    gap += EST_SAFE_LAG if safe_lag is None else int(safe_lag)
    start = pd.Timestamp(spec["data"]["start"])
    if issue is None:
        issue = pd.DatetimeIndex(sorted(set(issue_dates(spec, "dev")) | set(issue_dates(spec, "test"))))
    oi = ((issue - start).days).to_numpy()
    out = []
    for node, g in targets.groupby("node", sort=True):
        if nodes is not None and node not in nodes:
            continue
        g = g.set_index("date")
        full = pd.date_range(start, g.index.max(), freq="D")
        g = g.reindex(full)
        art = g["artefact"].fillna(True).astype(bool)
        s = g["T2"].where(~art)
        cov3_low = g["coverage3"].fillna(0) < float(spec["target"]["min_coverage"])
        s3 = g["T3"].where(~cov3_low)
        m28 = s.rolling(28, min_periods=20).mean().to_numpy()
        c7 = s3.rolling(7, center=True, min_periods=5).mean().to_numpy()
        m28ly = s3.rolling(28, min_periods=20).mean().to_numpy()
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
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["node", "origin", "horizon", "forecast"])


# ----------------------------------------------------------------------------- scores
def read_forecasts(fdir):
    p = os.path.join(os.path.expanduser(fdir), "node.csv.gz")
    if not os.path.exists(p):
        raise SystemExit(f"no node.csv.gz in {fdir}")
    df = pd.read_csv(p, dtype={"node": str})
    df["origin"] = pd.to_datetime(df["origin"])
    df["horizon"] = df["horizon"].astype(int)
    dup = df.duplicated(["node", "origin", "horizon"])
    if dup.any():
        raise SystemExit(f"{int(dup.sum())} duplicated node forecast rows in {fdir}")
    return df[["node", "origin", "horizon", "forecast"]]


def size_band(nodes):
    """substations: their stratum (large / mid); feeders: household bands."""
    out = nodes["stratum"].astype(str).where(nodes["level"] == "substation", "")
    hh = nodes["households"].astype(float)
    for lo, hi, lab in FEEDER_BANDS:
        out = out.where(~((nodes["level"] == "feeder") & (hh >= lo) & (hh < hi)), lab)
    return out


def common_rows(models, targets, spec, role, nodes):
    """Rows of the role's nodes and issue dates, clean target, forecast by every model -> long DataFrame."""
    names = list(models)
    want = set(nodes.loc[nodes["role"] == role, "node"])
    base = None
    for n in names:
        m = models[n]
        m = m[m["node"].isin(want)].rename(columns={"forecast": n})
        base = m if base is None else base.merge(m, on=["node", "origin", "horizon"], how="outer")
    s = spec["splits"][role]
    base = base[(base["origin"] >= s["issue_start"]) & (base["origin"] <= s["issue_end"])]
    base["date"] = base["origin"] + pd.to_timedelta(base["horizon"], unit="D")
    m = base.merge(targets[["node", "date", "y", "flag"]], on=["node", "date"], how="left")
    clean = m[~m["flag"].fillna(True).astype(bool) & np.isfinite(m["y"])]
    ok = clean[names].apply(np.isfinite).all(axis=1)
    missing = {n: int((~np.isfinite(clean[n])).sum()) for n in names}
    out = clean[ok].merge(nodes[["node", "level", "households", "stratum"]], on="node", how="left")
    return out, dict(expected=int(len(clean)), missing=missing)


def score(models, targets, spec, role, nodes):
    """models: {name: DataFrame node, origin, horizon, forecast} (B0 included) -> dict(per_node, summary, headline,
    by_band, data)."""
    rows, data = common_rows(models, targets, spec, role, nodes)
    names = list(models)
    nb = nodes.assign(band=size_band(nodes)).set_index("node")["band"]
    rows["band"] = rows["node"].map(nb)
    per, summ, band = [], [], []
    for n in names:
        rows[f"ae_{n}"] = (rows[n] - rows["y"]).abs()
        rows[f"e_{n}"] = rows[n] - rows["y"]
    g = rows.groupby(["level", "node", "horizon"])
    agg = g.agg(actual=("y", "sum"), days=("y", "size"), **{f"ae_{n}": (f"ae_{n}", "sum") for n in names},
                **{f"e_{n}": (f"e_{n}", "sum") for n in names}).reset_index()
    for n in names:
        per.append(pd.DataFrame({"model": n, "level": agg["level"], "node": agg["node"], "horizon": agg["horizon"],
                                 "days": agg["days"], "actual": agg["actual"],
                                 "sysmape": agg[f"ae_{n}"] / agg["actual"], "bias": agg[f"e_{n}"] / agg["actual"]}))
    per = pd.concat(per, ignore_index=True) if per else pd.DataFrame()
    if len(per):
        ref = per[per["model"] == REFERENCE].set_index(["node", "horizon"])["sysmape"] if REFERENCE in names else None
        for (n, lv, h), q in per.groupby(["model", "level", "horizon"]):
            r = rows[(rows["level"] == lv) & (rows["horizon"] == h)]
            beat = np.nan
            if ref is not None and n != REFERENCE:
                rr = ref.reindex(list(zip(q["node"], q["horizon"]))).to_numpy()
                beat = float((q["sysmape"].to_numpy() < rr).mean())
            summ.append(dict(model=n, level=lv, horizon=h, nodes=len(q), median=float(q["sysmape"].median()),
                             p75=float(q["sysmape"].quantile(0.75)),
                             pooled=float(r[f"ae_{n}"].sum() / r["y"].sum()) if r["y"].sum() > 0 else np.nan,
                             bias=float(r[f"e_{n}"].sum() / r["y"].sum()) if r["y"].sum() > 0 else np.nan,
                             median_abs_bias=float(q["bias"].abs().median()), beats_b0=beat))
        for (n, lv, bd, h), q in per.merge(nb.rename("band"), left_on="node", right_index=True)\
                .groupby(["model", "level", "band", "horizon"]):
            band.append(dict(model=n, level=lv, band=bd, horizon=h, nodes=len(q), median=float(q["sysmape"].median())))
    summ, band = pd.DataFrame(summ), pd.DataFrame(band)
    ctrl = list(spec["control_horizons"])
    by_month = _by_month(rows, names, ctrl)
    head = {}
    if len(summ):
        for (n, lv), q in summ[summ["horizon"].isin(ctrl)].groupby(["model", "level"]):
            head[f"{n}/{lv}"] = float(q["median"].mean())
    data.update(rows=int(len(rows)), nodes={lv: int(rows.loc[rows["level"] == lv, "node"].nunique()) for lv in LEVELS},
                issue_dates=int(rows["origin"].nunique()))
    return dict(per_node=per, summary=summ, by_band=band, by_month=by_month, headline=head, data=data, ctrl=ctrl,
                role=role)


def _by_month(rows, names, ctrl):
    """Per issue month, level and model: median per-node sysMAPE and pooled bias, both averaged over the control
    horizons (seasonal misses show up here; they partly cancel in the year's totals)."""
    q = rows[rows["horizon"].isin(ctrl)]
    if not len(q):
        return pd.DataFrame(columns=["model", "level", "month", "median", "bias"])
    q = q.assign(month=q["origin"].dt.strftime("%Y-%m"))
    out = []
    for n in names:
        g = q.groupby(["level", "month", "node", "horizon"]).agg(y=("y", "sum"), ae=(f"ae_{n}", "sum"),
                                                                 e=(f"e_{n}", "sum")).reset_index()
        g["sm"] = g["ae"] / g["y"]
        med = g.groupby(["level", "month", "horizon"])["sm"].median().groupby(["level", "month"]).mean()
        pb = g.groupby(["level", "month", "horizon"]).apply(lambda x: x["e"].sum() / x["y"].sum(), include_groups=False)\
            .groupby(["level", "month"]).mean()
        out.append(pd.DataFrame({"model": n, "median": med, "bias": pb}).reset_index())
    return pd.concat(out, ignore_index=True)


def report(res, models):
    ctrl, s = res["ctrl"], res["summary"]
    d = res["data"]
    lines = [f"NODE SCORES ({res['role']}): {d['nodes'].get('substation', 0)} substations, {d['nodes'].get('feeder', 0)} "
             f"feeders, {d['issue_dates']} issue dates, {d['rows']:,} node-days (expected {d['expected']:,}; missing "
             f"forecasts {d['missing']})", "",
             "median per-node sysMAPE (pooled over all nodes and dates in brackets) | bias pooled | share of nodes "
             "better than B0", ""]
    if not len(s):
        return "\n".join(lines + ["(no rows)"])
    for lv in LEVELS:
        q = s[s["level"] == lv]
        if not len(q):
            continue
        lines.append(f"{lv.upper()}")
        lines.append(f"  {'model':24s} " + " ".join(f"{'h' + str(h):>19s}" for h in ctrl) + f" {'headline':>9s}")
        for m in models:
            r = q[q["model"] == m].set_index("horizon")
            cells = []
            for h in ctrl:
                if h in r.index:
                    x = r.loc[h]
                    cells.append(f"{x['median']:6.2%} ({x['pooled']:6.2%})"[:19].rjust(19))
                else:
                    cells.append(" " * 19)
            hd = res["headline"].get(f"{m}/{lv}", np.nan)
            lines.append(f"  {m:24s} " + " ".join(cells) + f" {hd:9.2%}")
        for m in models:
            r = q[q["model"] == m].set_index("horizon")
            if m == REFERENCE or not len(r):
                continue
            lines.append(f"  {m:24s} bias " + " ".join(f"h{h} {r.loc[h, 'bias']:+.1%}" for h in ctrl if h in r.index)
                         + " | beats B0 " + " ".join(f"h{h} {r.loc[h, 'beats_b0']:.0%}" for h in ctrl if h in r.index))
        lines.append("")
    bm = res.get("by_month")
    if bm is not None and len(bm):
        for lv in LEVELS:
            q = bm[bm["level"] == lv]
            if not len(q):
                continue
            med = q.pivot(index="month", columns="model", values="median")
            bia = q.pivot(index="month", columns="model", values="bias")
            ms = [m for m in models if m in med.columns]
            lines.append(f"BY ISSUE MONTH, {lv} (median per-node sysMAPE | pooled bias; mean over the control horizons)")
            lines.append("  month   " + " ".join(f"{m[:16]:>16s}" for m in ms) + "  | " +
                         " ".join(f"{m[:10]:>10s}" for m in ms))
            for mo in med.index:
                lines.append(f"  {mo} " + " ".join(f"{med.loc[mo, m]:16.2%}" for m in ms) + "  | " +
                             " ".join(f"{bia.loc[mo, m]:+10.1%}" for m in ms))
            lines.append("")
    b = res["by_band"]
    if len(b):
        lines.append("BY SIZE (median per-node sysMAPE, mean over the control horizons)")
        bb = b[b["horizon"].isin(ctrl)].groupby(["level", "band", "model"])["median"].mean().unstack("model")
        cnt = b[b["horizon"].isin(ctrl)].groupby(["level", "band"])["nodes"].max()
        for (lv, bd), r in bb.iterrows():
            lines.append(f"  {lv:10s} {bd:8s} n={int(cnt.loc[(lv, bd)]):3d}  " +
                         "  ".join(f"{m} {r[m]:.2%}" for m in models if m in r.index and np.isfinite(r[m])))
    return "\n".join(lines)


def save(res, out_dir, text):
    os.makedirs(out_dir, exist_ok=True)
    for k in ("per_node", "summary", "by_band", "by_month"):
        res[k].to_csv(os.path.join(out_dir, f"{k}.csv"), index=False, float_format="%.6f")
    open(os.path.join(out_dir, "report.txt"), "w").write(text + "\n")
    json.dump(dict(headline=res["headline"], data=res["data"], role=res["role"]),
              open(os.path.join(out_dir, "headline.json"), "w"), indent=2, default=str)
