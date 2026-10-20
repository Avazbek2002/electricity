"""Scores from the workers' node sums: system WAPE per level, household MAE, the headline number."""
import numpy as np
import pandas as pd

import node_syswape as ns

LEVELS = ("transformer", "feeder", "substation", "district")
BANDS = ns.BANDS


def score(results, models, horizons, min_households, headline):
    """-> dict(summary, by_size, nodes, mae, headline{model: value}, data)."""
    results = [r for r in results if "acc" in r]
    tabs = [ns.node_table(ns.merge(results, lvl, models), lvl, models) for lvl in LEVELS]
    nm = ns.merge(results, "district", models)
    one = {k: nm[k].sum(0, keepdims=True) for k in ("A", "N", "households", "scored")}
    one["names"] = np.array(["national"])
    one["P"] = {k: v.sum(0, keepdims=True) for k, v in nm["P"].items()}
    tabs.append(ns.node_table(one, "national", models))
    t = pd.concat(tabs, ignore_index=True)
    ok = t[~t["unassigned"] & (t["households_per_day"] >= min_households) & (t["actual"] > 0)].copy()

    def summ(g):
        return pd.Series({"pooled": g["abs_err"].sum() / g["actual"].sum(), "median": g["sys_wape"].median(),
                          "p90": g["sys_wape"].quantile(0.9), "bias": g["signed"].sum() / g["actual"].sum(),
                          "nodes": int(len(g)), "median_households_per_day": g["households_per_day"].median()})
    summary = ok.groupby(["level", "model", "horizon"]).apply(summ, include_groups=False).reset_index()
    ok["band"] = pd.cut(ok["households_per_day"], list(BANDS) + [np.inf], right=False,
                        labels=[f"{a}-{b - 1}" for a, b in zip(BANDS, BANDS[1:])] + [f"{BANDS[-1]}+"])
    by_size = (ok[ok["level"].isin(["transformer", "feeder", "substation"])]
               .groupby(["level", "band", "model", "horizon"], observed=True).apply(summ, include_groups=False).reset_index())
    rows = []
    for m in models:
        tot = sum(r["mae"][m] for r in results)
        for j, h in enumerate(horizons):
            rows.append(dict(model=m, horizon=h, mae=tot[0, j] / max(tot[1, j], 1), days=int(tot[1, j])))
    mae = pd.DataFrame(rows)
    head = {}
    for m in models:
        x = summary[(summary["level"] == headline["level"]) & (summary["model"] == m)
                    & summary["horizon"].isin(headline["horizons"])]
        head[m] = float(x[headline.get("stat", "pooled")].mean()) if len(x) else float("nan")
    data = dict(households=int(sum(r.get("customers", 0) for r in results)),
                forecasts=int(sum(r.get("windows", 0) for r in results)),
                gaps=int(sum(r.get("gaps", 0) for r in results)), districts=len(results))
    cols = ["level", "node", "model", "horizon", "households", "households_per_day", "days", "sys_wape", "bias"]
    return dict(summary=summary, by_size=by_size, nodes=ok[cols], mae=mae, headline=head, data=data)


def metrics_dict(sc, model):
    """Flat MLflow metrics for one model: <level>/syswape_h<h>, <level>/syswape_median_h<h>, <level>/bias_h<h>,
    household/mae_h<h>, headline."""
    m = {"headline": sc["headline"][model]}
    s = sc["summary"][sc["summary"]["model"] == model]
    for r in s.itertuples():
        m[f"{r.level}/syswape_h{r.horizon}"] = float(r.pooled)
        m[f"{r.level}/bias_h{r.horizon}"] = float(r.bias)
        if r.level != "national":
            m[f"{r.level}/syswape_median_h{r.horizon}"] = float(r.median)
    for r in sc["mae"][sc["mae"]["model"] == model].itertuples():
        m[f"household/mae_h{r.horizon}"] = float(r.mae)
    return {k: v for k, v in m.items() if np.isfinite(v)}


def print_report(sc, models, headline):
    d = sc["data"]
    print(f"\n[data] {d['households']:,} households in {d['districts']} districts; {d['forecasts']:,} household forecasts; "
          f"{d['gaps']:,} forecast values filled by persistence (NaN or negative)")
    print(f"\n=== headline: {headline['level']} system WAPE ({headline.get('stat', 'pooled')}), mean of horizons "
          f"{headline['horizons']} ===")
    for m in models:
        print(f"  {m:24s} {sc['headline'][m]:.2%}")
    s = sc["summary"]
    for lvl in ("national", "district", "substation", "feeder", "transformer"):
        x = s[s["level"] == lvl]
        if x.empty:
            continue
        print(f"\n=== {lvl}: system WAPE pooled | {int(x['nodes'].max()):,} nodes, median "
              f"{x['median_households_per_day'].median():,.0f} households per day ===")
        print(x.pivot(index="model", columns="horizon", values="pooled").reindex(models)
              .map(lambda v: f"{v:6.1%}").to_string())
    print("\n=== household MAE (kWh per day) ===")
    print(sc["mae"].pivot(index="model", columns="horizon", values="mae").reindex(models)
          .map(lambda v: f"{v:6.3f}").to_string())
