#!/usr/bin/env python
"""
Compare several evaluated models on exactly the same days, in one Weights & Biases run (and local HTML / CSV).

    python scripts/compare_models.py --runs ~/work/eval/runs/<catboost run> ~/work/eval/runs/<xgboost run> \
        ~/work/eval/runs/<gam run> ~/work/eval/runs/<nlinear_x run> --name "sample validation, 4 models"

Each --runs folder is an evaluate.py run (forecasts/<model>/district.csv.gz + forecast_meta.json). All must be on the
same test set and districts. The common scorer drops a district row missing for any model for all of them, so every
number compares the same days.

Logged (job_type "compare"):
  - one chart per split and metric with a line per model over horizons 1..60: sysWAPE (national), WAPE (district
    rows), MAE (kWh per district-day), bias -- and B0;
  - every level (household .. national) per model at the control horizons: table + grouped bars;
  - window samples: national actual vs every model and B0, 60 days after 10-12 issue dates per split, with the
    observed daily max / min temperature below;
  - tables: scores by horizon, levels.
Outputs also go to --out-dir (default ~/work/eval/compare/<time>): scores.csv, levels.csv, html/*.html.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO)
from harness import testset as ts, wandb_log  # noqa: E402

METRICS = (("sysmape", "sysWAPE (national)"), ("district_sysmape", "WAPE (district rows)"),
           ("district_mae", "MAE (kWh per district-day)"), ("bias", "bias (national)"))


def model_dir(run):
    fd = os.path.join(run, "forecasts")
    names = [d for d in sorted(os.listdir(fd)) if not d.endswith("_unscaled")]
    if len(names) != 1:
        raise SystemExit(f"{run}: expected one model folder in forecasts/, found {names}")
    return names[0], os.path.join(fd, names[0])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="evaluate.py run folders (the first one leads the charts)")
    p.add_argument("--test-set", default=os.path.join(REPO, "configs", "test_set.yaml"))
    p.add_argument("--test-dir", default=None, help="default: the one recorded in the first run")
    p.add_argument("--config", default=os.path.join(REPO, "configs", "eval.yaml"))
    p.add_argument("--name", default=None, help="W&B run name")
    p.add_argument("--with-unscaled", action="store_true", help="also compare each model's unscaled forecast")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-wandb", action="store_true")
    a = p.parse_args(argv)
    spec = ts.load_spec(a.test_set)
    runs = [os.path.expanduser(r.rstrip("/")) for r in a.runs]
    dirs, metas = {}, {}
    for r in runs:
        name, d = model_dir(r)
        if name in dirs:
            name = f"{name}@{os.path.basename(r)[:15]}"
        dirs[name] = d
        metas[name] = json.load(open(os.path.join(d, "forecast_meta.json")))
        if a.with_unscaled and os.path.isdir(d + "_unscaled"):
            dirs[f"{name}_unscaled"] = d + "_unscaled"
    first = next(iter(metas.values()))
    test_dir = os.path.expanduser(a.test_dir or first.get("test_dir") or "")
    if not test_dir or not os.path.exists(os.path.join(test_dir, "manifest.json")):
        raise SystemExit("pass --test-dir (the built test set these runs were scored on)")
    for n, m in metas.items():
        if m.get("test_set_sha256") != spec["_sha256"]:
            raise SystemExit(f"{n} was made for another test set definition")
        if m.get("districts") != first.get("districts"):
            raise SystemExit(f"{n} covers other districts than {next(iter(metas))}: compare runs of the same scope")
    names = list(dirs)
    main_name = names[0]
    res = ts.score(dirs[main_name], test_dir, spec, others={n: dirs[n] for n in names[1:]})
    res["model"] = main_name
    out = os.path.expanduser(a.out_dir or os.path.join("~/work/eval/compare", time.strftime("%Y%m%d-%H%M%S")))
    os.makedirs(os.path.join(out, "html"), exist_ok=True)
    scores = pd.concat([s["scores"] for s in res["splits"].values()])
    scores.to_csv(os.path.join(out, "scores.csv"), index=False)
    lv = []
    for n, r in zip(names, runs + [None] * len(names)):
        f = os.path.join(r, "scores", "levels.csv") if r else None
        if f and os.path.exists(f) and not n.endswith("_unscaled"):
            lv.append(pd.read_csv(f).assign(model=n))
    levels = pd.concat(lv, ignore_index=True) if lv else pd.DataFrame()
    if len(levels):
        levels.to_csv(os.path.join(out, "levels.csv"), index=False)
    weather = None
    wf = os.path.join(runs[0], "scores", "weather_daily.csv")
    if os.path.exists(wf):
        weather = pd.read_csv(wf, parse_dates=["date"])
    models = names + ["B0"]
    colours = wandb_log.colours_for(models, main_name)

    # ---- console summary: the same days for everyone
    ctrl = res["ctrl"]
    for split, s in res["splits"].items():
        t = s["scores"][s["scores"]["horizon"].isin(ctrl)]
        if not t["days"].sum():
            continue
        print(f"\n=== {split}: sysWAPE (national) on the same days ===")
        print(t.pivot(index="model", columns="horizon", values="sysmape").reindex(models).map(lambda x: f"{x:6.2%}")
              .to_string())
        print(f"=== {split}: WAPE (district rows) ===")
        print(t.pivot(index="model", columns="horizon", values="district_sysmape").reindex(models)
              .map(lambda x: f"{x:6.2%}").to_string())
        miss = {k: v for k, v in s["missing"].items() if v}
        if miss:
            print(f"  rows dropped because a model had no forecast: {miss}")

    # ---- figures (plotly) and W&B
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    figs = {}
    for split, s in res["splits"].items():
        t = s["scores"]
        if not t["days"].sum():
            continue
        fig = make_subplots(rows=2, cols=2, subplot_titles=[m[1] for m in METRICS], vertical_spacing=0.12)
        for i, (col, title) in enumerate(METRICS):
            for m in models:
                g = t[t["model"] == m].sort_values("horizon")
                fig.add_trace(go.Scatter(x=g["horizon"], y=g[col], name=m, legendgroup=m, showlegend=i == 0,
                                         line=dict(color=colours[m], width=2, dash="dash" if m == "B0" else None),
                                         hovertemplate=f"{m} h%{{x}}: %{{y:.4g}}<extra></extra>"),
                              row=i // 2 + 1, col=i % 2 + 1)
            fig.update_xaxes(title_text="horizon (days)", row=i // 2 + 1, col=i % 2 + 1)
            if col != "district_mae":
                fig.update_yaxes(tickformat=".0%" if col != "bias" else ".1%", row=i // 2 + 1, col=i % 2 + 1)
        fig.update_layout(title=f"{split}: every model on the same days", template="plotly_white", height=700,
                          hovermode="x unified")
        figs[f"compare/{split}/metrics_by_horizon"] = fig
        if len(levels):
            q = levels[levels["split"] == split]
            hs = sorted(q["horizon"].unique())
            if len(q):
                fl = make_subplots(rows=1, cols=len(hs), subplot_titles=[f"h{h}" for h in hs], shared_yaxes=True)
                order = [x for x in ts.LEVELS if x in set(q["level"])]
                for j, h in enumerate(hs):
                    for m in names:
                        g = q[(q["horizon"] == h) & (q["model"] == m)].set_index("level").reindex(order)
                        fl.add_trace(go.Bar(x=order, y=g["wape"], name=m, legendgroup=m, showlegend=j == 0,
                                            marker_color=colours[m]), row=1, col=j + 1)
                fl.update_yaxes(tickformat=".0%")
                fl.update_layout(title=f"{split}: WAPE by level (national = sysWAPE)", barmode="group",
                                 template="plotly_white", height=420)
                figs[f"compare/{split}/levels"] = fl
    for k, f in figs.items():
        f.write_html(os.path.join(out, "html", k.replace("/", "_") + ".html"), include_plotlyjs="cdn")
    print(f"\n[compare] {len(names)} models: {', '.join(names)} | outputs in {out}")

    cfg = yaml.safe_load(open(a.config))
    wcfg = dict(cfg.get("wandb") or {})
    if a.no_wandb:
        wcfg["enabled"] = False
    run = wandb_log.start(wcfg, a.name or f"compare: {', '.join(names)}", f"{spec['version']} compare", "compare",
                          dict(models=names, runs=runs, test_set=spec["version"], test_dir=test_dir,
                               scope=first.get("scope"), weather={n: metas[n].get("weather") for n in metas}),
                          ["compare"] + names)
    if run is None:
        return
    import wandb
    try:
        for k, f in figs.items():
            run.log({k: wandb.Plotly(f)})
        hz = list(range(1, int(spec["horizons"]) + 1))
        for split, s in res["splits"].items():                          # native W&B line charts, one line per model
            t = s["scores"]
            if not t["days"].sum():
                continue
            for col, title in METRICS:
                ys = [t[t["model"] == m].set_index("horizon").reindex(hz)[col].tolist() for m in models]
                run.log({f"compare/{split}/{col}": wandb.plot.line_series(xs=hz, ys=ys, keys=models,
                                                                           title=f"{split}: {title}", xname="horizon")})
        n = wandb_log.log_windows(run, res, weather, int(wcfg.get("windows_per_split", 12)), out_dir=out)
        run.log({"tables/scores_by_horizon": wandb.Table(dataframe=scores)})
        if len(levels):
            run.log({"tables/levels": wandb.Table(dataframe=levels.astype({"level": str}))})
        for split in res["splits"]:
            t = res["splits"][split]["scores"]
            for m in models:
                q = t[(t["model"] == m) & t["horizon"].isin(ctrl)]
                if q["days"].sum():
                    run.summary[f"{split}/{m}/headline"] = float(q["sysmape"].mean())
        print(f"[wandb] comparison logged: {len(figs)} figures, {n} window charts -> {getattr(run, 'url', 'offline')}")
    finally:
        run.finish()


if __name__ == "__main__":
    main()
