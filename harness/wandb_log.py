"""Weights & Biases experiment tracking for the standard evaluation (evaluate.py validate / test, compare_models.py).

What a run logs (district / national aggregates only -- no customer key, no household row ever leaves the pod):
  - per-horizon curves, x = `horizon` (1..60): <split>/sysWAPE (national), <split>/WAPE (district rows),
    <split>/MAE (kWh per district-day), <split>/national_MAE, <split>/bias, and the same for B0 under <split>/B0/...
    -> in the project workspace the runs of different models overlay on the same chart (one line per model);
  - every level at the control horizons: <split>/level/<household|transformer|feeder|substation|district|national>/
    {WAPE,MAE,bias} (x = horizon), plus a table;
  - window samples (10-12 issue dates per split): the national actual vs the model and B0 over the 60 days after the
    issue date, with the observed daily max / min temperature of those days in a panel below (interactive Plotly;
    a PNG if Plotly is missing);
  - summary: headline, ТЗ pass/fail, substitution test; config: model settings, test set, commit, scope, weather.

Setup: `pip install wandb plotly`, `wandb login` (or WANDB_API_KEY in the environment; never in the repo). A
self-hosted server: WANDB_BASE_URL. WANDB_MODE=offline writes locally (sync later), WANDB_MODE=disabled turns it off.
configs/eval.yaml `wandb:` sets project / entity. A tracking failure only warns: scores are in the run folder.
"""
import os

import numpy as np
import pandas as pd

from harness import testset as ts

CURVE_COLS = {"sysWAPE": "sysmape", "WAPE": "district_sysmape", "MAE": "district_mae", "bias": "bias",
              "national_MAE": "mae"}
ACTUAL = "#0b0b0b"
PALETTE = ["#eb6834", "#2a78d6", "#6250d6", "#eda100", "#e87ba4", "#008300", "#e34948"]
B0_COLOUR = "#1baf7a"
TMAX, TMIN = "#52514e", "#9a9893"           # neutral inks: the weather panel never borrows a model's colour


def start(wcfg, name, group, job_type, config, tags):
    """A W&B run, or None (disabled, not installed, not logged in, unreachable): never stops the evaluation."""
    wcfg = wcfg or {}
    if not wcfg.get("enabled", True) or os.getenv("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed (pip install wandb plotly): this run is not tracked in W&B")
        return None
    try:
        run = wandb.init(project=wcfg.get("project", "electricity-forecast"), entity=wcfg.get("entity"), name=name,
                         group=group, job_type=job_type, config=config, tags=[str(t) for t in tags if t],
                         dir=os.path.expanduser(wcfg.get("dir", "~/work/eval")), reinit="create_new")
    except TypeError:                                                   # older wandb: reinit=True
        run = wandb.init(project=wcfg.get("project", "electricity-forecast"), entity=wcfg.get("entity"), name=name,
                         group=group, job_type=job_type, config=config, tags=[str(t) for t in tags if t],
                         dir=os.path.expanduser(wcfg.get("dir", "~/work/eval")), reinit=True)
    except Exception as e:                                              # noqa: BLE001
        print(f"[wandb] WARNING: not tracked ({type(e).__name__}: {str(e)[:200]})")
        return None
    for split in ("validation", "test"):                                # x axis of every per-horizon metric
        run.define_metric(f"{split}/*", step_metric="horizon")
    run.define_metric("horizon")
    return run


def _safe(what, fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:                                              # noqa: BLE001
        print(f"[wandb] WARNING: {what} not logged ({type(e).__name__}: {str(e)[:200]})")
        return None


# ----------------------------------------------------------------------------- metrics
def horizon_rows(res, main=None):
    """{horizon: {key: value}} for every split, model and level (model keys without a prefix for `main`)."""
    main = main or res["model"]
    rows = {}
    for split, s in res["splits"].items():
        for r in s["scores"].to_dict("records"):
            if not r["days"]:
                continue
            pre = f"{split}/" + ("" if r["model"] == main else f"{r['model']}/")
            d = rows.setdefault(int(r["horizon"]), {})
            for key, col in CURVE_COLS.items():
                if np.isfinite(r[col]):
                    d[pre + key] = float(r[col])
    lv = res.get("levels")
    if lv is not None:
        for r in lv.to_dict("records"):
            d = rows.setdefault(int(r["horizon"]), {})
            for k, name in (("wape", "WAPE"), ("mae", "MAE"), ("bias", "bias")):
                if np.isfinite(r[k]):
                    d[f"{r['split']}/level/{r['level']}/{name}"] = float(r[k])
    return rows


def log_scores(run, res, extra_summary=None):
    import wandb
    for h, d in sorted(horizon_rows(res).items()):                     # one log per horizon: x = horizon
        run.log({"horizon": h, **d})
    m = res["model"]
    summ = {}
    for split in res["splits"]:
        t = res["splits"][split]["scores"]
        q = t[t["horizon"].isin(res["ctrl"])]
        if q["days"].sum():
            summ[f"{split}/headline"] = ts.headline(res, split)
            for r in q.to_dict("records"):
                pre = f"{split}/" + ("" if r["model"] == m else f"{r['model']}/")
                summ[f"{pre}sysWAPE_h{r['horizon']}"] = r["sysmape"]
                summ[f"{pre}WAPE_h{r['horizon']}"] = r["district_sysmape"]
                summ[f"{pre}MAE_h{r['horizon']}"] = r["district_mae"]
    if len(res["splits"]["test"]["panel"]["national"]):
        for r in res["spec_check"].to_dict("records"):
            summ[f"test/spec_pass_h{r['horizon']}"] = bool(r["passed"])
        summ["test/spec_pass_all"] = bool(res["spec_check"]["passed"].all())
    summ.update(extra_summary or {})
    for k, v in summ.items():
        if v is not None and (not isinstance(v, float) or np.isfinite(v)):
            run.summary[k] = v
    lv = res.get("levels")
    if lv is not None and len(lv):
        t = lv.copy()
        t["level"] = t["level"].astype(str)
        t.insert(0, "model", m)
        run.log({"tables/levels": wandb.Table(dataframe=t)})
    sc = pd.concat([s["scores"] for s in res["splits"].values()])
    run.log({"tables/scores_by_horizon": wandb.Table(dataframe=sc)})


# ----------------------------------------------------------------------------- window charts
def weather_frame(results):
    """Daily observed max / min temperature, mean over the run's districts (from the harness workers)."""
    parts = [r for r in results if "tminmax" in r]
    if not parts:
        return None
    W = np.stack([r["tminmax"] for r in parts])                          # (districts, 2, T)
    with np.errstate(all="ignore"):
        m = np.nanmean(W, axis=0)
    dates = pd.date_range(parts[0]["dates0"], periods=W.shape[2], freq="D")
    return pd.DataFrame({"date": dates, "tmin": m[0], "tmax": m[1]})


def window_figure(g, models, colours, weather, title):
    """Plotly: top = national actual and forecasts (MWh/day), bottom = observed daily max / min temperature."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    g = g.sort_values("horizon")
    o = pd.Timestamp(g["origin"].iloc[0])
    full = pd.DataFrame({"horizon": np.arange(1, int(g["horizon"].max()) + 1)})
    g = full.merge(g, on="horizon", how="left")                         # gaps stay gaps
    dates = o + pd.to_timedelta(g["horizon"], unit="D")
    has_w = weather is not None
    fig = make_subplots(rows=2 if has_w else 1, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3] if has_w else [1],
                        vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=dates, y=g["y"] / 1e3, name="actual (T2)", line=dict(color=ACTUAL, width=3),
                             hovertemplate="%{x|%d %b %Y}: %{y:,.1f} MWh<extra>actual</extra>"), row=1, col=1)
    for m in models:
        if m in g and g[m].notna().any():
            dash = "dash" if m == "B0" else ("dot" if m.endswith("_unscaled") else None)
            fig.add_trace(go.Scatter(x=dates, y=g[m] / 1e3, name=m, line=dict(color=colours[m], width=2, dash=dash),
                                     hovertemplate="%{x|%d %b %Y}: %{y:,.1f} MWh<extra>" + m + "</extra>"), row=1, col=1)
    if has_w:
        w = weather.set_index("date").reindex(dates)
        fig.add_trace(go.Scatter(x=dates, y=w["tmax"].to_numpy(), name="daily max temp (observed)",
                                 line=dict(color=TMAX, width=1.5), hovertemplate="%{y:.1f} °C<extra>max</extra>"), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates, y=w["tmin"].to_numpy(), name="daily min temp (observed)",
                                 line=dict(color=TMIN, width=1.5, dash="dot"), hovertemplate="%{y:.1f} °C<extra>min</extra>"), row=2, col=1)
        fig.update_yaxes(title_text="°C", row=2, col=1)
    fig.update_yaxes(title_text="MWh / day", row=1, col=1)
    fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=14)), template="plotly_white", height=520,
                      hovermode="x unified", legend=dict(orientation="h", y=-0.12), margin=dict(l=60, r=20, t=60, b=40))
    return fig


def colours_for(models, main):
    """Fixed colours: the run's model first, B0 green, any other model the next palette slot (in the given order)."""
    out, k = {}, 1
    for m in models:
        if m == main:
            out[m] = PALETTE[0]
        elif m == "B0":
            out[m] = B0_COLOUR
        else:
            out[m] = PALETTE[k % len(PALETTE)]
            k += 1
    return out


def log_windows(run, res, weather, per_split=12, out_dir=None):
    """Window samples per split (issue dates from plots.pick_windows), every scored model on one chart."""
    import wandb
    from harness import plots
    models = list(res["models"])
    colours = colours_for(models, res["model"])
    n = 0
    for split, s in res["splits"].items():
        nat = s["panel"]["national"]
        for o in plots.pick_windows(nat, per_split):
            g = nat[nat["origin"] == o]
            err = ", ".join(f"{m} {float((g[m] - g['y']).abs().sum() / g['y'].sum()):.1%}" for m in models if m in g)
            title = f"{split}: issued {pd.Timestamp(o).date()} | sysWAPE over the window: {err}"
            key = f"windows/{split}/{pd.Timestamp(o).date()}"
            try:
                fig = window_figure(g, models, colours, weather, title)
                if out_dir:
                    os.makedirs(os.path.join(out_dir, "windows_html"), exist_ok=True)
                    fig.write_html(os.path.join(out_dir, "windows_html", f"{split}_{pd.Timestamp(o).date()}.html"),
                                   include_plotlyjs="cdn")
                run.log({key: wandb.Plotly(fig)})
            except ImportError:                                         # no plotly: the matplotlib PNG
                png = os.path.join(out_dir or ".", "windows", f"{split}_{pd.Timestamp(o).date()}.png")
                if os.path.exists(png):
                    run.log({key: wandb.Image(png)})
            n += 1
    return n


def log_run(wcfg, name, group, job_type, config, tags, res, weather=None, out_dir=None, extra_summary=None,
            files=()):
    """Everything of one scored run; returns the W&B run URL or None."""
    run = start(wcfg, name, group, job_type, config, tags)
    if run is None:
        return None
    try:
        _safe("scores", log_scores, run, res, extra_summary)
        n = _safe("window charts", log_windows, run, res, weather, int((wcfg or {}).get("windows_per_split", 12)),
                  out_dir)
        fs = [f for f in files if os.path.exists(f)]
        if fs:                                                          # aggregate tables / report (no customer keys)
            def art():
                import wandb
                a = wandb.Artifact(f"scores-{run.id}", type="evaluation", description="district / national scores")
                for f in fs:
                    a.add_file(f)
                run.log_artifact(a)
            _safe("score files", art)
        url = getattr(run, "url", None)
        print(f"[wandb] logged {name}: per-horizon metrics, levels, {n or 0} window charts -> {url or 'offline'}")
        return url
    finally:
        run.finish()
