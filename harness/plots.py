"""Charts of an evaluation run (logged to MLflow with the scores):

    figures/metrics_by_horizon.png    sysWAPE, WAPE, MAE and bias over every horizon 1..60: the model vs B0
    windows/<split>_<issue date>.png  window samples: the national (sum of districts) actual vs the model and B0 over
                                      the 60 days after the issue date, 12 issue dates per split (one per month)
    windows/overview_<split>.png      the same windows on one page

Static PNG (matplotlib, Agg). Colours: a validated categorical palette; every series also has its own line style, so
identity does not rest on colour alone. The numbers are in scores/scores.csv and scores/national_daily.csv."""
import os

import numpy as np
import pandas as pd

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
ACTUAL = ("actual (T2)", INK, "-")
MODEL_STYLE = ("#eb6834", "-")
REF_STYLE = ("#1baf7a", "--")
EXTRA = [("#2a78d6", "-."), ("#6250d6", ":"), ("#eda100", "-."), ("#e87ba4", ":")]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax, title=None, pct=False):
    import matplotlib.ticker as mt
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if pct:
        ax.yaxis.set_major_formatter(mt.PercentFormatter(1.0, decimals=0))
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK)


def _styles(models, main):
    st, k = {}, 0
    for m in models:
        if m == main:
            st[m] = MODEL_STYLE
        elif m == "B0":
            st[m] = REF_STYLE
        else:
            st[m] = EXTRA[k % len(EXTRA)]
            k += 1
    return st


def _save(fig, path, legend_rect=0.0, top=1.0):
    plt = _plt()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, legend_rect, 1, top))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def metrics_by_horizon(res, out_dir):
    """One row per split: sysWAPE, WAPE, MAE, bias over h = 1..60, every scored model."""
    plt = _plt()
    splits = list(res["splits"])
    panels = [("sysmape", "sysWAPE (national)", True), ("district_sysmape", "WAPE (district rows)", True),
              ("district_mae", "MAE (kWh per district-day)", False), ("bias", "bias (national)", True)]
    st = _styles(res["models"], res["model"])
    fig, axes = plt.subplots(len(splits), len(panels), figsize=(4.4 * len(panels), 3.2 * len(splits)), squeeze=False)
    for i, sp in enumerate(splits):
        t = res["splits"][sp]["scores"]
        for j, (col, title, pct) in enumerate(panels):
            ax = axes[i][j]
            for m in res["models"]:
                g = t[t["model"] == m].sort_values("horizon")
                c, ls = st[m]
                ax.plot(g["horizon"], g[col], color=c, linestyle=ls, linewidth=1.8, label=m)
            if col == "bias":
                ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.set_xlabel("horizon (days)", fontsize=8, color=MUTED)
            ax.set_xlim(1, int(t["horizon"].max()))
            _style(ax, f"{sp}: {title}", pct=pct)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), fontsize=9, frameon=False)
    return _save(fig, os.path.join(out_dir, "figures", "metrics_by_horizon.png"), legend_rect=0.06)


def pick_windows(nat, per_split=12, at_least=10):
    """Issue dates for the window charts: one per month, the issue date nearest the 15th among those with most of the
    month's best horizon coverage; if gaps (e.g. summer 2024) leave fewer than `at_least`, more issue dates are added
    evenly from the rest (at least 7 days from the others)."""
    cnt = nat.groupby("origin")["horizon"].nunique()
    if cnt.empty:
        return []
    d = cnt.reset_index(name="n")
    d = d[d["n"] >= 10]
    d["month"] = d["origin"].dt.to_period("M")
    d["far"] = (d["origin"].dt.day - 15).abs()
    d = d[d["n"] >= 0.75 * d.groupby("month")["n"].transform("max")]
    best = d.sort_values(["month", "far"]).groupby("month").head(1)
    if len(best) > per_split:
        best = best.iloc[np.linspace(0, len(best) - 1, per_split).round().astype(int)]
    picks = list(best["origin"])
    rest = d[~d["origin"].isin(picks)].sort_values("origin")
    for o in rest["origin"].iloc[np.linspace(0, len(rest) - 1, min(len(rest), 4 * at_least)).round().astype(int)] \
            if len(rest) else []:
        if len(picks) >= at_least:
            break
        if all(abs((o - p).days) >= 7 for p in picks):
            picks.append(o)
    return sorted(picks)


def _window(ax, g, models, st, title):
    import matplotlib.dates as mdates
    g = g.set_index("horizon").reindex(range(1, int(g["horizon"].max()) + 1))   # gaps break the lines
    o = g["origin"].dropna().iloc[0]
    dates = o + pd.to_timedelta(g.index, unit="D")
    ax.plot(dates, g["y"] / 1e3, color=INK, linewidth=2.2, label=ACTUAL[0])
    for m in models:
        c, ls = st[m]
        ax.plot(dates, g[m] / 1e3, color=c, linestyle=ls, linewidth=1.7, label=m)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.set_ylabel("MWh / day", fontsize=8, color=MUTED)
    _style(ax, title)


def window_samples(res, out_dir, per_split=12):
    """One PNG per sampled issue date and an overview page per split. Returns the files written."""
    plt = _plt()
    models = [m for m in res["models"]]
    st = _styles(models, res["model"])
    made = []
    for sp, s in res["splits"].items():
        nat = s["panel"]["national"]
        picks = pick_windows(nat, per_split)
        if not picks:
            continue
        for o in picks:
            g = nat[nat["origin"] == o]
            fig, ax = plt.subplots(figsize=(8, 3.6))
            err = {m: float((g[m] - g["y"]).abs().sum() / g["y"].sum()) for m in models}
            _window(ax, g, models, st, f"{sp}: issued {pd.Timestamp(o).date()} | sysWAPE over the window: " +
                    ", ".join(f"{m} {e:.1%}" for m, e in err.items()))
            ax.legend(fontsize=8, frameon=False, ncol=len(models) + 1)
            made.append(_save(fig, os.path.join(out_dir, "windows", f"{sp}_{pd.Timestamp(o).date()}.png")))
        ncol = 4
        nrow = int(np.ceil(len(picks) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.0 * nrow), squeeze=False)
        for ax in axes.ravel()[len(picks):]:
            ax.set_visible(False)
        for ax, o in zip(axes.ravel(), picks):
            _window(ax, nat[nat["origin"] == o], models, st, f"issued {pd.Timestamp(o).date()}")
        h, lab = axes[0][0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=len(lab), fontsize=9, frameon=False)
        fig.suptitle(f"{sp}: window samples, national (sum of the districts), 60 days after the issue date",
                     x=0.01, ha="left", fontsize=11, color=INK)
        made.append(_save(fig, os.path.join(out_dir, "windows", f"overview_{sp}.png"), legend_rect=0.04,
                          top=0.95))
    return made
