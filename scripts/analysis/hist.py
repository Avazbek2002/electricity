#!/usr/bin/env python3
"""
households_histogram.py - histogram of the number of households per transformer (national run).

Reads transformers_kept.csv (written by transformer_clean.py into <work-dir>/clean/) and draws two panels:
  left   all households linked to each transformer (n_households), every transformer in the run
  right  stable-core households (n_core), the ones the forecast is built on, transformers that were kept

Usage
  python households_histogram.py --work-dir ~/work/tf_national_v2
  python households_histogram.py --csv ~/work/tf_national_v2/clean/transformers_kept.csv --out hh_hist.png --bin 10 --xmax 400
"""
import argparse
import os

import numpy as np
import pandas as pd

INK, MUTED, GRID = "#1f2430", "#6b7280", "#e5e7eb"
C_ALL, C_CORE = "#2563b8", "#2f855a"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--csv", default=None, help="transformers_kept.csv (default <work-dir>/clean/transformers_kept.csv)")
    ap.add_argument("--out", default=None, help="png path (default next to the csv)")
    ap.add_argument("--bin", type=int, default=10, help="bin width in households")
    ap.add_argument("--xmax", type=int, default=400, help="right edge of the axis; transformers above it are counted in a note")
    a = ap.parse_args()
    path = os.path.expanduser(a.csv or os.path.join(a.work_dir or ".", "clean", "transformers_kept.csv"))
    k = pd.read_csv(path)
    out = os.path.expanduser(a.out or os.path.join(os.path.dirname(os.path.abspath(path)), "households_per_transformer.png"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

    allh = k["n_households"].dropna().astype(int)
    core = k.loc[k["kept"].astype(bool), "n_core"].dropna().astype(int)
    bins = np.arange(0, a.xmax + a.bin, a.bin)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax, s, col, title in ((axes[0], allh, C_ALL, "All households linked to the transformer"),
                              (axes[1], core, C_CORE, "Stable-core households (used in the forecast)")):
        ax.hist(s[s < a.xmax], bins=bins, color=col, edgecolor="#fcfcfb", linewidth=1.0, alpha=0.95)
        med, p10, p90 = s.median(), s.quantile(0.1), s.quantile(0.9)
        ax.axvline(med, color=INK, lw=1.2, ls="--")
        top = ax.get_ylim()[1]
        ax.text(med + a.xmax * 0.01, top * 0.97, f"median {med:.0f}", va="top", ha="left", fontsize=9.5, color=INK)
        above = int((s > a.xmax).sum())
        note = f"{len(s):,} transformers  ·  10th–90th percentile {p10:.0f}–{p90:.0f}  ·  max {s.max():,}"
        if above:
            note += f"\n{above} above {a.xmax} (not shown)"
        ax.text(0.98, 0.95, note, transform=ax.transAxes, ha="right", va="top", fontsize=9, color=MUTED, linespacing=1.5)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("households per transformer")
        ax.set_xlim(0, a.xmax)
        ax.grid(axis="y", color=GRID, lw=0.8), ax.set_axisbelow(True)
    axes[0].set_ylabel(f"transformers per {a.bin}-household bin")
    fig.text(0.01, -0.02, "National run: transformers need at least 20 households with daily readings to be drawn, so the left edge is a cut-off, "
             "not the true minimum.\nThe draw spreads transformers over coatos (square-root rule), so this is the run's sample, not a census.",
             fontsize=8.5, color=MUTED, va="top")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[hist] {len(allh):,} transformers, households per transformer: median {allh.median():.0f}, "
          f"mean {allh.mean():.1f}, p10 {allh.quantile(.1):.0f}, p90 {allh.quantile(.9):.0f}, max {allh.max():,}")
    print(f"[hist] core households (kept transformers): median {core.median():.0f}, p10 {core.quantile(.1):.0f}, p90 {core.quantile(.9):.0f}")
    print(f"[hist] wrote {out}")


if __name__ == "__main__":
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core"))
    from mlflow_utils import run_script
    run_script(main, "electricity/analysis")
