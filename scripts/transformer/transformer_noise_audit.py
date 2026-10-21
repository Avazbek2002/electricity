#!/usr/bin/env python3
"""
Noise audit for transformer-level forecasting: what is in the data that a forecast should not learn from.

Run it on the output of `transformer_forecast.py prepare` (no Spark, no model, a few minutes):

    python scripts/transformer/transformer_noise_audit.py --work-dir ~/work/tf

It measures, on the household x day matrix and the transformer panel built from it:

  1. system-wide days      every transformer loses readings on the same days (ingestion outages)
  2. usable transformers   how many transformers have enough clean days to form windows at all
  1b. regional events      all transformers of one coato lose readings together while the country is normal
                           (a district-level network or reading problem; NOT counted as a transformer's own gap)
  3. local gaps            a transformer loses a share of its households for a while: dropout-and-return
                           (possible outage / concentrator), step-down, step-up (rollout, joiners), chronic
  4. churn and composition households joining / leaving, and whether those households consume differently
  5. household glitches    isolated spikes / dips (the same causal repair the household model uses) and the
                           share of a transformer-day they make up
  6. heavy consumers       one household that dominates its transformer's total
  7. dips                  transformer-days far below their own +-14 day median while coverage is normal
                           (outage-like events; load shedding; holidays when many transformers share them)
  8. small-sample noise    day-to-day noise against household count: the floor no forecast can beat
  9. calendar              weekday profile and the dates many transformers move together
 10. outage candidates     isolated dropout-and-return episodes: one transformer silent, its neighbours reporting

Outputs in <work-dir>/audit/: coverage_heatmap.png, noise_vs_size.png, audit_*.csv and audit_masks.npz (the flags
per transformer-day, ready to be used as masks), and the printed report (also in run_log.txt).
It describes the data; it changes nothing.
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core")))  # the shared modules
import transformer_forecast as tf                                    # noqa: E402

INK, MUTED, GRID, ACCENT = tf.INK, tf.MUTED, tf.GRID, tf.C_BU


def runs(mask):
    """1-D bool -> [(start, end_exclusive)] of consecutive True."""
    d = np.diff(np.r_[0, np.asarray(mask, dtype=int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def bands(n_hh, q=4):
    s = pd.Series(n_hh)
    code = pd.qcut(s.rank(method="first"), q, labels=False)
    names = {c: f"Q{c + 1}: {int(s[code == c].min())}-{int(s[code == c].max())} hh" for c in range(q)}
    return code.map(names).to_numpy(), [names[c] for c in range(q)]


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def fmt_dates(d0, a, b):
    return f"{(d0 + pd.Timedelta(days=int(a))):%Y-%m-%d} .. {(d0 + pd.Timedelta(days=int(b) - 1)):%Y-%m-%d} ({b - a} d)"


# -----------------------------------------------------------------------------
def audit(args):
    wp = tf.wp_module()
    out = os.path.join(args.work_dir, "audit")
    os.makedirs(out, exist_ok=True)
    info, hh, kwh = tf.load_world(args.work_dir)
    panel = tf.build_panel(wp, info, hh, kwh, args)
    n_tr, T = panel.inten.shape
    samp = os.path.join(args.work_dir, "transformers_sampled.csv")
    if os.path.exists(samp):                                         # is the panel the whole transformer, or a thinned slice of it?
        sd = pd.read_csv(samp, dtype={"transformer_id": str}).set_index("transformer_id")
        col = "n_daily" if "n_daily" in sd else "n_on_id"
        rr = (pd.Series(panel.n_hh, index=panel.ids.astype(str)) / sd[col].reindex(panel.ids.astype(str)).to_numpy()).dropna()
        print(f"[check] households in the panel vs households with daily readings on the same transformer id today: median "
              f"{rr.median():.0%} (10th pct {rr.quantile(.1):.0%}); transformers in the panel {len(rr)} of {len(sd)} sampled")
        if rr.median() < 0.7:
            print("[check] WARNING: the panel holds only a fraction of each transformer's households. Everything below describes a "
                  "thinned slice (coverage steps are coarse, sizes are too small). Re-run prepare so it reads every batch.")
    d0 = panel.d0
    dates = d0 + pd.to_timedelta(np.arange(T), unit="D")
    ids = panel.ids.astype(str)
    n_hh = panel.n_hh.astype(float)
    cover, inten = panel.cover, panel.inten
    n_rep = cover * n_hh[:, None]
    mincov = args.min_coverage
    band, band_names = bands(n_hh, args.size_bands)
    flags = {}

    # coverage of the whole sample and of each coato (the regional layer needs the coatos before the system days are decided)
    coato_of = (panel.rows.groupby("transformer_id")["coato_code"].agg(lambda x: x.mode().iloc[0]).reindex(ids).to_numpy()
                if "coato_code" in panel.rows else np.array(["?"] * n_tr))
    coatos, cidx = np.unique(coato_of.astype(str), return_inverse=True)
    roll = lambda v: pd.Series(v).rolling(61, center=True, min_periods=15).median().to_numpy()
    cov_c, ref_c, checked = {}, {}, []
    for c in range(len(coatos)):
        m = cidx == c
        if len(coatos) > 1 and m.sum() >= args.min_region_transformers:
            cov_c[c] = n_rep[m].sum(0) / n_hh[m].sum()
            ref_c[c] = roll(cov_c[c])
            checked.append(c)

    # ---- 1. system-wide days -------------------------------------------------------------------------------
    section("1. SYSTEM-WIDE DAYS (the same days are bad for every transformer)")
    cov_sys = n_rep.sum(0) / n_hh.sum()
    ref = roll(cov_sys)
    system_bad = (cov_sys < args.system_floor) | (cov_sys < args.system_drop * ref) | (cov_sys == 0)
    if len(checked) >= 4:                      # country-wide only if most coatos dropped; one big coato must not look like "the system"
        share_c = np.mean([(cov_c[c] < args.system_floor) | (cov_c[c] < args.system_drop * ref_c[c]) | (cov_c[c] == 0) for c in checked], axis=0)
        n_pooled = int(system_bad.sum())
        system_bad &= share_c >= args.system_share
        print(f"(pooled coverage dropped on {n_pooled} days; {int(system_bad.sum())} of them in at least {args.system_share:.0%} of "
              f"the {len(checked)} checked coatos -> system-wide; the rest are regional, see 1b)")
    ep = runs(system_bad)
    print(f"{int(system_bad.sum())} of {T} days have much less than the usual share of households reporting "
          f"(all transformers together; usual level {np.nanmedian(cov_sys):.0%}; a day counts when coverage is below "
          f"{args.system_floor:.0%} or below {args.system_drop:.0%} of its own 61-day median). "
          f"A steady low level (e.g. 75% for months) is a coverage regime, not a system day.")
    rows = []
    for a, b in ep:
        mc = float(cov_sys[a:b].mean())
        rows.append(dict(start=dates[a].date(), end=dates[b - 1].date(), days=b - a, min_coverage=cov_sys[a:b].min(),
                         mean_coverage=mc, severity="blackout (<30%)" if mc < 0.3 else "partial (30-60%)" if mc < 0.6 else "dip (>=60%)"))
        print(f"  {fmt_dates(d0, a, b)}: coverage {cov_sys[a:b].min():.0%} .. {cov_sys[a:b].max():.0%}")
    pd.DataFrame(rows).to_csv(os.path.join(out, "audit_system_days.csv"), index=False)
    print("-> not a property of any transformer: mask these days for every transformer (training targets and inputs).")
    flags["system_bad"] = system_bad

    # ---- 1b. regional events ------------------------------------------------------------------------------
    regional_bad = np.zeros((n_tr, T), dtype=bool)
    section("1b. REGIONAL EVENTS (a whole coato loses readings while the country is normal)")
    reg_rows = []
    if len(coatos) < 2:
        print("single coato in this work dir: no regional layer (the system days above are all there is).")
    else:
        usable_c = len(checked)
        for c in checked:
            m = cidx == c
            cov_r, ref_r = cov_c[c], ref_c[c]
            low_share = (cover[m] < mincov).mean(0)
            rb = (cov_r < args.region_drop * ref_r) & (low_share >= args.region_share) & ~system_bad
            regional_bad[m] = rb[None, :]
            for a, b in runs(rb):
                reg_rows.append(dict(coato_code=coatos[c], start=dates[a].date(), end=dates[b - 1].date(), days=b - a,
                                     n_transformers=int(m.sum()), mean_coverage=float(cov_r[a:b].mean()),
                                     usual_coverage=float(np.nanmedian(ref_r[a:b])), share_transformers_low=float(low_share[a:b].mean())))
        reg = pd.DataFrame(reg_rows)
        print(f"{len(coatos)} coatos in the sample; {usable_c} have >= {args.min_region_transformers} transformers and are checked.")
        print(f"a coato-day is regional when its coverage is below {args.region_drop:.0%} of its own 61-day median AND at least "
              f"{args.region_share:.0%} of its transformers are below {mincov:.0%}, on a day that is not system-wide.")
        if len(reg):
            reg["weight"] = reg["days"] * reg["n_transformers"]
            reg.sort_values("weight", ascending=False).drop(columns="weight").to_csv(os.path.join(out, "audit_regional_events.csv"), index=False)
            print(f"{len(reg)} regional episodes in {reg['coato_code'].nunique()} coatos; by length: "
                  + ", ".join(f"{k} d: {v}" for k, v in reg["days"].clip(upper=14).value_counts().sort_index().items())
                  + " (14 = 14+)")
            print("  largest (days x transformers): " + "; ".join(
                f"coato {r.coato_code} {r.start}..{r.end} ({r.days} d, {r.n_transformers} tr, {r.mean_coverage:.0%} vs usual {r.usual_coverage:.0%})"
                for r in reg.sort_values("weight", ascending=False).head(8).itertuples()))
            per = reg.groupby("coato_code")["days"].sum().sort_values(ascending=False)
            print("  coatos with the most regional days: " + ", ".join(f"{c} {v}" for c, v in per.head(8).items()))
        else:
            print("no regional episodes.")
        print(f"-> {regional_bad.mean():.1%} of transformer-days sit inside a regional event. These are excluded from 'local' gaps below: "
              "they are not about one transformer. For outage work they are the interesting feed/substation-scale events.")
    flags["regional_bad"] = regional_bad

    # ---- 2. usable transformers ----------------------------------------------------------------------------
    section("2. USABLE TRANSFORMERS (enough clean days to form 90-day input + 60-day target windows)")
    ok = np.isfinite(inten)
    share_ok = ok.mean(1)
    o_args = SimpleNamespace(origin_stride=1)
    r_tr, r_o, tgt_ok = tf.windows(wp, panel, tf.origin_grid(wp, panel, o_args))
    n_win = np.bincount(r_tr, minlength=n_tr)
    n_full = np.bincount(r_tr[tgt_ok], minlength=n_tr)
    print(f"transformers with >= {args.min_households} households: {n_tr}; households per transformer: "
          f"median {np.median(n_hh):.0f}, 90th pct {np.percentile(n_hh, 90):.0f}, max {n_hh.max():.0f}")
    for lo, hi, lab in [(0, 0, "no complete window"), (1, 99, "1-99"), (100, 299, "100-299"), (300, 10 ** 9, ">= 300")]:
        m = (n_full >= lo) & (n_full <= hi)
        print(f"  windows with complete input+target {lab:>20}: {int(m.sum()):5d} transformers "
              f"({m.mean():5.1%}), {int(n_hh[m].sum()):7,} households")
    print(f"  usable days (coverage >= {mincov:.0%}): median transformer {np.median(share_ok):.0%}; "
          f"{int((share_ok < 0.5).sum())} transformers below 50%")
    pd.DataFrame({"transformer_id": ids, "n_hh": n_hh, "usable_day_share": share_ok, "complete_windows": n_full,
                  "size_band": band}).to_csv(os.path.join(out, "audit_transformers.csv"), index=False)
    print("-> transformers with no complete window drop out of BOTH methods silently: the results describe the rest.")
    dead = n_full < args.min_windows
    flags["too_few_windows"] = dead

    # ---- 3. local gaps ------------------------------------------------------------------------------------
    section("3. LOCAL GAPS (one transformer loses households while the system is normal)")
    local_bad = (cover < mincov) & ~system_bad[None, :] & ~regional_bad
    gap_rows = []
    for k in range(n_tr):
        for a, b in runs(local_bad[k]):
            good = ~(local_bad[k] | system_bad | regional_bad[k])
            pre = cover[k, max(0, a - 14):a][good[max(0, a - 14):a]]
            post = cover[k, b:b + 14][good[b:b + 14]] if b < T else np.array([])
            before = float(np.median(pre)) if len(pre) else np.nan
            after = float(np.median(post)) if len(post) else np.nan
            if share_ok[k] < 0.5:
                kind = "chronic (transformer mostly below threshold)"
            elif a == 0 or not np.isfinite(before):
                kind = "series starts low (joiners / rollout)"
            elif b >= T or not np.isfinite(after):
                kind = "step-down at the end (leavers or ongoing outage)"
            elif after >= mincov:
                kind = "dropout and return"
            else:
                kind = "step-down, stays low"
            gap_rows.append(dict(transformer_id=ids[k], coato_code=coato_of[k], n_hh=int(n_hh[k]), start=dates[a].date(), end=dates[b - 1].date(),
                                 days=b - a, mean_coverage=float(cover[k, a:b].mean()), min_coverage=float(cover[k, a:b].min()),
                                 coverage_before=before, coverage_after=after, kind=kind))
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(os.path.join(out, "audit_local_gaps.csv"), index=False)
    print(f"{int(local_bad.sum()):,} transformer-days ({local_bad.mean():.1%}) below {mincov:.0%} coverage on a day that is normal for "
          f"the country{' and for its coato' if regional_bad.any() else ''}.")
    if len(gaps):
        gaps["length"] = pd.cut(gaps["days"], [0, 3, 7, 30, 10 ** 6], labels=["1-3 d", "4-7 d", "8-30 d", "> 30 d"])
        print("\nepisodes by kind and length:")
        print(pd.crosstab(gaps["kind"], gaps["length"]).to_string())
        dr = gaps[gaps["kind"] == "dropout and return"]
        if len(dr):
            print(f"\n'dropout and return' episodes: {len(dr)} on {dr['transformer_id'].nunique()} transformers; "
                  f"median {dr['days'].median():.0f} days, median coverage inside {dr['mean_coverage'].median():.0%} "
                  f"(vs {dr['coverage_before'].median():.0%} before)")
            print("  longest: " + "; ".join(f"{r.transformer_id} {r.start}..{r.end} ({r.days} d, {r.mean_coverage:.0%})"
                                            for r in dr.sort_values("days", ascending=False).head(5).itertuples()))
        ep_by_band = (gaps.assign(band=gaps["transformer_id"].map(dict(zip(ids, band))))
                      .groupby("band").size().reindex(band_names).fillna(0))
        print("\nepisodes per size band: " + ", ".join(f"{b} {int(v)}" for b, v in ep_by_band.items()))
    print("-> these days are NOT demand: a target built on them mixes 'households missing' with 'households using less'. "
          "Mask them; a dropout-and-return episode is also an outage candidate (see section 10).")
    flags["local_bad"] = local_bad

    # ---- 4. churn and composition -------------------------------------------------------------------------
    section("4. CHURN AND COMPOSITION (households that join / leave, and whether they differ)")
    first_day = np.full(len(hh), -1)
    last_day = np.full(len(hh), -1)
    for a in range(0, kwh.shape[0], 5000):
        f = np.isfinite(kwh[a:a + 5000])
        any_ = f.any(1)
        first_day[a:a + 5000] = np.where(any_, f.argmax(1), -1)
        last_day[a:a + 5000] = np.where(any_, T - 1 - f[:, ::-1].argmax(1), -1)
    anyrep = (cover > 0).any(0)
    t_start = int(anyrep.argmax()) if anyrep.any() else 0
    t_end = int(T - 1 - anyrep[::-1].argmax()) if anyrep.any() else T - 1
    late, early = np.zeros(n_tr), np.zeros(n_tr)
    ratio = np.full(n_tr, np.nan)
    heavy_rows, glitch = [], np.zeros((n_tr, T))
    masked_days = np.zeros((n_tr, T))
    n_spike = n_dip = 0
    top_share = np.full((n_tr, T), np.nan)
    t_loop = time.time()
    for k, tid in enumerate(panel.ids):
        if k and k % 100 == 0:
            el = time.time() - t_loop
            print(f"  [sections 4-6] {k:,}/{n_tr:,} transformers, {el / 60:.1f} min elapsed, about {el / k * (n_tr - k) / 60:.1f} min left", flush=True)
        rows = panel.hh_rows[tid]
        block = np.asarray(kwh[rows], dtype="float64")
        late[k] = (first_day[rows] > t_start + args.churn_days).mean()
        early[k] = (last_day[rows] < t_end - args.churn_days).mean()
        fin = np.isfinite(block)
        core = fin.mean(1) >= 0.85
        if core.sum() >= 3 and (~core).sum() >= 3:
            mc = np.nanmean(np.where(fin[core], block[core], np.nan), 0)
            mn = np.nanmean(np.where(fin[~core], block[~core], np.nan), 0)
            v = np.isfinite(mc) & np.isfinite(mn)
            if v.sum() >= 30:
                ratio[k] = mn[v].sum() / mc[v].sum()
        tot = np.nansum(block, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            top_share[k] = np.where(tot > 0, np.nanmax(np.where(fin, block, -1), 0) / tot, np.nan)
        hm = np.nanmean(np.where(fin, block, np.nan), 1)
        med = np.nanmedian(hm)
        for j in np.flatnonzero(hm > args.heavy_ratio * med):
            heavy_rows.append(dict(transformer_id=ids[k], meter_id=hh["meter_id"].iloc[rows[j]], mean_kwh=hm[j],
                                   transformer_median_hh=med, ratio=hm[j] / med, n_hh=int(n_hh[k])))
        for x in block:                                                  # 5. glitch repair, same function as training
            yc, _, msk = wp.clean_daily_series(x)
            diff = x - yc
            glitch[k] += np.abs(np.nan_to_num(diff))
            masked_days[k] += msk
            n_spike += int((diff > 0).sum())
            n_dip += int((diff < 0).sum())
    churny = (late + early) > args.churn_share
    print(f"households whose first reading is > {args.churn_days} d after the data starts: {late.mean():.1%} on average per transformer "
          f"(late joiners); whose last reading is > {args.churn_days} d before the end: {early.mean():.1%} (early leavers).")
    print(f"transformers where more than {args.churn_share:.0%} of households join or leave: {int(churny.sum())} "
          f"({churny.mean():.1%}); median household count of those: {np.median(n_hh[churny]) if churny.any() else float('nan'):.0f}")
    v = np.isfinite(ratio)
    if v.any():
        print(f"composition: on days when both report, households that do NOT report steadily use {np.nanmedian(ratio):.2f}x "
              f"what the steady ones use (median over {int(v.sum())} transformers; 10th-90th pct "
              f"{np.nanpercentile(ratio, 10):.2f} .. {np.nanpercentile(ratio, 90):.2f}). "
              f"Far from 1.00 means: when they join or leave, the per-household mean shifts for reasons that are not demand.")
    print("-> a level shift from churn is a data artefact. Options: a stable core panel, or active-household scaling.")
    flags["churny"] = churny
    pd.DataFrame({"transformer_id": ids, "late_joiner_share": late, "early_leaver_share": early,
                  "noncore_to_core_ratio": ratio}).to_csv(os.path.join(out, "audit_churn.csv"), index=False)

    section("5. HOUSEHOLD GLITCHES (isolated spikes and dips inside the households' own series)")
    tot_day = inten * n_rep
    with np.errstate(invalid="ignore", divide="ignore"):
        gshare = np.where(np.isfinite(tot_day) & (tot_day > 0), glitch / tot_day, np.nan)
    print(f"repaired by the household cleaner: {n_spike:,} spikes, {n_dip:,} dips; "
          f"{int(masked_days.sum()):,} household-days sit in a clustered anomaly (the household model drops those windows).")
    g = np.nan_to_num(gshare)
    print(f"transformer-days where glitches are > 2% of the total: {(g > 0.02).sum() / max(ok.sum(), 1):.2%}; "
          f"> 5%: {(g > 0.05).sum() / max(ok.sum(), 1):.2%}")
    for b in band_names:
        m = band == b
        print(f"  {b:>22}: share of days with glitches > 5% of total = {(g[m] > 0.05).sum() / max(ok[m].sum(), 1):.2%}")
    print("-> the same repair is applied to households in the bottom-up path; the direct path sums raw values, so "
          "spikes enter its series. Cleaning households before aggregating is the fix.")
    flags["glitch_share"] = gshare

    section("6. HEAVY CONSUMERS (one household dominating a transformer)")
    heavy = pd.DataFrame(heavy_rows)
    print(f"households using more than {args.heavy_ratio:.0f}x their transformer's median household: {len(heavy)} "
          f"on {heavy['transformer_id'].nunique() if len(heavy) else 0} transformers")
    ts = np.nanmedian(top_share, axis=1)
    for b in band_names:
        print(f"  {b:>22}: median share of the transformer total from its single biggest household = {np.nanmedian(ts[band == b]):.1%}")
    if len(heavy):
        heavy.sort_values("ratio", ascending=False).to_csv(os.path.join(out, "audit_heavy_consumers.csv"), index=False)
        print("  largest: " + "; ".join(f"{r.meter_id} {r.ratio:.0f}x on {r.transformer_id}" for r in heavy.sort_values("ratio", ascending=False).head(5).itertuples()))
    print("-> a shop or workshop on a residential transformer makes its intensity move with ONE customer. Candidates for exclusion.")

    # ---- 7. dips -------------------------------------------------------------------------------------------
    section("7. DIPS (usable days far below the transformer's own recent level: outage-like events)")
    s = pd.DataFrame(inten.T)
    refm = s.rolling(29, center=True, min_periods=15).median().to_numpy().T
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = inten / refm
    dip = np.isfinite(rel) & (rel < args.dip_ratio) & ~system_bad[None, :]
    share_other = (dip.sum(0)[None, :] - dip) / max(n_tr - 1, 1)
    local_dip = dip & (share_other < 0.1)
    common_dip = dip & (share_other >= 0.1)
    print(f"transformer-days below {args.dip_ratio:.0%} of their own 29-day median: {int(dip.sum())} "
          f"(local {int(local_dip.sum())}, shared with >= 10% of transformers {int(common_dip.sum())})")
    rows = []
    for k in range(n_tr):
        for a, b in runs(local_dip[k]):
            lost = float(np.nansum((refm[k, a:b] - inten[k, a:b]) * n_hh[k]))
            rows.append(dict(transformer_id=ids[k], n_hh=int(n_hh[k]), start=dates[a].date(), end=dates[b - 1].date(), days=b - a,
                             depth=float(1 - np.nanmean(rel[k, a:b])), lost_kwh=lost))
    dips = pd.DataFrame(rows)
    if len(dips):
        dips.sort_values("lost_kwh", ascending=False).to_csv(os.path.join(out, "audit_dip_candidates.csv"), index=False)
        print(f"local dip events: {len(dips)} on {dips['transformer_id'].nunique()} transformers; by length: "
              + ", ".join(f"{k} d: {v}" for k, v in dips["days"].clip(upper=7).value_counts().sort_index().items()))
        print("  biggest by energy lost: " + "; ".join(f"{r.transformer_id} {r.start}..{r.end} ({r.days} d, -{r.depth:.0%}, {r.lost_kwh:,.0f} kWh)"
                                                     for r in dips.sort_values("lost_kwh", ascending=False).head(5).itertuples()))
    cd = np.flatnonzero(common_dip.sum(0) >= max(3, 0.1 * n_tr))
    if len(cd):
        print("  days where >= 10% of transformers dip together (holiday, weather shock or data): "
              + ", ".join(f"{dates[i]:%Y-%m-%d}" for i in cd[:15]) + (" ..." if len(cd) > 15 else ""))
    print("-> a dip that is local, 1-5 days long and has normal coverage is the signature to inspect for an outage. "
          "For a demand forecast, mask them in the TARGET (or you teach the model to expect outages); keep them for outage analysis.")
    flags["local_dip"], flags["common_dip"] = local_dip, common_dip

    # ---- 8. small-sample noise ----------------------------------------------------------------------------
    section("8. SMALL-SAMPLE NOISE (day-to-day wobble that more modelling cannot remove)")
    c7 = pd.DataFrame(inten.T).rolling(7, center=True, min_periods=5).mean().to_numpy().T
    with np.errstate(invalid="ignore", divide="ignore"):
        resid = inten / c7 - 1
    resid[:, system_bad] = np.nan
    resid[local_dip | common_dip] = np.nan
    n_fin = np.isfinite(resid).sum(0)
    sysdev = np.where(n_fin >= 5, np.nanmedian(np.where(np.isfinite(resid), resid, np.nan), axis=0), np.nan)
    idio = resid - sysdev[None, :]                       # what is left after what ALL transformers do that day

    def robust_sd(x):
        x = x[np.isfinite(x)]
        return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) > 20 else np.nan
    noise = np.array([robust_sd(resid[k]) for k in range(n_tr)])
    noise_idio = np.array([robust_sd(idio[k]) for k in range(n_tr)])
    print("day-to-day wobble, robust s.d. in % of the transformer's level; 'own' = after removing what all transformers do that day")
    print(f"{'band':>22} {'transformers':>13} {'total wobble':>13} {'own wobble':>11}")
    for b in band_names:
        m = band == b
        print(f"{b:>22} {int(m.sum()):13d} {np.nanmedian(noise[m]):13.1%} {np.nanmedian(noise_idio[m]):11.1%}")
    v = np.isfinite(noise_idio) & (noise_idio > 0)
    if v.sum() > 5:
        slope = np.polyfit(np.log(n_hh[v]), np.log(noise_idio[v]), 1)[0]
        small = n_hh <= 15
        print(f"own wobble ~ households^{slope:.2f}  (pure sampling noise would be -0.50). "
              f"A transformer with <= 15 households wobbles about "
              f"{np.nanmedian(noise_idio[small]) if small.any() else float('nan'):.0%} a day on its own.")
    print("-> the 'total' part contains weather and calendar, which a model can learn; the 'own' part is mostly sampling "
          "noise, the floor for that transformer. Judge small transformers on 7-day or weekly totals, or use --min-households.")
    pd.DataFrame({"transformer_id": ids, "n_hh": n_hh, "wobble_total": noise, "wobble_own": noise_idio,
                  "size_band": band}).to_csv(os.path.join(out, "audit_noise_by_transformer.csv"), index=False)
    noise = noise_idio

    # ---- 9. calendar and weather ---------------------------------------------------------------------------
    section("9. WHAT ALL TRANSFORMERS DO TOGETHER (weekday, weather, holidays)")
    dow = np.asarray(dates.dayofweek)
    prof = np.array([np.nanmedian(sysdev[dow == d]) for d in range(7)])
    print("weekday profile (median over days, relative to the 7-day mean): "
          + "  ".join(f"{n} {v:+.1%}" for n, v in zip(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], prof)))
    y = sysdev - prof[dow]
    unexpl = y.copy()
    try:
        clim, wdf = tf.load_climate(wp, SimpleNamespace(work_dir=args.work_dir), None)
        wts = pd.Series(n_hh, index=coato_of.astype(str)).groupby(level=0).sum()          # households of the transformers in the panel
        piv = wdf[wdf["coato_id"].isin(wts.index)].pivot_table(index="d", columns="coato_id", values="t")
        if piv.shape[1] == 0:
            raise ValueError("none of the sampled coatos is in the weather table")
        w_ = wts.reindex(piv.columns).fillna(0.0)
        wc = ((piv * w_).sum(axis=1, min_count=1) / (piv.notna() * w_).sum(axis=1)).reindex(dates)
        print(f"(weather: household-weighted mean temperature of {piv.shape[1]} of {len(wts)} sampled coatos)")
        hd, cd = np.maximum(wp.HDD_BASE - wc, 0), np.maximum(wc - wp.CDD_BASE, 0)
        X = np.c_[(hd - hd.rolling(7, center=True, min_periods=5).mean()).to_numpy(),
                  (cd - cd.rolling(7, center=True, min_periods=5).mean()).to_numpy()]
        vm = np.isfinite(y) & np.isfinite(X).all(1)
        A = np.c_[np.ones(vm.sum()), X[vm]]
        coef, *_ = np.linalg.lstsq(A, y[vm], rcond=None)
        fit = A @ coef
        r2 = 1 - ((y[vm] - fit) ** 2).sum() / ((y[vm] - y[vm].mean()) ** 2).sum()
        unexpl = np.full(T, np.nan)
        unexpl[vm] = y[vm] - fit
        print(f"weather explains {r2:.0%} of the days' common movement (+{coef[1]:.1%} per extra heating degree-day vs the week, "
              f"{coef[2]:+.1%} per cooling degree-day).")
    except Exception as e:                                                # noqa: BLE001
        print(f"(weather regression skipped: {type(e).__name__}: {e})")
    thr = max(0.03, 3 * robust_sd(unexpl)) if np.isfinite(unexpl).any() else 0.03
    cand = np.flatnonzero(np.isfinite(unexpl) & ~system_bad & (np.abs(unexpl) >= thr))
    cand = cand[np.argsort(-np.abs(unexpl[cand]))][:15]
    print(f"days when transformers move together by more than {thr:.1%} that weekday and weather do not explain:")
    for i in sorted(cand):
        print(f"  {dates[i]:%Y-%m-%d} ({dates[i]:%a}): {unexpl[i]:+.1%}")
    if not len(cand):
        print("  none")
    print("-> match these against public holidays (Navruz, the two Hayit holidays, 1 Sept, New Year). "
          "A weekday / holiday feature for the target days turns this from noise into signal; the household model has none.")


    # ---- 10. outage candidates ----------------------------------------------------------------------------
    section("10. OUTAGE CANDIDATES (one transformer silent for days, its neighbours reporting, then it comes back)")
    cand = gaps[(gaps["kind"] == "dropout and return") & (gaps["days"] >= args.outage_min_days)
                & (gaps["mean_coverage"] < args.outage_coverage)].copy() if len(gaps) else pd.DataFrame()
    if len(cand):
        kidx = {t: i for i, t in enumerate(ids)}
        nb, conc, kind_, nsil = [], [], [], []
        for r in cand.itertuples():
            k = kidx[r.transformer_id]
            a = (pd.Timestamp(r.start) - d0).days
            b = (pd.Timestamp(r.end) - d0).days + 1
            mates = (cidx == cidx[k]) if len(coatos) > 1 else np.ones(n_tr, dtype=bool)
            mates[k] = False
            ref_kind = "coato"
            if mates.sum() < 3:                                    # too few sampled neighbours: compare with the whole sample
                mates, ref_kind = np.ones(n_tr, dtype=bool), "country"
                mates[k] = False
            nb.append(float(n_rep[mates, a:b].sum() / (n_hh[mates].sum() * (b - a))) if mates.sum() >= 3 else np.nan)
            silent = (cover[mates, a:b].mean(1) < mincov)
            conc.append(float(silent.mean()) if mates.sum() >= 3 else np.nan)
            nsil.append(int(silent.sum()))
            kind_.append(ref_kind)
        cand["neighbour_coverage"] = nb
        cand["share_neighbours_also_silent"] = conc
        cand["compared_with"] = kind_
        cand["n_neighbours_also_silent"] = nsil
        cand["isolated"] = np.where(cand["compared_with"] == "coato", cand["n_neighbours_also_silent"] == 0,
                                    cand["share_neighbours_also_silent"] <= args.isolated_share)
        yrs = (T / 365.25)
        print(f"{len(cand)} episodes of >= {args.outage_min_days} days with < {args.outage_coverage:.0%} of households reporting, "
              f"then a return, on {cand['transformer_id'].nunique()} of {n_tr} transformers "
              f"({cand['transformer_id'].nunique() / n_tr:.1%}); {int(cand['isolated'].sum())} are isolated "
              f"(no other transformer of their coato silent at the same time; when the coato has fewer than 4 sampled transformers, at most "
              f"{args.isolated_share:.0%} of the whole sample), {int((~cand['isolated']).sum())} coincide with others.")
        iso = cand[cand["isolated"]]
        print(f"isolated episodes per transformer-year: {len(iso) / (n_tr * yrs):.3f}; "
              f"by length: " + ", ".join(f"{lab} {int(((iso['days'] >= lo) & (iso['days'] <= hi)).sum())}"
                                         for lo, hi, lab in [(3, 7, "3-7 d"), (8, 14, "8-14 d"), (15, 30, "15-30 d"), (31, 10 ** 6, ">30 d")]))
        m_ = pd.to_datetime(iso["start"]).dt.month.value_counts().reindex(range(1, 13)).fillna(0).astype(int)
        print("isolated episode starts by calendar month: " + " ".join(f"{mo:02d}:{v}" for mo, v in m_.items()))
        rep_ = iso.groupby("transformer_id").size().sort_values(ascending=False)
        print(f"transformers with >= 3 isolated episodes (repeat offenders: failing unit or gateway rather than one-off): "
              f"{int((rep_ >= 3).sum())}; top: " + ", ".join(f"{t} x{v}" for t, v in rep_.head(8).items()))
        if "n_hh" in iso:
            print(f"households affected by isolated episodes: median {iso['n_hh'].median():.0f} per episode, "
                  f"total household-episodes {int(iso['n_hh'].sum()):,}")
        cand.sort_values(["isolated", "days"], ascending=[False, False]).to_csv(os.path.join(out, "audit_outage_candidates.csv"), index=False)
    else:
        print("no candidates at these settings.")
    print("-> a candidate is NOT proof of an outage (a dead reading gateway looks the same): it is the list to check against a utility log "
          "or a known incident. Without ground truth, repeat offenders and shared-with-neighbour episodes need different explanations.")

    np.savez_compressed(os.path.join(out, "audit_masks.npz"), transformer_id=ids, dates=dates.values.astype("datetime64[D]"),
                        system_bad=system_bad, regional_bad=regional_bad, coato_code=coato_of.astype(str), local_bad=local_bad, local_dip=local_dip, common_dip=common_dip,
                        glitch_share=np.nan_to_num(gshare, nan=0.0).astype("float32"), churny=churny, too_few_windows=dead)

    make_figures(out, cover, cov_sys, n_hh, dates, noise, mincov, ids)
    print(f"\nwrote the CSVs, audit_masks.npz and two figures to {out}")


# -----------------------------------------------------------------------------
def make_figures(out, cover, cov_sys, n_hh, dates, noise, mincov, ids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    order = np.argsort(n_hh)
    cmap = LinearSegmentedColormap.from_list("gap", ["#1b4f93", "#9bbbe0", "#f4f6fa"])     # low coverage = dark, full = near white
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(11, 7.2), gridspec_kw=dict(height_ratios=[1, 4], hspace=0.08), sharex=True)
    x = mdates.date2num(dates.to_pydatetime())
    a0.plot(dates, cov_sys * 100, color=ACCENT, lw=1.6)
    a0.axhline(mincov * 100, color=MUTED, lw=1, ls=(0, (4, 3)))
    a0.set_ylim(0, 105), a0.set_ylabel("system\ncoverage, %"), a0.grid(axis="y", color=GRID), a0.set_axisbelow(True)
    a0.set_title("Share of households reporting, by transformer and day  (dark = missing; one row per transformer, small at the bottom)",
                 loc="left", fontsize=10.5)
    a1.imshow(cover[order], aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
              extent=[x[0] - 0.5, x[-1] + 0.5, len(order), 0])
    a1.xaxis_date()
    a1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    a1.set_ylabel("transformers, sorted by size")
    a1.set_yticks([])
    fig.savefig(os.path.join(out, "coverage_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    v = np.isfinite(noise) & (noise > 0)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.scatter(n_hh[v], noise[v] * 100, s=14, color=ACCENT, alpha=0.55, linewidths=0)
    if v.sum() > 5:
        xs = np.geomspace(n_hh[v].min(), n_hh[v].max(), 50)
        ref = np.median(noise[v] * np.sqrt(n_hh[v]))
        ax.plot(xs, ref / np.sqrt(xs) * 100, color=INK, lw=1.4, ls=(0, (4, 3)), label="pure sampling noise (1/sqrt(n))")
        ax.legend(frameon=False, fontsize=9)
    from matplotlib.ticker import FuncFormatter
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel("households on the transformer"), ax.set_ylabel("day-to-day noise, % of level")
    ax.grid(color=GRID), ax.set_axisbelow(True)
    ax.set_title("Small transformers are noisy by arithmetic", loc="left", fontsize=11)
    fig.tight_layout(), fig.savefig(os.path.join(out, "noise_vs_size.png"), dpi=150), plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--min-households", type=int, default=5)
    p.add_argument("--max-households", type=int, default=1500)
    p.add_argument("--min-hh-days", type=int, default=30)
    p.add_argument("--min-coverage", type=float, default=0.8)
    p.add_argument("--gap-fill", type=int, default=3)
    p.add_argument("--max-transformers", type=int, default=0)
    p.add_argument("--size-bands", type=int, default=4)
    p.add_argument("--system-drop", type=float, default=0.9, help="a day is system-wide bad if coverage < this x its 61-day median")
    p.add_argument("--system-floor", type=float, default=0.6, help="... or if overall coverage is below this (a steady 70-79%% level is a regime, not an event)")
    p.add_argument("--system-share", type=float, default=0.5, help="with >= 4 checked coatos: a day is system-wide only if this share of coatos dropped too")
    p.add_argument("--min-region-transformers", type=int, default=8, help="coatos with fewer sampled transformers get no regional check")
    p.add_argument("--region-drop", type=float, default=0.75, help="regional day: coato coverage < this x its own 61-day median ...")
    p.add_argument("--region-share", type=float, default=0.4, help="... and at least this share of its transformers below --min-coverage")
    p.add_argument("--isolated-share", type=float, default=0.1, help="an outage candidate is isolated if at most this share of its comparison transformers were silent too")
    p.add_argument("--outage-min-days", type=int, default=3, help="outage candidate: at least this many days ...")
    p.add_argument("--outage-coverage", type=float, default=0.3, help="... with less than this share of households reporting")
    p.add_argument("--churn-days", type=int, default=90, help="a household that starts this long after the data, or stops this long before its end, counts as churn")
    p.add_argument("--churn-share", type=float, default=0.2, help="transformer flagged if this share of households join or leave")
    p.add_argument("--heavy-ratio", type=float, default=5.0, help="household using more than this x its transformer's median household")
    p.add_argument("--dip-ratio", type=float, default=0.6, help="dip = usable day below this x its 29-day median")
    p.add_argument("--min-windows", type=int, default=100, help="transformers with fewer complete windows count as unusable")
    args = p.parse_args(argv)
    args.work_dir = os.path.abspath(os.path.expanduser(args.work_dir))
    with tf.logging_to(os.path.join(args.work_dir, "run_log.txt")):
        audit(args)


if __name__ == "__main__":
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core"))
    from mlflow_utils import run_script
    run_script(main, "electricity/data-prep", stage="transformer_audit")
