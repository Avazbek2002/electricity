#!/usr/bin/env python3
"""
Cleaning stage for transformer-level work: turns the household x day matrix from `transformer_forecast.py prepare`
into one clean, flagged series per transformer, plus a list of silence / outage events.

    python scripts/transformer/transformer_clean.py --work-dir ~/work/tf_national_v2

It applies what `transformer_noise_audit.py` found, in this order:

  1. system days        national collection outages (same rule as the audit)             -> flag 1
  2. regional events    a whole coato silent while the country is normal                -> flag 2
  3. household glitches isolated spikes / dips repaired, clustered anomalies dropped (the household model's own
                        `clean_daily_series`), BEFORE households are added up
  4. heavy consumers    households using more than --heavy-ratio x their transformer's median are left out
  5. stable core panel  only households that report on at least --core-presence of the transformer's observed days.
                        Households that join, leave or report only now and then use ~0.76x what steady ones use, so
                        counting them makes the level jump when they appear. The series is the core's mean per
                        reporting household (intensity), and n_core is fixed per transformer
  6. gaps               a day is a gap when core coverage is below --gap-ratio x the transformer's OWN usual coverage
                        (61-day rolling median): a transformer that always reports 70% is not "in a gap" every day   -> flag 4
  7. outside            before the transformer's first / after its last observed day                                    -> flag 8
  8. local dips         intensity far below its own recent level while reporting normally, and NOT shared with the
                        transformer's coato peers or the country (holidays and weather are shared)                   -> flag 16

No look-ahead (for forecast evaluation): `--fit-end <date>` chooses heavy consumers, the stable core and churn from days up
to that date only, and `--reference trailing` (the default) compares each day with the days BEFORE it when flagging system
days, regional events, gaps and dips. Set --fit-end to the earliest train cutoff of the forecast run:

    python scripts/transformer/transformer_clean.py --work-dir ~/work/tf_national_v2 --fit-end 2025-03-01

Outputs in <work-dir>/clean/:
  transformer_clean.npz    ids, coato, dates, intensity (cleaned), intensity_raw, n_core, cover, flags (uint8 bits above)
  transformers_kept.csv    one row per transformer: sizes, core share, usable share, complete 90+60 windows, why dropped
  core_households.csv      meter_id, transformer_id, role (core / noncore / heavy / short)
  outage_events.csv        silent stretches (core coverage < --event-cover for >= --event-min-days) and local dips, classified
                           isolated / coincident / regional / national, with kWh at risk
  clean_report.txt         what was removed and how much remains

Flags are bits: 1 system, 2 regional, 4 gap, 8 outside, 16 dip.  usable_input = no bit of (1|2|4|8);
usable_target = also no bit 16 (set --keep-dips to treat dips as usable).

About the events (read this before using them): the daily table is built so that reading intervals longer than 3 days are
dropped and 4-day flat runs are nulled, so silences shorter than 4 days do not exist in the data and boundaries are good to about
+-3 days. A silent transformer is NOT proof of an outage: a dead reading gateway looks identical. Use the list to check
against a utility log.
"""
import argparse
import os
import sys
import time
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core")))  # the shared modules
import transformer_forecast as tf                                   # noqa: E402

SYSTEM, REGIONAL, GAP, OUTSIDE, DIP = 1, 2, 4, 8, 16


def runs(mask):
    d = np.diff(np.r_[0, np.asarray(mask, dtype=int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def rolling_median(a, win, minp, trailing=True):
    """a (n, T) -> rolling median along time, NaN-aware.
    trailing=True: the median of the `win` days BEFORE each day (the day itself excluded), so a day is judged only against
    its past, as a live system would judge it. trailing=False: centred on the day (the old behaviour; it looks ahead)."""
    df = pd.DataFrame(np.atleast_2d(a).T)
    if trailing:
        df = df.shift(1)
    out = df.rolling(win, center=not trailing, min_periods=minp).median().to_numpy().T
    return out if np.ndim(a) > 1 else out[0]


# -----------------------------------------------------------------------------------------------------------------
def national_and_regional(n_rep, n_hh, cidx, n_coatos, dates, args):
    """Same rules as the audit (sections 1 and 1b). -> system_bad (T,), regional_bad (n_tr, T), regional events"""
    T = n_rep.shape[1]
    cover = n_rep / n_hh[:, None]
    trailing = args.reference == "trailing"
    roll = lambda v: rolling_median(v, 61, 15, trailing)                                                # noqa: E731
    cov_c, ref_c, checked = {}, {}, []
    for c in range(n_coatos):
        m = cidx == c
        if n_coatos > 1 and m.sum() >= args.min_region_transformers:
            cov_c[c] = n_rep[m].sum(0) / n_hh[m].sum()
            ref_c[c] = roll(cov_c[c])
            checked.append(c)
    cov_sys = n_rep.sum(0) / n_hh.sum()
    ref = roll(cov_sys)
    system_bad = (cov_sys < args.system_floor) | (cov_sys < args.system_drop * ref) | (cov_sys == 0)
    if len(checked) >= 4:
        share_c = np.mean([(cov_c[c] < args.system_floor) | (cov_c[c] < args.system_drop * ref_c[c]) | (cov_c[c] == 0)
                           for c in checked], axis=0)
        system_bad &= share_c >= args.system_share
    regional = np.zeros((len(n_rep), T), dtype=bool)
    rows = []
    for c in checked:
        m = cidx == c
        low_share = (cover[m] < args.region_low_cover).mean(0)
        rb = (cov_c[c] < args.region_drop * ref_c[c]) & (low_share >= args.region_share) & ~system_bad
        regional[m] = rb[None, :]
        for a, b in runs(rb):
            rows.append(dict(coato=c, start=dates[a], end=dates[b - 1], days=b - a))
    return system_bad, regional, rows


# -----------------------------------------------------------------------------------------------------------------
def clean(args):
    wp = tf.wp_module()
    out = os.path.join(args.work_dir, "clean")
    os.makedirs(out, exist_ok=True)
    t_start = time.time()
    info, hh, kwh = tf.load_world(args.work_dir)
    if str(info.get("source", "")).startswith("household_enriched") and not args.clean_households:
        args.no_clean_households = True
        print(f"[source] {info['source']}: households are already cleaned (Layer-1 spike, flat-run and gap rules), so the "
              f"household glitch repair is skipped and clean kwh is used as delivered (--clean-households to run it anyway)")
    T = kwh.shape[1]
    d0 = pd.Timestamp(info["start"])
    dates = d0 + pd.to_timedelta(np.arange(T), unit="D")
    hh = hh.reset_index(drop=True)
    print(f"[load] {len(hh):,} households x {T:,} days from {args.work_dir}")

    if args.fit_end:
        fe = int(np.searchsorted(dates, pd.Timestamp(args.fit_end), side="right")) - 1
        if fe < 60:
            raise SystemExit(f"--fit-end {args.fit_end} leaves fewer than 60 days of data (the panel starts {dates[0].date()})")
        print(f"[causal] heavy consumers, stable core and churn are chosen on days up to {dates[fe].date()} only; "
              f"thresholds use {'trailing' if args.reference == 'trailing' else 'CENTRED (looks ahead)'} references")
    else:
        fe = T - 1
        print("[causal] WARNING: no --fit-end, so the stable core and heavy consumers are chosen on ALL days, test months included. "
              "Pass --fit-end <the earliest train cutoff you will use> for a forecast evaluation without look-ahead.")

    # ---- which households and transformers take part ---------------------------------------------------------
    # reading days up to the fit end: a household that only appears later is unknown when the model is trained
    fin_days = np.concatenate([np.isfinite(kwh[a:a + 5000, :fe + 1]).sum(axis=1) for a in range(0, kwh.shape[0], 5000)])
    size_on_id = hh.groupby("transformer_id")["meter_id"].transform("size").to_numpy()
    role = np.where(fin_days < args.min_hh_days, "short", "").astype(object)
    role[size_on_id > args.max_households] = "placeholder"
    in_panel = role == ""
    h = hh[in_panel].reset_index().rename(columns={"index": "row"})
    sizes = h.groupby("transformer_id")["row"].size()
    ids = np.array(sorted(sizes[sizes >= args.min_households].index, key=str), dtype=object)
    if args.max_transformers:
        ids = np.sort(np.random.default_rng(0).choice(ids, size=min(args.max_transformers, len(ids)), replace=False)).astype(object)
    groups = h.groupby("transformer_id")
    hh_rows = {t: groups.get_group(t)["row"].to_numpy() for t in ids}
    n_tr = len(ids)
    n_hh = np.array([len(hh_rows[t]) for t in ids], dtype=float)
    coato_of = np.array([hh["coato_code"].iloc[hh_rows[t]].mode().iloc[0] if "coato_code" in hh else "?" for t in ids]).astype(str)
    coatos, cidx = np.unique(coato_of, return_inverse=True)
    print(f"[panel] {int((role == 'short').sum()):,} households with < {args.min_hh_days} reading days and "
          f"{int((role == 'placeholder').sum()):,} on placeholder ids left out; {n_tr:,} transformers with >= {args.min_households} "
          f"households, {len(coatos)} coatos")

    # ---- 1-2. system days and regional events (needs every transformer's raw coverage) ------------------------------
    n_rep_raw = np.zeros((n_tr, T))
    for k, t in enumerate(ids):
        n_rep_raw[k] = np.isfinite(kwh[hh_rows[t]]).sum(0)
    system_bad, regional, reg_rows = national_and_regional(n_rep_raw, n_hh, cidx, len(coatos), dates, args)
    print(f"[masks] system days: {int(system_bad.sum())} of {T}; regional transformer-days: {regional.mean():.2%} "
          f"({len(reg_rows)} coato episodes)")
    if system_bad.mean() > 0.15:
        cov_sys = n_rep_raw.sum(0) / n_hh.sum()
        yr = pd.Series(cov_sys, index=dates).groupby(dates.year).median()
        print(f"[masks] WARNING: {system_bad.mean():.0%} of days are flagged as national outages. The absolute floor "
              f"--system-floor {args.system_floor} is probably above this panel's normal coverage (median by year: "
              + ", ".join(f"{y} {v:.0%}" for y, v in yr.items())
              + "). Observed-only labels (enriched dataset) lower coverage in the older years; re-run with a floor below "
              f"those levels, e.g. --system-floor {max(0.1, round(0.6 * float(yr.min()), 2))}")
    excl = [(np.searchsorted(dates, pd.Timestamp(a)), np.searchsorted(dates, pd.Timestamp(b), side="right")) for a, b in wp.EXCLUDE_RANGES]

    # ---- 3-5. household cleaning, heavy consumers, stable core, transformer series -------------------------------------
    inten = np.full((n_tr, T), np.nan)
    inten_raw = np.full((n_tr, T), np.nan)
    cover = np.zeros((n_tr, T))
    n_core = np.zeros(n_tr, dtype=int)
    n_heavy = np.zeros(n_tr, dtype=int)
    first_obs = np.full(n_tr, -1)
    last_obs = np.full(n_tr, -1)
    dropped = {}
    n_rep_core = np.zeros((n_tr, T))
    spikes = dips = masked_hd = 0
    t_loop = time.time()
    for k, tid in enumerate(ids):
        if k and k % 100 == 0:
            el = time.time() - t_loop
            print(f"  [households] {k:,}/{n_tr:,} transformers, {el / 60:.1f} min, about {el / k * (n_tr - k) / 60:.1f} min left", flush=True)
        rows = hh_rows[tid]
        block = np.asarray(kwh[rows], dtype="float64")
        cl = block.copy()
        if not args.no_clean_households:
            for i in range(len(rows)):
                yc, rep, msk = wp.clean_daily_series(block[i])
                yc = np.array(yc, dtype="float64")
                yc[msk.astype(bool)] = np.nan                       # clustered anomalies: dropped, as in the household model
                spikes += int((rep & (block[i] > yc)).sum())
                dips += int((rep & (block[i] < yc)).sum())
                masked_hd += int(msk.sum())
                cl[i] = yc
        # who is heavy / core / churn is decided on days up to the fit end only (fe), as a model trained then would know it
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            hm = np.nanmean(np.where(np.isfinite(block[:, :fe + 1]), block[:, :fe + 1], np.nan), axis=1)
            med = np.nanmedian(hm)
        heavy = hm > args.heavy_ratio * med if med > 0 else np.zeros(len(rows), dtype=bool)
        n_heavy[k] = int(heavy.sum())
        for i in np.flatnonzero(heavy):
            role[rows[i]] = "heavy"
        fin = np.isfinite(cl) & ~heavy[:, None]
        n_light = int((~heavy).sum())
        frac = fin.sum(0) / max(n_light, 1)
        obs = np.flatnonzero((frac >= args.observed_share) & ~system_bad)
        if len(obs) == 0:
            dropped[tid] = "never observed"
            continue
        first_obs[k], last_obs[k] = obs.min(), obs.max()                 # the whole period: only used to flag days with no data
        obs_fit = obs[obs <= fe]
        if len(obs_fit) == 0:
            dropped[tid] = "not observed before the fit end"
            continue
        first_fit, last_fit = obs_fit.min(), obs_fit.max()
        elig = ~system_bad & ~regional[k]
        elig[:first_fit] = False
        elig[last_fit + 1:] = False
        for a, b in excl:
            elig[a:b] = False
        if elig.sum() < 60:
            dropped[tid] = "fewer than 60 observed days before the fit end"
            continue
        pres = fin[:, elig].sum(1) / elig.sum()
        thr_p = min(args.core_presence, 0.8 * float(np.median(pres[~heavy]))) if (~heavy).any() else args.core_presence
        fin_fit = fin[:, :fe + 1]
        anyf = fin_fit.any(1)
        f_day = np.where(anyf, fin_fit.argmax(1), T)                    # never seen by the fit end -> not core
        l_day = np.where(anyf, fe - fin_fit[:, ::-1].argmax(1), -1)
        churn = (f_day > first_fit + args.churn_days) | (l_day < last_fit - args.churn_days)      # joined / left before the fit end
        core = (pres >= thr_p) & ~churn & ~heavy
        for i in np.flatnonzero(~heavy):
            role[rows[i]] = "core" if core[i] else ("churn" if churn[i] else "intermittent")
        n_core[k] = int(core.sum())
        if n_core[k] < args.min_core:
            dropped[tid] = "core too small"
            continue
        fc = np.isfinite(cl[core])
        nr = fc.sum(0)
        n_rep_core[k] = nr
        cover[k] = nr / n_core[k]
        with np.errstate(invalid="ignore", divide="ignore"):
            inten[k] = np.where(nr > 0, np.where(fc, cl[core], 0).sum(0) / nr, np.nan)
            fr = np.isfinite(block[core])
            nrr = fr.sum(0)
            inten_raw[k] = np.where(nrr > 0, np.where(fr, block[core], 0).sum(0) / nrr, np.nan)
    kept = n_core >= args.min_core
    kept &= np.array([t not in dropped for t in ids])
    print(f"[households] repaired {spikes:,} spikes and {dips:,} dips, dropped {masked_hd:,} household-days in clustered anomalies; "
          f"{int(n_heavy.sum()):,} heavy consumers left out; {int(kept.sum()):,} of {n_tr:,} transformers have a stable core of >= {args.min_core}")

    # ---- 6-8. flags -----------------------------------------------------------------------------------------------
    flags = np.zeros((n_tr, T), dtype=np.uint8)
    flags[:, system_bad] |= SYSTEM
    flags[regional] |= REGIONAL
    for k in range(n_tr):
        if first_obs[k] >= 0:
            flags[k, :first_obs[k]] |= OUTSIDE
            flags[k, last_obs[k] + 1:] |= OUTSIDE
        else:
            flags[k, :] |= OUTSIDE
        for a, b in excl:
            flags[k, a:b] |= SYSTEM
    trailing = args.reference == "trailing"
    good = (flags & (SYSTEM | REGIONAL | OUTSIDE)) == 0
    cover_good = np.where(good & kept[:, None], cover, np.nan)
    ref = rolling_median(cover_good, 61, 15, trailing)                                # each transformer's own usual coverage
    thr = np.where(np.isfinite(ref), args.gap_ratio * ref, args.gap_floor)           # the floor only applies where no own baseline exists
    gap = good & kept[:, None] & ((cover < thr) | (n_rep_core < args.min_reporting))
    flags[gap] |= GAP
    usable_in = (flags & (SYSTEM | REGIONAL | GAP | OUTSIDE)) == 0
    intensity_ok = np.where(usable_in & np.isfinite(inten), inten, np.nan)

    # local dips: on RAW core intensity (the glitch cleaner would repair a 1-day outage away), relative to own median,
    # and not shared with coato peers / the country (shared = holiday, weather, collection)
    raw_ok = np.where(usable_in & np.isfinite(inten_raw), inten_raw, np.nan)
    refm = rolling_median(raw_ok, 29, 15, trailing)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = raw_ok / refm
    dip = np.isfinite(rel) & (rel < args.dip_ratio)
    n_valid = np.maximum(np.isfinite(rel).sum(0), 1)
    nat_share = (dip.sum(0) / n_valid)[None, :]
    peer_share = np.zeros_like(rel)
    for c in range(len(coatos)):
        m = cidx == c
        if m.sum() >= 5:
            tot = dip[m].sum(0)
            peer_share[m] = (tot[None, :] - dip[m]) / (m.sum() - 1)
    local_dip = dip & (nat_share < args.dip_national_share) & (peer_share < args.dip_peer_share)
    flags[local_dip] |= DIP
    print(f"[flags] share of transformer-days: system {np.mean(flags & SYSTEM > 0):.1%}, regional {np.mean(flags & REGIONAL > 0):.1%}, "
          f"gap {np.mean(flags & GAP > 0):.1%}, outside the observed period {np.mean(flags & OUTSIDE > 0):.1%}, local dip {np.mean(flags & DIP > 0):.2%}")

    # ---- usable windows (90 input + 60 target) and the 'kept' table -------------------------------------------------
    usable_t = usable_in if args.keep_dips else (usable_in & ((flags & DIP) == 0))
    series = np.where(usable_t & np.isfinite(inten), inten, np.nan)
    series = tf.fill_short_gaps(series, args.gap_fill)
    fake = SimpleNamespace(filled=series, n_days=T)
    r_tr, r_o, tgt_ok = tf.windows(wp, fake, tf.origin_grid(wp, fake, SimpleNamespace(origin_stride=1)))
    n_win = np.bincount(r_tr[tgt_ok], minlength=n_tr)
    share_use = np.where(kept, np.mean(np.isfinite(series), axis=1), 0.0)
    why = np.array([dropped.get(t, "") for t in ids], dtype=object)
    why[(why == "") & ~kept] = "core too small"
    kt = pd.DataFrame(dict(transformer_id=ids, coato_code=coato_of, n_households=n_hh.astype(int), n_core=n_core, n_heavy=n_heavy,
                           core_share=np.where(n_hh > 0, n_core / n_hh, np.nan), usable_day_share=share_use,
                           complete_windows=np.where(kept, n_win, 0), kept=kept, reason_dropped=why))
    kt.to_csv(os.path.join(out, "transformers_kept.csv"), index=False)
    hh_out = hh.assign(role=role)[["meter_id", "transformer_id", "role"]]
    hh_out[hh_out["transformer_id"].isin(set(ids))].to_csv(os.path.join(out, "core_households.csv"), index=False)

    # ---- events -------------------------------------------------------------------------------------------------
    ev = find_events(args, ids, coato_of, cidx, dates, cover, n_core, inten, inten_raw, ref, flags, kept, first_obs, last_obs,
                     local_dip, rel, system_bad)
    ev.to_csv(os.path.join(out, "outage_events.csv"), index=False)

    np.savez_compressed(os.path.join(out, "transformer_clean.npz"), transformer_id=ids.astype(str), coato_code=coato_of,
                        dates=dates.values.astype("datetime64[D]"), intensity=np.where(usable_t, inten, np.nan).astype("float32"),
                        intensity_all=inten.astype("float32"), intensity_raw=inten_raw.astype("float32"),
                        n_core=n_core, cover=cover.astype("float32"), flags=flags, kept=kept,
                        fit_end=np.array(str(dates[fe].date())), reference=np.array(args.reference))
    report(args, out, ids, kt, flags, kept, dates, ev, n_win, T, system_bad, regional, n_hh, role)
    print(f"\nwrote {out} in {(time.time() - t_start) / 60:.1f} min")


# -----------------------------------------------------------------------------------------------------------------
def find_events(args, ids, coato_of, cidx, dates, cover, n_core, inten, inten_raw, ref, flags, kept, first_obs, last_obs,
                local_dip, rel, system_bad):
    n_tr, T = cover.shape
    rows = []
    usable = (flags & (SYSTEM | REGIONAL | OUTSIDE)) == 0
    silent = (cover < args.event_cover) & usable & kept[:, None]
    # core size varies; a "silent" day also needs at least min_reporting households missing to be meaningful
    total = inten * n_core[:, None]
    for k in np.flatnonzero(kept):
        s = silent[k] | (system_bad & (np.arange(T) >= first_obs[k]) & (np.arange(T) <= last_obs[k]))   # bridge national outages
        for a, b in runs(s):
            sl = silent[k, a:b]
            if sl.sum() < args.event_min_days:
                continue
            idx = np.flatnonzero(sl)
            a, b = a + idx[0], a + idx[-1] + 1                                                          # trim system days at the edges
            days = int(sl.sum())
            before = cover[k, max(0, a - 28):a][usable[k, max(0, a - 28):a] & ~silent[k, max(0, a - 28):a]]
            after = cover[k, b:b + 28][usable[k, b:b + 28] & ~silent[k, b:b + 28]]
            cb = float(np.median(before)) if len(before) >= 3 else np.nan
            ca = float(np.median(after)) if len(after) >= 3 else np.nan
            if b > last_obs[k] - 2:
                kind = "ongoing at the end of the data"
            elif not np.isfinite(ca):
                kind = "no usable days after"
            elif np.isfinite(cb) and ca < 0.7 * cb:
                kind = "step-down (does not recover)"
            elif not np.isfinite(cb):
                kind = "starts silent"
            else:
                kind = "silence and return"
            reg_share = float(((flags[k, a:b] & REGIONAL) > 0).mean())
            mates = (cidx == cidx[k]) & kept
            mates[k] = False
            compared = "coato"
            if mates.sum() < 3:
                mates, compared = kept.copy(), "country"
                mates[k] = False
            m_sil = (silent[mates, a:b].sum(1) >= 0.5 * (b - a)) if mates.any() else np.zeros(0, bool)
            n_sil = int(m_sil.sum())
            share = float(m_sil.mean()) if len(m_sil) else np.nan
            if reg_share >= 0.5:
                cls = "regional (coato-wide)"
            elif (compared == "coato" and n_sil == 0) or (compared == "country" and share <= args.isolated_share):
                cls = "isolated"
            else:
                cls = "coincident (neighbours silent too)"
            base = np.r_[total[k, max(0, a - 28):a], total[k, b:b + 28]]
            base = base[np.isfinite(base)]
            exp_day = float(np.median(base)) if len(base) >= 5 else np.nan
            rows.append(dict(transformer_id=ids[k], coato_code=coato_of[k], n_core=int(n_core[k]), type="silence", start=dates[a].date(),
                             end=dates[b - 1].date(), days=days, mean_cover=float(cover[k, a:b][silent[k, a:b]].mean()),
                             cover_before=cb, cover_after=ca, kind=kind, classification=cls, compared_with=compared,
                             n_peers_silent=n_sil, share_peers_silent=share, expected_kwh_per_day=exp_day,
                             kwh_at_risk=exp_day * days if np.isfinite(exp_day) else np.nan,
                             start_weekday=dates[a].day_name()[:3], return_weekday=dates[min(b, T - 1)].day_name()[:3]))
        # local dips with normal coverage: short outage-like events the silence rule cannot see
        for a, b in runs(local_dip[k]):
            e = ((flags[k, a:b] & DIP) > 0)
            base = total[k, max(0, a - 28):a]
            base = base[np.isfinite(base)]
            exp_day = float(np.median(base)) if len(base) >= 5 else np.nan
            drop = float(1 - np.nanmean(rel[k, a:b]))
            rows.append(dict(transformer_id=ids[k], coato_code=coato_of[k], n_core=int(n_core[k]), type="dip", start=dates[a].date(),
                             end=dates[b - 1].date(), days=b - a, mean_cover=float(cover[k, a:b].mean()), cover_before=np.nan,
                             cover_after=np.nan, kind=f"consumption {drop:.0%} below own level, meters reporting",
                             classification="isolated", compared_with="coato/country", n_peers_silent=0, share_peers_silent=0.0,
                             expected_kwh_per_day=exp_day, kwh_at_risk=exp_day * drop * (b - a) if np.isfinite(exp_day) else np.nan,
                             start_weekday=dates[a].day_name()[:3], return_weekday=dates[min(b, T - 1)].day_name()[:3]))
    return pd.DataFrame(rows)


def report(args, out, ids, kt, flags, kept, dates, ev, n_win, T, system_bad, regional, n_hh, role):
    lines = []
    P = lambda s="": lines.append(s)                                                       # noqa: E731
    P("=" * 78), P("CLEANING REPORT"), P("=" * 78)
    ro = pd.Series(role[role != ""]).value_counts()
    P(f"households by role: " + ", ".join(f"{k} {v:,}" for k, v in ro.items()))
    P(f"transformers: {len(ids):,} considered, {int(kept.sum()):,} kept ({kept.mean():.1%}); dropped: "
      + ", ".join(f"{k} {v}" for k, v in kt.loc[~kt['kept'], 'reason_dropped'].value_counts().items()))
    kk = kt[kt["kept"]]
    if len(kk):
        P(f"kept transformers: households median {kk['n_households'].median():.0f}, core median {kk['n_core'].median():.0f} "
          f"({kk['core_share'].median():.0%} of households); usable days median {kk['usable_day_share'].median():.0%}")
        for lo, hi, lab in [(0, 0, "no complete window"), (1, 99, "1-99"), (100, 299, "100-299"), (300, 10 ** 9, ">= 300")]:
            m = (kk["complete_windows"] >= lo) & (kk["complete_windows"] <= hi)
            P(f"  complete 90+60-day windows {lab:>20}: {int(m.sum()):5d} transformers ({m.mean():5.1%}), {int(kk.loc[m, 'n_core'].sum()):8,} core households")
        yr = pd.DataFrame(np.isfinite(np.load(os.path.join(out, "transformer_clean.npz"))["intensity"][kept]).T, index=dates).groupby(dates.year).mean().mean(axis=1)
        P("usable share of days by year (kept transformers): " + ", ".join(f"{y} {v:.0%}" for y, v in yr.items()))
    P("")
    P("flags, share of transformer-days: " + ", ".join(f"{n} {np.mean((flags & b) > 0):.1%}"
                                                         for n, b in [("system", SYSTEM), ("regional", REGIONAL), ("gap", GAP), ("outside", OUTSIDE), ("dip", DIP)]))
    if len(ev):
        P(""), P("=" * 78), P("EVENTS"), P("=" * 78)
        s = ev[ev["type"] == "silence"]
        d = ev[ev["type"] == "dip"]
        P(f"silences (core coverage < {args.event_cover:.0%} for >= {args.event_min_days} days): {len(s)} on {s['transformer_id'].nunique()} transformers")
        if len(s):
            P(pd.crosstab(s["classification"], s["kind"]).to_string())
            iso = s[(s["classification"] == "isolated") & (s["kind"] == "silence and return")]
            yrs = pd.to_datetime(iso["start"]).dt.year.value_counts().sort_index()
            P(f"isolated silence-and-return: {len(iso)}; by year: " + ", ".join(f"{y}: {v}" for y, v in yrs.items()))
            P("  by length: " + ", ".join(f"{lab} {int(((iso['days'] >= lo) & (iso['days'] <= hi)).sum())}"
                                           for lo, hi, lab in [(4, 7, "4-7 d"), (8, 14, "8-14 d"), (15, 30, "15-30 d"), (31, 10 ** 6, ">30 d")]))
            rep = iso.groupby("transformer_id").size().sort_values(ascending=False)
            P(f"  transformers with >= 3 isolated silences: {int((rep >= 3).sum())}; top: " + ", ".join(f"{t} x{v}" for t, v in rep.head(8).items()))
            top = iso.sort_values("kwh_at_risk", ascending=False).head(5)
            P("  largest by kWh at risk: " + "; ".join(f"{r.transformer_id} {r.start}..{r.end} ({r.days} d, ~{r.kwh_at_risk:,.0f} kWh)" for r in top.itertuples()))
        P(f"local dips with normal reporting (not shared with peers): {len(d)} on {d['transformer_id'].nunique()} transformers; "
          f"by length 1 d: {int((d['days'] == 1).sum())}, 2-3 d: {int(d['days'].between(2, 3).sum())}, >= 4 d: {int((d['days'] >= 4).sum())}")
        P("-> boundaries are good to ~+-3 days (reading cycle); silences shorter than 4 days do not exist in this data; a silent "
          "transformer is not proof of an outage.")
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(out, "clean_report.txt"), "w") as fh:
        fh.write(txt + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--fit-end", default=None,
                   help="choose heavy consumers, the stable core and churn from days up to this date only (YYYY-MM-DD). Set it to the "
                        "EARLIEST train cutoff of the forecast run (the first --folds cutoff), so no test-period data shapes the panel. "
                        "Default: all days (looks ahead)")
    p.add_argument("--reference", choices=["trailing", "centred"], default="trailing",
                   help="system-day, regional, gap and dip thresholds compare a day with the median of the days BEFORE it (trailing, "
                        "causal) or of the days around it (centred, the old behaviour)")
    p.add_argument("--min-households", type=int, default=20, help="transformers with fewer households (with enough readings) are not considered")
    p.add_argument("--max-households", type=int, default=1500)
    p.add_argument("--min-hh-days", type=int, default=30)
    p.add_argument("--max-transformers", type=int, default=0, help="debug: random subset")
    p.add_argument("--no-clean-households", action="store_true", help="skip the household glitch repair")
    p.add_argument("--clean-households", action="store_true",
                   help="run the household glitch repair even on the enriched dataset, which is already cleaned (default there: off)")
    p.add_argument("--heavy-ratio", type=float, default=5.0, help="household using more than this x the transformer's median household is left out")
    p.add_argument("--core-presence", type=float, default=0.7, help="a household is core if it reports on this share of the transformer's observed days "
                   "(lowered to 0.8 x the transformer's median household when the whole transformer reports less)")
    p.add_argument("--churn-days", type=int, default=90, help="a household whose first reading is this long after the transformer's first observed day, or last this long before its last, is churn")
    p.add_argument("--min-core", type=int, default=15, help="smallest stable core a transformer can have")
    p.add_argument("--observed-share", type=float, default=0.3, help="a day counts as observed when this share of its households report")
    p.add_argument("--gap-ratio", type=float, default=0.75, help="gap day: core coverage below this x the transformer's own usual coverage ...")
    p.add_argument("--gap-floor", type=float, default=0.4, help="... (where a transformer has no baseline yet: below this absolute coverage)")
    p.add_argument("--min-reporting", type=int, default=8, help="... or fewer than this many core households reporting")
    p.add_argument("--gap-fill", type=int, default=3, help="interpolate gaps up to this many days in the series used for windows")
    p.add_argument("--dip-ratio", type=float, default=0.6)
    p.add_argument("--dip-national-share", type=float, default=0.03, help="a dip is shared (not local) if more than this share of transformers dip that day")
    p.add_argument("--dip-peer-share", type=float, default=0.25, help="... or more than this share of its coato peers")
    p.add_argument("--keep-dips", action="store_true", help="treat local dips as usable (default: masked in the target)")
    p.add_argument("--event-cover", type=float, default=0.3, help="silence: core coverage below this")
    p.add_argument("--event-min-days", type=int, default=4, help="silence: at least this many days (shorter ones cannot exist in this data)")
    p.add_argument("--isolated-share", type=float, default=0.1)
    p.add_argument("--system-drop", type=float, default=0.9)
    p.add_argument("--system-floor", type=float, default=0.6)
    p.add_argument("--system-share", type=float, default=0.5)
    p.add_argument("--min-region-transformers", type=int, default=8)
    p.add_argument("--region-drop", type=float, default=0.75)
    p.add_argument("--region-share", type=float, default=0.4)
    p.add_argument("--region-low-cover", type=float, default=0.8)
    args = p.parse_args(argv)
    args.work_dir = os.path.abspath(os.path.expanduser(args.work_dir))
    with tf.logging_to(os.path.join(args.work_dir, "run_log.txt")):
        clean(args)


if __name__ == "__main__":
    main()