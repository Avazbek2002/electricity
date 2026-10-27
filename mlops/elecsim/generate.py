"""Synthetic operator data in the source schema (mlops/sql/001_source_schema.sql).

    python -m elecsim.generate --dsn postgresql://... --config elecsim/config.yaml --clock-to 2024-12-31

Everything the source system will ever contain is generated once into sim.pending_* with the date it ARRIVES; the clock
(elecsim.clock) then releases each day's arrivals into the operator tables, so the pipeline sees the data the way the
live system would: readings a day (sometimes weeks) late, open reading gaps at the right edge, meters replaced,
households connected, disconnected and moved between feeders.

The truth behind the readings (daily kWh per household) is aggregated to sim.truth_node_daily for diagnostics only.
"""
import argparse
import datetime as dt
import io
import json
import os

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SQL = os.path.join(os.path.dirname(HERE), "sql", "001_source_schema.sql")
ASKUE, MANUAL = 4, 1
EID_FITR = ["2022-05-02", "2023-04-21", "2024-04-10", "2025-03-30", "2026-03-20"]
EID_ADHA = ["2022-07-09", "2023-06-28", "2024-06-16", "2025-06-06", "2026-05-27"]
FIXED = [(1, 1), (3, 8), (3, 21), (5, 9), (9, 1), (10, 1), (12, 8)]
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_config(path=None):
    with open(path or os.path.join(HERE, "config.yaml")) as f:
        return _plain(yaml.safe_load(f))


def _plain(x):
    """YAML dates (keys and values) -> ISO strings, so the config is JSON-safe and stored with the clock."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_plain(v) for v in x]
    return x.isoformat() if isinstance(x, (dt.date, dt.datetime)) else x


# ----------------------------------------------------------------------------- calendar and weather
def calendar_factor(days, cfg):
    c = cfg["calendar"]
    f = np.array([c["weekday_factor"][WD[d.dayofweek]] for d in days])
    hol = np.zeros(len(days), bool)
    for m, d in FIXED:
        hol |= (days.month == m) & (days.day == d)
    for e in EID_FITR + EID_ADHA:
        hol |= days == pd.Timestamp(e)
    f = np.where(hol, f * c["holiday_factor"], f)
    for e in EID_FITR:
        eve = (days >= pd.Timestamp(e) - pd.Timedelta(days=3)) & (days < pd.Timestamp(e))
        f = np.where(eve, f * c["eid_eve_factor"], f)
    for d, v in (c.get("national_events") or {}).items():
        f = np.where(days == pd.Timestamp(d), f * float(v), f)
    return f


def weather(days, districts, cfg, rng):
    """Observed daily mean / min / max per district (seasonal cycle + AR(1) anomalies shared regionally)."""
    w = cfg["weather"]
    doy = days.dayofyear.to_numpy()
    season = w["annual_mean"] - w["annual_amplitude"] * np.cos(2 * np.pi * (doy - 15) / 365.25)
    common = np.zeros(len(days))
    for i in range(1, len(days)):
        common[i] = w["ar1_rho"] * common[i - 1] + rng.normal(0, w["ar1_sd"])
    out = {}
    for c in districts:
        own = np.zeros(len(days))
        for i in range(1, len(days)):
            own[i] = 0.6 * own[i - 1] + rng.normal(0, 0.8)
        t = season + rng.normal(0, w["district_offset_sd"]) + common + own
        spread = 5.5 + 2.0 * np.sin(2 * np.pi * (doy - 100) / 365.25) + rng.normal(0, 1.0, len(days))
        out[c] = pd.DataFrame({"date": days, "t_mean": t, "t_min": t - spread / 2 - 1, "t_max": t + spread / 2 + 1})
    return out


def forecasts(days, wx, cfg, rng):
    """AIFS / FCN3-like runs issued 00 UTC on the issue date: truth + per (model, district, season) bias + lead error."""
    w = cfg["weather"]
    L = int(w["forecast_lead_days"])
    sd = np.linspace(w["forecast_error_sd"][0], w["forecast_error_sd"][1], L + 1)
    sel = str(w.get("forecast_issue_days", "odd"))
    if sel == "odd":
        issue = days[days.day % 2 == 1]
    elif sel == "all":
        issue = days
    else:
        issue = days[days.day.isin([int(x) for x in sel.split(",")])]
    rows = []
    season = lambda d: (d.month % 12) // 3                    # noqa: E731
    for model in w["forecast_models"]:
        for c, obs in wx.items():
            bias = rng.normal(0, w["forecast_bias_sd"], 4)
            o = obs.set_index("date")
            for d in issue:
                lead = np.arange(L + 1)
                tgt = d + pd.to_timedelta(lead, unit="D")
                ok = tgt <= days[-1]
                e = np.cumsum(rng.normal(0, 1, L + 1)) / np.sqrt(np.arange(1, L + 2)) * sd + bias[season(d)]
                tt = o.reindex(tgt[ok])
                rows.append(pd.DataFrame({"model": model, "issue_date": d, "coato_code": c, "lead_day": lead[ok],
                                          "t_mean": tt["t_mean"].to_numpy() + e[ok], "t_min": tt["t_min"].to_numpy() + e[ok],
                                          "t_max": tt["t_max"].to_numpy() + e[ok]}))
    f = pd.concat(rows, ignore_index=True)
    f["arrived_at"] = f["issue_date"] + pd.Timedelta(hours=7)   # available the morning of the issue date
    return f


# ----------------------------------------------------------------------------- network and households
def network(cfg, rng):
    n = cfg["network"]
    feeders, subs = [], []
    for c in n["districts"]:
        for s in range(n["substations_per_district"]):
            sid = int(f"{c}{s + 1:02d}")
            subs.append(dict(substation_id=sid, coato_code=c))
            for f in range(n["feeders_per_substation"]):
                feeders.append(dict(fider_id=sid * 100 + f + 1, substation_id=sid, coato_code=c))
    return pd.DataFrame(feeders), pd.DataFrame(subs)


def households(cfg, feeders, days, rng):
    h, n = cfg["households"], cfg["network"]
    start, end = days[0], days[-1]
    rows = []
    for fd in feeders.itertuples():
        k0 = int(rng.integers(n["households_per_feeder"][0], n["households_per_feeder"][1] + 1))
        con = [start - pd.Timedelta(days=int(x)) for x in rng.integers(100, 6000, k0)]
        years = (end - start).days / 365.25
        k_new = rng.poisson(k0 * h["growth_per_year"] * years)
        con += [start + pd.Timedelta(days=int(x)) for x in rng.integers(1, (end - start).days, k_new)]
        for c in con:
            rows.append(dict(fider_id=fd.fider_id, substation_id=fd.substation_id, coato_code=fd.coato_code,
                             connect=c, transformer_id=fd.fider_id * 10 + int(rng.integers(1, n["transformers_per_feeder"] + 1))))
    H = pd.DataFrame(rows)
    m = len(H)
    H["consumer_id"] = 1_000_000 + rng.permutation(m * 3)[:m]
    life = rng.exponential(1 / h["disconnect_per_year"], m) * 365.25
    disc = H["connect"] + pd.to_timedelta(life.astype(int), unit="D")
    H["disconnect"] = disc.where(disc <= end, pd.NaT)
    H["base"] = rng.lognormal(*h["base_kwh_lognormal"], m)
    H["gas"] = rng.random(m) < h["gas_share"]
    bh_g, bh_e = h["heat_beta_gas"], h["heat_beta_electric"]
    H["beta_h"] = np.clip(np.where(H["gas"], rng.normal(*bh_g, m), rng.normal(*bh_e, m)), 0, None)
    H["beta_c"] = np.clip(rng.normal(*h["cool_beta"], m), 0, None)
    H["housing_type_id"] = np.where(rng.random(m) < h["apartment_share"], 2, 1)
    H["residents"] = rng.integers(1, 8, m)
    return H.sort_values("consumer_id").reset_index(drop=True)


def remaps(cfg, H, feeders, days, rng):
    """Blocks of households moved to a feeder of another substation: (consumer_id, date, new fider, new substation)."""
    ev = cfg["events"]
    out = []
    for _ in range(int(ev["remaps"])):
        a = feeders.sample(1, random_state=int(rng.integers(1 << 30))).iloc[0]
        cand = feeders[(feeders["coato_code"] == a.coato_code) & (feeders["substation_id"] != a.substation_id)]
        if not len(cand):
            continue
        b = cand.sample(1, random_state=int(rng.integers(1 << 30))).iloc[0]
        edge = min(300, len(days) // 4)
        when = days[0] + pd.Timedelta(days=int(rng.integers(edge, len(days) - edge)))
        members = H[(H["fider_id"] == a.fider_id) & (H["connect"] < when)]
        k = int(len(members) * rng.uniform(*ev["remap_share"]))
        for cid in members.sample(k, random_state=int(rng.integers(1 << 30)))["consumer_id"]:
            out.append(dict(consumer_id=cid, date=when, fider_id=b.fider_id, substation_id=b.substation_id))
    return pd.DataFrame(out, columns=["consumer_id", "date", "fider_id", "substation_id"])


def truth(H, wx, cal, days, cfg, rng):
    """True daily kWh, households x days (0 outside the connection)."""
    h = cfg["households"]
    idx = {c: i for i, c in enumerate(wx)}
    T = np.vstack([wx[c]["t_mean"].to_numpy() for c in wx])
    hdd, cdd = np.maximum(18 - T, 0), np.maximum(T - 24, 0)
    ci = H["coato_code"].map(idx).to_numpy()
    load = (1 + H["beta_h"].to_numpy()[:, None] * hdd[ci] + H["beta_c"].to_numpy()[:, None] * cdd[ci]) * cal[None, :]
    y = H["base"].to_numpy()[:, None] * load * rng.lognormal(-h["daily_noise_sd"] ** 2 / 2, h["daily_noise_sd"], load.shape)
    di = np.arange(len(days))
    c0 = ((H["connect"] - days[0]).dt.days).to_numpy()
    c1 = ((H["disconnect"].fillna(days[-1] + pd.Timedelta(days=1)) - days[0]).dt.days).to_numpy()
    active = (di[None, :] >= c0[:, None]) & (di[None, :] < c1[:, None])
    return np.where(active, y, 0.0).astype(np.float32), active


# ----------------------------------------------------------------------------- readings
def _lag(rng, dist, n):
    k = np.array(list(dist.keys()), dtype=int)
    p = np.array(list(dist.values()), dtype=float)
    return rng.choice(k, n, p=p / p.sum())


def readings(H, Y, active, days, cfg, rng):
    """Meter readings with arrival dates, meters (with replacements) and per-household reading regime."""
    r = cfg["readings"]
    regime = pd.Timestamp(cfg["households"]["regime_change"])
    D = len(days)
    years = D / 365.25
    a0, a1 = r["askue_share_start"], r["askue_share_end"]
    rows, meters = [], []
    next_meter = 50_000_000
    for i, hh in enumerate(H.itertuples()):
        act = np.flatnonzero(active[i])
        if not len(act):
            continue
        first, last = act[0], act[-1]
        # automatic reading from day `ask` on (meters migrate: the share grows from a0 to a1)
        u = rng.random()
        ask = 0 if u < a0 else (D + 1 if u > a1 else int((u - a0) / (a1 - a0) * D))
        coef = float(rng.choice(r["coefficient_choices"]))
        # meter replacements
        k = rng.poisson(r["meter_replacement_per_year"] * years)
        cuts = sorted(set(int(x) for x in rng.integers(first + 30, max(first + 31, last), k))) if last - first > 60 else []
        bounds = [first] + cuts + [last + 1]
        days_read = np.zeros(D, bool)
        man = rng.integers(*r["manual_interval_days"])
        off = int(rng.integers(0, man))
        days_read[np.arange(first + off, D, man)] = True                       # manual schedule
        auto = (np.arange(D) >= ask) & (rng.random(D) < r["askue_daily_success"])
        n_out = rng.poisson(r["askue_outage_rate_per_year"] * years)
        for s in rng.integers(0, D, n_out):                                    # ASKUE outages
            auto[s:s + int(rng.integers(*r["askue_outage_days"]))] = False
        days_read = np.where(np.arange(D) >= ask, auto, days_read)
        for m, (b0, b1) in enumerate(zip(bounds[:-1], bounds[1:])):
            meter = next_meter
            next_meter += 1
            meters.append(dict(consumer_id=hh.consumer_id, meter_id=meter, start=days[b0], end=days[b1 - 1], coef=coef,
                               askue_from=days[min(ask, D - 1)] if ask <= D else pd.NaT))
            seg = np.arange(b0, b1)
            sel = seg[days_read[seg]]
            sel = np.unique(np.r_[b0, sel, b1 - 1])                            # installation and final readings
            start_reg = float(rng.integers(0, 30_000)) * 1000 * 100 / coef
            cum = np.cumsum(Y[i, b0:b1].astype(float))                         # kWh to the end of each day
            reg = start_reg + cum[sel - b0] * 1000 * 100 / coef                # stored register units
            whole = days[sel] < regime                                         # old regime: whole kWh
            unit = 1000 * 100 / coef
            reg = np.where(whole, np.floor(reg / unit) * unit, np.round(reg, 3))
            src = np.where(sel >= ask, ASKUE, MANUAL)
            lag = np.where(src == ASKUE, _lag(rng, r["arrival_lag_askue"], len(sel)), _lag(rng, r["arrival_lag_manual"], len(sel)))
            st = np.where(rng.random(len(sel)) < r["bad_status_rate"], 2, 1)
            rows.append(pd.DataFrame({"consumer_id": hh.consumer_id, "meter_id": meter, "coato_code": hh.coato_code,
                                      "reading_date": days[sel], "reading_a_plus": reg, "cycle_a_plus": 0,
                                      "reading_source_type_id": src, "meter_reading_status_id": st,
                                      "arrival_date": days[sel] + pd.to_timedelta(lag, unit="D")}))
    R = pd.concat(rows, ignore_index=True)
    dup = R.sample(frac=r["duplicate_rate"], random_state=int(rng.integers(1 << 30)))
    R = pd.concat([R, dup], ignore_index=True)
    R = R.sort_values(["arrival_date", "reading_date", "meter_id"], kind="stable").reset_index(drop=True)
    R["seq"] = np.arange(len(R))
    R["id"] = 0
    return R, pd.DataFrame(meters)


# ----------------------------------------------------------------------------- metadata
def metadata(H, M, RM, days, cfg):
    """Pending current-snapshot changes (consumers) and monthly histories (consumer_histories)."""
    snaps = []
    for hh in H.itertuples():
        base = dict(id=hh.consumer_id, coato_code=hh.coato_code, transformer_id=hh.transformer_id,
                    additional_substation_id=hh.substation_id, tariff_id=1, accounting_status_id=1,
                    housing_type_id=hh.housing_type_id, housing_category_id=1, gaz_connection=bool(hh.gas),
                    central_heating=False, residents_amount=int(hh.residents), application_date=hh.connect.date(),
                    contraction_date=hh.connect.date())
        ms = M[M["consumer_id"] == hh.consumer_id].sort_values("start")
        if not len(ms):
            continue
        for k, m in enumerate(ms.itertuples()):
            snaps.append(dict(base, meter_id=m.meter_id, consumer_status_id=1, effective_date=max(m.start, days[0]) if k else hh.connect))
        if pd.notna(hh.disconnect):
            snaps.append(dict(base, meter_id=ms["meter_id"].iloc[-1], consumer_status_id=2, effective_date=hh.disconnect))
    S = pd.DataFrame(snaps)
    for ev in RM.itertuples():                                                  # remaps change the substation from then on
        later = S[(S["id"] == ev.consumer_id)].copy()
        if not len(later):
            continue
        last = later.sort_values("effective_date").iloc[[-1]].copy()
        last["additional_substation_id"] = ev.substation_id
        last["effective_date"] = ev.date
        S = pd.concat([S, last], ignore_index=True)
    S["consumer_status_update_at"] = pd.to_datetime(S["effective_date"])
    S["updated_at"] = pd.to_datetime(S["effective_date"]) + pd.Timedelta(hours=6)
    # monthly histories: state at the start of each month the customer is connected
    months = pd.date_range(days[0], days[-1], freq="MS")
    fider = {hh.consumer_id: hh.fider_id for hh in H.itertuples()}
    moved = {ev.consumer_id: (ev.date, ev.fider_id) for ev in RM.itertuples()}
    hist = []
    for m in M.itertuples():
        hh_disc = H.loc[H["consumer_id"] == m.consumer_id, "disconnect"].iloc[0]
        for p in months[(months >= m.start.to_period("M").to_timestamp()) & (months <= m.end)]:
            f = fider[m.consumer_id]
            if m.consumer_id in moved and p >= moved[m.consumer_id][0]:
                f = moved[m.consumer_id][1]
            hist.append(dict(consumer_id=m.consumer_id, meter_id=m.meter_id, period=p.date(),
                             coato_code=str(m.consumer_id)[:0] or None, transformation_coefficient=m.coef,
                             meter_capacity=60, meter_type_id=1,
                             askue_connection=bool(pd.notna(m.askue_from) and p >= m.askue_from),
                             tariff_id=1, consumer_status_id=1, fider_id=f,
                             disconnection_date=hh_disc.date() if pd.notna(hh_disc) else None,
                             meter_installation_date=m.start.date()))
    Hh = pd.DataFrame(hist)
    Hh["coato_code"] = Hh["consumer_id"].map(H.set_index("consumer_id")["coato_code"])
    Hh = Hh.drop_duplicates(["consumer_id", "period"], keep="last")            # a replacement mid-month: the new meter
    return S, Hh


def truth_nodes(H, Y, active, RM, days):
    """True daily kWh and households per substation and feeder (current mapping on each day)."""
    D = len(days)
    fid = np.repeat(H["fider_id"].to_numpy()[:, None], D, axis=1)
    sid = np.repeat(H["substation_id"].to_numpy()[:, None], D, axis=1)
    pos = {c: i for i, c in enumerate(H["consumer_id"])}
    for ev in RM.itertuples():
        i, d = pos[ev.consumer_id], (ev.date - days[0]).days
        fid[i, d:], sid[i, d:] = ev.fider_id, ev.substation_id
    co = H["coato_code"].to_numpy()
    out = []
    for kind, ids in (("S", sid), ("F", fid)):
        for v in np.unique(ids):
            m = ids == v
            c = co[np.argmax(m.any(axis=1))]
            out.append(pd.DataFrame({"node": f"{kind}:{c}:{v}", "date": days, "kwh": (Y * m).sum(axis=0),
                                     "households": (active & m).sum(axis=0)}))
    return pd.concat(out, ignore_index=True)


def lookups(cfg):
    src = pd.DataFrame({"id": [1, 2, 3, 4], "code": ["MANUAL", "CONTROLLER", "PHONE", "ASKUE"],
                        "name_uz": ["Qo'lda", "Nazoratchi", "Telefon", "ASKUE"], "name_ru": ["Вручную", "Контролёр", "Телефон", "АСКУЭ"],
                        "name_crl": ["Қўлда", "Назоратчи", "Телефон", "АСКУЭ"], "status": 1})
    st = pd.DataFrame({"id": [1, 2], "code": ["NORMAL", "SUSPICIOUS"], "name_uz": ["Normal", "Shubhali"],
                       "name_ru": ["Нормальный", "Подозрительный"], "name_crl": ["Нормал", "Шубҳали"], "status": 1})
    co = pd.DataFrame({"code": cfg["network"]["districts"], "parent_code": "901", "level_id": 3,
                       "location_id": range(len(cfg["network"]["districts"])), "status": 1})
    co["name_uz"] = "Synthetic district " + co["code"]
    return src, st, co[["code", "parent_code", "name_uz", "level_id", "location_id", "status"]]


def generate(cfg):
    rng = np.random.default_rng(int(cfg["seed"]))
    days = pd.date_range(cfg["start"], cfg["end"], freq="D")
    feeders, subs = network(cfg, rng)
    wx = weather(days, cfg["network"]["districts"], cfg, rng)
    cal = calendar_factor(days, cfg)
    H = households(cfg, feeders, days, rng)
    RM = remaps(cfg, H, feeders, days, rng)
    Y, active = truth(H, wx, cal, days, cfg, rng)
    R, M = readings(H, Y, active, days, cfg, rng)
    S, Hh = metadata(H, M, RM, days, cfg)
    obs = pd.concat([w.assign(coato_code=c) for c, w in wx.items()], ignore_index=True)
    obs["arrived_at"] = obs["date"] + pd.Timedelta(days=1, hours=3)            # yesterday's weather, early morning
    fc = forecasts(days, wx, cfg, rng)
    src, st, co = lookups(cfg)
    return dict(households=H, remaps=RM, readings=R, meters=M, consumers=S, histories=Hh, weather_obs=obs,
                weather_fc=fc, feeders=feeders, truth=truth_nodes(H, Y, active, RM, days), lookups=(src, st, co))


# ----------------------------------------------------------------------------- loading
def _copy(cur, table, df, cols):
    buf = io.StringIO()
    df[cols].to_csv(buf, index=False, header=False, na_rep="\\N", date_format="%Y-%m-%d")
    buf.seek(0)
    with cur.copy(f"COPY {table} ({', '.join(cols)}) FROM STDIN WITH (FORMAT csv, NULL '\\N')") as cp:
        while chunk := buf.read(1 << 20):
            cp.write(chunk)


def load(dsn, data, cfg, reset=True):
    import psycopg
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        if reset:
            for s in ("household_reading", "household_billing", "household_consumer_new", "weather", "sim"):
                cur.execute(f"DROP SCHEMA IF EXISTS {s} CASCADE")
        cur.execute(open(SQL).read())
        src, st, co = data["lookups"]
        lk = ["id", "code", "name_uz", "name_ru", "name_crl", "status"]
        _copy(cur, "household_reading.reading_source_types", src, lk)
        _copy(cur, "household_reading.meter_reading_statuses", st, lk)
        _copy(cur, "household_billing.coato_branches", co, list(co.columns))
        _copy(cur, "household_billing.network_feeders", data["feeders"], ["fider_id", "substation_id", "coato_code"])
        rc = ["id", "consumer_id", "meter_id", "coato_code", "reading_date", "reading_a_plus", "cycle_a_plus",
              "reading_source_type_id", "meter_reading_status_id", "arrival_date", "seq"]
        _copy(cur, "sim.pending_readings", data["readings"], rc)
        cc = ["id", "coato_code", "transformer_id", "additional_substation_id", "tariff_id", "meter_id",
              "accounting_status_id", "consumer_status_id", "consumer_status_update_at", "housing_type_id",
              "housing_category_id", "gaz_connection", "central_heating", "residents_amount", "application_date",
              "contraction_date", "updated_at", "effective_date"]
        _copy(cur, "sim.pending_consumers", data["consumers"], cc)
        _copy(cur, "sim.pending_histories", data["histories"], list(data["histories"].columns))
        _copy(cur, "sim.pending_weather_obs", data["weather_obs"],
              ["coato_code", "date", "t_mean", "t_min", "t_max", "arrived_at"])
        _copy(cur, "sim.pending_weather_fc", data["weather_fc"],
              ["model", "issue_date", "coato_code", "lead_day", "t_mean", "t_min", "t_max", "arrived_at"])
        _copy(cur, "sim.truth_node_daily", data["truth"], ["node", "date", "kwh", "households"])
        start = pd.Timestamp(cfg["start"]) - pd.Timedelta(days=1)
        cur.execute("INSERT INTO sim.clock (id, sim_date, next_reading_id, config) VALUES (1, %s, 1, %s)",
                    (start.date(), json.dumps(cfg, default=str)))
        conn.commit()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dsn", default=os.environ.get("SOURCE_DSN"), help="PostgreSQL DSN (env SOURCE_DSN)")
    p.add_argument("--config", default=None)
    p.add_argument("--clock-to", default=None, help="release all arrivals up to this date after loading")
    a = p.parse_args(argv)
    if not a.dsn:
        raise SystemExit("no DSN: --dsn or SOURCE_DSN")
    cfg = load_config(a.config)
    t0 = dt.datetime.now()
    data = generate(cfg)
    print(f"[sim] {len(data['households']):,} households, {len(data['meters']):,} meters, "
          f"{len(data['readings']):,} readings, {len(data['remaps'])} remapped households, "
          f"{len(data['weather_fc']):,} forecast rows ({(dt.datetime.now() - t0).seconds} s)", flush=True)
    load(a.dsn, data, cfg)
    print("[sim] loaded into sim.pending_*", flush=True)
    if a.clock_to:
        from elecsim import clock
        clock.advance(a.dsn, a.clock_to)


if __name__ == "__main__":
    main()
