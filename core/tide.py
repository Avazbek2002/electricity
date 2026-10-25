"""TiDE (Das et al. 2023, "Long-term Forecasting with TiDE") for household daily kWh, trained on daily AND interval labels.

    past y (90 days, scaled by s), its mask, coarse-day flag ─┐
    covariates of the 90 past + 60 target days ──► feature projection (per day, ResBlock -> r) ─┤
    static: log s, input coverage, coarse share, [regime], housing / district embeddings ──────┤
                                                    dense encoder (ResBlocks) ─► dense decoder ─► (60 x p)
                                     temporal decoder per target day: [decoder_t, projected covariates_t] -> 1
                                                    + linear skip from the past 90 days to the 60 outputs
The model forecasts every household day o + 1 .. o + 60 (direct, multi-output). Loss per window (scaled units):
    daily rows      huber(yhat_t - y_t)                         observed target days
    interval rows   len x huber((sum_{t in I} yhat_t - Y_I) / len)  reading intervals wholly inside the target window
                    (the Gaussian weight of a sum of len days; an interval says nothing about its daily shape)
    weight          old_weight for windows issued before the regime cutoff, x s^weight_power (energy weighting)
Missing target days have no row. Early stopping on validation households (a stable 5 % of customers, never trained on)
issued in the last 84 days before the cutoff, partial target windows included up to the refit date - 1.

Training data: core/tide_data.py (read per district in worker processes; customer keys never leave them).
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

import tide_data as td

DEFAULTS = dict(lookback=90, buckets=[0, 1, 2, 3, 4], stride=7, stride_new=1, val_stride=7, year_ago=True, min_input_days=60, min_recent_days=20,
                min_target_days=20, coarse_inputs="old", regime_cutoff="2024-08-01", old_weight=1.0, regime_flag=False,
                interval_labels=True, val_households=0.05, exclude_ranges=[["2023-01-01", "2023-02-28"]], min_scale=0.05,
                model=dict(hidden=512, enc_layers=2, dec_layers=2, decoder_dim=16, temporal_hidden=64, cov_proj=4,
                           cov_hidden=64, dropout=0.2, layernorm=True, house_dim=4, dist_dim=8),
                train=dict(batch=4096, lr=1e-3, weight_decay=1e-4, epochs=40, windows_per_epoch=2_000_000, patience=4,
                           loss="mse", huber_delta=1.0, weight_power=1.0, val_windows=200_000, seed=42, device="auto", amp=True,
                           fallback_epochs=8, min_val_windows=1000, max_store_gb=60),
                predict=dict(device="cpu", batch=8192, threads=None))


def merged(cfg):
    out = json.loads(json.dumps(DEFAULTS))
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------------------- the network
class ResBlock(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout, layernorm):
        super().__init__()
        self.lin1, self.lin2 = nn.Linear(d_in, d_hidden), nn.Linear(d_hidden, d_out)
        self.skip = nn.Linear(d_in, d_out)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out) if layernorm else nn.Identity()

    def forward(self, x):
        return self.norm(self.drop(self.lin2(torch.relu(self.lin1(x)))) + self.skip(x))


class TiDE(nn.Module):
    def __init__(self, L, H, n_cov, n_stat, n_dist, mc):
        super().__init__()
        self.L, self.H, self.p = L, H, int(mc["decoder_dim"])
        r, hid, dr, ln = int(mc["cov_proj"]), int(mc["hidden"]), float(mc["dropout"]), bool(mc["layernorm"])
        self.proj = ResBlock(n_cov, int(mc["cov_hidden"]), r, dr, ln)
        self.house = nn.Embedding(td.HOUSING_IDS, int(mc["house_dim"]))
        self.dist = nn.Embedding(n_dist + 1, int(mc["dist_dim"]))         # row 0 = a district not seen in training
        d_in = 5 * L + 2 * H + (L + H) * r + n_stat + int(mc["house_dim"]) + int(mc["dist_dim"])
        enc = [ResBlock(d_in, hid, hid, dr, ln)] + [ResBlock(hid, hid, hid, dr, ln) for _ in range(int(mc["enc_layers"]) - 1)]
        dec = [ResBlock(hid, hid, hid, dr, ln) for _ in range(int(mc["dec_layers"]) - 1)] + \
              [ResBlock(hid, hid, H * self.p, dr, ln)]
        self.encoder, self.decoder = nn.Sequential(*enc), nn.Sequential(*dec)
        self.temporal = ResBlock(self.p + r + 2, int(mc["temporal_hidden"]), 1, dr, ln)   # + year-ago value, mask
        self.skip = nn.Linear(L, H)

    def forward(self, b):
        cov = torch.cat([b["cov_p"], b["cov_f"]], 1)                       # (B, L + H, n_cov)
        f = self.proj(cov)                                                 # (B, L + H, r)
        x = torch.cat([b["y"], b["m"], b["c"], b["ly_p"], b["ly_pm"], b["ly_f"], b["ly_fm"], f.flatten(1), b["stat"],
                       self.house(b["house"]), self.dist(b["dist"])], 1)
        g = self.decoder(self.encoder(x)).view(-1, self.H, self.p)
        ly = torch.stack([b["ly_f"], b["ly_fm"]], 2)
        out = self.temporal(torch.cat([g, f[:, self.L:, :], ly], 2)).squeeze(-1)
        return out + self.skip(b["y"])


def huber(r, delta):
    if delta is None:                                  # squared error: the forecast targets the MEAN (node sums add up)
        return 0.5 * r * r
    a = r.abs()
    return torch.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))


def window_loss(pred, b, tc, old_weight):
    """-> (loss, dict of row counts) for one batch (scaled units)."""
    delta = None if tc.get("loss", "mse") == "mse" else float(tc["huber_delta"])
    daily = huber(pred - b["yt"], delta) * b["mt"]
    C = torch.cat([torch.zeros_like(pred[:, :1]), torch.cumsum(pred, 1)], 1)       # C[:, j] = sum of the first j days
    pos = torch.arange(pred.shape[1], device=pred.device)[None, :]
    st = (pos - b["il"] + 1).clamp(min=0)
    s_int = C[:, 1:] - torch.gather(C, 1, st)
    ln = b["il"].clamp(min=1).float()
    inter = ln * huber((s_int - b["isum"]) / ln, delta) * b["im"]
    w = torch.where(b["old"], torch.full_like(b["s"], float(old_weight)), torch.ones_like(b["s"]))
    if float(tc["weight_power"]):
        w = w * b["s"].clamp(min=0.05) ** float(tc["weight_power"])
    w = w[:, None]
    den = (w * (b["mt"] + ln * b["im"])).sum().clamp(min=1e-6)
    return (w * (daily + inter)).sum() / den, dict(daily=float(b["mt"].sum()), interval=float(b["im"].sum()))


# ----------------------------------------------------------------------------- data assembly
def _to_device(st, device):
    out = {}
    for k, v in st.items():
        t = torch.from_numpy(np.ascontiguousarray(v))
        out[k] = t.to(device)
    return out


def assemble(parts, spec, cfg):
    """District results -> one store (numpy), window lists, per-district normals."""
    d0, d_end = pd.Timestamp(spec.start), pd.Timestamp(spec.data_end)
    T = (d_end - d0).days + 1
    Tp = T + td.H + 2
    parts = [p for p in parts if p.get("n")]
    N = sum(p["n"] for p in parts)
    gb = N * Tp * 11 / 1e9
    print(f"[tide] store: {N:,} households x {Tp} days = {gb:.1f} GB")
    if gb > float(cfg["train"]["max_store_gb"]):
        raise SystemExit(f"[tide] store {gb:.1f} GB > train.max_store_gb: use fewer customer buckets")
    st = dict(val=np.full((N, Tp), np.nan, np.float32), kind=np.zeros((N, Tp), np.int8), coff=np.zeros((N, Tp), np.int8),
              ilen=np.zeros((N, Tp), np.int8), isum=np.full((N, Tp), np.nan, np.float32),
              house=np.zeros(N, np.int64), dist=np.zeros(N, np.int64))
    dates = pd.date_range(d0, periods=Tp, freq="D")
    cov = np.zeros((len(parts) + 1, Tp, td.N_COV), np.float32)
    norms, coatos = {}, []
    tr_h, tr_o, va_h, va_o = [], [], [], []
    a = 0
    for i, p in enumerate(sorted(parts, key=lambda q: q["code"]), start=1):
        n = p["n"]
        for k in ("val", "kind", "coff", "ilen", "isum"):
            st[k][a:a + n, :T] = p["store"][k]
        st["house"][a:a + n] = p["house"]
        st["dist"][a:a + n] = i
        wt = np.full((3, Tp), np.nan)
        wt[:, :T] = p["wt"]
        nm = td.normals(p["wt"], dates[:T])
        norms[p["code"]] = nm
        coatos.append(p["code"])
        cov[i] = td.covariates(dates, wt, nm)
        tr_h.append(p["train"][0] + a)
        tr_o.append(p["train"][1])
        va_h.append(p["val"][0] + a)
        va_o.append(p["val"][1])
        a += n
        for k in ("store", "wt"):
            p.pop(k, None)
    st["cov"] = cov
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, np.int64)          # noqa: E731
    return st, (cat(tr_h), cat(tr_o)), (cat(va_h), cat(va_o)), norms, coatos


# ----------------------------------------------------------------------------- fit
def fit(spec, artifacts_dir, cfg_in):
    import concurrent.futures as cf
    import multiprocessing as mp
    import enriched_io as eio
    cfg = merged(cfg_in)
    tc = cfg["train"]
    if len(spec.roots) != 1 or spec.roots[0].get("env_prefix"):
        raise SystemExit("tide trains on the developer zone only (one root, normal key)")
    d0 = pd.Timestamp(spec.start)
    idx = lambda d: int((pd.Timestamp(d) - d0).days)                              # noqa: E731
    rc = dict(uri=spec.roots[0]["uri"], start=spec.start, end=spec.data_end, buckets=cfg["buckets"],
              lookback=int(cfg["lookback"]), stride=int(cfg["stride"]), val_stride=int(cfg["val_stride"]),
              stride_new=int(cfg["stride_new"]), year_ago=bool(cfg["year_ago"]),
              min_input_days=int(cfg["min_input_days"]), min_recent_days=int(cfg["min_recent_days"]),
              min_target_days=int(cfg["min_target_days"]), coarse_inputs=cfg["coarse_inputs"],
              interval_labels=bool(cfg["interval_labels"]), regime_idx=idx(cfg["regime_cutoff"]),
              val_households=float(cfg["val_households"]), exclude_ranges=cfg["exclude_ranges"],
              train_hi=idx(spec.data_end) - 1, val_lo=idx(spec.val_start), seed=int(tc["seed"]))
    fs, root = eio.open_root(rc["uri"])
    codes = eio.finished_districts(fs, root, set(cfg["coatos"]) if cfg.get("coatos") else None)
    workers = int(cfg.get("workers") or spec.workers)
    print(f"[tide] reading {len(codes)} districts ({workers} workers), customer buckets {cfg['buckets']}, "
          f"{spec.start} .. {spec.data_end}; coarse inputs: {cfg['coarse_inputs']}, interval labels: "
          f"{cfg['interval_labels']}", flush=True)
    t0 = time.time()
    parts, stats = [], {}
    with cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(td.district_worker, c, rc): c for c in codes}
        for k, fu in enumerate(cf.as_completed(futs), 1):
            r = fu.result()
            parts.append(r)
            for kk, v in (r.get("stats") or {}).items():
                stats[kk] = stats.get(kk, 0) + v
            if k % 20 == 0 or k == len(codes) or k <= 2:
                print(f"[tide] {k}/{len(codes)} districts read, {(time.time() - t0) / 60:.1f} min", flush=True)
    st, tr, va, norms, coatos = assemble(parts, spec, cfg)
    del parts
    print(f"[tide] windows: train {len(tr[0]):,} ({stats.get('old_train', 0):,} issued before {cfg['regime_cutoff']}), "
          f"validation {len(va[0]):,} | intervals: {stats.get('est_runs', 0):,} estimated runs, "
          f"{stats.get('lg_intervals', 0):,} long gaps ({stats.get('lg_dropped', 0):,} long-gap runs dropped)", flush=True)
    if not len(tr[0]):
        raise SystemExit("[tide] no training windows")
    dev = tc["device"]
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)
    torch.manual_seed(int(tc["seed"]))
    rng = np.random.default_rng(int(tc["seed"]))
    S = _to_device(st, device)
    del st
    gcfg = dict(rc, regime_flag=bool(cfg["regime_flag"]), min_scale=float(cfg["min_scale"]),
                regime_cutoff=str(cfg["regime_cutoff"]))
    L, H = int(cfg["lookback"]), td.H
    net = TiDE(L, H, td.N_COV, td.N_STAT, len(coatos), cfg["model"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    trh, tro = torch.from_numpy(tr[0]).to(device), torch.from_numpy(tr[1]).to(device)
    if len(va[0]) > int(tc["val_windows"]):
        pick = rng.choice(len(va[0]), int(tc["val_windows"]), replace=False)
        va = (va[0][pick], va[1][pick])
    vah, vao = torch.from_numpy(va[0]).to(device), torch.from_numpy(va[1]).to(device)
    B = int(tc["batch"])
    amp = bool(tc["amp"]) and device.type == "cuda"
    use_val = len(va[0]) >= int(tc["min_val_windows"])
    epochs = int(tc["epochs"]) if use_val else int(tc["fallback_epochs"])
    if not use_val:
        print(f"[tide] WARNING: only {len(va[0])} validation windows (< train.min_val_windows): no early stopping, "
              f"{epochs} epochs")
    best, best_state, bad, curves = np.inf, None, 0, []
    per_epoch = min(int(tc["windows_per_epoch"]), len(tr[0]))
    for ep in range(epochs):
        net.train()
        te = time.time()
        perm = torch.randperm(len(trh), device=device)[:per_epoch]
        tot, nb, rows = 0.0, 0, dict(daily=0.0, interval=0.0)
        for i in range(0, per_epoch, B):
            j = perm[i:i + B]
            b = td.gather(S, trh[j], tro[j], gcfg, torch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                pred = net(b).float()
            loss, cnt = window_loss(pred, b, tc, cfg["old_weight"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
            for k in rows:
                rows[k] += cnt[k]
        v = evaluate(net, S, vah, vao, gcfg, tc, cfg, device, amp) if len(vah) else {}
        curves.append(dict(epoch=ep + 1, train_loss=tot / max(nb, 1), seconds=time.time() - te,
                           lr=opt.param_groups[0]["lr"], **rows, **v))
        print(f"[tide] epoch {ep + 1}: train {tot / max(nb, 1):.4f} | " +
              (f"val {v['val_loss']:.4f}, household WAPE {v['val_wape']:.3f}, bias {v['val_bias']:+.3f}, "
               f"interval WAPE {v['val_interval_wape']:.3f} | " if v else "") +
              f"rows daily {rows['daily']:,.0f} interval {rows['interval']:,.0f} | {time.time() - te:.0f} s", flush=True)
        if use_val:
            sched.step(v["val_loss"])
            if v["val_loss"] < best - 1e-5:
                best, bad = v["val_loss"], 0
                best_state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= int(tc["patience"]):
                    print(f"[tide] early stop: best validation loss {best:.4f} at epoch {ep + 1 - bad}")
                    break
    if best_state is None:
        best_state = {k: t.detach().cpu() for k, t in net.state_dict().items()}
    torch.save(best_state, os.path.join(artifacts_dir, "tide_weights.pt"))
    np.savez(os.path.join(artifacts_dir, "tide_normals.npz"), **{f"c{c}": v for c, v in norms.items()})
    info = dict(config=cfg, gather=gcfg, coatos=coatos, n_stat=td.N_STAT, curves=curves, stats=stats,
                counts=dict(train=int(len(tr[0])), val=int(len(va[0])), households=int(S["val"].shape[0])),
                best_epoch=int(np.argmin([c.get("val_loss", np.inf) for c in curves]) + 1) if use_val else len(curves),
                fit_minutes=(time.time() - t0) / 60)
    json.dump(info, open(os.path.join(artifacts_dir, "tide_info.json"), "w"), indent=2, default=str)
    print(f"[tide] saved {artifacts_dir} ({info['fit_minutes']:.1f} min)")


@torch.no_grad()
def evaluate(net, S, vah, vao, gcfg, tc, cfg, device, amp):
    net.eval()
    tot, n, ae, a, e, iae, ia = 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0
    B = int(tc["batch"])
    for i in range(0, len(vah), B):
        b = td.gather(S, vah[i:i + B], vao[i:i + B], gcfg, torch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pred = net(b).float()
        loss, _ = window_loss(pred, b, tc, cfg["old_weight"])
        tot += float(loss)
        n += 1
        s = b["s"][:, None]
        P, Y = pred * s, b["yt"] * s
        ae += float(((P - Y).abs() * b["mt"]).sum())
        e += float(((P - Y) * b["mt"]).sum())
        a += float((Y * b["mt"]).sum())
        C = torch.cat([torch.zeros_like(P[:, :1]), torch.cumsum(P, 1)], 1)
        pos = torch.arange(P.shape[1], device=device)[None, :]
        si = C[:, 1:] - torch.gather(C, 1, (pos - b["il"] + 1).clamp(min=0))
        iy = b["isum"] * s
        iae += float(((si - iy).abs() * b["im"]).sum())
        ia += float((iy * b["im"]).sum())
    return dict(val_loss=tot / max(n, 1), val_wape=ae / max(a, 1e-9), val_bias=e / max(a, 1e-9),
                val_interval_wape=iae / ia if ia > 0 else float("nan"))


# ----------------------------------------------------------------------------- serving
class Predictor:
    def __init__(self, artifacts_dir, cfg_in=None):
        info = json.load(open(os.path.join(artifacts_dir, "tide_info.json")))
        self.cfg = merged(info["config"])
        if cfg_in and cfg_in.get("predict"):
            self.cfg["predict"].update(cfg_in["predict"])
        self.gcfg = info["gather"]
        self.coato_idx = {c: i + 1 for i, c in enumerate(info["coatos"])}
        nz = np.load(os.path.join(artifacts_dir, "tide_normals.npz"))
        self.norms = {k[1:]: nz[k] for k in nz.files}
        pc = self.cfg["predict"]
        self.device = torch.device(pc["device"] if pc["device"] != "auto" else
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        if pc.get("threads"):
            torch.set_num_threads(int(pc["threads"]))
        self.net = TiDE(int(self.cfg["lookback"]), td.H, td.N_COV, int(info["n_stat"]), len(info["coatos"]),
                        self.cfg["model"])
        self.net.load_state_dict(torch.load(os.path.join(artifacts_dir, "tide_weights.pt"), map_location="cpu"))
        self.net.to(self.device).eval()

    @torch.no_grad()
    def predict(self, panel):
        code = str(panel.coato)
        norm = self.norms.get(code)
        if norm is None:                                   # a district not in training: normals from its own history
            norm = td.normals(panel.weather, panel.dates)
        st, h, o, c = td.store_from_panel(panel, norm, self.coato_idx.get(code, 0), self.gcfg)
        out = np.zeros((panel.n, td.H))
        B = int(self.cfg["predict"]["batch"])
        S = {k: torch.from_numpy(np.ascontiguousarray(v)).to(self.device) for k, v in st.items()}
        for i in range(0, panel.n, B):
            hh = torch.from_numpy(h[i:i + B]).to(self.device)
            oo = torch.from_numpy(o[i:i + B]).to(self.device)
            b = td.gather(S, hh, oo, c, torch, self.device)
            p = self.net(b).float() * b["s"][:, None]
            out[i:i + B] = p.clamp(min=0).cpu().numpy()
        return out
