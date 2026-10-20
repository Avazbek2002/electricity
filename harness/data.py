"""Reading one district for the harness: every root (developer zone, vault) into one customer x day matrix."""
import os
import shlex

import numpy as np
import pandas as pd

MISSING = -(2 ** 62)
CRED_SUFFIXES = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "S3_ENDPOINT")


def load_env_file(path):
    """`export NAME=value` / `NAME=value` lines -> os.environ (used for the vault key; never printed)."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise SystemExit(f"vault key file {path} not found")
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        k, _, v = line.partition("=")
        if k.strip():
            vals = shlex.split(v) or [""]
            os.environ[k.strip()] = vals[0]


_HIDDEN = {}


def scrub_credentials():
    """Remove every S3 credential from this process's environment (before a model's predict runs). They are kept in
    this module, so the next district in the same worker process can read again (restore_credentials)."""
    for k in list(os.environ):
        if k.endswith(CRED_SUFFIXES):
            _HIDDEN[k] = os.environ.pop(k)


def restore_credentials():
    """Put back the credentials scrub_credentials removed (start of each district: worker processes are reused)."""
    os.environ.update(_HIDDEN)


def read_district(code, roots, start, end, labels):
    """roots: [{"uri", "env_prefix", "start", "end"}] read in order, each only over its own dates.
    -> dict(keys, M (n, T) float32 kWh on kept labels, wt (3, T) t_mean/t_min/t_max, housing (n,), network (n, 3) int64
    transformer/feeder/substation (MISSING if unknown; placeholder transformers -> -1), dates (T,))."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as pads
    import enriched_io as eio
    d0, d1 = pd.Timestamp(start), pd.Timestamp(end)
    T = (d1 - d0).days + 1
    e0 = int((d0 - pd.Timestamp("1970-01-01")).days)
    lab = pa.array(list(labels), type=pa.string())
    key_of, keys = {}, []
    cap = 4096
    M = np.full((cap, T), np.nan, dtype=np.float32)
    house = np.full(cap, np.nan)
    hday = np.full(cap, -1, dtype=np.int64)
    net = np.full((cap, 3), MISSING, dtype=np.int64)
    wt = np.full((3, T), np.nan)
    for rt in roots:
        fs, root = eio.open_root(rt["uri"], rt.get("env_prefix", ""))
        try:
            dset = eio.district_daily(fs, root, code)
        except (FileNotFoundError, OSError):
            continue
        a, b = max(d0, pd.Timestamp(rt.get("start", start))), min(d1, pd.Timestamp(rt.get("end", end)))
        if a > b:
            continue
        want = ["customer_key", "date", "kwh", "label_state", "housing_type_id", "t_mean", "t_min", "t_max",
                "transformer_id", "feeder_id", "substation_id", "placeholder_transformer"]
        cols = [c for c in want if c in dset.schema.names]
        flt = (pads.field("date") >= a.date()) & (pads.field("date") <= b.date())
        for bt in dset.scanner(columns=cols, filter=flt, batch_size=262_144, use_threads=True).to_batches():
            if not bt.num_rows:
                continue
            de = pc.dictionary_encode(bt.column("customer_key"))
            idx = np.empty(len(de.dictionary), dtype=np.int64)
            for j, s in enumerate(de.dictionary.to_pylist()):
                v = key_of.get(s)
                if v is None:
                    v = key_of[s] = len(keys)
                    keys.append(s)
                idx[j] = v
            c = idx[de.indices.to_numpy(zero_copy_only=False)]
            if len(keys) > cap:
                new_cap = max(len(keys), 2 * cap)
                M = np.vstack([M, np.full((new_cap - cap, T), np.nan, dtype=np.float32)])
                house = np.r_[house, np.full(new_cap - cap, np.nan)]
                hday = np.r_[hday, np.full(new_cap - cap, -1, dtype=np.int64)]
                net = np.vstack([net, np.full((new_cap - cap, 3), MISSING, dtype=np.int64)])
                cap = new_cap
            day = pc.cast(pc.cast(bt.column("date"), pa.date32()), pa.int32()).to_numpy(zero_copy_only=False).astype(np.int64) - e0
            for i, col in enumerate(("t_mean", "t_min", "t_max")):
                if col in cols:
                    np.fmax.at(wt[i], day, np.asarray(bt.column(col).to_numpy(zero_copy_only=False), dtype=float))
            if "housing_type_id" in cols:
                hcol = bt.column("housing_type_id")
                try:
                    hv = pc.cast(hcol, pa.float64()).to_numpy(zero_copy_only=False)
                except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                    hv = pd.to_numeric(pd.Series(hcol.to_pylist()), errors="coerce").to_numpy(float)
                m = np.isfinite(hv) & (day >= hday[c])
                hday[c[m]], house[c[m]] = day[m], hv[m]
            for j, col in enumerate(("transformer_id", "feeder_id", "substation_id")):
                if col in cols:
                    v = pc.fill_null(pc.cast(bt.column(col), pa.int64()), MISSING).to_numpy(zero_copy_only=False)
                    m = (net[c, j] == MISSING) & (v != MISSING)
                    net[c[m], j] = v[m]
            if "placeholder_transformer" in cols:
                ph = pc.fill_null(bt.column("placeholder_transformer"), False).to_numpy(zero_copy_only=False)
                net[c[ph], 0] = -1
            kv = bt.column("kwh")
            ok = pc.and_(pc.fill_null(pc.is_in(bt.column("label_state"), value_set=lab), False),
                         pc.is_valid(kv)).to_numpy(zero_copy_only=False)
            M[c[ok], day[ok]] = kv.to_numpy(zero_copy_only=False)[ok].astype(np.float32)
    n = len(keys)
    return dict(keys=np.asarray(keys, dtype=object), M=M[:n], wt=wt, housing=house[:n], network=net[:n],
                dates=pd.date_range(d0, d1, freq="D"))
