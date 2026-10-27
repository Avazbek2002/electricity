#!/usr/bin/env python
"""Full-year line chart of node forecasts against the actual node target, one horizon at a time.

    python scripts/analysis/node_year_chart.py --role dev \
        --forecasts ~/work/eval/runs/<..._xgboost_cal_edd_nodes-dev>/forecasts/xgboost_cal_edd \
                    ~/work/eval/runs/<..._nlinear_x_nodes-dev>/forecasts/nlinear_x \
                    ~/work/eval/node_blend_dev/forecasts/blend_cv
    # labels (LABEL=folder) tell several blends apart (each blend folder is called blend_cv / blend):
    python scripts/analysis/node_year_chart.py --role dev --forecasts \
        blend_5pct=~/work/eval/node_blend_dev/forecasts/blend_cv blend_cal=~/work/eval/node_blend_dev_cal/forecasts/blend_cv

For a horizon h, the point on target day d is the forecast issued on d - h (so every line is a forecast made h days
ahead). B0 is added. Writes <out>/year_chart.html (self-contained: pick the node, the horizon and the level; the legend
shows each model's sysMAPE on what is plotted) and year_total_<level>.png (the sum of all nodes of the role, one panel
per --horizons, default h1 / h7 / h14 / h30 / h60). Node aggregates only (nodes of >= 50 households), no customer keys.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
import harness  # noqa: E402,F401
from harness import nodeset as ns  # noqa: E402

TOTAL = "ALL (sum of the role's nodes)"
PALETTE = ["#2a6fdb", "#e8590c", "#2f9e44", "#ae3ec9", "#f59f00", "#1098ad"]


def series(rows, names, horizons):
    """-> {level: {node: {"d": dates, "y": actual, "h<k>": {model: values}}}} on the common rows."""
    out = {}
    for lv, q in rows.groupby("level"):
        out[lv] = {}
        groups = [(TOTAL, q)] + list(q.groupby("node"))
        for node, g in groups:
            g = g[g["horizon"].isin(horizons)]
            dates = np.sort(g["date"].unique())
            idx = pd.DatetimeIndex(dates)
            d = {"d": [x.strftime("%Y-%m-%d") for x in idx]}
            ys = {}
            for h in horizons:
                gh = g[g["horizon"] == h].groupby("date")[["y"] + names].sum(min_count=1).reindex(idx)
                d[f"h{h}"] = {n: _r(gh[n]) for n in names}
                ys[h] = gh["y"]
            # the actual is the same for every horizon on single nodes; for the total it depends on which nodes have
            # a row, so it is kept per horizon
            d["y"] = {f"h{h}": _r(ys[h]) for h in horizons}
            d["hh"] = int(g["households"].iloc[0]) if node != TOTAL and "households" in g else None
            out[lv][node] = d
    return out


def _r(s):
    return [None if not np.isfinite(v) else round(float(v), 1) for v in s.to_numpy(float)]


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Node forecasts, full year</title>
<script>%(plotly)s</script>
<style>body{font:14px system-ui,sans-serif;margin:16px;background:#fff;color:#222}select{font-size:14px;margin-right:12px}
#c{width:100%%;height:640px}.note{color:#666;font-size:12px;margin-top:6px}</style></head><body>
<h3 style="margin:0 0 8px">Node forecasts vs actual, %(role)s, %(start)s .. %(end)s%(lp)s</h3>
<label>Level <select id="lv"></select></label><label>Node <select id="nd"></select></label>
<label>Horizon <select id="hz"></select></label>
<div id="c"></div>
<div class="note">Point on day d = forecast issued on d - h. Actual = node target (T2/T3 rules of the spec). Legend: sysMAPE
over the plotted days. Grey bands: Saturdays and Sundays.</div>
<script>
const D=%(data)s, H=%(horizons)s, M=%(models)s, C=%(colors)s;
const lv=document.getElementById('lv'), nd=document.getElementById('nd'), hz=document.getElementById('hz');
for(const k of Object.keys(D)) lv.add(new Option(k,k));
for(const h of H) hz.add(new Option('h'+h,'h'+h));
function nodes(){nd.innerHTML='';for(const k of Object.keys(D[lv.value])){const hh=D[lv.value][k].hh;
  nd.add(new Option(hh?k+' ('+hh+' households)':k,k));}}
function smape(f,y){let a=0,s=0;for(let i=0;i<y.length;i++){if(y[i]==null||f[i]==null)continue;a+=Math.abs(f[i]-y[i]);s+=y[i];}
  return s>0?(100*a/s).toFixed(2)+'%%':'-';}
function draw(){const s=D[lv.value][nd.value], h=hz.value, y=s.y[h];
  const tr=[{x:s.d,y:y,name:'actual',mode:'lines',line:{color:'#222',width:2}}];
  M.forEach((m,i)=>tr.push({x:s.d,y:s[h][m],name:m+' ('+smape(s[h][m],y)+')',mode:'lines',
    line:{color:C[i%%C.length],width:1.4,dash:m==='B0'?'dot':'solid'}}));
  const sh=[];for(const d of s.d){const w=new Date(d+'T00:00:00Z').getUTCDay();if(w===0||w===6)
    sh.push({type:'rect',xref:'x',yref:'paper',x0:d+' 00:00',x1:d+' 23:59',y0:0,y1:1,fillcolor:'#000',opacity:0.05,line:{width:0}});}
  Plotly.react('c',tr,{margin:{t:30,r:20,l:60,b:40},hovermode:'x unified',yaxis:{title:'kWh per day'},
    legend:{orientation:'h',y:1.08},shapes:sh,title:{text:nd.value+' | '+h,font:{size:13}}},{responsive:true});}
lv.onchange=()=>{nodes();draw()}; nd.onchange=draw; hz.onchange=draw; nodes(); draw();
</script></body></html>"""


def png_total(data, names, out, horizons=(1, 7, 14, 30, 60)):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[charts] no matplotlib ({e}): PNG skipped")
        return
    for lv, nodes in data.items():
        s = nodes[TOTAL]
        hs = [f"h{h}" for h in horizons if f"h{h}" in s]
        fig, axes = plt.subplots(len(hs), 1, figsize=(14, 3.2 * len(hs)), sharex=True)
        axes = np.atleast_1d(axes)
        x = pd.to_datetime(s["d"])
        for ax, h in zip(axes, hs):
            y = np.array([np.nan if v is None else v for v in s["y"][h]])
            ax.plot(x, y / 1e3, color="#222", lw=1.8, label="actual")
            for i, m in enumerate(names):
                f = np.array([np.nan if v is None else v for v in s[h][m]])
                ok = np.isfinite(f) & np.isfinite(y)
                e = np.abs(f[ok] - y[ok]).sum() / y[ok].sum() * 100
                ax.plot(x, f / 1e3, color=PALETTE[i % len(PALETTE)], lw=1.1, ls=":" if m == "B0" else "-",
                        label=f"{m} ({e:.2f}%)")
            ax.set_title(f"{lv}: sum of all nodes, {h}", fontsize=10, loc="left")
            ax.set_ylabel("MWh per day")
            ax.legend(fontsize=8, ncol=len(names) + 1, loc="upper right")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(out, f"year_total_{lv}.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        print(f"[charts] {p}")


def _not_leak_proof(fd):
    """'' if the forecast folder is leak-proof, else the reason."""
    mp = os.path.join(fd, "forecast_meta.json")
    if not os.path.exists(mp):
        return "no forecast_meta.json"
    m = json.load(open(mp))
    if "components" in m:                                     # a blend (scripts/node_blend.py)
        return "" if m.get("leak_proof") else (f"blend '{m.get('model')}' is not leak-proof (use blend_exp from runs with "
                                              f"safe node scaling)")
    sc = m.get("scaling") or {}
    if int(sc.get("safe_lag", 0)) < ns.EST_SAFE_LAG:
        return (f"node scaling window ends on o - 1 (safe_lag {sc.get('safe_lag', 0)}): rerun the node run with the "
                f"current harness (window ends on o - 1 - {ns.EST_SAFE_LAG})")
    st = m.get("substitution_test")
    if isinstance(st, dict) and not st.get("passed", False):
        return "substitution test not passed"
    return ""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forecasts", nargs="+", required=True, help="node forecast folders (node.csv.gz), optionally LABEL=folder; default label = folder name")
    p.add_argument("--role", default="dev", choices=["dev", "test"])
    p.add_argument("--spec", default=os.path.join(REPO, "configs", "test_set_substations_2025.yaml"))
    p.add_argument("--test-dir", default=None)
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 7, 14, 30, 60])
    p.add_argument("--no-b0", action="store_true")
    p.add_argument("--year", type=int, default=None, help="keep target days of this year only (e.g. 2025)")
    p.add_argument("--leak-proof", action="store_true",
                   help="refuse forecasts that are not leak-proof: model runs need node scaling with safe_lag >= "
                        "nodeset.EST_SAFE_LAG, blends must be blend_exp built from such runs")
    p.add_argument("--out", default="~/work/analysis/year_chart")
    a = p.parse_args(argv)
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    spec = ns.load_spec(a.spec)
    test_dir = os.path.expanduser(a.test_dir or os.path.join("~/work/test_set", spec["version"]))
    targets, nodes = ns.load_targets(test_dir), ns.load_nodes(test_dir)
    models = {}
    for fd in a.forecasts:
        label, _, path = fd.rpartition("=") if "=" in fd else ("", "", fd)
        fd = os.path.expanduser(path.rstrip("/"))
        name = label or os.path.basename(fd)
        if name in models:
            name = f"{name}@{os.path.basename(os.path.dirname(os.path.dirname(fd)))[:15]}"
        if a.leak_proof:
            why = _not_leak_proof(fd)
            if why:
                raise SystemExit(f"[leak-proof] {fd}: {why}")
        models[name] = ns.read_forecasts(fd)
    if not a.no_b0:
        want = set(nodes.loc[nodes["role"] == a.role, "node"])
        b0 = ns.b0_forecasts(targets, spec, nodes=want, issue=ns.issue_dates(spec, a.role))
        models[ns.REFERENCE] = b0[["node", "origin", "horizon", "forecast"]]
    names = list(models)
    rows, info = ns.common_rows(models, targets, spec, a.role, nodes)
    if a.year:
        rows = rows[rows["date"].dt.year == a.year]
    if not len(rows):
        raise SystemExit(f"no common rows (missing forecasts {info['missing']})")
    print(f"[rows] {len(rows):,} node-days x horizons on {rows['node'].nunique()} nodes, "
          f"{rows['date'].min():%Y-%m-%d} .. {rows['date'].max():%Y-%m-%d}")
    data = series(rows, names, a.horizons)
    from plotly.offline import get_plotlyjs
    html = HTML % dict(lp=" (leak-proof)" if a.leak_proof else "", plotly=get_plotlyjs(), data=json.dumps(data, separators=(",", ":")), horizons=json.dumps(a.horizons),
                       models=json.dumps(names), colors=json.dumps(PALETTE), role=a.role,
                       start=f"{rows['date'].min():%Y-%m-%d}", end=f"{rows['date'].max():%Y-%m-%d}")
    p_html = os.path.join(out, "year_chart.html")
    open(p_html, "w").write(html)
    print(f"[charts] {p_html} ({os.path.getsize(p_html) / 1e6:.1f} MB)")
    png_total(data, names, out, a.horizons)
    return out


if __name__ == "__main__":
    main()
