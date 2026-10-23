#!/usr/bin/env python
"""
Team scorer for the shared test-set release (README.md next to it). Needs only numpy, pandas and pyyaml.

    python scorer/score.py --set standard --template ~/my_model --sample             # rows to fill (5 districts)
    python scorer/score.py --set standard --forecasts ~/my_model                      # score (B0 on the same rows)
    python scorer/score.py --set s25 --forecasts ~/my_model --compare xgb=~/xgb_run   # several models, same rows
    python scorer/score.py --set s25 --track weather --forecasts ~/my_wx_model       # AIFS / FCN3 issue dates only

--set      standard = the guide's split (validation 2024, test 2025); s25 = supplementary (validation 2025-H1, test
           2025-H2). Each has its own spec.yaml; a submission names the one it was made for (forecast_meta.json).
--track    all (default) = every issue date; weather = only the issue dates that have an AIFS / FCN3 reforecast
           (weather/issue_dates.csv). Models with and without weather are compared on the weather track.
Submission folder: forecast_meta.json + district.csv[.gz] with columns node, origin, horizon, forecast
(forecast = the district's T2 in kWh on day origin + horizon, issued on `origin` with data up to origin - 1).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    import testset as ts                                  # the release: scorer/testset.py next to this file
except ImportError:                                       # the repository: harness/testset.py
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))
    from harness import testset as ts

import pandas as pd  # noqa: E402

ROOT_DEFAULT = os.path.normpath(os.path.join(HERE, ".."))   # the release folder (scorer/..)


def weather_dates(root, set_name):
    p = os.path.join(root, "weather", "issue_dates.csv")
    if not os.path.exists(p):
        raise SystemExit(f"no {p}: this release has no weather track")
    w = pd.read_csv(p, parse_dates=["issue_date"])
    w = w[w[f"split_{set_name}"].notna() & w["aifs"] & w["fcn3"]]
    return pd.DatetimeIndex(w["issue_date"])


def restrict_issue_dates(dates):
    """Score only these issue dates, for every model alike (the common panel is built from the split rows)."""
    keep = set(pd.DatetimeIndex(dates))
    orig = ts.split_rows

    def split_rows(spec, split, df):
        d = orig(spec, split, df)
        return d[d["origin"].isin(keep)]
    ts.split_rows = split_rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=ROOT_DEFAULT, help="the release folder (default: the folder above scorer/)")
    p.add_argument("--set", required=True, choices=["standard", "s25"])
    p.add_argument("--track", default="all", choices=["all", "weather"])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--forecasts", help="folder with forecast_meta.json and district.csv[.gz]")
    g.add_argument("--template", help="write the rows a submission contains into this folder")
    p.add_argument("--sample", action="store_true", help="template: the guide's 5 districts only")
    p.add_argument("--districts", default=None, help="template: comma list of districts")
    p.add_argument("--target", default="T2", choices=["T2", "T3"], help="T3 = experimental (reported separately)")
    p.add_argument("--compare", action="append", default=[], help="name=forecast_dir of another model, same rows")
    p.add_argument("--out-dir", default=None, help="where to write the scores (default <forecasts>/scores_<set>_<track>)")
    a = p.parse_args()
    root = os.path.expanduser(a.root)
    set_dir = os.path.join(root, a.set)
    spec = ts.load_spec(os.path.join(set_dir, "spec.yaml"))
    ts.check_same_spec(spec, ts.read_manifest(set_dir))
    dates = weather_dates(root, a.set) if a.track == "weather" else None
    if a.template:
        all_d = pd.read_csv(os.path.join(set_dir, "districts.csv"), dtype=str)["district"].tolist()
        dist = spec["sample_districts"] if a.sample else (a.districts.split(",") if a.districts else all_d)
        out = os.path.expanduser(a.template)
        n = ts.write_template(spec, out, dist)
        if dates is not None:
            t = pd.read_csv(os.path.join(out, "district.csv.gz"), dtype={"node": str}, parse_dates=["origin"])
            t = t[t["origin"].isin(set(dates))]
            t["origin"] = t["origin"].dt.date
            t.to_csv(os.path.join(out, "district.csv.gz"), index=False)
            n = len(t)
        print(f"[template] {out}: {n:,} rows ({len(dist)} districts, {a.set}, track {a.track})")
        print("  fill `forecast` (district T2 in kWh), set `model`, `weather`, `level` in forecast_meta.json, then "
              "score with --forecasts")
        return
    if dates is not None:
        restrict_issue_dates(dates)
        print(f"[score] weather track: {len(dates)} issue dates with an AIFS and FCN3 forecast")
    others = {}
    for c in a.compare:
        name, _, path = c.partition("=")
        others[name] = os.path.expanduser(path)
    fdir = os.path.expanduser(a.forecasts)
    res = ts.score(fdir, set_dir, spec, target=a.target, others=others)
    text = ts.report(res, spec)
    if dates is not None:
        text = "TRACK: weather (AIFS / FCN3 issue dates only)\n" + text
    print(text)
    out = os.path.expanduser(a.out_dir or os.path.join(fdir, f"scores_{a.set}_{a.track}"
                                                       + ("" if a.target == "T2" else f"_{a.target}")))
    ts.save_scores(res, out, text)
    print(f"\n[score] written to {out}")


if __name__ == "__main__":
    main()
