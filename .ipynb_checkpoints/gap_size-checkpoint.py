import sys, numpy as np, pandas as pd
path = sys.argv[1] if len(sys.argv) > 1 else "transformer_daily.csv"
MINCOV, MINLEN = 0.8, 4                      # a gap = coverage below 80% for at least 4 days in a row
d = pd.read_csv(path, parse_dates=["calendar_day"], dtype={"transformer_id": str})
rep = d.pivot_table(index="transformer_id", columns="calendar_day", values="n_reporting", aggfunc="sum")
rep = rep.reindex(columns=pd.date_range(rep.columns.min(), rep.columns.max())).fillna(0)   # no row = nobody reported
size = d.groupby("transformer_id")["n_ever"].max().reindex(rep.index)
cov = rep.div(size, axis=0)
sys_cov = rep.sum() / size.sum()                                                          # all transformers together
sys_bad = sys_cov < 0.9 * sys_cov.rolling(61, center=True, min_periods=15).median()
bad = (cov < MINCOV).loc[:, ~sys_bad.values]                                              # drop system-wide outage days
run = bad.astype(int).T.rolling(MINLEN).min().T.fillna(0).to_numpy().astype(int)          # 1 on day 4+ of a bad run
episodes = (np.diff(np.c_[np.zeros(len(run)), run], axis=1) == 1).sum(1)                  # runs of >= MINLEN days
years = bad.shape[1] / 365.25
t = pd.DataFrame({"n_ever": size, "episodes": episodes, "bad_day_share": bad.mean(axis=1)}, index=rep.index)
t = t[t["n_ever"] >= 5]
t["band"] = pd.cut(t["n_ever"], [0, 20, 50, 100, 200, 300, 10 ** 6], right=False,
                   labels=["5-19", "20-49", "50-99", "100-199", "200-299", "300+"])
out = t.groupby("band", observed=True).agg(transformers=("n_ever", "size"),
        episodes_per_transformer_year=("episodes", lambda s: s.sum() / max(len(s), 1) / years),
        share_with_any_gap=("episodes", lambda s: (s > 0).mean()),
        median_bad_day_share=("bad_day_share", "median"))
print(f"gaps = coverage < {MINCOV:.0%} for >= {MINLEN} days in a row, system-wide outage days removed ({int(sys_bad.sum())} days)")
print(out.round(3).to_string())
print(f"\nSpearman correlation of size with episodes per transformer: {t['n_ever'].corr(t['episodes'], method='spearman'):+.2f}")