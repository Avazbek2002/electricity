# tide_obs: tide_agg with observed-only inputs (diagnostic)

Same as tide_agg (interval labels, MSE, year-ago inputs, daily new-regime windows) except `coarse_inputs: none` and
`min_input_days: 40` for training windows.

Why: in run 2 (49bcd7f) both TiDE variants over-forecast every substation by ~30 % for issue dates Jan-Mar 2025 and
+18 % in April, then under-forecast October by ~19 %. Hypothesis: interval days as inputs exist only before 2024-08, so
the coarse flag / coarse share tell the model which regime a window is from; in the 2025-01 and 2025-04 refits the
new-regime windows all target the autumn heating rise, so "no coarse inputs" was learned as "consumption rises".
Serving windows never have coarse inputs. Old windows also got interval days in their year-ago inputs, which a served
window never has. With observed-only inputs an old window looks like a served one (apart from whole-kWh rounding).

Read: the Jan-Apr rows of the by-month table. If the +30 % is gone, the confound was the cause.
