"""
Item 3, complement -- coverage of the pipeline's OWN CIs (the controls and
learner it chose) on new realizations: the 50 tigramite scenarios of the
new-seeds run (Results_Seeds/Run_20261014_222930, seeds 1001-1010, frozen
configuration), against the same reference as Table_S9_Tigramite_Controls.md.
Nothing is re-estimated: reads the CIs stored in patch_tau10/results.csv.

Usage:
    python generators/coverage_new_seeds.py   (origin SEEDS_RUN, the run of main_seed_replication.py)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dml_controls_tigramite as D  # noqa: E402

RUN = D.O.require("SEEDS_RUN")
ref, truth = D.references()
rows = []
n_true = 0
for sc in D.TG._BENCHMARK_SCENARIO_DEFINITIONS:
    for sd in range(1001, 1011):
        base = os.path.join(RUN, f"benchmark_tigramite_{sc}_seed{sd}")
        src = pd.read_csv(os.path.join(base, "source.csv")).set_index("Name")
        res = pd.read_csv(os.path.join(base, "patch_tau10", "results.csv"))
        res = res[res["significant"].astype(str) == "True"]
        for (sc_, s, t), rf in ref.items():
            if sc_ != sc:
                continue
            n_true += 1
            (se_, sp), (te, tp) = s.split("__"), t.split("__")
            m = res[(res.source_entity == se_) & (res.source_property == sp)
                    & (res.target_entity == te) & (res.target_property == tp)]
            if not len(m) or int(m.iloc[0].lag) != rf["lag"]:
                continue        # only the links reported at the true lag, as in panel 7
            r = m.iloc[0]
            refv = rf["b"] * src.loc[se_, sp].std() / src.loc[te, tp].std()
            rows.append(dict(scm=sc, seed=sd, link=f"{s}->{t}", form=D.FORM[rf["form"]],
                             ref=refv, th=r.causal_effect, lo=r.ci_95_lower, hi=r.ci_95_upper,
                             learner=str(r.dml_model_used).split("[")[0]))
d = pd.DataFrame(rows)
d["cov"] = (d.lo <= d.ref) & (d.ref <= d.hi)
s = (d.hi - d.lo) / 3.92
d["z"] = (d.th - d.ref) / s * np.sign(d.ref)
d["sign_ok"] = np.sign(d.th) == np.sign(d.ref)
d["relerr"] = (d.th.abs() - d.ref.abs()) / d.ref.abs()
d["abserr"] = (d.th - d.ref).abs()
print(f"true links: {n_true}; reported at the true lag: {len(d)}")
print(f"coverage {d['cov'].mean():.1%}  mean z {d.z.mean():+.2f}  SD z {d.z.std():.2f}  "
      f"correct sign {d.sign_ok.mean():.1%}  mean rel. error of |theta| {d.relerr.mean():+.1%}  "
      f"MAE {d.abserr.mean():.3f}")
print(d.groupby("form")["cov"].agg(["mean", "count"]).round(3))
ps = d.groupby("seed")["cov"].agg(["sum", "count"])
ps["frac"] = ps["sum"] / ps["count"]
print(ps)
print(d.learner.value_counts())
# bootstrap CI with the seed as the unit (preserves the correlation between SCMs of the same seed)
rng = np.random.default_rng(20261023)
seeds = ps.index.values
bs = []
for _ in range(20000):
    pick = rng.choice(seeds, len(seeds), replace=True)
    bs.append(ps.loc[pick, "sum"].sum() / ps.loc[pick, "count"].sum())
print("95% bootstrap CI by seed:", np.percentile(bs, [2.5, 97.5]).round(3))
d.to_csv(D.O.out("tigramite_coverage_new_seeds.csv"), index=False)
