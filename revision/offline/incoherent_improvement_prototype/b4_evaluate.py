# Provenance: 4th analysis of 2026-11-15 (section 11 of ACHADOS). Read-only. POST HOC.
# Evaluates the criteria K1, K2 and K12 (b4_criteria.py) as a switch for applying CA, in two domains fixed beforehand:
#   D1: only in the incoherent regime with fallback to Stage 6 (where the guard already flags the output as low confidence);
#   D2: in any regime.
# Outside the domain, or with the criterion off, the current output stays. Thresholds: K1 > 0; K2 >= 0 (fixed beforehand).
# Input: the CSV written by b4_criteria.py. Supplementary Table S33.3, last row.
# Usage: python revision/offline/incoherent_improvement_prototype/b4_evaluate.py res_b4_criterios.csv
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

D = pd.read_csv(sys.argv[1])
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
D["fam"] = D.cenario.str.replace("benchmark_|real_", "", regex=True).str.split("_").str[:2].str.join("_")
D.loc[D.cenario.str.contains("causal_chamber"), "fam"] = "causal_chamber"
D["gain"] = D.f1_CA - D.f1_atual
D["c_K1"] = D.K1 > 0
D["c_K2"] = D.K2 >= 0
D["c_K12"] = D.c_K1 & D.c_K2
print("scenarios with estimable OLS:", len(D), "| per set:", D.conj.value_counts().to_dict())
print("\n== Values of the criteria per family and regime (means), with the gain of CA")
print(D.groupby(["grupo", "fam"]).agg(scen=("n", "size"), gain=("gain", "mean"), K1=("K1", "mean"), K1min=("K1", "min"), K1max=("K1", "max"),
                                      K2=("K2", "mean"), K2min=("K2", "min"), K2max=("K2", "max")).round(3).to_string())
inc = D.grupo == "incoh/stage6"
print("\n== Separation within D1 (incoherent with fallback): does the criterion switch on where CA helps?")
S = D[inc]
for c in ["c_K1", "c_K2", "c_K12"]:
    on = S[S[c]]
    off = S[~S[c]]
    print("  %-5s on in %2d of %2d | mean gain where on %+.3f (%d+/%d-) | gain that would be lost where off %+.3f (%d+/%d-)" % (
        c[2:], len(on), len(S), on.gain.mean() if len(on) else np.nan, (on.gain > 1e-9).sum(), (on.gain < -1e-9).sum(),
        off.gain.mean() if len(off) else np.nan, (off.gain > 1e-9).sum(), (off.gain < -1e-9).sum()))
print(S[["conj", "cenario", "n", "T", "f1_atual", "f1_CA", "f1_varlingam", "K1", "K1_frac", "K2", "K2_frac"]]
      .assign(cenario=lambda x: x.cenario.str[-30:]).sort_values("K1").round(3).to_string(index=False))


def summary(domain, crit):
    out = []
    for set_name, S in D.groupby("conj", sort=False):
        applies = S[crit] & (domain[S.index] if domain is not None else True)
        f = np.where(applies, S.f1_CA, S.f1_atual)
        d = f - S.f1_atual
        p = wilcoxon(f, S.f1_atual).pvalue if (np.abs(d) > 1e-12).any() else 1.0
        out.append("%s: applies %d/%d, F1 %.4f -> %.4f (%d+/%d-; p=%.3g)" % (set_name, int(applies.sum()), len(S), S.f1_atual.mean(), f.mean(),
                                                                             (d > 1e-9).sum(), (d < -1e-9).sum(), p))
    return " | ".join(out)


print("\n== Effect of the rule (only on the scenarios with estimable OLS; the others stay the same)")
for domain_name, dom in [("D1 incoherent", inc), ("D2 all", None)]:
    for c in ["c_K1", "c_K2", "c_K12"]:
        print("  %-13s %-4s %s" % (domain_name, c[2:], summary(dom, c)))
    D["c_always"] = True
    print("  %-13s %-4s %s" % (domain_name, "none", summary(dom, "c_always")))
print("\n== D2: where each criterion switches on outside the incoherent regime (family: on/scenarios, mean gain where on)")
for c in ["c_K1", "c_K2", "c_K12"]:
    S = D[~inc]
    parts = []
    for fam, g in S.groupby("fam"):
        parts.append("%s %d/%d %+.3f" % (fam, g[c].sum(), len(g), g.gain[g[c]].mean() if g[c].any() else 0.0))
    print(" ", c[2:], "|", "; ".join(parts))
