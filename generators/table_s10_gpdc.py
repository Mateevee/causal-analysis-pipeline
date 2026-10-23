"""
Generates the body of Table S21 (GPDC check) of the supplement, at both windows,
from the main run (pipeline, PCMCI and PCMCI+ with ParCorr) and the GPDC run
(PCMCI-GPDC and PCMCI+-GPDC), with the current configuration (main_gpdc.py).

Output: Table_S21_GPDC.md (pasted into the "Table S21" section of the supplement).
"""
import os
import re
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import origins as O  # noqa: E402

O.require("MAIN_RUN")
GPDC = O.require("GPDC_RUN")   # in the paper, Results_GPDC/Run_20261002_210059
OUT = O.out("Table_S21_GPDC.md")
EPS = 1e-9
FAMILY = [("logistic_maps", "Logistic maps"), ("lorenz96", "Lorenz-96"),
          ("tigramite_quadratic", "tigramite quadratic"), ("tigramite_saturation", "tigramite saturation"),
          ("tigramite_threshold", "tigramite threshold"), ("tigramite_large_mixed", "tigramite large-mixed"),
          ("tigramite_baseline_linear", "tigramite linear-baseline"), ("tigramite_climate_walker", "Walker Circulation"),
          ("pems_bay", "PeMS-BAY"), ("sachs", "Sachs")]


def family(sc):
    s = O.canonical(sc).replace("benchmark_", "").replace("real_", "")
    for k, v in FAMILY:
        if s.startswith(k):
            return v


def f1_table(suf):
    rows = {}
    for sc in sorted(os.listdir(GPDC)):
        g = pd.read_csv(os.path.join(GPDC, sc, "validation_summary.csv")).set_index("model")
        m = pd.read_csv(os.path.join(O.find(sc), "validation_summary.csv")).set_index("model")
        rows[sc] = {"Pipeline": m.loc["patch" + suf, "f1"], "PCMCI": m.loc["pcmci_tigramite" + suf, "f1"],
                    "PCMCI+": m.loc["pcmciplus" + suf, "f1"], "PCMCI-GPDC": g.loc["pcmci_gpdc" + suf, "f1"],
                    "PCMCI+-GPDC": g.loc["pcmciplus_gpdc" + suf, "f1"]}
    return pd.DataFrame(rows).T


def cmp(a, b):
    return "W" if a > b + EPS else ("L" if a < b - EPS else "T")


def section(suf, label, per_realization):
    d = f1_table(suf)
    out = [f"### {label}\n", "*Head-to-head counts, per realization, from the pipeline's side:*\n",
           "| Pipeline vs | Pipeline higher | Tied | Pipeline lower |", "|---|---|---|---|"]
    for m in ["PCMCI", "PCMCI+", "PCMCI-GPDC", "PCMCI+-GPDC"]:
        c = Counter(cmp(a, b) for a, b in zip(d.Pipeline, d[m]))
        name = m + (" (ParCorr)" if "GPDC" not in m else "")
        out.append(f"| {name} | {c['W']} | {c['T']} | {c['L']} |")
    out += ["", "*Effect of switching each method from ParCorr to GPDC, per realization:*\n",
            "| Method | F1 rises | F1 unchanged | F1 falls | Mean F1, ParCorr → GPDC |", "|---|---|---|---|---|"]
    for a, b in [("PCMCI", "PCMCI-GPDC"), ("PCMCI+", "PCMCI+-GPDC")]:
        c = Counter(cmp(y, x) for x, y in zip(d[a], d[b]))
        out.append(f"| {a} | {c['W']} | {c['T']} | {c['L']} | {d[a].mean():.3f} → {d[b].mean():.3f} |")
    out += ["", f"The pipeline's mean F1 over the same thirty-six realizations is {d.Pipeline.mean():.3f}.\n",
            "*Standing of the pipeline against the PCMCI family (best of its configurations on each realization), before and after the two GPDC configurations are added (wins / ties / losses):*\n",
            "| Family | n | ParCorr only | ParCorr and GPDC |", "|---|---|---|---|"]
    d["fam"] = [family(s) for s in d.index]
    d["only"] = [cmp(p, max(a, b)) for p, a, b in zip(d.Pipeline, d.PCMCI, d["PCMCI+"])]
    d["all"] = [cmp(p, max(a, b, c, e)) for p, a, b, c, e in zip(d.Pipeline, d.PCMCI, d["PCMCI+"], d["PCMCI-GPDC"], d["PCMCI+-GPDC"])]
    tot = [Counter(), Counter()]
    for _, fam in FAMILY:
        g = d[d.fam == fam]
        c1, c2 = Counter(g.only), Counter(g["all"])
        tot[0].update(c1); tot[1].update(c2)
        out.append(f"| {fam} | {len(g)} | {c1['W']} / {c1['T']} / {c1['L']} | {c2['W']} / {c2['T']} / {c2['L']} |")
    out.append(f"| **Total** | **{len(d)}** | **{tot[0]['W']} / {tot[0]['T']} / {tot[0]['L']}** | "
               f"**{tot[1]['W']} / {tot[1]['T']} / {tot[1]['L']}** |")
    ch = d[d.only != d["all"]]
    names = {"W": "lead", "T": "tie", "L": "loss"}
    if len(ch):
        items = []
        for s, r in ch.iterrows():
            best = max(("PCMCI-GPDC", r["PCMCI-GPDC"]), ("PCMCI+-GPDC", r["PCMCI+-GPDC"]), key=lambda x: x[1])
            items.append(f"{s.replace('benchmark_', '').replace('real_', '')} ({names[r.only]} → {names[r['all']]}; "
                         f"{best[0]} {best[1]:.4f} against the pipeline's {r.Pipeline:.4f})")
        out.append("\nThe realizations whose standing changes are " + "; ".join(items) + ".")
    else:
        out.append("\nNo realization changes standing when the GPDC configurations are added.")
    if per_realization:
        out += ["", "*Per-realization F1 (best of each row in bold):*\n",
                "| Realization | Pipeline | PCMCI (ParCorr) | PCMCI+ (ParCorr) | PCMCI-GPDC | PCMCI+-GPDC |",
                "|---|---|---|---|---|---|"]
        cols = ["Pipeline", "PCMCI", "PCMCI+", "PCMCI-GPDC", "PCMCI+-GPDC"]
        for s, r in d.iterrows():
            mx = max(r[c] for c in cols)
            cells = [f"**{r[c]:.3f}**" if abs(r[c] - mx) < EPS and mx > 0 else f"{r[c]:.3f}" for c in cols]
            out.append(f"| {s.replace('benchmark_', '').replace('real_', '')} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


txt = section("_tau10", "S21.1 — τ_max = 10 (primary window)", True) + "\n" + \
      section("", "S21.2 — τ_max = 24 (sensitivity window)", False)
open(OUT, "w", encoding="utf-8").write(O.canonical(txt))
print(txt[:3000])
