"""
Generates ALL the numbers of the paper from the runs on disk, in a single report
per window (Paper_Numbers_tau24.md, Paper_Numbers_tau10.md), with one JSON per window. No number is
typed by hand: each section of the report corresponds to a passage of the paper,
as listed in the paper's Supplementary Reproducibility Index.

How to use it when new data arrive:
  1. Set the origins in generators/origins.py (MAIN_RUN, HELDOUT_RUN, ...).
  2. python generators/paper_numbers.py
  3. Check section "0. Provenance" (which run each model came from) and
     update the text of the paper accordingly.

Conventions (the same as the paper's, Section 4.4):
  - Unit: ordered pair of entities. F1 recomputed from TP/FP/FN.
  - Aggregate per group = pooled (sum of TP/FP/FN over the group's realizations).
  - Win/tie/loss per group compares pooled F1 over the realizations on which
    BOTH methods ran (status ok).
  - Trivial floor per group: p = sum(n_causal)/sum(n_universe); F1 = 2p/(1+p).
"""
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ----------------------------------------------------------------------------
# CONFIGURATION -- the runs come from origins.py (MAIN_RUN, HELDOUT_RUN and the
# runs of the separate experiments); nothing is read from a fixed path.
# ----------------------------------------------------------------------------
import origins as O  # noqa: E402

OUT_DIR = O.OUT_DIR
# Run with SHD, AUROC and AUPRC per scenario (same configuration as the reported
# one); only the AUPRC of section 7 comes from it. See generators/metrics_shd_auc.py.
# Without METRICS_RUN, section 7 comes out without the AUPRC.
METRICS_CSV = (os.path.join(O.METRICS_RUN, "auroc_auprc_shd_per_scenario.csv")
               if O.METRICS_RUN else None)

# Main run (main.py): pipeline and the five baselines. In the paper,
# Run_20261004_232204 (58 scenarios, two windows: the 51 development ones and
# TEP, PeMS-BAY, Sachs, Walker and PetShop). Baselines with their native
# decision (the method's graph).
PIPELINE_RUN = O.require("MAIN_RUN")
PIPELINE_MODEL = "patch"
PIPELINE_ALT = "patch_alt"   # pipeline at the OTHER window (section 11)

# Sources per window: canonical model -> list of (run, model name in
# validation_summary.csv). Uses the FIRST source in which the model ran with
# status ok; the provenance is printed in section 0 of each report. The
# held-out runs (main_heldout.py; in the paper, Results_Extra/Run_20261008_221315
# for BEELINE, DREAM3, RCAEval and EasyVista and
# Results_Extra/Run_20261009_causal_chamber) come BEFORE the main one, so that
# DREAM3 is read from them.
O.require("HELDOUT_RUN")   # the held-out results are the paper's headline
_HO = O.heldout_runs()
# GPDC (main_gpdc.py; in the paper, Results_GPDC/Run_20261002_210059). Optional.
_RG = O.GPDC_RUN


def _sources(name):
    return [(r, name) for r in _HO] + [(PIPELINE_RUN, name)]


def _gpdc(name):
    return [(_RG, name)] if _RG else []


SOURCES = {
    "24": {
        "patch": _sources("patch"),
        "patch_alt": _sources("patch_tau10"),
        "pcmci_tigramite": _sources("pcmci_tigramite"),
        "pcmciplus": _sources("pcmciplus"),
        "dynotears": _sources("dynotears"),
        "cuts_plus": _sources("cuts_plus"),
        "varlingam": _sources("varlingam"),
        "pcmci_gpdc": _gpdc("pcmci_gpdc"),
        "pcmciplus_gpdc": _gpdc("pcmciplus_gpdc"),
    },
    "10": {
        "patch": _sources("patch_tau10"),
        "patch_alt": _sources("patch"),
        "pcmci_tigramite": _sources("pcmci_tigramite_tau10"),
        "pcmciplus": _sources("pcmciplus_tau10"),
        "dynotears": _sources("dynotears_tau10"),
        "cuts_plus": _sources("cuts_plus_tau10"),
        "varlingam": _sources("varlingam_tau10"),
        "pcmci_gpdc": _gpdc("pcmci_gpdc_tau10"),
        "pcmciplus_gpdc": _gpdc("pcmciplus_gpdc_tau10"),
    },
}
# Folder holding the pipeline's original.csv at each window (ablation, section 9, AUPRC)
PIPELINE_FOLDER = {"24": "patch", "10": "patch_tau10"}
WINDOW = "24"   # redefined in __main__ for each window

BASELINES = [("pcmci_tigramite", "PCMCI"), ("pcmciplus", "PCMCI+"),
             ("dynotears", "DYNOTEARS"), ("cuts_plus", "CUTS+"),
             ("varlingam", "VARLiNGAM")]
GPDC = [("pcmci_gpdc", "PCMCI-GPDC"), ("pcmciplus_gpdc", "PCMCI+-GPDC")]
LABEL = dict([(PIPELINE_MODEL, "Pipeline")] + BASELINES + GPDC)
EPS = 1e-9

ORIGINAL11 = ["logistic_maps_bifurcation", "logistic_maps_chain",
              "logistic_maps_convergence", "logistic_maps_with_isolated",
              "lorenz96_K8", "lorenz96_K10", "tigramite_baseline_linear",
              "tigramite_quadratic", "tigramite_threshold",
              "tigramite_saturation", "tigramite_large_mixed"]
CAUSALTIME = ["causaltime_medical", "causaltime_pm25", "causaltime_traffic"]
REAL4 = ["tep_normal", "alarm_icu", "pems_bay_traffic", "sachs_proteins"]
WALKER = ["tigramite_climate_walker"]
NEW2 = ["netsim_sim3", "petshop"]
# External datasets of the extra runs (ten groups, 32 scenarios)
EXTRA = ["beeline_HSC", "beeline_VSC", "dream3", "rcaeval", "easyvista_antivirus", "easyvista_mom",
         "easyvista_storm", "easyvista_web", "causal_chamber_wt", "causal_chamber_lt"]
# Groups that took no part in the development of the pipeline (Section 4.5):
# five datasets, seven scenarios. NetSim stays in the development suite,
# because the coherence-guard exception was introduced while observing the
# pipeline on it.
HELDOUT = ["tep_normal", "pems_bay_traffic", "sachs_proteins"] + WALKER + ["petshop"] + EXTRA
SYNTH = [f"synthetic_var_{i}" for i in range(1, 6)]
ESTAB18 = ORIGINAL11 + CAUSALTIME + REAL4
PREC19 = ESTAB18 + WALKER
# Every group of the paper (36).
ALL_GROUPS = SYNTH + ESTAB18 + WALKER + NEW2 + EXTRA


def group_of(scen):
    s = re.sub(r"^(benchmark_|real_)", "", O.canonical(scen))
    if s.startswith("petshop_"):
        return "petshop"
    for g in EXTRA:
        if s.startswith(g):
            return g
    return re.sub(r"_(seed|sample|subj)\d+$", "", s)


def family_of(group):
    for k, v in [("synthetic", "Synthetic VAR"), ("logistic", "Logistic maps"),
                 ("lorenz", "Lorenz-96"), ("climate_walker", "Walker"),
                 ("tigramite", "tigramite SCM"), ("causaltime", "CausalTime"),
                 ("netsim", "NetSim"), ("petshop", "PetShop"), ("beeline", "BEELINE"),
                 ("dream3", "DREAM3"), ("rcaeval", "RCAEval"), ("easyvista", "EasyVista"),
                 ("causal_chamber", "Causal Chambers")]:
        if k in group:
            return v
    return "Real-world"


def f1(tp, fp, fn):
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def prf(c):
    tp, fp, fn = c
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, f1(tp, fp, fn)


def run_of(rows, scen, model=None):
    """Run the model (by default the pipeline) came from in this scenario."""
    return rows[scen][model or PIPELINE_MODEL]["run"].rsplit(":", 1)[0]


def core(rows):
    """Only the 58 scenarios of the main run: the ablations and the precision per
    path (sections 8, 8c and 9) were not redone on the EXTRA groups."""
    return {s: mm for s, mm in rows.items() if group_of(s) not in EXTRA}


def cmp(a, b):
    return "W" if a > b + EPS else ("L" if a < b - EPS else "T")


# ----------------------------------------------------------------------------
# LOADING
# ----------------------------------------------------------------------------
def read_vs(run, scen):
    p = os.path.join(run, scen, "validation_summary.csv")
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p)
    return {r["model"]: r for _, r in df.iterrows()}


def load():
    """rows[scen][canonical model] = dict(tp,fp,fn,n_causal,n_universe,n_pred,run)"""
    src = SOURCES[WINDOW]
    runs = {run for lst in src.values() for run, _ in lst}
    scens = sorted({d for run in runs for d in os.listdir(run)
                    if os.path.isdir(os.path.join(run, d))
                    and group_of(d) in ALL_GROUPS})
    rows = defaultdict(dict)
    prov = defaultdict(Counter)
    cache = {}

    def vs_of(run, scen):
        if (run, scen) not in cache:
            cache[(run, scen)] = read_vs(run, scen)
        return cache[(run, scen)]

    for scen in scens:
        for canon, lst in src.items():
            for run, name in lst:
                v = vs_of(run, scen)
                if name in v and v[name]["status"] == "ok":
                    r = v[name]
                    rows[scen][canon] = dict(
                        tp=int(r["TP"]), fp=int(r["FP"]), fn=int(r["FN"]),
                        n_causal=int(r["n_causal"]), n_universe=int(r["n_universe"]),
                        n_pred=int(r["n_predicted"]), run=f"{run}:{name}")
                    prov[canon][f"{O.label(run)}:{name}"] += 1
                    break
    return rows, prov


def pooled(rows, groups, model, only_with=None):
    """TP/FP/FN summed per group; with only_with, only the realizations on which
    that other model also ran."""
    out = {}
    for g in groups:
        c = [0, 0, 0]
        n = 0
        for scen, mm in rows.items():
            if group_of(scen) != g or model not in mm:
                continue
            if only_with and only_with not in mm:
                continue
            c[0] += mm[model]["tp"]; c[1] += mm[model]["fp"]; c[2] += mm[model]["fn"]
            n += 1
        if n:
            out[g] = tuple(c)
    return out


def head2head(rows, groups, a, b):
    ga = pooled(rows, groups, a, only_with=b)
    gb = pooled(rows, groups, b, only_with=a)
    res = {g: cmp(f1(*ga[g]), f1(*gb[g])) for g in ga if g in gb}
    return res, ga, gb


# ----------------------------------------------------------------------------
# REPORT
# ----------------------------------------------------------------------------
class Rep:
    def __init__(self):
        self.lines, self.data = [], {}

    def h(self, t):
        self.lines += ["", t, ""]

    def p(self, t=""):
        self.lines.append(t)

    def table(self, header, body):
        self.p("| " + " | ".join(header) + " |")
        self.p("|" + "---|" * len(header))
        for r in body:
            self.p("| " + " | ".join(str(x) for x in r) + " |")

    def set(self, k, v):
        self.data[k] = v


def fmt(x, d=4):
    return f"{x:.{d}f}"


def wtl(res):
    c = Counter(res.values())
    return c["W"], c["T"], c["L"]


def main():
    rows, prov = load()
    R = Rep()
    R.p(f"# Numbers of the paper -- generated by generators/paper_numbers.py")
    R.p(f"Common window tau_max = {WINDOW}. Pipeline: `{O.label(PIPELINE_RUN)}`; sources of each "
        "model in section 0.")

    # 0. provenance ---------------------------------------------------------
    R.h("## 0. Provenance (which run each model came from)")
    R.table(["Model", "Scenarios", "Per run"],
            [[LABEL.get(m, m), sum(c.values()),
              ", ".join(f"{k}: {v}" for k, v in c.items())]
             for m, c in prov.items()])
    missing = [s for s in rows if PIPELINE_MODEL not in rows[s]]
    R.p(f"\nScenarios without the pipeline: {missing or 'none'}")
    R.set("provenance", {m: dict(c) for m, c in prov.items()})

    P = PIPELINE_MODEL
    grp_all = sorted({group_of(s) for s in rows})

    # 1. headline: pipeline vs PCMCI -----------------------------------------
    R.h("## 1. Headline: pipeline vs BH-PCMCI (pooled F1 per group)")
    res, gp, gb = head2head(rows, ESTAB18, P, "pcmci_tigramite")
    v, e, d = wtl(res)
    R.p(f"- Established groups with both methods ({len(res)} of 18): matches or beats in **{v + e} of {len(res)}** "
        f"(wins {v}, ties {e}, loses {d}). Loses on: "
        f"{[g for g, x in res.items() if x == 'L']}")
    r11, _, _ = head2head(rows, ORIGINAL11, P, "pcmci_tigramite")
    r14, _, _ = head2head(rows, ORIGINAL11 + CAUSALTIME, P, "pcmci_tigramite")
    R.p(f"- 11 original groups: {sum(1 for x in r11.values() if x != 'L')} of {len(r11)} | "
        f"14 groups (with CausalTime): {sum(1 for x in r14.values() if x != 'L')} of {len(r14)}")
    rp = {}
    for g in PREC19:
        a = pooled(rows, [g], P, "pcmci_tigramite").get(g)
        b = pooled(rows, [g], "pcmci_tigramite", P).get(g)
        if a and b:
            rp[g] = cmp(prf(a)[0], prf(b)[0])
    R.p(f"- Precision strictly higher than PCMCI's on **{sum(1 for x in rp.values() if x == 'W')} "
        f"of {len(rp)}** groups; non-wins: {[g for g, x in rp.items() if x != 'W']}")
    R.set("headline", dict(w=v, t=e, l=d, n=len(res), prec_wins=sum(1 for x in rp.values() if x == 'W'),
                           prec_n=len(rp)))

    # 2. pooled aggregates ----------------------------------------------------
    R.h("## 2. Pooled aggregates (Section 5.2)")
    body = []
    for name, groups in [("11 original groups", ORIGINAL11),
                         ("14 groups (with CausalTime)", ORIGINAL11 + CAUSALTIME)]:
        # pipeline on every realization; comparison with PCMCI only on the common ones
        for m, other, tag in [(P, None, "all"), (P, "pcmci_tigramite", "common with PCMCI"),
                              ("pcmci_tigramite", P, "common with PCMCI")]:
            vals = list(pooled(rows, groups, m, only_with=other).values())
            if not vals:
                continue
            c = np.sum(vals, axis=0)
            n = sum(1 for s_, mm in rows.items() if group_of(s_) in groups and m in mm
                    and (other is None or other in mm))
            pr, rc, ff = prf(c)
            body.append([name, LABEL[m], tag, n, fmt(pr), fmt(rc), fmt(ff), *map(int, c)])
    R.table(["Set", "Model", "Realizations", "n", "P", "R", "F1", "TP", "FP", "FN"], body)
    c14 = np.sum(list(pooled(rows, ORIGINAL11 + CAUSALTIME, P).values()), axis=0)
    cct = np.sum(list(pooled(rows, CAUSALTIME, P).values()), axis=0)
    R.p(f"\nCausalTime accounts for {int(cct[1])} of {int(c14[1])} FP and "
        f"{int(cct[2])} of {int(c14[2])} FN of the pipeline on the 14 groups.")

    # 3. pipeline vs each baseline --------------------------------------------
    R.h("## 3. Pipeline vs each baseline, per group (18 established + Walker + NetSim + PetShop)")
    body = []
    for m, lab in BASELINES + GPDC:
        r, _, _ = head2head(rows, PREC19 + NEW2 + EXTRA, P, m)
        if r:
            vv, ee, dd = wtl(r)
            body.append([lab, len(r), vv, ee, dd, [g for g, x in r.items() if x == "L"]])
    R.table(["Baseline", "Groups", "Pipeline wins", "Ties", "Loses", "Groups on which it loses"], body)
    # best baseline per group
    best = {}
    for g in PREC19 + NEW2 + EXTRA:
        pg = pooled(rows, [g], P).get(g)
        if not pg:
            continue
        cand = [(f1(*pooled(rows, [g], m)[g]), LABEL[m]) for m, _ in BASELINES
                if g in pooled(rows, [g], m)]
        if cand:
            b = max(cand)
            best[g] = (cmp(f1(*pg), b[0]), f1(*pg), b)
    vv, ee, dd = wtl({g: x[0] for g, x in best.items()})
    R.p(f"\nAgainst the best of the five baselines in each group: wins {vv}, ties {ee}, loses {dd} "
        f"(of {len(best)}).")
    # per realization, against the best of the five and against each one (F1 to 4 places)
    from scipy.stats import binomtest
    F4 = lambda mm, m: round(f1(mm[m]["tp"], mm[m]["fp"], mm[m]["fn"]), 4)
    sc_res, losses = Counter(), []
    for scen, mm in sorted(rows.items()):
        cand = [(F4(mm, m), LABEL[m]) for m, _ in BASELINES if m in mm]
        if P in mm and cand:
            b = max(cand)
            x = cmp(F4(mm, P), b[0])
            sc_res[x] += 1
            if x == "L":
                losses.append((scen.replace("benchmark_", ""), F4(mm, P), b))
    wv, lv = sc_res["W"], sc_res["L"]
    R.p(f"\nPer realization, against the best of the five: wins {wv}, ties {sc_res['T']}, loses {lv}; "
        f"sign test p = {binomtest(wv, wv + lv, 0.5).pvalue:.2g}")
    for x in losses:
        R.p(f"  - loses: {x}")
    for m, lab in BASELINES:
        c = Counter(cmp(F4(mm, P), F4(mm, m)) for mm in rows.values() if P in mm and m in mm)
        if c:
            R.p(f"- vs {lab} per realization: wins {c['W']}, ties {c['T']}, loses {c['L']}; "
                f"sign test p = {binomtest(c['W'], c['W'] + c['L'], 0.5).pvalue:.2g}")

    # 3b. Table 3 of the paper: score per family against the best of the five
    R.h("## 3b. Table 3 -- score per family against the best baseline of each scenario")

    def fam3(scen):
        g = group_of(scen)
        for k, v in [("synthetic", "VAR synthetic"), ("climate_walker", "Walker Circulation"),
                     ("tigramite", "tigramite SCM"), ("lorenz", "Lorenz-96"),
                     ("logistic", "Coupled logistic maps"), ("causaltime", "CausalTime"),
                     ("tep", "Tennessee Eastman"), ("sachs", "Sachs"), ("alarm", "ALARM"),
                     ("pems", "PeMS-BAY"), ("netsim", "NetSim"), ("petshop", "PetShop"),
                     ("beeline", "BEELINE"), ("dream3", "DREAM3"), ("rcaeval", "RCAEval"),
                     ("easyvista", "EasyVista"), ("causal_chamber", "Causal Chambers")]:
            if k in g:
                return v
        return g
    fams = defaultdict(lambda: {"res": Counter(), "p": [], "b": []})
    for scen, mm in rows.items():
        cand = [F4(mm, m) for m, _ in BASELINES if m in mm]
        if P not in mm or not cand:
            continue
        f = fams[fam3(scen)]
        f["res"][cmp(F4(mm, P), max(cand))] += 1
        f["p"].append(F4(mm, P)); f["b"].append(max(cand))
    order3 = ["VAR synthetic", "tigramite SCM", "Lorenz-96", "Coupled logistic maps", "CausalTime",
              "ALARM", "NetSim", "PetShop",
              "Tennessee Eastman", "Walker Circulation", "Sachs", "PeMS-BAY",
              "BEELINE", "DREAM3", "RCAEval", "EasyVista", "Causal Chambers"]
    body, tot = [], Counter()
    for k in order3:
        if k in fams:
            f = fams[k]; tot += f["res"]
            body.append([f"{k} ({len(f['p'])})", f"{f['res']['W']} / {f['res']['T']} / {f['res']['L']}",
                         f"{np.mean(f['p']):.3f} vs {np.mean(f['b']):.3f}",
                         f"{max(0.0, max(b - a for a, b in zip(f['p'], f['b']))):.3f}"])
    body.append([f"**Total ({sum(tot.values())})**", f"**{tot['W']} / {tot['T']} / {tot['L']}**", "", ""])
    R.table(["Family (scenarios)", f"tau_max = {WINDOW}: W / T / L", "Mean F1, pipeline vs best baseline",
             "Largest shortfall to the best baseline"], body)
    for name, keep in [("development", lambda s: group_of(s) not in HELDOUT),
                       ("held-out", lambda s: group_of(s) in HELDOUT)]:
        a = [(F4(mm, P), max(F4(mm, m) for m, _ in BASELINES if m in mm)) for s, mm in rows.items()
             if keep(s) and P in mm]
        c = Counter(cmp(x, y) for x, y in a)
        R.p(f"- Subtotal {name} ({len(a)}): {c['W']} / {c['T']} / {c['L']}; mean F1 "
            f"{np.mean([x for x, _ in a]):.3f} vs {np.mean([y for _, y in a]):.3f}; largest shortfall to the best baseline "
            f"{max(y - x for x, y in a):.3f}")
    lo = Counter()
    for s, mm in rows.items():
        b = max(F4(mm, m) for m, _ in BASELINES if m in mm)
        if F4(mm, P) < b - EPS:
            for m, lab in BASELINES:
                if m in mm and abs(F4(mm, m) - b) <= EPS:
                    lo[(lab, "held-out" if group_of(s) in HELDOUT else "development")] += 1
    R.p("- Baseline ahead in the losses (ties between baselines count for all of them): "
        + ", ".join(f"{k[0]} [{k[1]}] {v}" for k, v in sorted(lo.items(), key=lambda x: -x[1])))

    # 3c. two-sided paired Wilcoxon, Holm over the five baselines ----------------
    # Unit: realization (F1 to 4 places) or family (mean of the differences in the
    # fam3 family). The best of five is a separate comparison, without Holm.
    R.h("## 3c. Two-sided paired Wilcoxon against each baseline (Holm over the five)")
    no_ct = lambda s: group_of(s) not in CAUSALTIME
    best5 = lambda mm: max(F4(mm, m) for m, _ in BASELINES if m in mm)
    for name, keep in [("all realizations", lambda s: True), ("without CausalTime", no_ct)]:
        for unit in ["realization", "family"]:
            body, pv = [], {}
            for m, lab in BASELINES + [("best5", "Best of five")]:
                pairs = [(s, F4(mm, P), best5(mm) if m == "best5" else F4(mm, m))
                         for s, mm in rows.items() if keep(s) and P in mm and (m == "best5" or m in mm)]
                if unit == "family":
                    diffs = defaultdict(list)
                    for s, a, b in pairs:
                        diffs[fam3(s)].append(a - b)
                    d = np.array([np.mean(v) for v in diffs.values()])
                else:
                    d = np.array([a - b for _, a, b in pairs])
                c = Counter("W" if x > EPS else ("L" if x < -EPS else "T") for x in d)
                p = wilcoxon(d).pvalue if np.any(np.abs(d) > EPS) else 1.0
                if m != "best5":
                    pv[lab] = p
                body.append([lab, len(d), f"{c['W']}/{c['T']}/{c['L']}", f"{np.mean(d):+.3f}", f"{p:.2g}"])
            # Holm over the five baselines
            ordered = sorted(pv.items(), key=lambda x: x[1])
            holm, acc = {}, 0.0
            for i, (lab, p) in enumerate(ordered):
                acc = max(acc, min(1.0, p * (len(ordered) - i)))
                holm[lab] = acc
            for r in body:
                r.append(f"{holm[r[0]]:.2g}" if r[0] in holm else "-")
            R.p(f"\n### {name}, unit = {unit}\n")
            R.table(["Against", "n", "Pipeline higher/equal/lower", "Mean dF1", "Wilcoxon p (two-sided)",
                     "Holm p (5 baselines)"], body)
            R.set(f"wilcoxon_{'all' if name.startswith('all') else 'no_causaltime'}_{unit}",
                  {r[0]: {"p": r[4], "holm": r[5]} for r in body})

    # 3f. bootstrap with the scenario group as the unit -------------------------
    R.h("## 3f. F1 difference with the scenario group as the unit (Section 5.1, Table S8)")
    rng = np.random.default_rng(0)
    for name, keep in [("all scenarios", lambda s: True),
                       ("development", lambda s: group_of(s) not in HELDOUT),
                       ("held-out", lambda s: group_of(s) in HELDOUT)]:
        body, pw_of = [], {}
        for m, lab in BASELINES + [("best5", "Best of five")]:
            pairs = [(s, F4(mm, P) - (best5(mm) if m == "best5" else F4(mm, m)))
                     for s, mm in sorted(rows.items()) if keep(s) and P in mm and (m == "best5" or m in mm)]
            byg, byf = defaultdict(list), defaultdict(list)
            for s_, d_ in pairs:
                byg[group_of(s_)].append(d_); byf[fam3(s_)].append(d_)
            groups = list(byg)
            boots = [np.concatenate([byg[g] for g in rng.choice(groups, len(groups))]).mean() for _ in range(5000)]
            lo_, hi_ = np.percentile(boots, [2.5, 97.5])
            fm = np.array([np.mean(v) for v in byf.values()])
            ahead, behind = int((fm > EPS).sum()), int((fm < -EPS).sum())
            ps = binomtest(ahead, ahead + behind).pvalue if ahead + behind else 1.0
            pw = wilcoxon(fm).pvalue if (np.abs(fm) > EPS).any() else 1.0
            if m != "best5":
                pw_of[lab] = pw
            body.append([lab, len(pairs), len(groups), f"{np.mean([d_ for _, d_ in pairs]):+.3f}",
                         f"[{lo_:+.3f}, {hi_:+.3f}]", f"{ahead} / {len(fm) - ahead - behind} / {behind}",
                         f"{ps:.2g}", f"{pw:.2g}"])
        holm, acc = {}, 0.0
        for i, (lab, p) in enumerate(sorted(pw_of.items(), key=lambda kv: kv[1])):
            acc = max(acc, min(1.0, p * (len(pw_of) - i)))
            holm[lab] = acc
        for r in body:
            r.append(f"{holm[r[0]]:.2g}" if r[0] in holm else "-")
        R.p(f"\n### {name}\n")
        R.table(["Against", "Scenarios", "Groups", "Mean F1 difference", "95% CI (bootstrap over scenario groups)",
                 "Families ahead / level / behind", "Family sign test p", "Family Wilcoxon p (two-sided)",
                 "Holm-adjusted p"], body)

    # 3d. naive combinations of the five baselines -------------------------------
    R.h("## 3d. Naive combinations of the five baselines (Section 5.2, Table S8.1)")
    ensemble(R, rows)

    # 3e. shortfall to the best method of each scenario ---------------------------
    R.h("## 3e. Shortfall to the best method of each scenario (Highlights, abstract, Section 5.2, Table 3)")
    shortfall(R, rows)

    # 3g. mean rank and one-to-one comparison per subset --------------------------
    R.h("## 3g. Mean rank (Friedman, Nemenyi) and one-to-one comparison per subset "
        "(Highlights, abstract, Section 5.1, Table 2a)")
    ranking(R, rows)

    # 3h. relative shortfall and shortfall restricted to scenarios above the floor -
    R.h("## 3h. Relative shortfall to the best method and shortfall on the scenarios above the trivial floor "
        "(Sections 4.4, 5.1 and 6.2, Table 2a, Table S8.3)")
    relative_shortfall(R, rows)

    # 4. Table 3 -----------------------------------------------------------
    R.h("## 4. Real benchmarks, NetSim and PetShop (pooled P/R/F1 per group)")
    body = []
    for g in REAL4 + NEW2 + EXTRA:
        for m in [P] + [x for x, _ in BASELINES]:
            c = pooled(rows, [g], m).get(g)
            if c:
                body.append([g, LABEL[m], *map(fmt, prf(c))])
    R.table(["Benchmark", "Model", "P", "R", "F1"], body)
    wc = pooled(rows, WALKER, P).get(WALKER[0])
    if wc:
        R.p(f"\nWalker: pipeline P/R/F1 = {', '.join(map(fmt, prf(wc)))}; TP/FP/FN = {wc}")

    # 5. synthetics (5.1) ----------------------------------------------------
    R.h("## 5. Internal synthetic generator (Section 5.1, Figure 4)")
    body = []
    for g in SYNTH:
        for m in [P] + [x for x, _ in BASELINES]:
            c = pooled(rows, [g], m).get(g)
            if c:
                body.append([g, LABEL[m], *map(fmt, prf(c)), *c])
    R.table(["Scenario", "Model", "P", "R", "F1", "TP", "FP", "FN"], body)
    for m, lab in BASELINES:
        r, _, _ = head2head(rows, SYNTH, P, m)
        R.p(f"- vs {lab}: wins/ties/loses = {wtl(r)}")

    # 6. zeros ---------------------------------------------------------------
    R.h("## 6. Realizations with F1 = 0 (Section 5.3)")
    for m in [P] + [x for x, _ in BASELINES]:
        z = [s for s, mm in rows.items() if m in mm and f1(mm[m]["tp"], mm[m]["fp"], mm[m]["fn"]) == 0]
        n = sum(1 for mm in rows.values() if m in mm)
        R.p(f"- {LABEL[m]}: {len(z)} of {n} | {[x.replace('benchmark_', '') for x in z]}")
    ps = [f1(mm[P]["tp"], mm[P]["fp"], mm[P]["fn"]) for mm in rows.values() if P in mm]
    R.p(f"- Lowest F1 of the pipeline: {min(ps):.4f}")

    # 6b. mean precision and recall per realization -----------------------------
    R.h("## 6b. Mean precision and recall per realization (Section 5.3)")
    methods = [P] + [x for x, _ in BASELINES]
    PR = lambda mm, m: prf((mm[m]["tp"], mm[m]["fp"], mm[m]["fn"]))[:2]
    body = []
    for name, keep in [("all", lambda s: True),
                       ("without CausalTime", lambda s: group_of(s) not in CAUSALTIME),
                       ("CausalTime only", lambda s: group_of(s) in CAUSALTIME)]:
        for m in methods:
            v = [PR(mm, m) for s, mm in rows.items() if keep(s) and m in mm]
            if v:
                body.append([name, LABEL[m], len(v), fmt(np.mean([x[0] for x in v]), 3),
                             fmt(np.mean([x[1] for x in v]), 3)])
    R.table(["Realizations", "Model", "n", "Mean P", "Mean R"], body)
    # per family: does the pipeline have P below and R above the most precise baseline?
    body = []
    for k in order3:
        ss = [s for s in rows if fam3(s) == k and P in rows[s]]
        med = {m: np.mean([PR(rows[s], m) for s in ss if m in rows[s]], axis=0)
               for m in methods if any(m in rows[s] for s in ss)}
        bp = max((m for m in med if m != P), key=lambda m: med[m][0])
        trades = med[P][0] < med[bp][0] - EPS and med[P][1] > med[bp][1] + EPS
        body.append([f"{k} ({len(ss)})", fmt(med[P][0], 3), fmt(med[P][1], 3), LABEL[bp],
                     fmt(med[bp][0], 3), fmt(med[bp][1], 3), "**yes**" if trades else "no"])
    R.p()
    R.table(["Family", "P pipeline", "R pipeline", "Most precise baseline", "P", "R",
             "Pipeline trades precision for recall"], body)
    for m, lab in BASELINES:
        c = Counter(cmp(PR(mm, P)[0], PR(mm, m)[0]) for mm in rows.values() if P in mm and m in mm)
        R.p(f"- precision vs {lab} per realization: wins {c['W']}, ties {c['T']}, loses {c['L']}")

    # 7. trivial floor ---------------------------------------------------------
    R.h("## 7. Trivial floor (Section 5.3, Table S23)")
    body, clears = [], Counter()
    nobody = []
    for g in ALL_GROUPS:
        rs = [mm for s, mm in rows.items() if group_of(s) == g and P in mm]
        if not rs:
            continue
        pv = sum(mm[P]["n_causal"] for mm in rs) / sum(mm[P]["n_universe"] for mm in rs)
        floor = 2 * pv / (1 + pv)
        f_models = {}
        for m in [P] + [x for x, _ in BASELINES]:
            c = pooled(rows, [g], m).get(g)
            if c:
                f_models[m] = f1(*c)
                if f1(*c) > floor + EPS:
                    clears[m] += 1
        anyc = any(v > floor + EPS for v in f_models.values())
        if not anyc:
            nobody.append(g)
        pp = prf(pooled(rows, [g], P)[g])
        body.append([g, fmt(pv, 3), fmt(floor, 3), fmt(f_models[P], 3),
                     "yes" if f_models[P] > floor + EPS else "**no**",
                     fmt(max(f_models.values()), 3), "yes" if anyc else "**no**",
                     fmt(pp[0], 3)])
    R.table(["Group", "p", "Floor", "F1 pipeline", "Pipeline clears", "Best F1", "Any clears",
             "Pipeline precision"], body)
    R.p("\nGroups each method clears: " + ", ".join(f"{LABEL[m]} {c}" for m, c in clears.items()))
    R.p(f"Groups no method clears: {nobody}")
    R.set("floor", dict(clears={LABEL[m]: c for m, c in clears.items()}, nobody=nobody))
    # AUPRC of the pipeline on the groups nobody clears (Tables S23.3 and S16.3): read
    # from the metrics run (METRICS_CSV), which repeats the reported one with the
    # current code and is checked against it by generators/metrics_shd_auc.py.
    # Ranking by the pair's max |penalized_effect|; never-scored pairs tied at zero.
    R.p("\nAUPRC of the pipeline on these groups (mean over realizations; chance = prevalence):")
    if METRICS_CSV is None:
        R.p("- (METRICS_RUN not set in origins.py)")
        met = pd.DataFrame(columns=["model", "scenario", "auprc", "prevalence"])
    else:
        met = pd.read_csv(METRICS_CSV)
    met = met[met["model"] == PIPELINE_FOLDER[WINDOW]]
    for g in nobody:
        x = met[met["scenario"].map(group_of) == g]
        if len(x):
            R.p(f"- {g}: AUPRC {x['auprc'].mean():.3f} vs prevalence {x['prevalence'].mean():.3f}")

    # 8. Stage 7 ablation -------------------------------------------------------
    R.h("## 8. Stage 7 ablation (Section 5.4, Table 4, Table S25)")
    ablation(R, core(rows))

    # 8c. full pipeline minus one regime-dependent component -----------------------
    R.h("## 8c. Full pipeline without 8a-bis / without the coherence guard (Section 5.4, Table 4)")
    leave_one_out(R, core(rows))

    # 9. precision per path ---------------------------------------------------------
    R.h("## 9. Precision per acceptance path (Section 3.4.2, Table S1)")
    provenance(R, core(rows))

    # 9b. effect estimate on the reported pairs --------------------------------
    R.h("## 9b. Effect estimate on the reported pairs (Sections 3.5 and 3.7)")
    effect_coverage(R, rows)

    # 10. GPDC -----------------------------------------------------------------
    if any("pcmci_gpdc" in mm for mm in rows.values()):
        R.h("## 10. GPDC check (Section 5.2, Table S21)")
        gpdc(R, rows)

    # 11. robustness to the window ------------------------------------------------
    if PIPELINE_ALT and any(PIPELINE_ALT in mm for mm in rows.values()):
        R.h(f"## 11. Robustness to the window ({PIPELINE_MODEL} vs {PIPELINE_ALT})")
        a = [(f1(mm[P]["tp"], mm[P]["fp"], mm[P]["fn"]),
              f1(mm[PIPELINE_ALT]["tp"], mm[PIPELINE_ALT]["fp"], mm[PIPELINE_ALT]["fn"]))
             for mm in rows.values() if P in mm and PIPELINE_ALT in mm]
        c = Counter(cmp(y, x) for x, y in a)
        R.p(f"- Realizations: {len(a)} | Mean F1: {np.mean([x for x, _ in a]):.3f} ({P}) vs "
            f"{np.mean([y for _, y in a]):.3f} ({PIPELINE_ALT})")
        R.p(f"- {PIPELINE_ALT} higher/lower/equal: {c['W']}/{c['L']}/{c['T']}")

    # 12. tables per realization S10-S13 ---------------------------------------------
    R.h("## 12. Tables per realization (Tables S10-S13)")
    for name, pref in [("S10 logistic maps", "logistic"), ("S11 Lorenz-96", "lorenz"),
                       ("S12 tigramite", "tigramite_"), ("S13 CausalTime", "causaltime"),
                       ("NetSim", "netsim"), ("PetShop", "petshop"), ("BEELINE", "beeline"),
                       ("DREAM3", "dream3"), ("RCAEval", "rcaeval"), ("EasyVista", "easyvista"),
                       ("Causal Chambers", "causal_chamber")]:
        R.p(f"\n### {name}\n")
        body = []
        for s in sorted(rows):
            if pref not in s or group_of(s) in WALKER:
                continue
            for m in [P] + [x for x, _ in BASELINES]:
                if m in rows[s]:
                    c = (rows[s][m]["tp"], rows[s][m]["fp"], rows[s][m]["fn"])
                    body.append([s.replace("benchmark_", "").replace("real_", ""), LABEL[m], *map(lambda x: fmt(x, 3), prf(c)), *c])
        R.table(["Scenario", "Method", "P", "R", "F1", "TP", "FP", "FN"], body)

    with open(os.path.join(OUT_DIR, f"Paper_Numbers_tau{WINDOW}.md"), "w", encoding="utf-8") as f:
        f.write(O.canonical("\n".join(R.lines) + "\n"))
    with open(os.path.join(OUT_DIR, f"paper_numbers_tau{WINDOW}.json"), "w", encoding="utf-8") as f:
        f.write(O.canonical(json.dumps(R.data, indent=2, ensure_ascii=False, default=str)))
    print("ok ->", os.path.join(OUT_DIR, f"Paper_Numbers_tau{WINDOW}.md"))


# ----------------------------------------------------------------------------
# STAGE 7 ABLATION
# ----------------------------------------------------------------------------
def pair_table(run, scen, model):
    p = os.path.join(run, scen, model, "original.csv")
    g = os.path.join(run, scen, "ground_truth.csv")
    if not (os.path.exists(p) and os.path.exists(g)):
        return None, None
    gt = pd.read_csv(g)
    gt["pos"] = gt["causal_gt"].astype(str).str.lower().isin(["true", "1"])
    truth = gt.groupby(["source_entity", "target_entity"])["pos"].any()
    df = pd.read_csv(p)
    return df, truth


def num(s):
    return pd.to_numeric(s, errors="coerce")


def ablation(R, rows):
    rules = {
        "1 Stage 7 (DS)": lambda d: num(d["causal_penalty_score"]) == 1,
        "2 No filter": lambda d: pd.Series(True, index=d.index),
        "3 Cascade high/medium": lambda d: d["cascade_confidence"].isin(["high", "medium"]),
        "4 p_BH < 0.05": lambda d: num(d["pvalue_bh"]) < 0.05,
        "5 |t| > 1.96": lambda d: num(d["t_stat"]).abs() > 1.96,
        "9 p_BH<0.05 and CI excludes 0": lambda d: (num(d["pvalue_bh"]) < 0.05) & (
            (num(d["ci_95_lower"]) > 0) | (num(d["ci_95_upper"]) < 0)),
        "10 p_BH<0.05 and asymmetry>0": lambda d: (num(d["pvalue_bh"]) < 0.05) & (num(d["causal_asymmetry"]) > 0),
        "Final (DS + Stage 8)": lambda d: d["significant"].astype(str) == "True",
    }
    per = defaultdict(dict)
    npred = defaultdict(dict)
    prec = defaultdict(dict)
    rec = defaultdict(dict)
    skipped = []
    for scen in sorted(rows):
        df, truth = pair_table(PIPELINE_RUN, scen, PIPELINE_FOLDER[WINDOW])
        if df is None or len(df) == 0:
            skipped.append(scen)
            continue
        T = set(truth[truth].index)
        pairs = df[["source_entity", "target_entity"]].apply(tuple, axis=1)

        def score_sets(mask):
            S = set(pairs[mask.fillna(False)])
            tp = len(S & T); fp = len(S - T); fn = len(T - S)
            return (tp, fp, fn), len(S)

        for k, fn_ in rules.items():
            c, n = score_sets(fn_(df))
            per[k][scen] = f1(*c); npred[k][scen] = n
            prec[k][scen], rec[k][scen] = prf(c)[0], prf(c)[1]
        # oracles
        pmin = num(df["pvalue_bh"]).groupby(pairs).min()
        tmax = num(df["t_stat"]).abs().groupby(pairs).max()
        for k, sc, grid, better in [("6 p oracle", pmin, [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5], "lt"),
                                    ("7 |t| oracle", tmax, list(np.arange(0.5, 4.01, 0.25)), "gt")]:
            bestc, bestn = None, 0
            for th in grid:
                S = set(sc[(sc < th) if better == "lt" else (sc > th)].index)
                c = (len(S & T), len(S - T), len(T - S))
                if bestc is None or f1(*c) > f1(*bestc) + EPS:
                    bestc, bestn = c, len(S)
            per[k][scen] = f1(*bestc); npred[k][scen] = bestn
            prec[k][scen], rec[k][scen] = prf(bestc)[0], prf(bestc)[1]
        # 8: recall matched to DS
        r_ds = rec["1 Stage 7 (DS)"][scen]
        bestc, bestn = None, 0
        for th in sorted(tmax.dropna().unique(), reverse=True):
            S = set(tmax[tmax >= th].index)
            c = (len(S & T), len(S - T), len(T - S))
            if prf(c)[1] >= r_ds - EPS:
                if bestc is None or prf(c)[0] > prf(bestc)[0] + EPS:
                    bestc, bestn = c, len(S)
        if bestc is not None:
            per["8 |t| matched recall"][scen] = f1(*bestc); npred["8 |t| matched recall"][scen] = bestn
            prec["8 |t| matched recall"][scen], rec["8 |t| matched recall"][scen] = prf(bestc)[:2]
        # 11: |t| with an adaptive cut-off per dataset, the analogue of Stage 7's
        # adaptive floor (45th percentile of the candidates with |t| > 1.96; with
        # fewer than 10, only the fixed cut-off). Rule fixed before the computation
        # (analysis/stage7_vs_adaptive_rule/PREREGISTRATION.md).
        cand = tmax[tmax > 1.96]
        if len(cand) >= 10:
            cand = cand[cand > float(np.percentile(cand.values, 45.0))]
        S = set(cand.index)
        c = (len(S & T), len(S - T), len(T - S))
        per["11 |t| adaptive (P45)"][scen] = f1(*c); npred["11 |t| adaptive (P45)"][scen] = len(S)
        prec["11 |t| adaptive (P45)"][scen], rec["11 |t| adaptive (P45)"][scen] = prf(c)[:2]

    ds = per["1 Stage 7 (DS)"]
    order = ["1 Stage 7 (DS)", "2 No filter", "3 Cascade high/medium", "4 p_BH < 0.05", "5 |t| > 1.96",
             "6 p oracle", "7 |t| oracle", "8 |t| matched recall", "9 p_BH<0.05 and CI excludes 0",
             "10 p_BH<0.05 and asymmetry>0", "11 |t| adaptive (P45)", "Final (DS + Stage 8)"]
    pv = {}
    for k in order[1:]:
        common = [s for s in ds if s in per[k]]
        a = np.array([per[k][s] for s in common]); b = np.array([ds[s] for s in common])
        try:
            pv[k] = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
        except ValueError:
            pv[k] = 1.0
    # Holm
    it = sorted(pv.items(), key=lambda x: x[1]); m = len(it); run = 0; holm = {}
    for i, (k, p) in enumerate(it):
        run = max(run, min(1.0, p * (m - i))); holm[k] = run
    body = []
    for k in order:
        common = [s for s in ds if s in per[k]]
        fam = defaultdict(list)
        for s in common:
            fam[family_of(group_of(s))].append(per[k][s])
        c = Counter(cmp(per[k][s], ds[s]) for s in common)
        body.append([k, len(common), fmt(np.mean([prec[k][s] for s in common]), 3),
                     fmt(np.mean([rec[k][s] for s in common]), 3), fmt(np.mean([per[k][s] for s in common]), 3),
                     fmt(np.median([per[k][s] for s in common]), 3),
                     fmt(np.mean([np.mean(v) for v in fam.values()]), 3),
                     fmt(np.mean([npred[k][s] for s in common]), 1),
                     "" if k == order[0] else f"{np.mean([per[k][s] - ds[s] for s in common]):+.3f}",
                     "" if k == order[0] else f"{c['W']}/{c['T']}/{c['L']}",
                     "" if k == order[0] else f"{pv[k]:.2g}", "" if k == order[0] else f"{holm[k]:.2g}"])
    R.table(["Rule", "n", "Mean P", "Mean R", "Mean F1", "Median F1", "Mean F1 per family",
             "Pairs accepted", "dF1 vs DS", "Rule wins/ties/loses", "Wilcoxon p", "Holm p"], body)
    mf = lambda k: np.mean([per[k][s] for s in ds if s in per[k]])
    for fixed, oracle in [("4 p_BH < 0.05", "6 p oracle"), ("5 |t| > 1.96", "7 |t| oracle")]:
        den = mf(oracle) - mf(fixed)
        if abs(den) > EPS:
            R.p(f"- Fraction of the oracle's gain recovered by DS ({fixed} -> {oracle}): "
                f"{100 * (mf('1 Stage 7 (DS)') - mf(fixed)) / den:.0f}%")
    R.p(f"- Scenarios without pipeline output (left out of the ablation): {skipped or 'none'}")
    # F1 per scenario and rule (Table S25.2)
    R.h("### 8b. F1 per scenario (Table S25.2)")
    R.table(["Scenario"] + [k.split(" ")[0] if k[0].isdigit() else "Full" for k in order],
            [[s] + [fmt(per[k][s], 3) if s in per[k] else "—" for k in order] for s in sorted(ds)])


def leave_one_out(R, rows):
    """
    Full pipeline minus one component, recomputed offline from the
    intermediate decisions the run stores in original.csv:

      without 8a-bis : on the lagged rows on which the predictive competition was
                       applied, falls back to the decision made before it
                       ('significant_before_predictive');
      without guard  : on the scenarios of regime 'incoherent', the lagged rows
                       fall back to the decision of Stages 7-8a ('significant_before_guard');
      without 8b     : the lag-0 rows are no longer reported;
      without the time-axis rule : the lagged edges marked 'no_time_axis'
                       come back (8b stays as in the final output);
      without the guard exception : on the incoherent scenarios on which the past
                       explains little ('evidence_fallback' = 'stage7_8a'), the
                       lagged rows fall back to the general rule of the guard,
                       the decision of Stage 6 ('significant_stage6'). Where there
                       is no time axis the lagged rows are discarded anyway, and
                       the output does not change.

    The guard only counts as acting where the fallback to Stage 6 was applied
    ('evidence_fallback' other than 'stage7_8a').

    Order of the pipeline: 8a -> 8a-bis -> guards (incoherence, time axis) ->
    8b. Lag-0 rows (8b) and those discarded for lack of a time axis stay as in
    the final output. Does not capture second-order effects (8b deciding
    differently with another lagged graph).
    """
    VAR = ["Full", "Full without 8a-bis", "Full without the coherence guard",
           "Full without 8b", "Full without the time-axis rule",
           "Full without the guard exception"]
    per = {k: {} for k in VAR}
    pr = {k: {} for k in VAR}
    npred = {k: {} for k in VAR}
    acts = {k: [] for k in VAR[1:]}
    for scen in sorted(rows):
        df, truth = pair_table(PIPELINE_RUN, scen, PIPELINE_FOLDER[WINDOW])
        if df is None or len(df) == 0:
            continue
        T = set(truth[truth].index)
        pairs = df[["source_entity", "target_entity"]].apply(tuple, axis=1)
        final = df["significant"].astype(str) == "True"
        lagged = num(df["lag"]) >= 1
        no_axis = (df["structural_rejection_reason"].astype(str) == "no_time_axis"
                   if "structural_rejection_reason" in df else pd.Series(False, index=df.index))
        incoherent = "evidence_regime" in df and (df["evidence_regime"].astype(str) == "incoherent").any()

        no_pred = final.copy()
        if "significant_before_predictive" in df and not incoherent:
            m = lagged & df["significant_before_predictive"].notna() & ~no_axis
            if m.any():
                no_pred = pd.Series(np.where(m, df["significant_before_predictive"].astype(str) == "True", final),
                                    index=df.index)
                acts[VAR[1]].append(scen)
        exception = "evidence_fallback" in df and (df["evidence_fallback"].astype(str) == "stage7_8a").any()
        no_guard = final.copy()
        if incoherent and not exception and "significant_before_guard" in df:
            m = lagged & df["significant_before_guard"].notna() & ~no_axis
            no_guard = pd.Series(np.where(m, df["significant_before_guard"].astype(str) == "True", final),
                                 index=df.index)
            acts[VAR[2]].append(scen)
        # without 8b: the lag-0 (contemporaneous) rows are no longer reported
        lag0 = (num(df["lag"]) == 0) & final
        no_8b = final & ~lag0
        if lag0.any():
            acts[VAR[3]].append(scen)
        # without the time-axis rule: the lagged edges it discarded come back
        no_axis_rule = final | no_axis
        if no_axis.any():
            acts[VAR[4]].append(scen)

        # without the guard exception: the lagged rows fall back to the decision of Stage 6
        no_exc = final.copy()
        if exception:
            if not no_axis.any():
                no_exc = pd.Series(np.where(lagged, df["significant_stage6"].astype(str) == "True", final),
                                   index=df.index)
            acts[VAR[5]].append(scen)

        for k, mask in zip(VAR, [final, no_pred, no_guard, no_8b, no_axis_rule, no_exc]):
            S = set(pairs[mask])
            c = (len(S & T), len(S - T), len(T - S))
            per[k][scen] = f1(*c); pr[k][scen] = prf(c)[:2]; npred[k][scen] = len(S)

    full = per[VAR[0]]
    body = []
    for k in VAR:
        sc = sorted(full)
        row = [k, len(sc), fmt(np.mean([pr[k][s][0] for s in sc]), 3), fmt(np.mean([pr[k][s][1] for s in sc]), 3),
               fmt(np.mean([per[k][s] for s in sc]), 3), fmt(np.mean([npred[k][s] for s in sc]), 1)]
        if k == VAR[0]:
            row += ["", "", "", "", ""]
        else:
            d = np.array([per[k][s] - full[s] for s in sc])
            c = Counter(cmp(per[k][s], full[s]) for s in sc)
            p = wilcoxon(d).pvalue if np.any(np.abs(d) > EPS) else 1.0
            a = acts[k]
            row += [f"{np.mean(d):+.3f}", f"{c['W']}/{c['T']}/{c['L']}", f"{p:.2g}", len(a),
                    f"{np.mean([per[k][s] for s in a]):.3f} vs {np.mean([full[s] for s in a]):.3f}" if a else "-"]
        body.append(row)
    R.table(["Variant", "n", "Mean P", "Mean R", "Mean F1", "Pairs accepted", "dF1 vs full",
             "Variant wins/ties/loses", "Wilcoxon p (two-sided)", "Scenarios on which the component acts",
             "Mean F1 on those scenarios: variant vs full"], body)
    for k in VAR[1:]:
        R.p()
        R.p(f"{k} -- scenarios on which the component acts (F1 of the variant -> full F1):")
        for s in acts[k]:
            R.p(f"  - {s.replace('benchmark_', '').replace('real_', '')}: {per[k][s]:.4f} -> {full[s]:.4f}")
    R.set("leave_one_out", {k: {"f1": float(np.mean(list(per[k].values()))),
                                "acts": [s for s in acts.get(k, [])]} for k in VAR})


def ensemble(R, rows):
    """
    Naive combinations of the five baselines, scored as one more method.
    In each scenario, each baseline that ran (status ok) votes on each ordered
    pair of entities: it votes for the pair if it has at least one significant
    row from the source entity to the target entity, in any property and lag
    (the unit of Section 4.4). Four rules: strict majority (more than half of
    the voters), at least half, union (one vote) and intersection (all). The
    set of pairs of each baseline is read from results.csv and checked against
    the TP/FP/FN of validation_summary.csv.
    """
    from scipy.stats import binomtest
    rules = [("Strict majority", lambda v, n: 2 * v > n), ("At least half", lambda v, n: 2 * v >= n),
             ("Union", lambda v, n: v >= 1), ("Intersection", lambda v, n: v == n)]
    P = PIPELINE_MODEL
    src = SOURCES[WINDOW]
    F4 = lambda c: round(f1(*c), 4)
    per = {k: {} for k, _ in rules}
    pipe, best5, voters, mismatches = {}, {}, Counter(), 0
    for scen, mm in sorted(rows.items()):
        if P not in mm:
            continue
        gt = pd.read_csv(os.path.join(run_of(rows, scen), scen, "ground_truth.csv"))
        gt["pos"] = gt["causal_gt"].astype(str).str.lower().isin(["true", "1"])
        g = gt.groupby([gt["source_entity"].astype(str), gt["target_entity"].astype(str)])["pos"].any()
        T, U = set(g[g].index), set(g.index)
        sets = {}
        for m, _lab in BASELINES:
            if m not in mm:
                continue
            run, name = mm[m]["run"].rsplit(":", 1)
            d = pd.read_csv(os.path.join(run, scen, name, "results.csv"), low_memory=False)
            if "significant" in d.columns:
                d = d[d["significant"].astype(str) == "True"]
            S = set(zip(d["source_entity"].astype(str), d["target_entity"].astype(str))) & U
            sets[m] = S
            if (len(S & T), len(S - T), len(T - S)) != (mm[m]["tp"], mm[m]["fp"], mm[m]["fn"]):
                mismatches += 1
        n = len(sets)
        voters[n] += 1
        votes = Counter(p for S in sets.values() for p in S)
        for k, rule in rules:
            S = {p for p, v in votes.items() if rule(v, n)}
            per[k][scen] = F4((len(S & T), len(S - T), len(T - S)))
        pipe[scen] = F4((mm[P]["tp"], mm[P]["fp"], mm[P]["fn"]))
        best5[scen] = max(F4((mm[m]["tp"], mm[m]["fp"], mm[m]["fn"])) for m in sets)

    sc = sorted(pipe)
    a = np.array([pipe[s] for s in sc])
    R.p(f"Scenarios: {len(sc)}; voters per scenario: "
        + ", ".join(f"{k} baselines on {v}" for k, v in sorted(voters.items()))
        + f". Pair sets that do not reproduce validation_summary: {mismatches}.\n")
    body = []
    for k, _ in rules:
        b = np.array([per[k][s] for s in sc])
        c = Counter(cmp(x, y) for x, y in zip(a, b))
        pw = wilcoxon(a, b).pvalue if np.any(np.abs(a - b) > EPS) else 1.0
        body.append([k, f"{b.mean():.3f}", f"{a.mean():.3f}", f"{c['W']}/{c['T']}/{c['L']}",
                     f"{binomtest(c['W'], c['W'] + c['L']).pvalue:.2g}", f"{pw:.2g}"])
    R.table(["Combination", "Mean F1 of the combination", "Mean F1 of the pipeline", "Pipeline wins/ties/loses",
             "Sign test p", "Wilcoxon p (two-sided)"], body)
    b5 = np.array([best5[s] for s in sc])
    maj = np.array([per["Strict majority"][s] for s in sc])
    for tag, b in [("the best of five", b5), ("the best of five + strict majority", np.maximum(b5, maj))]:
        c = Counter(cmp(x, y) for x, y in zip(a, b))
        R.p(f"- Pipeline against {tag}: {c['W']}/{c['T']}/{c['L']} (mean F1 {a.mean():.3f} vs {b.mean():.3f})")
    c = Counter(cmp(x, y) for x, y in zip(maj, b5))
    R.p(f"- Strict majority against the best of five: {c['W']}/{c['T']}/{c['L']}")
    losses = [(s.replace("benchmark_", "").replace("real_", ""), pipe[s], per["Strict majority"][s])
              for s in sc if cmp(pipe[s], per["Strict majority"][s]) == "L"]
    R.p(f"- Pipeline loses to the strict majority on: {losses}")
    R.set("ensemble", {k: {"mean_f1": float(np.mean([per[k][s] for s in sc]))} for k, _ in rules})


def shortfall(R, rows):
    """
    Shortfall of each method to the best of the six in each scenario: F1 of the
    best method that ran on the scenario minus the F1 of the method (zero when it
    is the best). Measures regularity, not wins: on how many scenarios each method
    stays close to the best, on how many it falls far behind, and the worst case.
    Reported on all scenarios, on the development ones, on the external ones
    (HELDOUT) and without CausalTime.
    Trivial floor per scenario: p = n_causal / n_universe; F1 = 2p / (1 + p).
    """
    P = PIPELINE_MODEL
    methods = [P] + [m for m, _ in BASELINES]
    F4 = lambda mm, m: round(f1(mm[m]["tp"], mm[m]["fp"], mm[m]["fn"]), 4)
    F = {s: {m: F4(mm, m) for m in methods if m in mm} for s, mm in rows.items() if P in mm}
    floor = {}
    for s in F:
        r = rows[s][P]
        p = r["n_causal"] / r["n_universe"]
        floor[s] = 2 * p / (1 + p)
    external = lambda s: group_of(s) in HELDOUT
    subsets = [("All scenarios", sorted(F)),
               ("Development", sorted(s for s in F if not external(s))),
               ("External (outside development)", sorted(s for s in F if external(s))),
               ("Without CausalTime", sorted(s for s in F if "causaltime" not in s))]
    data = {}
    for name, sc in subsets:
        R.p(f"\n### {name} ({len(sc)} scenarios)\n")
        body = []
        for m in methods:
            ss = [s for s in sc if m in F[s]]
            if not ss:
                continue
            f = np.array([F[s][m] for s in ss])
            gap = np.array([max(F[s].values()) for s in ss]) - f
            fl = np.array([floor[s] for s in ss])
            d = dict(n=len(ss), mean_f1=float(f.mean()), mean_gap=float(gap.mean()),
                     max_gap=float(gap.max()), within_005=int((gap <= 0.05 + EPS).sum()),
                     within_010=int((gap <= 0.10 + EPS).sum()), beyond_020=int((gap > 0.20 + EPS).sum()),
                     zeros=int((f == 0).sum()), below_floor=int((f < fl - EPS).sum()),
                     best_or_tied=int((gap <= EPS).sum()))
            data.setdefault(name, {})[LABEL[m]] = d
            body.append([LABEL[m], d["n"], f"{d['mean_f1']:.3f}", f"{d['mean_gap']:.3f}", f"{d['max_gap']:.3f}",
                         d["within_005"], d["within_010"], d["beyond_020"], d["zeros"], d["below_floor"],
                         d["best_or_tied"]])
        R.table(["Method", "Scenarios", "Mean F1", "Mean shortfall", "Worst shortfall", "Within 0.05 of the best",
                 "Within 0.10 of the best", "More than 0.20 behind", "F1 = 0", "Below the trivial floor",
                 "Best or tied"], body)
        best = np.array([max(F[s][m] for m, _ in BASELINES if m in F[s]) for s in sc])
        a = np.array([F[s][P] for s in sc])
        c = Counter(cmp(x, y) for x, y in zip(a, best))
        R.p(f"\nPipeline against the best baseline: {c['W']} / {c['T']} / {c['L']} "
            f"(mean F1 {a.mean():.3f} vs {best.mean():.3f})")
    R.p("\n### Scenarios on which the pipeline is not the best\n")
    body = []
    for s in sorted(F, key=lambda s: F[s][P] - max(F[s].values())):
        g = max(F[s].values()) - F[s][P]
        if g > EPS:
            b = max(F[s], key=F[s].get)
            body.append([s.replace("benchmark_", "").replace("real_", ""), f"{F[s][P]:.3f}",
                         f"{F[s][b]:.3f}", LABEL[b], f"{g:.3f}", f"{floor[s]:.3f}"])
    R.table(["Scenario", "Pipeline", "Best", "Method", "Shortfall", "Trivial floor"], body)
    R.set("shortfall", data)


def ranking(R, rows):
    """
    Pipeline as a single choice against each baseline as a single choice.
    Mean rank of the methods (1 = highest F1; ties receive the mean rank),
    Friedman test and Nemenyi critical difference at 5% (Demsar, 2006):
    two methods are separated when their mean ranks differ by more than the CD.
    Three units, because VARLiNGAM does not run on every scenario and because
    the realizations of the same group are not independent:
      - scenario, six methods: only the scenarios on which all six ran;
      - scenario, five methods: every scenario, without VARLiNGAM;
      - group, six methods: mean F1 of each method in the group (same scenarios
        as the first), one value per group.
    Then the one-to-one comparison per scenario in each subset (the one of
    section 3c is only the one over all scenarios), with Holm over the five
    baselines.
    """
    from scipy.stats import binomtest, friedmanchisquare, rankdata, studentized_range
    P = PIPELINE_MODEL
    six = [P] + [m for m, _ in BASELINES]
    five = [m for m in six if m != "varlingam"]
    F4 = lambda mm, m: round(f1(mm[m]["tp"], mm[m]["fp"], mm[m]["fn"]), 4)
    external = lambda s: group_of(s) in HELDOUT
    every = sorted(s for s in rows if P in rows[s])
    subsets = [("All scenarios", every),
               ("Development", [s for s in every if not external(s)]),
               ("Held-out", [s for s in every if external(s)])]

    def block(M, methods):
        """M: units x methods. Mean rank, Friedman p and Nemenyi CD."""
        rk = np.array([rankdata(-r) for r in M]).mean(axis=0)
        k, n = M.shape[1], M.shape[0]
        cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
        return dict(n=n, rank={LABEL[m]: float(x) for m, x in zip(methods, rk)},
                    friedman_p=float(friedmanchisquare(*M.T).pvalue), cd=float(cd))

    data = {}
    for name, sc in subsets:
        with6 = [s for s in sc if all(m in rows[s] for m in six)]
        byg = defaultdict(list)
        for s in with6:
            byg[group_of(s)].append([F4(rows[s], m) for m in six])
        blocks = [("scenario, six methods", six, np.array([[F4(rows[s], m) for m in six] for s in with6])),
                  ("scenario, five methods (without VARLiNGAM)", five,
                   np.array([[F4(rows[s], m) for m in five] for s in sc])),
                  ("group, six methods", six, np.array([np.mean(v, axis=0) for v in byg.values()]))]
        res = {tag: block(M, methods) for tag, methods, M in blocks}
        data[name] = res
        R.p(f"\n### {name} ({len(sc)} scenarios): mean rank\n")
        body = []
        for m in six:
            row = [LABEL[m]]
            for tag, _, _ in blocks:
                b = res[tag]
                if LABEL[m] not in b["rank"]:
                    row.append("-")
                    continue
                diff = b["rank"][LABEL[m]] - b["rank"][LABEL[P]]
                row.append(f"{b['rank'][LABEL[m]]:.2f}" + ("" if m == P else " *" if diff > b["cd"] else ""))
            body.append(row)
        body.append(["Units (n)"] + [res[tag]["n"] for tag, _, _ in blocks])
        body.append(["Friedman p"] + [f"{res[tag]['friedman_p']:.2g}" for tag, _, _ in blocks])
        body.append(["Nemenyi CD (5%)"] + [f"{res[tag]['cd']:.3f}" for tag, _, _ in blocks])
        R.table(["Method"] + [tag for tag, _, _ in blocks], body)
        R.p("\n\\* mean rank worse than the pipeline's by more than the CD.")

        R.p(f"\n### {name} ({len(sc)} scenarios): pipeline against each baseline, per scenario\n")
        body, pv = [], {}
        for m, lab in BASELINES:
            d = np.array([F4(rows[s], P) - F4(rows[s], m) for s in sc if m in rows[s]])
            c = Counter("W" if x > EPS else ("L" if x < -EPS else "T") for x in d)
            ps = binomtest(c["W"], c["W"] + c["L"]).pvalue if c["W"] + c["L"] else 1.0
            pv[lab] = wilcoxon(d).pvalue if np.any(np.abs(d) > EPS) else 1.0
            body.append([lab, len(d), f"{c['W']}/{c['T']}/{c['L']}", f"{d.mean():+.3f}", f"{ps:.2g}", f"{pv[lab]:.2g}"])
        holm, acc = {}, 0.0
        for i, (lab, p) in enumerate(sorted(pv.items(), key=lambda kv: kv[1])):
            acc = max(acc, min(1.0, p * (len(pv) - i)))
            holm[lab] = acc
        for r in body:
            r.append(f"{holm[r[0]]:.2g}")
        R.table(["Against", "n", "Pipeline higher/equal/lower", "Mean dF1", "Sign test p",
                 "Wilcoxon p (two-sided)", "Holm p (5 baselines)"], body)
        data[name]["one_to_one"] = {r[0]: dict(n=r[1], wtl=r[2], dF1=r[3], sign_p=r[4], wilcoxon_p=r[5], holm_p=r[6])
                                    for r in body}
    R.set("ranking", data)


def relative_shortfall(R, rows):
    """
    The shortfall of section 3e is an absolute F1 difference: where the best F1
    is low it is small for any method. Two readings that do not depend on that,
    on the same subsets:
      - relative shortfall: (F1 of the best - F1 of the method) / F1 of the best,
        zero when the best F1 is zero; mean, and scenarios within 10% and 25%;
      - the measures of section 3e only on the scenarios on which the best of the
        methods that ran reaches the trivial floor (2p / (1 + p)).
    At the end, RCAEval on its own, where the best F1 does not exceed 0.22.
    """
    P = PIPELINE_MODEL
    methods = [P] + [m for m, _ in BASELINES]
    F4 = lambda mm, m: round(f1(mm[m]["tp"], mm[m]["fp"], mm[m]["fn"]), 4)
    F = {s: {m: F4(mm, m) for m in methods if m in mm} for s, mm in rows.items() if P in mm}
    floor = {}
    for s in F:
        p = rows[s][P]["n_causal"] / rows[s][P]["n_universe"]
        floor[s] = 2 * p / (1 + p)
    external = lambda s: group_of(s) in HELDOUT
    above = lambda s: max(F[s].values()) >= floor[s] - EPS
    subsets = [("All scenarios", sorted(F)),
               ("Development", sorted(s for s in F if not external(s))),
               ("Held-out", sorted(s for s in F if external(s))),
               ("RCAEval", sorted(s for s in F if group_of(s) == "rcaeval"))]
    R.p(f"Scenarios on which the best F1 is zero: {sum(1 for s in F if max(F[s].values()) <= EPS)}")

    def measures(ss, m):
        f = np.array([F[s][m] for s in ss])
        best = np.array([max(F[s].values()) for s in ss])
        gap = best - f
        rel = np.divide(gap, best, out=np.zeros_like(gap), where=best > EPS)
        return dict(n=len(ss), mean_gap=float(gap.mean()), mean_rel=float(rel.mean()),
                    within_010=int((gap <= 0.10 + EPS).sum()), rel_010=int((rel <= 0.10 + EPS).sum()),
                    rel_025=int((rel <= 0.25 + EPS).sum()))

    data = {}
    for name, sc in subsets:
        ac = [s for s in sc if above(s)]
        R.p(f"\n### {name} ({len(sc)} scenarios; the best method reaches the floor on {len(ac)})\n")
        body = []
        for m in methods:
            ss, sa = [s for s in sc if m in F[s]], [s for s in ac if m in F[s]]
            if not ss:
                continue
            d, a = measures(ss, m), (measures(sa, m) if sa else None)
            data.setdefault(name, {})[LABEL[m]] = dict(all=d, above_floor=a)
            body.append([LABEL[m], d["n"], f"{d['mean_gap']:.3f}", f"{100 * d['mean_rel']:.1f}%", d["within_010"],
                         d["rel_010"], d["rel_025"]]
                        + ([a["n"], f"{a['mean_gap']:.3f}", f"{100 * a['mean_rel']:.1f}%", a["within_010"],
                            a["rel_010"], a["rel_025"]] if a else ["-"] * 6))
        R.table(["Method", "Scenarios", "Mean shortfall", "Mean relative shortfall", "Within 0.10 of the best",
                 "Within 10% of the best", "Within 25% of the best", "Above the floor: scenarios", "Mean shortfall",
                 "Mean relative shortfall", "Within 0.10 of the best", "Within 10% of the best",
                 "Within 25% of the best"], body)
        best = [max(F[s].values()) for s in sc]
        R.p(f"\nBest F1 per scenario: minimum {min(best):.3f}, median {np.median(best):.3f}, maximum {max(best):.3f}")
    R.set("relative_shortfall", data)


def provenance(R, rows):
    tp, fp = Counter(), Counter()
    for scen in sorted(rows):
        df, truth = pair_table(PIPELINE_RUN, scen, PIPELINE_FOLDER[WINDOW])
        if df is None or len(df) == 0:
            continue
        T = set(truth[truth].index)
        acc = df[df["significant"].astype(str) == "True"]
        seen = set()
        for _, r in acc.iterrows():
            k = (r["source_entity"], r["target_entity"])
            if k in seen:
                continue
            seen.add(k)
            (tp if k in T else fp)[r["filter_method"]] += 1
    tot_fp = sum(fp.values()) or 1
    body = []
    for m in sorted(set(tp) | set(fp)):
        n = tp[m] + fp[m]
        body.append([m, n, tp[m], fp[m], fmt(tp[m] / n, 3) if n else "-", f"{100 * fp[m] / tot_fp:.1f}%"])
    R.table(["Path", "Accepted", "TP", "FP", "Precision", "Share of the FP"], body)


def effect_coverage(R, rows):
    """
    Does every reported pair (significant=True) have a complete effect estimate?
    Counts, over the rows of original.csv (one per property pair and lag):
    reported pairs, how many are lag 0, how many lack finite values for the five
    fields of the estimate (theta, standard error, 95% CI, p-value), and how many
    come out with the cascade's fallback estimate (cascade_confidence == 'low'),
    per family. For the lag-0 adjacencies, reported in both directions, checks
    whether the sign of theta agrees in both. Covers the 90 scenarios: each one
    reads the original.csv of the run the pipeline came from.
    """
    fields = ["causal_effect", "standard_error", "ci_95_lower", "ci_95_upper", "pvalue"]
    n = n0 = missing = cand = npairs = 0
    low = Counter()
    crosses = Counter()
    adj = {}
    for scen in sorted(rows):
        if PIPELINE_MODEL not in rows[scen]:
            continue
        run = rows[scen][PIPELINE_MODEL]["run"].rsplit(":", 1)[0]
        df, _truth = pair_table(run, scen, PIPELINE_FOLDER[WINDOW])
        if df is None or len(df) == 0:
            continue
        cand += int(num(df["causal_effect"]).notna().sum())
        s = df[df["significant"].astype(str) == "True"]
        lag = num(s["lag"])
        n += len(s); n0 += int((lag == 0).sum())
        npairs += len(set(zip(s["source_entity"], s["target_entity"])))
        est = s[fields].apply(num)
        missing += int((~np.isfinite(est)).any(axis=1).sum())
        fb = s[s["cascade_confidence"].astype(str) == "low"]
        for _, r in fb.iterrows():
            kind = "lag 0" if int(r["lag"]) == 0 else "lagged"
            key = (family_of(group_of(scen)), kind)
            low[key] += 1
            if float(r["ci_95_lower"]) <= 0.0 <= float(r["ci_95_upper"]):
                crosses[kind] += 1
        for _, r in s[lag == 0].iterrows():
            a_ = f"{r['source_entity']}__{r['source_property']}"
            b_ = f"{r['target_entity']}__{r['target_property']}"
            adj.setdefault((scen, tuple(sorted((a_, b_)))), []).append(float(r["causal_effect"]))
    R.p(f"- Reported pairs: {n:,} ({n - n0:,} lagged + {n0} lag 0); "
        f"without a complete estimate (theta, SE, 95% CI, p): **{missing}**")
    R.p(f"- Distinct ordered entity pairs among the reported ones: {npairs:,}")
    R.p(f"- Candidates with theta estimated by Stage 6 (including rejected ones): {cand:,}")
    tot = sum(low.values())
    R.p(f"- Reported with the fallback estimate (cascade_confidence = low): **{tot}** -> "
        + ", ".join(f"{k[0]} ({k[1]}): {v}" for k, v in sorted(low.items())))
    R.p(f"  - 95% CI includes zero on {crosses['lagged']} of the lagged ones and {crosses['lag 0']} of the lag-0 ones")
    both = [v for v in adj.values() if len(v) == 2]
    R.p(f"- Lag-0 adjacencies: {len(adj)} ({len(both)} reported in both directions); "
        f"sign of theta agrees in both directions on {sum(np.sign(v[0]) == np.sign(v[1]) for v in both)} of {len(both)}")
    R.set("effect_coverage", dict(reported=n, lag0=n0, missing=missing, fallback=tot,
                                  fallback_by={f"{k[0]}|{k[1]}": v for k, v in low.items()},
                                  ci_crosses_zero=dict(crosses), lag0_adjacencies=len(adj)))


def gpdc(R, rows):
    sc = [s for s, mm in rows.items() if "pcmci_gpdc" in mm and PIPELINE_MODEL in mm]
    F = lambda s, m: f1(rows[s][m]["tp"], rows[s][m]["fp"], rows[s][m]["fn"])
    P = PIPELINE_MODEL
    R.p(f"Realizations: {len(sc)}")
    body = []
    for m in ["pcmci_tigramite", "pcmciplus", "pcmci_gpdc", "pcmciplus_gpdc"]:
        c = Counter(cmp(F(s, P), F(s, m)) for s in sc if m in rows[s])
        body.append([LABEL[m], c["W"], c["T"], c["L"], fmt(np.mean([F(s, m) for s in sc if m in rows[s]]), 3)])
    R.table(["Pipeline vs", "Pipeline higher", "Tie", "Pipeline lower", "Mean F1 of the baseline"], body)
    R.p(f"Mean F1 of the pipeline: {np.mean([F(s, P) for s in sc]):.3f}")
    for a, b in [("pcmci_tigramite", "pcmci_gpdc"), ("pcmciplus", "pcmciplus_gpdc")]:
        c = Counter(cmp(F(s, b), F(s, a)) for s in sc)
        R.p(f"- {LABEL[a]} -> {LABEL[b]}: up {c['W']}, equal {c['T']}, down {c['L']}")
    before, after, changed = Counter(), Counter(), []
    fam = defaultdict(lambda: [Counter(), Counter()])
    for s in sc:
        parcorr = max(F(s, "pcmci_tigramite"), F(s, "pcmciplus"))
        allm = max(parcorr, F(s, "pcmci_gpdc"), F(s, "pcmciplus_gpdc"))
        x, y = cmp(F(s, P), parcorr), cmp(F(s, P), allm)
        before[x] += 1; after[y] += 1
        fam[family_of(group_of(s))][0][x] += 1; fam[family_of(group_of(s))][1][y] += 1
        if x != y:
            changed.append((s.replace("benchmark_", ""), x, y, round(F(s, P), 4), round(F(s, "pcmci_gpdc"), 4),
                            round(F(s, "pcmciplus_gpdc"), 4)))
    R.p(f"- Against the PCMCI family: ParCorr only {before['W']}/{before['T']}/{before['L']}; "
        f"with GPDC {after['W']}/{after['T']}/{after['L']}")
    for c in changed:
        R.p(f"  - changed: {c}")
    for m in ["pcmci_gpdc", "pcmciplus_gpdc", "pcmci_tigramite", "pcmciplus"]:
        every = sum(1 for s in sc if m in rows[s] and rows[s][m]["n_pred"] == rows[s][m]["n_universe"])
        none_ = sum(1 for s in sc if m in rows[s] and rows[s][m]["n_pred"] == 0)
        R.p(f"- {LABEL[m]}: declares every pair on {every}, none on {none_}")


if __name__ == "__main__":
    for WINDOW in SOURCES:
        main()
