"""
Generates the numbers specific to the five held-out datasets that came in last
(BEELINE, DREAM3, RCAEval, EasyVista, Causal Chambers; Sections 4.2, 4.5 and 5.2,
Table S15) from the held-out runs (main_heldout.py), in a single report
(Extra_Numbers.md). The aggregates of the paper (90 scenarios) come from
paper_numbers.py; from here come the pipeline's path per scenario (section 5),
the complete run of the thirteen candidate families (section 6) and Table S15.

Sources:
  - HELDOUT_RUN (origins.py): the held-out runs. In the paper,
    Results_Extra/Run_20261008_221315 (BEELINE HSC and VSC, DREAM3, RCAEval,
    EasyVista; 28 scenarios) and Results_Extra/Run_20261009_causal_chamber
    (Causal Chambers, 4 scenarios). Only the families of FAMILIES are included.
  - FULL_RUN (CANDIDATES_RUN; optional): the run with every extra source (13 families, 58 scenarios),
    from which the five were chosen; used only in section 6 (Table S15).
  - PAPER_RUN (MAIN_RUN): the 58 scenarios of the paper, for the aggregate of section 4.

Conventions the same as paper_numbers.py's (Section 4.4 of the paper): F1
recomputed from TP/FP/FN; shortfall to the best = F1 of the best of the six methods
minus the method's, on the scenarios on which the method ran (status ok); trivial
floor F1 = 2p/(1+p), p = n_causal/n_universe.

    python generators/extra_numbers.py
"""
import os
import re
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import origins as O  # noqa: E402

OUT = O.out("Extra_Numbers.md")

HELDOUT_RUNS = O.heldout_runs()
FULL_RUN = O.CANDIDATES_RUN
PAPER_RUN = O.require("MAIN_RUN")
if not HELDOUT_RUNS:
    O.require("HELDOUT_RUN")

MODELS = [("patch", "Pipeline"), ("pcmci_tigramite", "PCMCI"), ("pcmciplus", "PCMCI+"),
          ("dynotears", "DYNOTEARS"), ("cuts_plus", "CUTS+"), ("varlingam", "VARLiNGAM")]
BASELINES = [m for m, _ in MODELS[1:]]
LABEL = dict(MODELS)
EPS = 1e-9

# Order and label of the families of the additional block
FAMILIES = [("beeline_HSC", "BEELINE HSC"), ("beeline_VSC", "BEELINE VSC"), ("dream3", "DREAM3"),
            ("rcaeval", "RCAEval"), ("easyvista", "EasyVista"),
            ("causal_chamber_wt", "Causal Chambers, wind tunnel"),
            ("causal_chamber_lt", "Causal Chambers, light tunnel")]
FULL_FAMILIES = [("beeline_GSD", "BEELINE GSD"), ("beeline_mCAD", "BEELINE mCAD"),
                 ("causal_rivers", "CausalRivers"), ("dream4", "DREAM4"), ("ess_cryo", "ESS cryogenics"),
                 ("metr_la", "METR-LA"), ("nino34", "Niño 3.4 pairs"), ("tuebingen", "Tübingen pairs")]
# Scenarios of the paper left out of the aggregate (Section 4.2)
PAPER_SKIP = ("dream3",)


def family_of(scen):
    s = re.sub(r"^(benchmark_|real_)", "", scen)
    for k, _ in FAMILIES + FULL_FAMILIES:
        if s.startswith(k):
            return k
    return s


def f1(tp, fp, fn):
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def load(run, window):
    """rows[scen][model] = dict(tp, fp, fn, f1, p, r) for status ok only; meta[scen]."""
    root = run
    rows, meta = {}, {}
    for scen in sorted(os.listdir(root)):
        p = os.path.join(root, scen, "validation_summary.csv")
        if not os.path.isfile(p):
            continue
        df = pd.read_csv(p).set_index("model")
        rows[scen] = {}
        for m, _ in MODELS:
            name = m + "_tau10" if window == "10" else m
            if name in df.index and str(df.loc[name, "status"]) == "ok":
                tp, fp, fn = (int(df.loc[name, c]) for c in ("TP", "FP", "FN"))
                rows[scen][m] = dict(tp=tp, fp=fp, fn=fn, f1=f1(tp, fp, fn),
                                     p=tp / (tp + fp) if tp + fp else 0.0,
                                     r=tp / (tp + fn) if tp + fn else 0.0)
        any_row = df.iloc[0]
        meta[scen] = dict(n_causal=int(any_row["n_causal"]), n_universe=int(any_row["n_universe"]), run=run)
    return rows, meta


def dims(run, scen):
    src = pd.read_csv(os.path.join(run, scen, "source.csv"), usecols=[0, 1])
    return src.iloc[:, 1].nunique(), src.iloc[:, 0].nunique()


def load_heldout(window):
    """load() over the held-out runs, only the families of FAMILIES; the first run that has the scenario wins."""
    rows, meta = {}, {}
    keys = [k for k, _ in FAMILIES]
    for run in HELDOUT_RUNS:
        r, m = load(run, window)
        for s in r:
            if family_of(s) in keys and s not in rows:
                rows[s], meta[s] = r[s], m[s]
    return rows, meta


def best_of(row, models):
    v = [row[m]["f1"] for m in models if m in row]
    return max(v) if v else None


def shortfall_table(rows, scens, title):
    out = [f"\n**{title}** ({len(scens)} scenarios)\n",
           "| Method | Scenarios | Mean F1 | Mean shortfall | Largest shortfall | Within 0.05 | Within 0.10 | More than 0.20 | Best or tied | F1 = 0 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for m, lab in MODELS:
        ran = [s for s in scens if m in rows[s]]
        gap = [best_of(rows[s], [x for x, _ in MODELS]) - rows[s][m]["f1"] for s in ran]
        fs = [rows[s][m]["f1"] for s in ran]
        out.append(f"| {lab} | {len(ran)} | {sum(fs) / len(fs):.3f} | {sum(gap) / len(gap):.3f} | {max(gap):.3f} | "
                   f"{sum(g <= 0.05 + EPS for g in gap)} | {sum(g <= 0.10 + EPS for g in gap)} | "
                   f"{sum(g > 0.20 + EPS for g in gap)} | {sum(g <= EPS for g in gap)} | {sum(f <= EPS for f in fs)} |")
    return out


def wtl(rows, scens):
    w = t = l = 0
    for s in scens:
        d = rows[s]["patch"]["f1"] - best_of(rows[s], BASELINES)
        w, t, l = w + (d > EPS), t + (abs(d) <= EPS), l + (d < -EPS)
    return w, t, l


def family_table(rows, meta, scens, fams):
    out = ["| Family (scenarios) | Series | Samples | True links | W / T / L | Mean F1, pipeline vs best baseline | Largest shortfall | Baseline ahead in the losses | Above the floor (pipeline / best baseline) |",
           "|---|---|---|---|---|---|---|---|---|"]
    tot = defaultdict(float)
    for k, lab in fams:
        ss = [s for s in scens if family_of(s) == k]
        if not ss:
            continue
        n, T = zip(*(dims(meta[s]["run"], s) for s in ss))
        gt = [meta[s]["n_causal"] for s in ss]
        w, t, l = wtl(rows, ss)
        fp_ = [rows[s]["patch"]["f1"] for s in ss]
        fb = [best_of(rows[s], BASELINES) for s in ss]
        gap = [best_of(rows[s], [x for x, _ in MODELS]) - rows[s]["patch"]["f1"] for s in ss]
        ahead = defaultdict(int)
        for s in ss:
            b = best_of(rows[s], BASELINES)
            if b > rows[s]["patch"]["f1"] + EPS:
                for m in BASELINES:
                    if m in rows[s] and abs(rows[s][m]["f1"] - b) <= EPS:
                        ahead[LABEL[m]] += 1
        floor = [2 * (meta[s]["n_causal"] / meta[s]["n_universe"]) / (1 + meta[s]["n_causal"] / meta[s]["n_universe"]) for s in ss]
        cp = sum(rows[s]["patch"]["f1"] > fl + EPS for s, fl in zip(ss, floor))
        cb = sum(best_of(rows[s], BASELINES) > fl + EPS for s, fl in zip(ss, floor))
        rng = lambda v: f"{min(v)}" if min(v) == max(v) else f"{min(v)}-{max(v)}"
        out.append(f"| {lab} ({len(ss)}) | {rng(n)} | {rng(T)} | {rng(gt)} | {w} / {t} / {l} | "
                   f"{sum(fp_) / len(ss):.3f} vs {sum(fb) / len(ss):.3f} | {max(gap):.3f} | "
                   f"{', '.join(f'{a} {c}' for a, c in sorted(ahead.items(), key=lambda x: -x[1])) or '-'} | {cp} / {cb} of {len(ss)} |")
        for key, val in (("n", len(ss)), ("w", w), ("t", t), ("l", l), ("fp", sum(fp_)), ("fb", sum(fb)),
                         ("cp", cp), ("cb", cb)):
            tot[key] += val
        tot["gap"] = max(tot["gap"], max(gap))
    n = int(tot["n"])
    out.append(f"| **Total ({n})** | | | | **{int(tot['w'])} / {int(tot['t'])} / {int(tot['l'])}** | "
               f"{tot['fp'] / n:.3f} vs {tot['fb'] / n:.3f} | {tot['gap']:.3f} | | {int(tot['cp'])} / {int(tot['cb'])} of {n} |")
    return out


def family_means(rows, scens, fams, fail_note=None):
    out = ["| Family (scenarios) | " + " | ".join(lab for _, lab in MODELS) + " | Mean shortfall of the pipeline | Largest shortfall of the pipeline |",
           "|---|" + "---|" * (len(MODELS) + 2)]
    for k, lab in fams:
        ss = [s for s in scens if family_of(s) == k]
        if not ss:
            continue
        cells = []
        for m, _ in MODELS:
            ran = [s for s in ss if m in rows[s]]
            cells.append("-" if not ran else f"{sum(rows[s][m]['f1'] for s in ran) / len(ran):.3f}"
                         + ("" if len(ran) == len(ss) else f" ({len(ran)})"))
        ran = [s for s in ss if "patch" in rows[s]]
        gap = [best_of(rows[s], [x for x, _ in MODELS]) - rows[s]["patch"]["f1"] for s in ran]
        out.append(f"| {lab} ({len(ss)}) | " + " | ".join(cells) + f" | {sum(gap) / len(gap):.3f} | {max(gap):.3f} |")
    return out


def path_diag(run, scen, window):
    """From the pipeline's original.csv: path of the coherence guard and share of the detections at lag 0."""
    folder = "patch_tau10" if window == "10" else "patch"
    d = pd.read_csv(os.path.join(run, scen, folder, "original.csv"), low_memory=False)
    sig = d[d["significant"].astype(str).str.lower().isin(["true", "1", "1.0"])]
    lag0 = int((pd.to_numeric(sig["lag"], errors="coerce") == 0).sum())
    regime = d["evidence_regime"].dropna().astype(str).mode()
    fb = d["evidence_fallback"].dropna().astype(str).mode() if "evidence_fallback" in d.columns else pd.Series(dtype=str)
    r2 = pd.to_numeric(d.get("evidence_past_r2"), errors="coerce").median() if "evidence_past_r2" in d.columns else float("nan")
    return dict(rows=len(d), sig=len(sig), lag0=lag0, regime=regime.iloc[0] if len(regime) else "-",
                fallback=fb.iloc[0] if len(fb) else "-", past_r2=r2,
                stage6=int(d["significant_stage6"].astype(str).str.lower().isin(["true", "1", "1.0"]).sum()))


def report(window):
    L = [f"\n# Window tau_max = {window}\n"]
    rows, meta = load_heldout(window)
    scens = sorted(rows, key=lambda s: ([k for k, _ in FAMILIES].index(family_of(s)), s))

    L.append("## 1. Shortfall to the best of the six methods, additional block (Section 5.2, Table 2c)")
    L += shortfall_table(rows, scens, "Additional block")
    nofail = [s for s in scens if all(m in rows[s] for m, _ in MODELS)]
    L.append(f"\nVARLiNGAM does not run on: {[s for s in scens if 'varlingam' not in rows[s]]}")
    w, t, l = wtl(rows, scens)
    fb = sum(best_of(rows[s], BASELINES) for s in scens) / len(scens)
    fp_ = sum(rows[s]["patch"]["f1"] for s in scens) / len(scens)
    L.append(f"\nPipeline against the best baseline of each scenario: **{w} W / {t} T / {l} L**; mean F1 {fp_:.3f} against {fb:.3f}.")
    L.append("\nPipeline against each baseline (F1 higher / equal / lower, on the scenarios on which both run):\n")
    for m in BASELINES:
        ss = [s for s in scens if m in rows[s]]
        d = [rows[s]["patch"]["f1"] - rows[s][m]["f1"] for s in ss]
        L.append(f"- {LABEL[m]}: {sum(x > EPS for x in d)} / {sum(abs(x) <= EPS for x in d)} / {sum(x < -EPS for x in d)} of {len(ss)}; mean difference {sum(d) / len(d):+.3f}")

    L.append("\n## 2. Per family (Sections 4.2 and 5.2, Table 2c)\n")
    L += family_table(rows, meta, scens, FAMILIES)
    L.append("\nMean F1 of each method per family (in parentheses, scenarios on which the method runs when it is not all of them):\n")
    L += family_means(rows, scens, FAMILIES)

    L.append("\n## 3. Per scenario (Table S15)\n")
    L.append("| Scenario | Pairs | Links | Floor | " + " | ".join(lab for _, lab in MODELS) + " | Pipeline shortfall |")
    L.append("|---|---|---|---|" + "---|" * (len(MODELS) + 1))
    for s in scens:
        p = meta[s]["n_causal"] / meta[s]["n_universe"]
        best = best_of(rows[s], [x for x, _ in MODELS])
        cells = []
        for m, _ in MODELS:
            if m not in rows[s]:
                cells.append("-")
            else:
                v = rows[s][m]["f1"]
                cells.append(f"**{v:.3f}**" if abs(v - best) <= EPS else f"{v:.3f}")
        L.append(f"| {re.sub(r'^(benchmark_|real_)', '', s)} | {meta[s]['n_universe']} | {meta[s]['n_causal']} | "
                 f"{2 * p / (1 + p):.3f} | " + " | ".join(cells) + f" | {best - rows[s]['patch']['f1']:.3f} |")

    L.append("\n## 4. With the 58 scenarios of the paper (58 + 32)\n")
    pr, _ = load(PAPER_RUN, window)
    pr = {s: v for s, v in pr.items() if not any(k in s for k in PAPER_SKIP)}
    allrows = {**pr, **rows}
    L += shortfall_table(allrows, sorted(pr), "The 58 of the paper (check against Table 2a)")
    L += shortfall_table(allrows, sorted(allrows), "58 + 32")
    w, t, l = wtl(allrows, sorted(allrows))
    L.append(f"\nPipeline against the best baseline of each scenario, 58 + 32: {w} W / {t} T / {l} L.")

    L.append("\n## 5. Path of the pipeline per scenario (Section 5.2; coherence guard and contemporaneous stage)\n")
    L.append("| Scenario | Regime | Path | Past R2 | Pairs at Stage 6 | Reported rows | At lag 0 | TP | FP | FN |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    agg = defaultdict(lambda: defaultdict(int))
    for s in scens:
        d = path_diag(meta[s]["run"], s, window)
        r = rows[s]["patch"]
        L.append(f"| {re.sub(r'^(benchmark_|real_)', '', s)} | {d['regime']} | {d['fallback']} | "
                 f"{'-' if pd.isna(d['past_r2']) else format(d['past_r2'], '.2f')} | {d['stage6']} | {d['sig']} | {d['lag0']} | "
                 f"{r['tp']} | {r['fp']} | {r['fn']} |")
        a = agg[family_of(s)]
        a["sig"] += d["sig"]; a["lag0"] += d["lag0"]; a["n"] += 1
        a["fallback6"] += d["fallback"] == "stage6"
    L.append("\nPer family: rows reported at lag 0 / total; scenarios through the Stage 6 fallback:\n")
    for k, lab in FAMILIES:
        a = agg[k]
        L.append(f"- {lab}: {a['lag0']} / {a['sig']} at lag 0; Stage 6 fallback on {a['fallback6']} of {a['n']}")
    return L


def full_run():
    if not FULL_RUN:
        return ["\n# 6. Run with every extra source: CANDIDATES_RUN not set in origins.py\n"]
    L = ["\n# 6. Run with every extra source (Table S15b), tau_max = 10\n",
         f"`{O.label(FULL_RUN)}`: 58 scenarios, 13 families. The five of the additional block were chosen after this run.\n"]
    rows, meta = load(FULL_RUN, "10")
    scens = sorted(rows)
    miss = [s for s in scens if "patch" not in rows[s]]
    L.append(f"Pipeline without a result (error status in the run; counted as F1 = 0 below): {miss}\n")
    for s in miss:
        rows[s]["patch"] = dict(tp=0, fp=0, fn=meta[s]["n_causal"], f1=0.0, p=0.0, r=0.0)
    fams = sorted(FAMILIES[:5] + FULL_FAMILIES, key=lambda x: x[1])
    L += family_means(rows, scens, fams)
    L += shortfall_table(rows, scens, "Every extra source")
    kept = [s for s in scens if family_of(s) in [k for k, _ in FAMILIES]]
    L += shortfall_table(rows, kept, "The five chosen families, in this run")
    L += shortfall_table(rows, [s for s in scens if s not in kept], "The eight families not chosen")
    return L


S15_OUT = O.out("Table_S15_Extra.md")


def _en_shortfall(rows, scens, label):
    out = []
    for m, lab in MODELS:
        ran = [s for s in scens if m in rows[s]]
        gap = [best_of(rows[s], [x for x, _ in MODELS]) - rows[s][m]["f1"] for s in ran]
        fs = [rows[s][m]["f1"] for s in ran]
        out.append(f"| {label if m == 'patch' else ''} | {lab} | {len(ran)} | {sum(fs) / len(fs):.3f} | {sum(gap) / len(gap):.3f} | "
                   f"{max(gap):.3f} | {sum(g <= 0.05 + EPS for g in gap)} | {sum(g <= 0.10 + EPS for g in gap)} | "
                   f"{sum(g > 0.20 + EPS for g in gap)} |")
    return out


def supplement_table():
    """Table S15, ready for the supplement (Table_S15_Extra.md)."""
    L = ["## Table S15 — Per-Realization Results: BEELINE, DREAM3, RCAEval, EasyVista, and the Causal Chambers (32 realizations), and the Complete Run of Thirteen Candidate Families", "",
         "<!-- DATA: Table_S15_Extra.md, generated by generators/extra_numbers.py -->", "",
         "*Companion to Sections 4.5 and 5.2. Sources: the held-out runs "
         + ", ".join(f"`{O.label(r)}`" for r in HELDOUT_RUNS)
         + (f", and `{O.label(FULL_RUN)}` (the complete run, parts (3)–(4))" if FULL_RUN else "")
         + ", with the reported configuration "
         "(Tables S2 and S4) and the baseline settings of Table S6. F1 at entity-pair level (Section 4.4); the trivial floor is "
         "2p/(1 + p) for the prevalence p of true pairs in the scenario; the shortfall is the F1 of the best of the six methods "
         "minus the pipeline's. Bold marks the best method of the scenario; a dash, a method that did not run (VARLiNGAM's order "
         "selection fails on two RCAEval cases).*".replace("\\", "/"), ""]
    for tag, window in (("1", "10"), ("2", "24")):
        rows, meta = load_heldout(window)
        scens = sorted(rows, key=lambda s: ([k for k, _ in FAMILIES].index(family_of(s)), s))
        L += [f"**({tag}) τ_max = {window}**", "",
              "| Scenario | Candidate pairs | True links | Trivial floor | " + " | ".join(lab for _, lab in MODELS) + " | Pipeline shortfall |",
              "|---|---|---|---|" + "---|" * (len(MODELS) + 1)]
        for s in scens:
            p = meta[s]["n_causal"] / meta[s]["n_universe"]
            best = best_of(rows[s], [x for x, _ in MODELS])
            cells = ["–" if m not in rows[s] else
                     (f"**{rows[s][m]['f1']:.3f}**" if abs(rows[s][m]["f1"] - best) <= EPS else f"{rows[s][m]['f1']:.3f}")
                     for m, _ in MODELS]
            L.append(f"| {re.sub(r'^(benchmark_|real_)', '', s)} | {meta[s]['n_universe']} | {meta[s]['n_causal']} | "
                     f"{2 * p / (1 + p):.3f} | " + " | ".join(cells) + f" | {best - rows[s]['patch']['f1']:.3f} |")
        L.append("")

    if not FULL_RUN:
        with open(S15_OUT, "w", encoding="utf-8") as fh:
            fh.write(O.canonical("\n".join(L)))
        print("written (without CANDIDATES_RUN):", S15_OUT)
        return
    rows, meta = load(FULL_RUN, "10")
    scens = sorted(rows)
    miss = [s for s in scens if "patch" not in rows[s]]
    for s in miss:
        rows[s]["patch"] = dict(tp=0, fp=0, fn=meta[s]["n_causal"], f1=0.0, p=0.0, r=0.0)
    kept_keys = [k for k, _ in FAMILIES]
    fams = sorted(FAMILIES[:5] + FULL_FAMILIES, key=lambda x: x[1].lower())
    L += ["**(3) The complete run of the thirteen candidate families (58 scenarios), τ_max = 10: mean F1 by family.** "
          "*† marks the five families that are part of the evaluation suite (main-text Section 4.5). In parentheses, the number of "
          "scenarios on which a method runs when it is not all of them.*", "",
          "| Family (scenarios) | " + " | ".join(lab for _, lab in MODELS) + " | Pipeline: mean shortfall | Pipeline: largest shortfall |",
          "|---|" + "---|" * (len(MODELS) + 2)]
    for k, lab in fams:
        ss = [s for s in scens if family_of(s) == k]
        cells = []
        for m, _ in MODELS:
            ran = [s for s in ss if m in rows[s]]
            cells.append(f"{sum(rows[s][m]['f1'] for s in ran) / len(ran):.3f}" + ("" if len(ran) == len(ss) else f" ({len(ran)})"))
        gap = [best_of(rows[s], [x for x, _ in MODELS]) - rows[s]["patch"]["f1"] for s in ss]
        L.append(f"| {lab}{' †' if k in kept_keys else ''} ({len(ss)}) | " + " | ".join(cells)
                 + f" | {sum(gap) / len(gap):.3f} | {max(gap):.3f} |")
    kept = [s for s in scens if family_of(s) in kept_keys]
    L += ["", "**(4) Shortfall of each method to the best of the six in the complete run, τ_max = 10.**", "",
          "| Set | Method | Scenarios | Mean F1 | Mean shortfall | Largest shortfall | Within 0.05 of the best | Within 0.10 of the best | More than 0.20 behind |",
          "|---|---|---|---|---|---|---|---|---|"]
    L += _en_shortfall(rows, scens, "All thirteen families (58)")
    L += _en_shortfall(rows, kept, "The five in the suite (28)")
    L += _en_shortfall(rows, [s for s in scens if s not in kept], "The eight outside the suite (30)")
    names = ", ".join(re.sub(r"^(benchmark_|real_)", "", s) for s in miss)
    L += ["", f"*Notes. The pipeline returned an empty result on four scenarios of the complete run ({names}); they are scored "
          "as F1 = 0 in parts (3) and (4). The five families in the suite were run again for parts (1) and (2); the pipeline's F1 "
          "differs from the complete run in the third decimal on two RCAEval cases. The eight families outside the suite are the "
          "BEELINE networks GSD and mCAD (Pratapa et al., 2020); CausalRivers, river discharge in Bavaria with the river network "
          "as ground truth (Stein et al., 2025); the ten-gene DREAM4 in silico networks (Marbach et al., 2010); the cryoplant of "
          "the European Spallation Source, with a graph built by the plant's experts (Mogensen, Rathsman & Nilsson, 2024); "
          "METR-LA traffic speeds, with road-network connectivity as ground truth (Li et al., 2018); two pairs formed by the "
          "Niño 3.4 index and a regional climate series, with a direction taken from the literature; and ten of the Tübingen "
          "cause-effect pairs (Mooij et al., 2016), which are cross-sectional and have two variables each.*", "", "---", ""]
    with open(S15_OUT, "w", encoding="utf-8") as fh:
        fh.write(O.canonical("\n".join(L)))
    print("written:", S15_OUT)


if __name__ == "__main__":
    supplement_table()
    out = ["# Numbers of the additional families -- generated by generators/extra_numbers.py",
           "Held-out runs: " + ", ".join(f"`{O.label(r)}`" for r in HELDOUT_RUNS) + "; "
           f"every source: `{O.label(FULL_RUN) if FULL_RUN else '-'}`; paper: `{O.label(PAPER_RUN)}`."]
    for w in ("10", "24"):
        out += report(w)
    out += full_run()
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(O.canonical("\n".join(out) + "\n"))
    print("written:", OUT)
