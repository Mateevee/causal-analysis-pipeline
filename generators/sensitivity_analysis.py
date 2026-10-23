"""
Sensitivity of the pipeline to its cut-offs, from the run of
main_sensitivity.py (SENSITIVITY_RUN in origins.py). The protocol (grids, joint
draw, tolerance criterion) is the one of Config/sensitivity.py, locked in
<RUN>/protocol.json before the run, and is read from there: nothing of the
criterion is defined in this script.

  0. Scope: scenarios of the run against those of the paper; runs with errors.
  1. Control (default configuration) against the reported run.
  2. One cut-off at a time: each value of the grid of each cut-off.
  3. Joint variation (Latin hypercube): protocol criterion, distribution,
     conclusions of the paper that hold, responsible cut-offs, interaction.
  4. Scenarios: those on the boundary of the mark and those that change most.
  5. Criterion at other marks (post hoc analysis, outside the protocol).
  6. Seeds.
  7. Walker: the support floor and the outcome R2 (sentence of Section 5.5).

Scoring of the paper (Section 4.4): ordered pair of entities, any lag.
Baselines: those of the reported run (paper_numbers.load, window 10), without variation.

Criterion (protocol), with delta = max(tolerance, range of the mean F1 across
the seeds): R1 mean F1 within delta of the default on all scenarios, on the
development suite and on the held-out datasets; R2 the number of scenarios within
`within_mark` of the best of the six methods drops by at most `max_within_drop`;
R3 the mean shortfall of the pipeline to the best of the six stays below that of
every baseline.

Output: Sensitivity.md (report) and Table_S20_Sensitivity.md (tables for the
supplement).
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, studentized_range

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_numbers as na

import origins as O  # noqa: E402
RUN = O.require("SENSITIVITY_RUN")   # in the paper, Results_Sensitivity/Run_20261010_233817
OUT = os.path.join(na.OUT_DIR, "Sensitivity.md")
OUT_S = os.path.join(na.OUT_DIR, "Table_S20_Sensitivity.md")
MARKS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]   # section 5
WALKER = "real_tigramite_climate_walker"        # section 7
EPS = na.EPS
BL = [m for m, _ in na.BASELINES]
FIVE = [m for m in BL if m != "varlingam"]
short = lambda s: s.replace("benchmark_", "").replace("real_", "")


def load_run():
    """C[scenario][config] = (tp, fp, fn) of seed 0; seeds[offset][scenario] = the same
    for the default configuration; errors = runs with a status other than ok."""
    C, seeds, errors = {}, {}, 0
    for path in sorted(glob.glob(os.path.join(RUN, "runs", "*.jsonl"))):
        s = os.path.basename(path)[:-6]
        gt = pd.read_csv(os.path.join(RUN, "gt", s + ".csv"))
        for c in ("source_entity", "target_entity"):
            gt[c] = gt[c].astype(str).str.strip()
        pos = gt["causal_gt"].astype(str).str.lower().isin(["true", "1"])
        T = set(map(tuple, gt.loc[pos, ["source_entity", "target_entity"]].values))
        U = set(map(tuple, gt[["source_entity", "target_entity"]].values))
        C[s] = {}
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["status"] != "ok":
                errors += 1
                continue
            S = {(a, b) for a, b, _ in r["pairs"]} & U
            c = (len(S & T), len(S - T), len(T - S))
            if r["seed"] == 0:
                C[s][r["config"]] = c
            elif r["config"] == "default":
                seeds.setdefault(r["seed"], {})[s] = c
    return C, seeds, errors


def ranks(F, sc, methods, B):
    """Mean rank (pipeline first) and Nemenyi CD at 5%, as in paper_numbers.ranking."""
    M = np.array([[F[s]] + [B[s][m] for m in methods] for s in sc])
    rk = np.array([rankdata(-r) for r in M]).mean(axis=0)
    k = M.shape[1]
    cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * len(sc)))
    return rk, cd


def table(header, body):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(str(x) for x in r) + " |" for r in body]


def main():
    na.WINDOW = "10"
    rows, _ = na.load()
    P = na.PIPELINE_MODEL
    proto = json.load(open(os.path.join(RUN, "protocol.json"), encoding="utf-8"))
    TH, ROB = proto["protocol"]["thresholds"], proto["protocol"]["robustness"]
    cfg = pd.read_csv(os.path.join(RUN, "configs.csv")).set_index("config")
    C, seeds, errors = load_run()
    scens = sorted(C)
    f1 = lambda c: round(na.f1(*c), 4)
    ho = [s for s in scens if na.group_of(s) in na.HELDOUT]
    dev = [s for s in scens if s not in ho]
    B = {s: {m: f1((rows[s][m]["tp"], rows[s][m]["fp"], rows[s][m]["fn"])) for m in BL if m in rows[s]}
         for s in scens}
    six = [s for s in scens if len(B[s]) == len(BL)]
    F = {c: {s: f1(C[s][c]) for s in scens} for c in cfg.index}
    D = F["default"]
    gap = lambda c, s: max([F[c][s]] + list(B[s].values())) - F[c][s]
    within = lambda c, mark: sum(gap(c, s) <= mark + EPS for s in scens)
    mean = lambda c, sc: float(np.mean([F[c][s] for s in sc]))

    # delta of the protocol: the tolerance, or the range of the mean F1 across the seeds if larger
    seed_means = [mean("default", scens)] + [float(np.mean([f1(v[s]) for s in scens]))
                                              for v in seeds.values() if set(v) >= set(scens)]
    delta = max(ROB["mean_f1_tolerance"], max(seed_means) - min(seed_means))
    MARK, DROP = ROB["within_mark"], ROB["max_within_drop"]

    def stats(c):
        g = np.array([gap(c, s) for s in scens])
        bgap = min(np.mean([max([F[c][s]] + list(B[s].values())) - B[s][m] for s in scens if m in B[s]])
                   for m in BL)
        r5, cd5 = ranks(F[c], scens, FIVE, B)
        r6, cd6 = ranks(F[c], six, BL, B)
        h5, _ = ranks(F[c], ho, FIVE, B)
        ho6 = [s for s in ho if s in six]
        h6, _ = ranks(F[c], ho6, BL, B)
        d5, _ = ranks(F[c], dev, FIVE, B)
        above = all(np.mean([F[c][s] for s in scens if m in B[s]]) > np.mean([B[s][m] for s in scens if m in B[s]])
                    for m in BL)
        above_ho = all(np.mean([F[c][s] for s in ho if m in B[s]]) > np.mean([B[s][m] for s in ho if m in B[s]])
                       for m in BL)
        diff = np.array([F[c][s] - D[s] for s in scens])
        return dict(
            all=mean(c, scens), dev=mean(c, dev), ho=mean(c, ho),
            changed=sum(C[s][c] != C[s]["default"] for s in scens), loss=diff.min(), gain=diff.max(),
            w=int((g <= MARK + EPS).sum()), b20=int((g > 0.20 + EPS).sum()), mgap=g.mean(), bgap=bgap,
            rank5=r5[0], next5=r5[1:].min(), sep5=int((r5[1:] - r5[0] > cd5).sum()),
            rank6=r6[0], next6=r6[1:].min(), sep6=int((r6[1:] - r6[0] > cd6).sum()),
            first5=bool(r5[0] < r5[1:].min()), first6=bool(r6[0] < r6[1:].min()),
            first_ho5=bool(h5[0] < h5[1:].min()), first_ho6=bool(h6[0] < h6[1:].min()),
            rank_ho5=h5[0], first_dev5=bool(d5[0] < d5[1:].min()), above=above, above_ho=above_ho)

    S = pd.DataFrame({c: stats(c) for c in cfg.index}).T
    d = S.loc["default"]
    num = lambda col: S[col].astype(float)
    S["R1"] = ((num("all") - d["all"]).abs() <= delta + EPS) & ((num("dev") - d["dev"]).abs() <= delta + EPS) \
        & ((num("ho") - d["ho"]).abs() <= delta + EPS)
    S["R2"] = (d["w"] - num("w")) <= DROP
    S["R3"] = num("mgap") < num("bgap")
    S["ok"] = S.R1 & S.R2 & S.R3
    d = S.loc["default"]
    oat, joint = S[cfg.phase == "oat"], S[cfg.phase == "joint"]
    nj = len(joint)
    yn = lambda b: "yes" if b else "no"

    out = ["# Sensitivity to the pipeline's cut-offs -- generated by generators/sensitivity_analysis.py", "",
           f"Run: `{os.path.basename(RUN)}` (started {proto['started']}, commit {proto['git_commit'][:7]}); "
           f"reference model `{proto['protocol']['reference_model']}`."]
    sup = ["# Table S20 -- generated by generators/sensitivity_analysis.py (paste into the supplement)", ""]

    # ------------------------------------------------------------------ 0
    missing = sorted(s for s in rows if P in rows[s] and s not in C)
    out += ["", "## 0. Scope", "",
            f"- Scenarios in the run: {len(scens)} ({len(dev)} development, {len(ho)} held-out); in the paper: "
            f"{sum(P in rows[s] for s in rows)}.",
            f"- In the paper but not in the run ({len(missing)}): {', '.join(short(s) for s in missing) or 'none'}.",
            f"- Configurations: {len(cfg)} (1 default, {len(oat)} one cut-off at a time, {nj} joint); "
            f"runs: {len(scens) * len(cfg)}; with errors: {errors}.",
            f"- Scenarios on which the six methods ran: {len(six)}; held-out among them: {sum(s in six for s in ho)}.",
            f"- Seeds besides the reported one: {len(seeds)}. delta of criterion R1 = {delta:.3f}."]

    # ------------------------------------------------------------------ 1
    diff = [s for s in scens if (rows[s][P]["tp"], rows[s][P]["fp"], rows[s][P]["fn"]) != C[s]["default"]]
    ref = float(np.mean([f1((rows[s][P]["tp"], rows[s][P]["fp"], rows[s][P]["fn"])) for s in scens]))
    out += ["", "## 1. Control against the reported run", "",
            f"- Mean F1: control {d['all']:.4f}, reported {ref:.4f} (on the {len(scens)} scenarios of the run).",
            f"- Scenarios with different TP/FP/FN: {len(diff)}"]
    for s in diff:
        r = rows[s][P]
        out.append(f"  - {short(s)}: reported {(r['tp'], r['fp'], r['fn'])} -> control {C[s]['default']}")
    out.append("- Mean F1 of the baselines on the scenarios of the run on which each one ran: " + ", ".join(
        f"{lab} {np.mean([B[s][m] for s in scens if m in B[s]]):.3f} (n={sum(m in B[s] for s in scens)})"
        for m, lab in na.BASELINES) + ".")

    # ------------------------------------------------------------------ 2
    out += ["", "## 2. One cut-off at a time", "",
            f"Within tolerance: {int(oat.ok.sum())} of {len(oat)}. Default: mean F1 {d['all']:.3f} "
            f"(development {d['dev']:.3f}, held-out {d['ho']:.3f}), {int(d['w'])} scenarios within {MARK:.2f} "
            f"of the best method, {int(d['b20'])} more than 0.20 behind.", ""]
    head_rep = ["Cut-off", "Value", "Mean F1", "Devel.", "Held-out", "Scenarios that change", "Largest loss",
                "Largest gain", f"Within {MARK:.2f} of the best", "More than 0.20 behind", "R1", "R2", "R3", "Within"]
    head_sup = ["Cut-off", "Value", "Mean F1", "Development", "Held-out", "Scenarios whose output changes",
                "Largest loss", "Largest gain", f"Within {MARK:.2f} of the best method", "More than 0.20 behind",
                "Within tolerance"]
    body_rep, body_sup, summary_rep, summary_sup, prot_sup = [], [], [], [], []
    for th in TH:
        k, grid = th["key"], sorted(th["grid"])
        cid = lambda v: "default" if float(v) == float(th["default"]) else f"oat__{k}__{v:g}"
        for v in grid:
            r = S.loc[cid(v)]
            tag = " (default)" if cid(v) == "default" else ""
            body_rep.append([th["label"], f"{v:g}{tag}", f"{r['all']:.3f}", f"{r['dev']:.3f}", f"{r['ho']:.3f}",
                             int(r["changed"]), f"{r['loss']:+.3f}", f"{r['gain']:+.3f}", int(r["w"]), int(r["b20"]),
                             yn(r.R1), yn(r.R2), yn(r.R3), yn(r.ok)])
            body_sup.append([th["label"], f"{v:g}{tag}", f"{r['all']:.3f}", f"{r['dev']:.3f}", f"{r['ho']:.3f}",
                             int(r["changed"]), f"{r['loss']:+.3f}", f"{r['gain']:+.3f}", int(r["w"]), int(r["b20"]),
                             yn(r.ok)])
        sub = S.loc[[cid(v) for v in grid if cid(v) != "default"]]
        # widest run of consecutive grid values, around the default, within tolerance
        i0 = grid.index(th["default"])
        lo = hi = i0
        while lo > 0 and S.loc[cid(grid[lo - 1]), "ok"]:
            lo -= 1
        while hi < len(grid) - 1 and S.loc[cid(grid[hi + 1]), "ok"]:
            hi += 1
        row = [th["label"], f"{th['default']:g}", f"{grid[0]:g}–{grid[-1]:g}", f"{int(sub.ok.sum())} / {len(sub)}",
               f"{grid[lo]:g}–{grid[hi]:g}", f"{num('all')[sub.index].min():.3f}–{num('all')[sub.index].max():.3f}",
               int(num("changed")[sub.index].max()), f"{num('loss')[sub.index].min():+.3f}"]
        summary_rep.append(row)
        summary_sup.append(row)
        prot_sup.append([th["label"], "per dataset" if th["group"] == "regime" else "per candidate pair",
                         f"{th['default']:g}", ", ".join(f"{v:g}" for v in grid)])
    out += table(["Cut-off", "Default", "Grid", "Values within tolerance", "Range within tolerance",
                  "Mean F1 (min–max)", "Max. scenarios that change", "Largest loss on one scenario"], summary_rep)
    out += [""] + table(head_rep, body_rep)
    out += ["", f"Variants with mean F1 higher than the default's: development "
                f"{int((num('dev')[oat.index] > d['dev'] + EPS).sum())}, held-out "
                f"{int((num('ho')[oat.index] > d['ho'] + EPS).sum())} (of {len(oat)}). Mean F1 over the variants: "
                f"{num('all')[oat.index].min():.3f} to {num('all')[oat.index].max():.3f}."]

    sup += ["### S20.1 — Protocol: cut-offs, defaults, and grids", ""] + \
        table(["Cut-off", "Decision", "Default", "Grid"], prot_sup)
    sup += ["", "### S20.2 — One cut-off at a time: summary by cut-off", ""] + \
        table(["Cut-off", "Default", "Grid", "Settings within tolerance", "Range within tolerance",
               "Mean F1 (min–max)", "Most scenarios whose output changes", "Largest loss on one scenario"], summary_sup)
    sup += ["", "### S20.3 — One cut-off at a time: every setting", ""] + table(head_sup, body_sup)

    # ------------------------------------------------------------------ 3
    out += ["", "## 3. Joint variation", "",
            f"Within tolerance: {int(joint.ok.sum())} of {nj} ({joint.ok.mean():.1%}); required by the protocol: "
            f"{ROB['joint_min_fraction']:.0%}.",
            f"- R1 (mean F1 within {delta:.2f}): {int(joint.R1.sum())}; fails on {int((~joint.R1).sum())} "
            f"(all {int(((num('all')[joint.index] - d['all']).abs() > delta + EPS).sum())}, development "
            f"{int(((num('dev')[joint.index] - d['dev']).abs() > delta + EPS).sum())}, held-out "
            f"{int(((num('ho')[joint.index] - d['ho']).abs() > delta + EPS).sum())}).",
            f"- R2 (drop of at most {DROP} scenarios within {MARK:.2f} of the best): {int(joint.R2.sum())}; fails on "
            f"{int((~joint.R2).sum())}. Drop: median {int((d['w'] - num('w')[joint.index]).median())}, maximum "
            f"{int((d['w'] - num('w')[joint.index]).max())}.",
            f"- R3 (mean shortfall below that of every baseline): {int(joint.R3.sum())}.",
            f"- R2 and R3 without R1 (ceiling if the F1 tolerance grows with the seeds): {int((joint.R2 & joint.R3).sum())}.", ""]
    dist = [("Mean F1, all", "Mean F1, all scenarios", "all", 3), ("Mean F1, development", "Mean F1, development suite", "dev", 3),
            ("Mean F1, held-out", "Mean F1, held-out datasets", "ho", 3),
            (f"Scenarios within {MARK:.2f} of the best", f"Scenarios within {MARK:.2f} of the best method", "w", 0),
            ("Scenarios more than 0.20 behind", "Scenarios more than 0.20 behind", "b20", 0),
            ("Mean shortfall to the best", "Mean shortfall to the best method", "mgap", 3),
            ("Mean rank, five methods", f"Mean rank, five methods ({len(scens)} scenarios)", "rank5", 2),
            ("Mean rank of the runner-up, five methods", "Mean rank of the runner-up, five methods", "next5", 2),
            ("Mean rank, six methods", f"Mean rank, six methods ({len(six)} scenarios)", "rank6", 2),
            ("Mean rank of the runner-up, six methods", "Mean rank of the runner-up, six methods", "next6", 2),
            ("Scenarios that change", "Scenarios whose output changes", "changed", 0)]
    fm = lambda x, n: f"{x:.{n}f}"
    dist_rows = lambda i: [[lab[i], fm(d[c], n), fm(num(c)[joint.index].min(), n), fm(num(c)[joint.index].median(), n),
                            fm(num(c)[joint.index].max(), n)] for *lab, c, n in dist]
    out += table(["Measure", "Default", "Minimum", "Median", "Maximum"], dist_rows(0))
    concl = [("Best mean rank, five methods", f"Best mean rank, five methods ({len(scens)} scenarios)", "first5"),
             ("Best mean rank, six methods", f"Best mean rank, six methods ({len(six)} scenarios)", "first6"),
             ("Mean F1 above every baseline", "Mean F1 above every baseline", "above"),
             ("Mean shortfall below that of every baseline (R3)", "Mean shortfall below every baseline's (R3)", "R3"),
             ("Nemenyi: separated from all, six methods", "Separated from every baseline by the Nemenyi critical difference, six methods", None),
             ("Nemenyi: separated from all, five methods", "Separated from every baseline by the Nemenyi critical difference, five methods", None),
             ("Best mean rank on the development suite, five methods", f"Best mean rank on the development suite, five methods ({len(dev)} scenarios)", "first_dev5"),
             ("Best mean rank on the held-out datasets, five methods", f"Best mean rank on the held-out datasets, five methods ({len(ho)} scenarios)", "first_ho5"),
             ("Best mean rank on the held-out datasets, six methods", f"Best mean rank on the held-out datasets, six methods ({sum(s in six for s in ho)} scenarios)", "first_ho6"),
             ("Mean F1 above every baseline on the held-out datasets", "Mean F1 above every baseline on the held-out datasets", "above_ho")]
    val = lambda r, c, i: (r["sep6"] == len(BL) if i == 4 else r["sep5"] == len(FIVE)) if c is None else bool(r[c])
    concl_rows = lambda j: [[lab[j], yn(val(d, c, i)),
                             f"{sum(val(joint.loc[x], c, i) for x in joint.index)} / {nj}"]
                            for i, (*lab, c) in enumerate(concl)]
    out += ["", "### Conclusions of the paper in the joint configurations", ""] + \
        table(["Conclusion", "At the default", "Holds in"], concl_rows(0))
    loses = {m: sum(ranks(F[c], scens, FIVE, B)[0][i + 1] - S.loc[c, "rank5"] <= ranks(F[c], scens, FIVE, B)[1]
                    for c in joint.index) for i, m in enumerate(FIVE)}
    out += ["", f"Baseline that is no longer separated (five methods): "
                f"{ {dict(na.BASELINES)[m]: int(n) for m, n in loses.items() if n} }. Mean rank of the pipeline on the held-out datasets "
                f"(five methods): default {d['rank_ho5']:.2f}, joint {num('rank_ho5')[joint.index].min():.2f} to "
                f"{num('rank_ho5')[joint.index].max():.2f}."]

    keys = [th["key"] for th in TH]
    J = cfg.loc[joint.index, keys].astype(float)
    terc_rep, terc_sup = [], []
    for th in TH:
        q = pd.qcut(J[th["key"]], 3, labels=False)
        row = [th["label"]]
        for t in range(3):
            idx = q.index[q == t]
            row += [f"{joint.ok[idx].mean():.0%}"]
        for t in range(3):
            idx = q.index[q == t]
            row += [f"{num('w')[idx].mean():.1f}"]
        terc_rep.append(row)
        terc_sup.append(row)
    out += ["", "### By tercile of the value of each cut-off (within tolerance; scenarios within the mark)", ""] + \
        table(["Cut-off", "Within: low tercile", "middle", "high", "Scenarios: low tercile", "middle", "high"], terc_rep)
    rho, r2 = J["coherence_rho"], J["outcome_r2"]
    for lab, cond in (("rho >= 0.25", rho >= 0.25), ("outcome R2 <= 0.60", r2 <= 0.60),
                      ("both", (rho >= 0.25) & (r2 <= 0.60))):
        out.append(f"- Within tolerance with {lab}: {int(joint.ok[cond].sum())} of {int(cond.sum())} "
                   f"({joint.ok[cond].mean():.0%}) -- post hoc subset.")

    # interaction: sum of the one-at-a-time effects (linear interpolation of each curve) against the joint one
    pred = []
    for c in joint.index:
        p = d["all"]
        for th in TH:
            xs = sorted(th["grid"])
            ys = [S.loc["default" if x == th["default"] else f"oat__{th['key']}__{x:g}", "all"] for x in xs]
            p += np.interp(cfg.loc[c, th["key"]], xs, ys) - d["all"]
        pred.append(p)
    inter = num("all")[joint.index].values - np.array(pred)
    out += ["", f"Interaction (joint mean F1 minus the sum of the one-at-a-time effects): mean {inter.mean():+.4f}, "
                f"from {inter.min():+.4f} to {inter.max():+.4f}; above delta on {int((abs(inter) > delta).sum())} of {nj}."]

    sup += ["", "### S20.4 — Joint variation: the pre-specified criterion", "",
            f"Configurations within tolerance: **{int(joint.ok.sum())} of {nj} ({joint.ok.mean():.1%})**; required: "
            f"{ROB['joint_min_fraction']:.0%}.", ""] + \
        table(["Condition", "Configurations that meet it", "Configurations that fail it"],
              [[f"R1 — mean F1 within {delta:.2f} of the default (all scenarios, development suite, held-out datasets)",
                int(joint.R1.sum()), int((~joint.R1).sum())],
               [f"R2 — at most {DROP} fewer scenarios within {MARK:.2f} of the best of the six methods",
                int(joint.R2.sum()), int((~joint.R2).sum())],
               ["R3 — mean shortfall to the best method below that of every baseline",
                int(joint.R3.sum()), int((~joint.R3).sum())],
               ["All three", int(joint.ok.sum()), int((~joint.ok).sum())]])
    sup += ["", "### S20.5 — Joint variation: distribution over the 150 configurations", ""] + \
        table(["Measure", "Default", "Minimum", "Median", "Maximum"], dist_rows(1))
    sup += ["", "### S20.6 — Joint variation: conclusions of the main text", ""] + \
        table(["Conclusion", "At the defaults", "Holds in"], concl_rows(1))
    sup += ["", "### S20.7 — Joint variation by tercile of each cut-off", ""] + \
        table(["Cut-off", "Within tolerance: low tercile", "middle", "high",
               f"Scenarios within {MARK:.2f}: low tercile", "middle", "high"], terc_sup)

    # ------------------------------------------------------------------ 4
    g0 = {s: gap("default", s) for s in scens}
    bands = [("0 (best or tied)", "0 (best or tied)", lambda g: g <= EPS),
             ("up to 0.05", "up to 0.05", lambda g: EPS < g <= 0.05 + EPS),
             (f"0.05 to {MARK:.2f}", f"0.05 to {MARK:.2f}", lambda g: 0.05 + EPS < g <= MARK + EPS),
             (f"above {MARK:.2f}", f"above {MARK:.2f}", lambda g: g > MARK + EPS)]
    fx_rep, fx_sup = [], []
    for rep, sp, f in bands:
        sc = [s for s in scens if f(g0[s])]
        never = sum(all(C[s][c] == C[s]["default"] for c in joint.index) for s in sc)
        beyond = np.mean([sum(gap(c, s) > MARK + EPS for s in sc) for c in joint.index])
        beyond = "—" if f(1.0) else f"{beyond:.1f}"   # the last band is already beyond the mark
        fx_rep.append([rep, len(sc), never, beyond])
        fx_sup.append([sp, len(sc), never, beyond])
    out += ["", "## 4. Scenarios", "", "### By shortfall to the best method at the default", ""] + \
        table(["Shortfall at the default", "Scenarios", "Never change in the joint configurations",
               f"Beyond {MARK:.2f}, mean per joint configuration"], fx_rep)
    never = sum(all(C[s][c] == C[s]["default"] for c in cfg.index) for s in scens)
    out += ["", f"Scenarios whose output changes in none of the {len(cfg) - 1} configurations: {never} of {len(scens)}."]
    mov_rep, mov_sup = [], []
    for th in TH:
        for s in scens:
            cs = [c for c in oat.index if c.startswith(f"oat__{th['key']}__") and abs(F[c][s] - D[s]) >= 0.10]
            if not cs:
                continue
            worst = min(cs, key=lambda c: F[c][s] - D[s]) if min(F[c][s] - D[s] for c in cs) < 0 else \
                max(cs, key=lambda c: F[c][s] - D[s])
            vals = ", ".join(f"{float(c.rsplit('__', 1)[1]):g}" for c in cs)
            row = [th["label"], vals, f"`{short(s)}`", "held-out" if s in ho else "development",
                   f"{D[s]:.3f}", f"{F[worst][s]:.3f}", f"{F[worst][s] - D[s]:+.3f}"]
            mov_rep.append(row)
            mov_sup.append(row)
    out += ["", "### One cut-off at a time: scenarios with an F1 change of 0.10 or more (worst value)", ""] + \
        table(["Cut-off", "Values", "Scenario", "Set", "F1 at the default", "F1 at the variant", "Difference"], mov_rep)
    sup += ["", "### S20.8 — Scenarios by their shortfall to the best method at the defaults", ""] + \
        table(["Shortfall at the defaults", "Scenarios", "Unchanged in every joint configuration",
               f"Beyond {MARK:.2f}, mean per joint configuration"], fx_sup)
    sup += ["", "### S20.9 — One cut-off at a time: scenarios whose F1 changes by 0.10 or more", ""] + \
        table(["Cut-off", "Values", "Scenario", "Set", "F1 at the default", "F1 at the setting (largest change)",
               "Difference"], mov_sup)

    # ------------------------------------------------------------------ 5
    mk_rep, mk_sup = [], []
    for mark in MARKS:
        w0 = within("default", mark)
        r2m = pd.Series({c: (w0 - within(c, mark)) <= DROP for c in joint.index})
        ok = r2m & joint.R1 & joint.R3
        row = [f"{mark:.2f}" + (" (protocol)" if abs(mark - MARK) < EPS else ""), f"{w0} / {len(scens)}",
               int(r2m.sum()), f"{int(ok.sum())} ({ok.mean():.1%})"]
        mk_rep.append(row)
        mk_sup.append(row)
    out += ["", "## 5. Criterion at other marks (post hoc analysis, outside the protocol)", ""] + \
        table(["Mark", "Scenarios within the mark at the default", f"R2 (of {nj})", f"R1, R2 and R3 (of {nj})"], mk_rep)
    sup += ["", "### S20.10 — The criterion at other marks (post hoc)", ""] + \
        table(["Mark", "Scenarios within the mark at the defaults", f"R2 met (of {nj})",
               f"R1, R2, and R3 met (of {nj})"], mk_sup)

    # ------------------------------------------------------------------ 6
    out += ["", "## 6. Seeds", ""]
    if seeds:
        out.append(f"- Seeds run: {sorted(seeds)}; mean F1 per seed: " +
                   ", ".join(f"{m:.4f}" for m in seed_means) + f"; range {max(seed_means) - min(seed_means):.4f}.")
    else:
        out.append("- NOT run: the run only has the reported seed (offset 0). The delta of the criterion is the "
                   f"protocol's fixed tolerance ({ROB['mean_f1_tolerance']}).")

    # ------------------------------------------------------------------ 7
    # Walker: Section 5.5 shows the support of WCPAC -> CPAC 0.002 from the floor
    # with few candidates. What happens to the output when that floor varies, and
    # with the outcome R2 cut-off, which is what changes this scenario.
    walker = next((s for s in C if O.canonical(s) == WALKER), None)   # the run may hold a legacy ID
    if walker:
        rec = {}
        for line in open(os.path.join(RUN, "runs", walker + ".jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if r["seed"] == 0:
                rec[r["config"]] = r
        pairs_str = lambda ps: ", ".join(sorted({f"{p[0]} → {p[1]}" for p in ps})) or "none"
        body = []
        for key in ("floor_few_candidates", "outcome_r2"):
            th = next(t for t in TH if t["key"] == key)
            for v in sorted(th["grid"]):
                c = "default" if float(v) == float(th["default"]) else f"oat__{key}__{v:g}"
                t = rec[c]["trace"]
                body.append([th["label"], f"{v:g}" + (" (default)" if c == "default" else ""), t.get("support_floor"),
                             t.get("control_set"), "—" if c == "default" else yn("stage8a" in rec[c]["computed"]),
                             pairs_str(rec[c]["pairs"]), f"{F[c][walker]:.3f}"])
        t0 = rec["default"]["trace"]
        out += ["", "## 7. Walker: support floor with few candidates and outcome R2 (sentence of Section 5.5)", "",
                f"At the default: outcome R2 = {t0.get('outcome_r2'):.3f}, applied floor = {t0.get('support_floor')} "
                f"({t0.get('support_floor_n')} candidates in the calibration).", ""]
        out += table(["Cut-off", "Value", "Applied floor", "Control set", "Stage 8a recomputed (new input)",
                      "Final output", "F1"], body)

        # Outputs of Stage 8a stored in the run's cache for this scenario (one per distinct input):
        # causal_penalty_score = decision of Stage 7; rejected_phase2_structural = removed by 8a.
        import joblib
        ents = []
        for p in sorted(glob.glob(os.path.join(RUN, "cache", walker, "stage8a_*.pkl"))):
            if os.path.basename(p).startswith("stage8a_target"):
                continue
            df = joblib.load(p)
            sel = lambda m: df.loc[m, ["source_entity", "target_entity"]].values.tolist()
            rem = df["rejected_phase2_structural"] == True  # noqa: E712
            ents.append(dict(s7=sel(df["causal_penalty_score"] == 1), fin=sel(df["significant"] == True),  # noqa: E712
                             rem=[f"{r.source_entity} → {r.target_entity} ({str(r.structural_rejection_reason).split(' (')[0]})"
                                  for r in df[rem].itertuples()]))
        out += ["", "Outputs of Stage 8a in the run's cache (every configuration; one row per distinct input):", ""]
        out += table(["Accepted by Stage 7", "Removed by 8a (reason)", "Accepted after 8a"],
                     [[pairs_str(e["s7"]), "; ".join(e["rem"]) or "none", pairs_str(e["fin"])] for e in ents])
        fin0 = pairs_str(rec["default"]["pairs"])
        base = [e for e in ents if pairs_str(e["fin"]) == fin0 and not e["rem"]]
        alt = [e for e in ents if pairs_str(e["fin"]) == fin0 and e["rem"]]
        th = next(t for t in TH if t["key"] == "floor_few_candidates")
        same = all(rec[f"oat__floor_few_candidates__{v:g}"]["pairs"] == rec["default"]["pairs"]
                   for v in th["grid"] if float(v) != float(th["default"]))
        new = [f"{v:g}" for v in sorted(th["grid"]) if float(v) != float(th["default"])
               and "stage8a" in rec[f"oat__floor_few_candidates__{v:g}"]["computed"]]
        out += ["", f"- Final output equal to the default's at every value of the floor ({min(th['grid']):g} to {max(th['grid']):g}): {yn(same)}.",
                f"- Values of the floor at which 8a received an input different from the default's (recomputed): {', '.join(new) or 'none'}.",
                f"- Cache inputs with the default's output: {len(base)} without removal by 8a and {len(alt)} with removal"
                + (f" ({'; '.join(alt[0]['rem'])}; accepted by Stage 7: {pairs_str(alt[0]['s7'])})." if alt else ".")]

    open(OUT, "w", encoding="utf-8").write(O.canonical("\n".join(out) + "\n"))
    open(OUT_S, "w", encoding="utf-8").write(O.canonical("\n".join(sup) + "\n"))
    print("saved to", OUT, "and", OUT_S)


if __name__ == "__main__":
    main()
