"""
Generates Tables S8.2 (mean rank, Friedman, Nemenyi and one-to-one comparison
per scenario) and S8.3 (relative shortfall and shortfall on the scenarios above
the trivial floor) of the supplement, at both windows, from the JSON files
written by paper_numbers.py (keys `ranking` and `relative_shortfall`, sections
3g and 3h of the report). Run paper_numbers.py first.

Output: Table_S8_Ranking.md (pasted after S8.1 in the supplement).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import origins as O  # noqa: E402
ROOT = O.OUT_DIR   # where paper_numbers.py wrote the JSON files
OUT = O.out("Table_S8_Ranking.md")
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
STAR = chr(92) + "*"
METHODS = ["Pipeline", "PCMCI", "PCMCI+", "DYNOTEARS", "CUTS+", "VARLiNGAM"]
UNITS = [("scenario, six methods", "Scenarios, six methods"),
         ("scenario, five methods (without VARLiNGAM)", "Scenarios, five methods (without VARLiNGAM)"),
         ("group, six methods", "Scenario groups, six methods")]
SETS = [("All scenarios", "All scenarios"), ("Development", "Development suite"),
        ("Held-out", "Held-out datasets")]
WINDOWS = ["10", "24"]


def pf(x):
    s = x if isinstance(x, str) else f"{x:.2g}"
    if "e" in s:
        m, e = s.split("e")
        return f"{m} × 10{str(int(e)).translate(SUP)}"
    return s


def bold(cells, on):
    return [f"**{c}**" for c in cells] if on else cells


def s82(D):
    out = ["### S8.2 — Mean rank of the six methods and one-to-one comparison by scenario", "",
           "<!-- DATA: section 3g -->", "",
           "*Companion to Sections 4.4 and 5.1 (Table 3). Each method is taken as a single choice kept across the "
           "suite. In each scenario the methods are ranked by F1 (rank 1 for the highest; tied methods receive the "
           "mean of the ranks they span) and each is summarized by its mean rank; the Friedman test compares the mean "
           "ranks, and two methods are separated when their mean ranks differ by more than the Nemenyi critical "
           "difference at the 5% level, q₀.₀₅ √(k(k + 1)/(6N)) for k methods and N units (Demšar, 2006). Three units "
           "are reported: the scenarios on which all six methods run (the mean rank of main-text Table 3); every "
           "scenario, with VARLiNGAM, which does not run on all of them, left out; and the scenario group, each "
           "method entering with its mean F1 over the scenarios of the group on which all six run. An asterisk marks "
           "a baseline whose mean rank is behind the pipeline's by more than the critical difference. Panels (3) and "
           "(4) give the comparison of the pipeline with each baseline by scenario within each set; the sign test "
           "and the Wilcoxon signed-rank test are two-sided, and the Wilcoxon p-values are Holm-adjusted over the "
           "five baselines within each set. Source: `generators/paper_numbers.py`, section 3g of its report at each "
           "window; this table is written by `generators/table_s8_ranking.py`.*", ""]
    for lab, w in zip(["(1)", "(2)"], WINDOWS):
        out += [f"**{lab} Mean rank, τ_max = {w}**", "",
                "| Set | Method | " + " | ".join(e for _, e in UNITS) + " |", "|---|---|---|---|---|"]
        for sp, se in SETS:
            d = D[w]["ranking"][sp]
            for i, m in enumerate(METHODS):
                cells = []
                for up, _ in UNITS:
                    b = d[up]
                    if m not in b["rank"]:
                        cells.append("—")
                        continue
                    star = STAR if m != "Pipeline" and b["rank"][m] - b["rank"]["Pipeline"] > b["cd"] else ""
                    cells.append(f"{b['rank'][m]:.2f}{star}")
                row = bold([m] + cells, m == "Pipeline")
                out.append(f"| {se if i == 0 else ''} | " + " | ".join(row) + " |")
            out.append("|  | Units (N) | " + " | ".join(str(d[up]["n"]) for up, _ in UNITS) + " |")
            out.append("|  | Friedman test p | " + " | ".join(pf(d[up]["friedman_p"]) for up, _ in UNITS) + " |")
            out.append("|  | Nemenyi critical difference (5%) | "
                       + " | ".join(f"{d[up]['cd']:.2f}" for up, _ in UNITS) + " |")
        out.append("")
    for lab, w in zip(["(3)", "(4)"], WINDOWS):
        out += [f"**{lab} Pipeline against each baseline by scenario, τ_max = {w}**", "",
                "| Set | Against | Scenarios | Pipeline higher / equal / lower | Mean F1 difference | Sign test p | "
                "Wilcoxon p (two-sided) | Holm-adjusted p (five baselines) |", "|---|---|---|---|---|---|---|---|"]
        for sp, se in SETS:
            for i, m in enumerate(METHODS[1:]):
                r = D[w]["ranking"][sp]["one_to_one"][m]
                out.append(f"| {se if i == 0 else ''} | {m} | {r['n']} | {r['wtl'].replace('/', ' / ')} | "
                           f"{r['dF1'].replace('-', '−')} | {pf(r['sign_p'])} | {pf(r['wilcoxon_p'])} | "
                           f"{pf(r['holm_p'])} |")
        out.append("")
    return out


def s83(D):
    out = ["### S8.3 — Relative shortfall, and shortfall on the scenarios above the trivial floor", "",
           "<!-- DATA: section 3h -->", "",
           "*Companion to Sections 4.4, 5.1, and 6.2 (Table 3). The shortfall of main-text Table 3 is an absolute "
           "difference of F1 values and is small for every method where the best F1 is itself low. Two readings "
           "that do not depend on that scale are given here. The relative shortfall of a method on a scenario is "
           "its shortfall divided by the F1 of the best of the six methods on that scenario (the best F1 is above "
           "zero on every scenario); it is reported as its mean and as the number of scenarios on which it is at "
           "most 10% and at most 25%, two marks chosen after the results were seen, like those of Table 3. The "
           "right-hand columns repeat the measures on the scenarios on which the best method reaches the trivial "
           "floor of Section 4.4, F1 = 2p/(1 + p). RCAEval, where the best F1 of a scenario does not exceed 0.22, "
           "is shown on its own. Source: `generators/paper_numbers.py`, section 3h of its report at each window; "
           "this table is written by `generators/table_s8_ranking.py`.*", ""]
    sets = SETS + [("RCAEval", "RCAEval")]
    for lab, w in zip(["(1)", "(2)"], WINDOWS):
        out += [f"**{lab} τ_max = {w}**", "",
                "| Set | Method | Scenarios | Mean shortfall | Mean relative shortfall | Within 0.10 of the best | "
                "Within 10% of the best | Within 25% of the best | Above the floor: scenarios | Mean shortfall | "
                "Mean relative shortfall | Within 0.10 of the best | Within 10% of the best | "
                "Within 25% of the best |", "|" + "---|" * 14]
        for sp, se in sets:
            for i, m in enumerate(METHODS):
                if m not in D[w]["relative_shortfall"][sp]:
                    continue
                r = D[w]["relative_shortfall"][sp][m]
                cells = [m]
                for d in (r["all"], r["above_floor"]):
                    cells += ["—"] * 6 if d is None else [
                        str(d["n"]), f"{d['mean_gap']:.3f}", f"{100 * d['mean_rel']:.1f}%", str(d["within_010"]),
                        str(d["rel_010"]), str(d["rel_025"])]
                out.append(f"| {se if i == 0 else ''} | " + " | ".join(bold(cells, m == "Pipeline")) + " |")
        out.append("")
    return out


if __name__ == "__main__":
    D = {w: json.load(open(os.path.join(ROOT, f"paper_numbers_tau{w}.json"), encoding="utf-8")) for w in WINDOWS}
    txt = "\n".join(s82(D) + s83(D))
    open(OUT, "w", encoding="utf-8").write(txt)
    print("ok ->", OUT)
