"""
EXPANDED v2 version of Figure 5 — the same 21 benchmark groups as the
original version (make_figure5_expanded.py), but with an updated data source
(Results/Run_20261004_232204) and two additional models:

    PCMCI+   (pcmciplus)   — upgrade of PCMCI with detection of contemporaneous
                             relations (Runge, 2020)
    VARLiNGAM (varlingam)  — linear/non-Gaussian alternative (Hyvarinen
                             et al., 2010)

Generated as a SEPARATE file (suffix _v2) for side-by-side comparison
with the original 4-model version — it does not overwrite the file published
in the paper.

SAME visual style as the previous versions: line chart, 3 stacked panels,
alternating background bands per family. Figure width widened
proportionally to fit the 21 groups, as in the original.
"""
import os
import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
import pandas as pd
import sys
# Common window: "10" (paper, main) or "24" (supplement, sensitivity).
WINDOW = sys.argv[1] if len(sys.argv) > 1 else "10"
assert WINDOW in ("10", "24")
SUF = "_tau10" if WINDOW == "10" else ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import origins as O  # noqa: E402
O.require("MAIN_RUN")   # in the paper, Run_20261004_232204
OUT = O.fig(f"benchmark_performance_tau{WINDOW}.svg")

GROUPS = [
    ("Log. maps\nBifurcation",   [f"benchmark_logistic_maps_bifurcation_seed{s}"  for s in (1, 2, 3)]),
    ("Log. maps\nChain",         [f"benchmark_logistic_maps_chain_seed{s}"        for s in (1, 2, 3)]),
    ("Log. maps\nConverg.",      [f"benchmark_logistic_maps_convergence_seed{s}"  for s in (1, 2, 3)]),
    ("Log. maps\nIsolated var.", [f"benchmark_logistic_maps_with_isolated_seed{s}" for s in (1, 2, 3)]),
    ("Lorenz-96\nK=8",           [f"benchmark_lorenz96_K8_seed{s}"                for s in (1, 2, 3)]),
    ("Lorenz-96\nK=10",          [f"benchmark_lorenz96_K10_seed{s}"               for s in (1, 2, 3)]),
    ("tigramite\nLinear",        [f"benchmark_tigramite_baseline_linear_seed{s}"  for s in (11, 22, 33)]),
    ("tigramite\nQuadratic",     [f"benchmark_tigramite_quadratic_seed{s}"        for s in (11, 22, 33)]),
    ("tigramite\nStep",          [f"benchmark_tigramite_threshold_seed{s}"        for s in (11, 22, 33)]),
    ("tigramite\nSaturation",    [f"benchmark_tigramite_saturation_seed{s}"       for s in (11, 22, 33)]),
    ("tigramite\nMixed-12",      [f"benchmark_tigramite_large_mixed_seed{s}"      for s in (11, 22, 33)]),
    ("CausalTime\nMedical",      [f"benchmark_causaltime_medical_sample{s}"       for s in (0, 1, 2)]),
    ("CausalTime\nPM2.5",        [f"benchmark_causaltime_pm25_sample{s}"          for s in (0, 1, 2)]),
    ("CausalTime\nTraffic",      [f"benchmark_causaltime_traffic_sample{s}"       for s in (0, 1, 2)]),
    ("TEP",                      ["benchmark_tep_normal"]),
    ("ALARM\n(ICU)",             ["real_alarm_icu"]),
    ("PEMS-BAY\n(traffic)",      ["real_pems_bay_traffic"]),
    ("Sachs\n(proteins)",        ["real_sachs_proteins"]),
    ("Walker\nCirculation",      ["real_tigramite_climate_walker"]),
    ("NetSim\n(fMRI)",           [f"benchmark_netsim_sim3_subj{s}"                for s in (0, 1, 2)]),
    ("PetShop\n(cloud)",         [f"real_petshop_{r}" for r in ("low_traffic", "high_traffic", "temporal_traffic1")]),
]

METHODS = [
    ("patch" + SUF,           "Pipeline",  "#0072b2", "o"),
    ("pcmci_tigramite" + SUF, "PCMCI",     "#e69f00", "s"),
    ("pcmciplus" + SUF,       "PCMCI+",    "#009e73", "v"),
    ("dynotears" + SUF,       "DYNOTEARS", "#7f5aa2", "^"),
    ("cuts_plus" + SUF,       "CUTS+",     "#d55e00", "D"),
    ("varlingam" + SUF,       "VARLiNGAM", "#cc79a7", "X"),
]
METRICS = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]

FAMILIES = [
    ("Logistic maps",            0,  4, True),
    ("Lorenz-96",                4,  6, False),
    ("tigramite SCM",            6, 11, True),
    ("CausalTime",               11, 14, False),
    ("Third-party datasets", 14, 21, True),
]


def pooled(scens, model):
    tp = fp = fn = 0
    n_ok = 0
    for s in scens:
        df = pd.read_csv(os.path.join(O.find(s), "validation_summary.csv")).set_index("model")
        if model not in df.index:
            continue  # model not applicable/failed on this scenario (e.g.: VARLiNGAM on CausalTime)
        r = df.loc[model]
        tp += int(r.TP); fp += int(r.FP); fn += int(r.FN)
        n_ok += 1
    if n_ok == 0:
        return None  # no valid realization in the group -> gap in the chart (NaN)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "n_ok": n_ok, "n_total": len(scens)}


data = {m: [pooled(sc, m) for _, sc in GROUPS] for m, _, _, _ in METHODS}

# warning of partial/missing coverage per group (does not stop the figure generation)
for m, label, _, _ in METHODS:
    for (name, scens), d in zip(GROUPS, data[m]):
        if d is None:
            print(f"[WARNING] {label}: NO valid realization in '{name.replace(chr(10), ' ')}' "
                  f"({len(scens)} scenarios tested, 0 with a result) -> gap in the chart")
        elif d["n_ok"] < d["n_total"]:
            print(f"[WARNING] {label}: partial pooling in '{name.replace(chr(10), ' ')}' "
                  f"({d['n_ok']}/{d['n_total']} scenarios with a result)")

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["svg.fonttype"] = "path"

# proportional width: keeps ~53.8pt per group, as in the original with 14
PT_PER_GROUP = (811.4 - 58.0) / 14.0
PANEL_WIDTH = PT_PER_GROUP * len(GROUPS)
RIGHT_MARGIN = 828 - 811.4          # right margin of the original, preserved
W_PT = 58.0 + PANEL_WIDTH + RIGHT_MARGIN
H_PT = 691.2

W, H = W_PT / 72.0, H_PT / 72.0
fig = plt.figure(figsize=(W, H))

LEFT, RIGHT = 58.0 / W_PT, (58.0 + PANEL_WIDTH) / W_PT
WIDTH = RIGHT - LEFT
BOXES = [
    (LEFT, (691.2 - 240.0) / 691.2, WIDTH, (240.0 - 82.9) / 691.2),
    (LEFT, (691.2 - 420.7) / 691.2, WIDTH, (420.7 - 263.6) / 691.2),
    (LEFT, (691.2 - 601.3) / 691.2, WIDTH, (601.3 - 444.3) / 691.2),
]

x = list(range(len(GROUPS)))
n = len(GROUPS)
axes = []
for (met, title), box in zip(METRICS, BOXES):
    ax = fig.add_axes(box)
    axes.append(ax)

    for _, i0, i1, gray in FAMILIES:
        if gray:
            ax.axvspan(i0 - 0.5, i1 - 0.5, color="#f7f7f7", zorder=0, lw=0)
    for _, i0, _, _ in FAMILIES[1:]:
        ax.axvline(i0 - 0.5, color="#999999", lw=0.6, zorder=1)

    ax.grid(axis="y", color="#e0e0e0", lw=0.6, zorder=1)
    ax.set_axisbelow(True)

    for model, label, color, marker in METHODS:
        y = [(d[met] if d is not None else float("nan")) for d in data[model]]
        ax.plot(x, y, color=color, lw=1.3, marker=marker, markersize=5,
                markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.6,
                label=label, zorder=3, solid_capstyle="projecting")

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.03, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel(title, fontsize=9.5)
    ax.tick_params(labelsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([])

axes[-1].set_xticklabels([g[0] for g in GROUPS], fontsize=7.0)

for title, i0, i1, _ in FAMILIES:
    fs = 8.5 if (i1 - i0) >= 3 else 6.6
    axes[0].text((i0 + i1 - 1) / 2.0, 1.075, title,
                 transform=axes[0].get_xaxis_transform(),
                 ha="center", va="center", fontsize=fs, color="#444444")

fig.suptitle(
    f"Performance across established benchmark groups (τ_max = {WINDOW})",
    fontsize=11.5, y=(691.2 - 27.0) / 691.2)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, 0.006), columnspacing=1.3,
           handletextpad=0.5)

fig.savefig(OUT, format="svg")
print("saved to", OUT)
print(f"width: {W_PT:.1f}pt ({len(GROUPS)} groups)")
print()
for met, _ in METRICS:
    print(f"  {met}:")
    for model, label, _, _ in METHODS:
        print(f"    {label:10s} " + " ".join(
            (f"{d[met]:.3f}" if d is not None else "  -  ") for d in data[model]))
