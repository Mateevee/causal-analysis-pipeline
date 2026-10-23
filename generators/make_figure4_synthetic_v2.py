"""
Reconstruction of Figure 4 (synthetic_performance.svg) — 5 scenarios of the
internal synthetic VAR generator (Section 4.1), now with PCMCI+ and VARLiNGAM
added, and fixing the clipping at the top/bottom of the original file.

Cause of the clipping in the original file: viewBox of 828x259.2pt, but the
suptitle was anchored at y=3.17pt (almost touching the top) and the legend
at y=264.78pt — BEYOND the lower limit of the canvas (259.2pt). Both were
clipped when rendered. Here the figure uses constrained_layout, which
reserves space automatically for the suptitle and the figure legend,
removing the root cause instead of recomputing absolute boxes by hand.

Data source: Results/Run_20261004_232204, the same run used for the
expanded Figure 5 v2. A single realization per scenario (same convention as
the original file — there is no pooling of seeds here).

Generated as a separate file (suffix _v2) for side-by-side comparison
with the published file.
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
BASE = O.require("MAIN_RUN")   # in the paper, Run_20261004_232204
OUT = O.fig(f"synthetic_performance_tau{WINDOW}.svg")

# (x-axis label = number of true causal connections, scenario folder)
SCENARIOS = [
    ("2",  "benchmark_synthetic_var_1_seed11"),
    ("3",  "benchmark_synthetic_var_2_seed22"),
    ("17", "benchmark_synthetic_var_3_seed33"),
    ("40", "benchmark_synthetic_var_4_seed44"),
    ("58", "benchmark_synthetic_var_5_seed55"),
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

data = {}
for model, _, _, _ in METHODS:
    line = []
    for _, folder in SCENARIOS:
        df = pd.read_csv(os.path.join(BASE, folder, "validation_summary.csv")).set_index("model")
        if model not in df.index:
            line.append(None)
            print(f"[WARNING] {model}: no result in {folder}")
            continue
        r = df.loc[model]
        line.append({"precision": float(r.precision), "recall": float(r.recall), "f1": float(r.f1)})
    data[model] = line

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["svg.fonttype"] = "path"

W, H = 828 / 72.0, 300 / 72.0  # +41pt of height vs. the original (259.2pt), to fit title+legend
fig, axes = plt.subplots(1, 3, figsize=(W, H), constrained_layout=True)

x = list(range(len(SCENARIOS)))
for ax, (met, title) in zip(axes, METRICS):
    ax.grid(axis="y", color="#e0e0e0", lw=0.6, zorder=1)
    ax.set_axisbelow(True)

    for model, label, color, marker in METHODS:
        y = [(d[met] if d is not None else float("nan")) for d in data[model]]
        ax.plot(x, y, color=color, lw=1.3, marker=marker, markersize=5,
                markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.6,
                label=label, zorder=3)

    ax.set_xlim(-0.4, len(SCENARIOS) - 0.6)
    ax.set_ylim(-0.03, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title(title, fontsize=10.5)
    ax.tick_params(labelsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in SCENARIOS], fontsize=8.5)
    ax.set_xlabel("True causal connections\nin ground truth", fontsize=7.8, labelpad=4)

fig.suptitle(f"Performance across synthetic scenarios (τ_max = {WINDOW})", fontsize=12.0)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=6, fontsize=8.5,
           frameon=False, columnspacing=1.3, handletextpad=0.5)

fig.savefig(OUT, format="svg")
print("saved to", OUT)
print()
for met, _ in METRICS:
    print(f"  {met}:")
    for model, label, _, _ in METHODS:
        print(f"    {label:10s} " + " ".join(
            (f"{d[met]:.3f}" if d is not None else "  -  ") for d in data[model]))
