"""
Generates Figure 3 of the paper — architecture of Stage 7 (Dempster-Shafer
evidence fusion).

REASON FOR THIS VERSION (v2): the previous figure
(causal_effect_penalty_dempster_shafer.svg, 2026-08-15) drew in the
"Evidence Fusion" block the HIERARCHICAL rule of three families
(statistical/diagnostic/causal_soft, each one via Murphy, near-veto
entering directly) — which is the OPT-IN variant of the code
(COMBINATION_RULE='hierarchical'), NOT the default rule.

The actual default is `COMBINATION_RULE = "hybrid_gated"`
(Pipeline/SelectorFunc.py) -> `_combine_gated_hybrid`:

    if cascade_confidence == 'high' AND n_models_agree >= 3:
        -> _combine_light_hybrid  (Murphy only on the causal_soft cluster,
                                   then Yager)
    else:
        -> _combine_all(rule='yager')  (Yager directly on all of them)

This version fixes the fusion block and sets the count of the primary
layer to 11 active specialists (10 core sources drawn + the legacy
provenance prior, P11 in Table S6), so that 11 + 2 + 4 = 17 matches
Table S6.

Layout rebuilt from the original SVG (positions, sizes and colors
extracted from it), to keep visual fidelity.
"""
import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["svg.fonttype"] = "path"

W, H = 568.44, 775.584          # pt (identical to the original)
fig = plt.figure(figsize=(W / 72.0, H / 72.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)               # y grows downward, as in the SVG
ax.axis("off")
ax.add_patch(Rectangle((0, 0), W, H, fc="#ffffff", ec="none", zorder=0))

CX = 284.22                     # central axis of the figure

# colors extracted from the original SVG
BLUE, PURPLE, GREEN = "#0072b2", "#7f5aa2", "#009e73"
AMBER, SLATE = "#e69f00", "#455a7a"
ORANGE, DARK_GREEN, BROWN = "#d55e00", "#2f6b00", "#8a3300"
GRAY_T, GRAY_C = "#555555", "#333333"


def box(x0, y0, x1, y1, fc, ec, lw=1.0, z=2):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                           fc=fc, ec=ec, lw=lw, zorder=z))


def txt(x, y, s, fs, color=GRAY_C, weight="normal", ha="center", z=5):
    ax.text(x, y, s, fontsize=fs, color=color, fontweight=weight,
            ha=ha, va="center", zorder=z)


def line(x0, y0, x1, y1, color=GRAY_C, lw=0.9):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, zorder=1,
            solid_capstyle="butt")


def curve(p0, ctrl, p1, color=GRAY_C, lw=0.9):
    """Quadratic Bezier, like the 'elbow' connectors of the original."""
    ax.add_patch(PathPatch(Path([p0, ctrl, p1],
                                [Path.MOVETO, Path.CURVE3, Path.CURVE3]),
                           fc="none", ec=color, lw=lw, zorder=1))


# ── title ─────────────────────────────────────────────────────────────
txt(CX, 27.0, "Stage 7 — Evidence-Based Filtering", 13.5, "#222222", "bold")
txt(CX, 40.5, "Dempster-Shafer evidence-fusion architecture", 9.5, GRAY_T)

# ── input: DML output (Stage 6) ───────────────────────────────────────
box(120.1, 56.5, 448.3, 83.6, "#e0e0e0", GRAY_C)
txt(CX, 66.5, "DML output (Stage 6)", 9.0, "#222222", "bold")
txt(CX, 76.5, "θ̂, SE, CI, p-value, TE, bootstrap, ESS, lag, cascade agreement, ...",
    6.3, "#222222")

# ── structural gates ──────────────────────────────────────────────────
line(CX, 83.5, CX, 101.8)
box(120.1, 106.6, 448.3, 133.7, ORANGE, GRAY_C)
txt(CX, 116.5, "Structural Gates  (deterministic, domain-agnostic)",
    8.3, "#ffffff", "bold")
txt(CX, 126.5, "null / ill-defined effect  ·  invalid statistics  ·  near-zero marginal variance",
    6.0, "#ffffff")

# immediate-rejection branch
line(448.3, 120.1, 493.6, 120.1)
box(497.4, 108.2, 555.0, 132.0, BROWN, GRAY_C)
txt(526.2, 116.8, "reject", 7.0, "#ffffff", "bold")
txt(526.2, 125.6, "w = 0", 5.8, "#ffffff")
txt(470.0, 105.0, "fails", 6.6, BROWN)

# ── feature engineering ───────────────────────────────────────────────
line(CX, 133.6, CX, 152.4)
txt(300.0, 145.0, "passes", 6.6, DARK_GREEN, ha="left")
box(120.1, 156.7, 448.3, 183.7, "#999999", GRAY_C)
txt(CX, 166.7, "Feature Engineering", 9.0, "#ffffff", "bold")
txt(CX, 176.5, "derives CV(CI), robustness residual, ESS_min, lag coherence, asymmetry, ...",
    6.2, "#ffffff")

# ── envelope: evidence gathering ──────────────────────────────────────
line(CX, 183.7, CX, 192.8)
box(27.1, 194.4, 541.3, 444.4, "#f4f6f9", SLATE, z=1)
txt(CX, 204.0, "Evidence Gathering", 11.2, SLATE, "bold")
txt(CX, 213.3, "three independent belief-mass sources, computed in parallel from the Feature Engineering output",
    6.8, GRAY_T)

# ── primary layer ─────────────────────────────────────────────────────
box(49.0, 234.8, 519.4, 348.5, "#ffffff", BLUE)
txt(CX, 243.5, "Primary Evidence  —  11 active specialists (10 core sources shown)",
    9.2, BLUE, "bold")
txt(CX, 252.8, "each contributes an independent Dempster-Shafer belief mass  {H, ¬H, Θ}  from one non-overlapping evidence source",
    6.3, GRAY_T)

PRIM = [
    ("Transfer", "Entropy"), ("DML", "t-statistic"), ("Bootstrap", "sign agree."),
    ("Permutation", "p-value"), ("Robustness", "residual"),
    ("Residual", "mis-spec."), ("Effective", "sample size"), ("Lag", "coherence"),
    ("Directional", "asymmetry"), ("Cascade", "agreement"),
]
X0, WIDTH, STEP = 59.4, 82.1, 91.9
for i, (l1, l2) in enumerate(PRIM):
    col, row = i % 5, i // 5
    x = X0 + col * STEP
    y0 = 260.6 + row * 42.6
    box(x, y0, x + WIDTH, y0 + 35.7, "#e6f0f7", BLUE, lw=0.8, z=3)
    txt(x + WIDTH / 2, y0 + 12.5, l1, 6.4, BLUE, z=6)
    txt(x + WIDTH / 2, y0 + 23.0, l2, 6.4, BLUE, z=6)

# ── meta layer ────────────────────────────────────────────────────────
box(49.0, 360.3, 276.1, 430.4, "#ffffff", PURPLE)
txt(162.5, 368.5, "Meta Evidence", 8.4, PURPLE, "bold")
txt(162.5, 376.6, "2 of 5 implemented active", 6.1, GRAY_T)
for x0, (l1, l2) in zip((62.4, 167.4),
                        (("Cross-specialist", "directional consistency"),
                         ("Minimum non-statistical", "support check"))):
    box(x0, 388.3, x0 + 95.3, 422.3, "#efe8f5", PURPLE, lw=0.8, z=3)
    txt(x0 + 95.3 / 2, 399.5, l1, 6.2, PURPLE, z=6)
    txt(x0 + 95.3 / 2, 410.5, l2, 6.2, PURPLE, z=6)

# ── interaction layer ─────────────────────────────────────────────────
box(292.4, 360.3, 519.4, 430.4, "#ffffff", GREEN)
txt(405.9, 368.5, "Interaction Specialists", 8.2, GREEN, "bold")
txt(405.9, 376.6, "4 of 7 implemented active", 6.1, GRAY_T)
for x0, (l1, l2) in zip((297.8, 353.1, 408.3, 463.5),
                        (("Bootstrap ×", "residuals"), ("Robustness ×", "RESET"),
                         ("ESS ×", "bootstrap"), ("Weak consensus", "× no TE"))):
    box(x0, 388.3, x0 + 50.4, 422.3, "#e2f3ee", GREEN, lw=0.8, z=3)
    txt(x0 + 50.4 / 2, 399.5, l1, 5.5, GREEN, z=6)
    txt(x0 + 50.4 / 2, 410.5, l2, 5.5, GREEN, z=6)

# ── evidence fusion (FIXED BLOCK) ─────────────────────────────────────
line(CX, 444.3, CX, 472.3)
box(43.5, 478.2, 524.9, 537.6, "#ffffff", AMBER)
OCHRE = "#6b4e00"
txt(CX, 486.5, "Evidence Fusion  (gated hybrid — default rule)", 9.6, AMBER, "bold")
txt(CX, 497.8, "gated by the reliability of the underlying DML estimate: when cascade confidence = high AND ≥ 3 of 4 models agree on sign,",
    6.4, OCHRE)
txt(CX, 507.0, "four correlated specialists are first consolidated via Murphy’s average rule; otherwise no averaging step is applied",
    6.4, OCHRE)
txt(163.0, 517.2, "all resulting sources combined", 7.0, OCHRE)
txt(163.0, 526.9, "via Yager’s modified combination rule", 7.0, OCHRE)
txt(284.0, 522.0, "→", 15.0, OCHRE)
txt(410.0, 517.2, "conflict redirected to", 7.0, OCHRE)
txt(410.0, 526.9, "uncertainty (Θ), not discarded", 7.0, OCHRE)

# ── final decision ────────────────────────────────────────────────────
line(CX, 537.5, CX, 555.8)
box(92.8, 557.4, 475.7, 596.8, "#ffffff", GRAY_C)
txt(CX, 567.3, "Final Decision", 9.6, GRAY_C, "bold")
txt(CX, 578.0, "Support, Opposition, Uncertainty, Conflict, Consensus, Evidence Quality",
    6.5, GRAY_C)
txt(CX, 588.3, "accept iff  net-support ratio ≥ threshold  AND  support ≥ floor  AND  opposition < ceiling",
    6.7, GRAY_C)

# ── binary branching ──────────────────────────────────────────────────
line(CX, 596.8, CX, 613.4)
box(274.2, 605.2, 294.3, 625.0, "#ffffff", GRAY_C, z=4)
txt(CX, 615.1, "w", 8.0, GRAY_C, "bold", z=6)
curve((274.4, 623.7), (208.5, 579.1), (165.3, 642.9))
curve((294.1, 623.7), (359.9, 579.1), (403.1, 642.9))
txt(126.0, 609.0, "w = 1", 7.2, DARK_GREEN)
txt(443.0, 609.0, "w = 0", 7.2, BROWN)

box(65.4, 645.2, 262.4, 679.8, DARK_GREEN, GRAY_C)
txt(163.9, 657.5, "Accept", 9.5, "#ffffff", "bold")
txt(163.9, 670.0, "θ̂ preserved unchanged", 6.6, "#ffffff")

box(306.0, 645.2, 503.0, 679.8, BROWN, GRAY_C)
txt(404.5, 657.5, "Reject", 9.5, "#ffffff", "bold")
txt(404.5, 670.0, "pair discarded", 6.6, "#ffffff")

# ── footnotes ─────────────────────────────────────────────────────────
txt(CX, 694.0, "Mutually exclusive outcomes: an accepted pair is never also rejected, and vice versa.",
    7.4, GRAY_C)
txt(CX, 704.8, "Support, opposition and uncertainty can be recomputed per pair from the stored inputs.",
    7.0, GRAY_T)

import os  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import origins as O  # noqa: E402
OUT = O.fig("causal_effect_penalty_dempster_shafer_v2.svg")
fig.savefig(OUT, format="svg")
print("saved to", OUT)
