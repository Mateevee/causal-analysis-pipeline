"""
Figures 1 and 2 of the IEEE Access version of the paper (ferre1.pdf, ferre2.pdf),
drawn at the journal's final size: the column (3.35 in) or text width (6.97 in)
of the Access-Template-2024, Arial of 8 pt or more at final scale, no title
inside the figure (it is in the caption), fonts embedded in the PDF. Each figure
is written as ferreN.pdf (vector, the final figure file; IEEE names figures by
the first five letters of the first author's surname and the figure number) and
ferreN.png (600 dpi, the copy placed in the manuscript).

Usage:
    python generators/make_figures_ieee.py          # both figures
    python generators/make_figures_ieee.py 1        # Figure 1 only
    python generators/make_figures_ieee.py 2        # Figure 2 only

Figure 1 is the authors' diagram of the eight stages, in two rows; it reads no
run. Figure 2 computes KneeLagPenalty with the exact formula of the class
(Pipeline/penalties.py), with knee1 = 5 and knee2 = 10 (Supplementary Table
S4.5), at tau_max = 10; its tau_max = 24 counterpart (Figure S4) is drawn by
make_figure2_lag_penalty.py.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Rectangle
import numpy as np
from PIL import ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import origins as O  # noqa: E402

COL, FULL = 3.35, 6.97          # inches: column and text width of the template

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
})


def save(fig, n):
    fig.savefig(O.fig(f"ferre{n}.pdf"))
    fig.savefig(O.fig(f"ferre{n}.png"), dpi=600)
    plt.close(fig)
    print("ok ->", O.fig(f"ferre{n}.pdf"), "/ .png")


# ======================================================================== Figure 2
def knee_lag_penalty(tau, knee1=5, knee2=10, r1=0.02, r2=0.15, r3=0.5):
    """Exact formula from KneeLagPenalty.__call__ (Pipeline/penalties.py)."""
    pk1 = np.exp(-r1 * (knee1 - 1))
    pk2 = pk1 * np.exp(-r2 * (knee2 - knee1))
    return np.array([np.exp(-r1 * (t - 1)) if t <= knee1 else
                     pk1 * np.exp(-r2 * (t - knee1)) if t <= knee2 else
                     pk2 * np.exp(-r3 * (t - knee2)) for t in tau])


def fig2():
    tau_max, k1, k2 = 10, 5, 10
    tau = np.arange(1, tau_max + 1)
    fig, ax = plt.subplots(figsize=(COL, 3.0))
    ax.axvspan(0.5, k1 + 0.5, color="#EAF3DE", zorder=0, lw=0)
    ax.axvspan(k1 + 0.5, tau_max + 0.5, color="#FAEED0", zorder=0, lw=0)
    ax.plot(tau, np.ones_like(tau, dtype=float), color="#888888", lw=1.1, ls="--",
            label="No penalty (λ = 1)", zorder=3)
    ax.plot(tau, np.clip(1.0 - (tau - 1) / (tau_max - 1), 0, 1), color="#3B9B4A", lw=1.1, ls=":",
            label="Linear (illustrative)", zorder=3)
    ax.plot(tau, np.exp(-0.20 * (tau - 1)), color="#E8895A", lw=1.2, ls="-.",
            label="Exponential (rate 0.20)", zorder=3)
    y = knee_lag_penalty(tau)
    ax.plot(tau, y, color="#1F6FB2", lw=2.0, label="KneeLagPenalty (default)", zorder=4)
    # knee1 above the dashed line; knee2 in the gap between the default curve and the others
    for k, label, xy_txt, ha, va in ((k1, f"knee$_1$, λ = {y[k1 - 1]:.3f}", (k1, 1.04), "center", "bottom"),
                                     (k2, f"knee$_2$\nλ = {y[k2 - 1]:.3f}", (k2 - 0.2, 0.425), "right", "top")):
        ax.plot([k], [y[k - 1]], marker="o", color="#1F6FB2", markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        ax.text(*xy_txt, label, ha=ha, va=va, fontsize=8, color="#1F6FB2", zorder=6)
    ax.text((1 + k1) / 2, 1.20, f"zone 1 (τ 1–{k1})", ha="center", fontsize=8, color="#3B6D11")
    ax.text((k1 + 1 + tau_max) / 2, 1.20, f"zone 2 (τ {k1 + 1}–{tau_max})", ha="center", fontsize=8,
            color="#8A6D00")
    ax.set_xlim(0.5, tau_max + 0.5)
    ax.set_ylim(-0.03, 1.32)
    ax.set_xticks(tau)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Lag τ (periods)")
    ax.set_ylabel("Penalty weight λ(τ)")
    ax.grid(axis="y", color="#e5e5e5", lw=0.5, zorder=1)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
               handlelength=2.2, columnspacing=1.0, handletextpad=0.4, labelspacing=0.3,
               bbox_to_anchor=(0.54, 0.0))
    fig.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.30)
    save(fig, 2)
    print("   ", {k: round(float(v), 4) for k, v in zip(tau, y)})


# ======================================================================== Figure 1
# Text is wrapped by measuring it in the same Arial that matplotlib draws with.
_FONT = {bold: ImageFont.truetype(font_manager.findfont(
             font_manager.FontProperties(family="Arial", weight="bold" if bold else "normal"),
             fallback_to_default=False), 1000)
         for bold in (False, True)}


def width(s, pt, bold=False):
    return _FONT[bold].getlength(s) * pt / 1000.0


def wrap(text, pt, maxw, bold=False):
    lines, current = [], ""
    for word in text.split():
        cand = (current + " " + word).strip()
        if width(cand, pt, bold) <= maxw or not current:
            current = cand
        else:
            lines.append(current)
            current = word
    return lines + ([current] if current else [])


# Box content: (badge, title, fill, edge, items); an item is ("t", text) or, when the
# decision is taken from the data alone, ("d", text), drawn with a diamond.
BOXES = {
    "1": ("1", "Active-property detection", "#EAF3DE", "#3B6D11",
          [("t", "Identifies pairs (entity, property), eliminating series that are almost always zero "
                 "before causal analysis.")]),
    "2": ("2", "Time-series preprocessing", "#E6F1FB", "#185FA5",
          [("t", "Structural zeros, bounded interpolation, normalization.")]),
    "3": ("3", "Cross-correlation screening", "#EEEDFE", "#3C3489",
          [("t", "(lagged Pearson/Xcorr)"), ("t", "Fast linear pair filter.")]),
    "4": ("4", "Transfer Entropy with permutation test", "#EEEDFE", "#3C3489",
          [("t", "Nonlinear causal detection.")]),
    "5": ("5", "Rescue", "#EEEDFE", "#3C3489",
          [("t", "(Granger + Pearson bootstrap)"), ("t", "Recovers pairs lost to sparsity."),
           ("d", "Lag-multiplicity correction only if the data has a time axis.")]),
    "6": ("6", "Cascaded DML", "#FAEEDA", "#854F0B",
          [("t", "Bayesian Ridge → Random Forest → KRR → Nadaraya-Watson"),
           ("t", "Effect θ estimation (CI, p-value)."),
           ("d", "Controls: target's past + other candidate parents; blanket if near-deterministic.")]),
    "7": ("7", "Evidence-based filtering", "#FBEAF0", "#9D174D",
          [("t", "Dempster-Shafer fusion of the evidence of seventeen specialists defines w in {0,1} "
                 "(configurable)."),
           ("d", "Support floor calibrated per dataset.")]),
    "8a": ("8a", "Structural pruning", "#E3F6F1", "#0E7C66",
           [("t", "Parent competition (conditional MI), edge orientation, BIC pruning."),
            ("d", "8a-bis: predictive parent competition if near-deterministic (may abstain).")]),
    "g": ("◆", "Regime guards", "#F1F3F5", "#495057",
          [("d", "TE and DML evidence disagree: keep the pairs with a significant Stage-6 effect "
                 "(flagged), unless the past explains little."),
           ("d", "No time axis: drop every lagged edge.")]),
    "8b": ("8b", "Contemporaneous completion", "#E3F6F1", "#0E7C66",
           [("t", "Adds lag-0 adjacencies (PC-stable on VAR innovations), unoriented."),
            ("t", "Lag-0 DML effect estimate."),
            ("d", "Only if the past explains little of the series.")]),
}
PLUGGABLE = {"7"}
GROUPS = [("Correlation pre-filter", "Lag definition · pluggable λ(τ) penalization",
           "FDR-BH across pairs · max-statistic permutation over lags", "#534AB7", "#26215C", ["3", "4", "5"]),
          ("Graph-level structural adjustment", "Acts on the whole graph · no new lagged pairs", None,
           "#0E7C66", "#04342C", ["8a", "g", "8b"])]
ROWS = [["1", "2", "3", "4", "5"], ["6", "7", "8a", "g", "8b"]]
DIAMOND_COLOR = "#A64B0A"
PT_T, PT_C, LH_T, LH_C = 8.5, 8.0, 10.0, 9.4      # title and body: size and line height (pt)


def diamond(ax, x, y, s=2.6, color=DIAMOND_COLOR):
    ax.add_patch(Polygon([(x, y - s), (x + s, y), (x, y + s), (x - s, y)], closed=True, color=color, lw=0,
                         zorder=6))


def plug(ax, x, y, color):
    """Icon of a pluggable component: a square with a plug."""
    ax.add_patch(FancyBboxPatch((x, y), 11, 11, boxstyle="round,pad=0,rounding_size=2", fc="white",
                                ec=color, lw=0.8, zorder=6))
    ax.add_patch(Rectangle((x + 3.2, y + 3.0), 4.6, 3.6, fc="none", ec=color, lw=0.8, zorder=7))
    for dx in (4.4, 6.6):
        ax.plot([x + dx, x + dx], [y + 1.6, y + 3.0], color=color, lw=0.8, zorder=7)
    ax.plot([x + 5.5, x + 5.5], [y + 6.6, y + 9.2], color=color, lw=0.8, zorder=7)


def box_layout(key, w):
    _, title, _, _, items = BOXES[key]
    maxw = w - 9
    blocks = [("title", line) for line in wrap(title, PT_T, maxw, True)]
    for kind, text in items:
        lines = wrap(text, PT_C, maxw - (7 if kind == "d" else 0))
        blocks.append(("gap", None))
        blocks += [("d1" if (kind == "d" and i == 0) else "c", line) for i, line in enumerate(lines)]
    height = 6 + 14 + 4                                # margin, badge, space
    for kind, _ in blocks:
        height += LH_T if kind == "title" else (3 if kind == "gap" else LH_C)
    return blocks, height + 6


def fig1():
    W = FULL * 72
    mx, channel, g = 3.0, 12.0, 11.0
    w = (W - mx - channel - 4 * g) / 5
    head, foot, pad_g = 26.0, 13.0, 4.0               # group header and footer, group padding
    lay = {k: box_layout(k, w) for k in BOXES}
    row_h = [max(lay[k][1] for k in row) for row in ROWS]
    y = 4.0
    y_r1 = y + head                                   # top of the boxes of row 1
    y_end1 = y_r1 + row_h[0] + foot + pad_g
    y_channel = y_end1 + 7
    y_r2 = y_channel + 7 + head
    y_end2 = y_r2 + row_h[1] + pad_g
    H = y_end2 + 22
    fig = plt.figure(figsize=(FULL, H / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    pos = {}
    for ri, (row, y_top) in enumerate(zip(ROWS, (y_r1, y_r2))):
        for j, k in enumerate(row):
            pos[k] = (mx + j * (w + g), y_top, w, row_h[ri])
    # groups
    for title, sub, footer, color, color_t, members in GROUPS:
        x0 = pos[members[0]][0] - pad_g
        x1 = pos[members[-1]][0] + w + pad_g
        yt = pos[members[0]][1] - head
        yb = pos[members[0]][1] + pos[members[0]][3] + (foot if footer else 0) + pad_g
        ax.add_patch(FancyBboxPatch((x0, yt), x1 - x0, yb - yt, boxstyle="round,pad=0,rounding_size=7",
                                    fc="none", ec=color, lw=1.0, ls=(0, (4, 2.5)), zorder=1))
        xc = (x0 + x1) / 2
        ax.text(xc, yt + 4, title, ha="center", va="top", fontsize=9, fontweight="bold", color=color_t)
        ax.text(xc, yt + 15, sub, ha="center", va="top", fontsize=8, fontstyle="italic", color=color)
        if footer:
            ax.text(xc, yb - pad_g - 2, footer, ha="center", va="bottom", fontsize=8, color=color_t)
        plug(ax, x1 - 15, yt + 4, color)
    # boxes
    for k, (x, yt, wk, hk) in pos.items():
        badge, _, fc, ec, _ = BOXES[k]
        ax.add_patch(FancyBboxPatch((x, yt), wk, hk, boxstyle="round,pad=0,rounding_size=5", fc=fc, ec=ec,
                                    lw=0.9, zorder=2))
        xc = x + wk / 2
        ax.add_patch(Circle((xc, yt + 6 + 7), 7, color=ec, zorder=3))
        if badge == "◆":
            diamond(ax, xc, yt + 13, 3.6, "white")
        else:
            ax.text(xc, yt + 13, badge, ha="center", va="center", fontsize=8, fontweight="bold",
                    color="white", zorder=4)
        if k in PLUGGABLE:
            plug(ax, x + wk - 14, yt + 3, ec)
        yc = yt + 6 + 14 + 4
        for kind, text in lay[k][0]:
            if kind == "title":
                ax.text(xc, yc, text, ha="center", va="top", fontsize=PT_T, fontweight="bold", color=ec)
                yc += LH_T
            elif kind == "gap":
                yc += 3
            else:
                ax.text(xc + (3.5 if kind == "d1" else 0), yc, text, ha="center", va="top", fontsize=PT_C,
                        color="#222222")
                if kind == "d1":
                    diamond(ax, xc + 3.5 - width(text, PT_C) / 2 - 5, yc + 3.6)
                yc += LH_C

    # arrows (axes in points: head drawn as a 4.5 x 3.6 pt triangle)
    def arrow(xs, ys):
        ax.plot(xs, ys, color="#555555", lw=0.9, zorder=5, solid_capstyle="butt")
        (xa, ya), (xb, yb) = (xs[-2], ys[-2]), (xs[-1], ys[-1])
        d = np.hypot(xb - xa, yb - ya)
        ux, uy = (xb - xa) / d, (yb - ya) / d
        c, half = 4.5, 1.8
        ax.add_patch(Polygon([(xb, yb), (xb - c * ux - half * uy, yb - c * uy + half * ux),
                              (xb - c * ux + half * uy, yb - c * uy - half * ux)], closed=True,
                             color="#555555", lw=0, zorder=6))

    for row in ROWS:
        for a, b in zip(row, row[1:]):
            xa, ya, wa, ha_ = pos[a]
            yy = ya + ha_ / 2
            arrow([xa + wa, pos[b][0]], [yy, yy])
    x5, y5, w5, h5 = pos["5"]
    x6, y6, w6, _ = pos["6"]
    xr = W - channel / 2
    arrow([x5 + w5, xr, xr, x6 + w6 / 2, x6 + w6 / 2],
          [y5 + h5 / 2, y5 + h5 / 2, y_channel, y_channel, y6])
    # legend
    yl = H - 13
    plug(ax, mx + 2, yl - 5.5, "#534AB7")
    ax.text(mx + 18, yl, "Pluggable/configurable stage", va="center", fontsize=8, color="#222222")
    xl = mx + 18 + width("Pluggable/configurable stage", 8) + 26
    diamond(ax, xl, yl, 3.2)
    ax.text(xl + 7, yl, "Decided from the data alone (no ground truth, no dataset name)", va="center",
            fontsize=8, color="#222222")
    save(fig, 1)
    print(f"    {FULL:.2f} x {H / 72:.2f} in; box {w:.1f} pt")


if __name__ == "__main__":
    for n in sys.argv[1:] or ["1", "2"]:
        {"1": fig1, "2": fig2}[n]()
