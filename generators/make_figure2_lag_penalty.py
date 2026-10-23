"""
Figure 2 (lag-pen-en_tau10.svg) and Figure S3 (lag-pen-en_tau24.svg) --
lag-penalization functions, computed with the actual formula of
KneeLagPenalty (Pipeline/penalties.py).

Usage:
    python generators/make_figure2_lag_penalty.py        # tau_max = 10 (main window, Figure 2)
    python generators/make_figure2_lag_penalty.py 24     # tau_max = 24 (sensitivity, Figure S3)

The paper's main window became tau_max = 10; the previous version of this
figure (lag-pen-en.svg) was drawn only for tau_max = 24. The knees
(knee1=5, knee2=10) do not depend on the window: with tau_max = 10 the axis
ends at knee2 and the third zone (tau > 10) is never used; it only appears
at the window of 24.

History: the original file (lag-pen.svg) was in Portuguese and used
knee1=4, knee2=6 (the class's constructor defaults, never used in
production) instead of knee1=5, knee2=10 (Supplementary Table S4.5).

The KneeLagPenalty curve is computed with the exact formula of the class
(three exponential zones, continuous at the knees), not redrawn by hand.
The other curves (simple exponential, linear decay, no penalization) are
illustrative of pluggable alternatives, in the same spirit as the original
file.
"""
import os
import sys

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
import numpy as np

TAU_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 10
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import origins as O  # noqa: E402
OUT = O.fig(f"lag-pen-en_tau{TAU_MAX}.svg")

KNEE1, KNEE2 = 5, 10
RATE1, RATE2, RATE3 = 0.02, 0.15, 0.5


def knee_lag_penalty(tau, knee1=KNEE1, knee2=KNEE2, r1=RATE1, r2=RATE2, r3=RATE3):
    """Exact formula from KneeLagPenalty.__call__ (Pipeline/penalties.py)."""
    pk1 = np.exp(-r1 * (knee1 - 1))
    pk2 = pk1 * np.exp(-r2 * (knee2 - knee1))
    out = np.empty_like(tau, dtype=float)
    for i, t in enumerate(tau):
        if t <= knee1:
            out[i] = np.exp(-r1 * (t - 1))
        elif t <= knee2:
            out[i] = pk1 * np.exp(-r2 * (t - knee1))
        else:
            out[i] = pk2 * np.exp(-r3 * (t - knee2))
    return out


def exponential_penalty(tau, rate=0.20):
    """Exact formula from ExponentialLagPenalty.__call__."""
    return np.exp(-rate * (tau - 1))


def linear_penalty(tau, tau_max=TAU_MAX):
    """Illustrative straight-line decay to 0 at tau_max (not a named
    pipeline class -- shown, as in the original figure, purely as a
    contrasting shape a user could plug in via the same interface)."""
    return np.clip(1.0 - (tau - 1) / (tau_max - 1), 0.0, 1.0)


def uniform_penalty(tau):
    return np.ones_like(tau, dtype=float)


tau = np.arange(1, TAU_MAX + 1)
y_knee = knee_lag_penalty(tau)
y_exp = exponential_penalty(tau)
y_lin = linear_penalty(tau)
y_uni = uniform_penalty(tau)
has_zone3 = TAU_MAX > KNEE2

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["svg.fonttype"] = "path"

fig, ax = plt.subplots(figsize=(680 / 72.0, 490 / 72.0))

# background bands per zone (green/yellow/red), as in the original
ax.axvspan(0.5, KNEE1 + 0.5, color="#EAF3DE", zorder=0, lw=0)
ax.axvspan(KNEE1 + 0.5, min(KNEE2, TAU_MAX) + 0.5, color="#FAEED0", zorder=0, lw=0)
if has_zone3:
    ax.axvspan(KNEE2 + 0.5, TAU_MAX + 0.5, color="#FADCD9", zorder=0, lw=0)

ax.plot(tau, y_uni, color="#999999", lw=1.4, ls="--", label="No penalization (λ = 1)", zorder=3)
ax.plot(tau, y_lin, color="#3B9B4A", lw=1.4, label="Linear decay (illustrative)", zorder=3)
ax.plot(tau, y_exp, color="#E8895A", lw=1.6, label="Simple exp. (rate = 0.20)", zorder=3)
ax.plot(tau, y_knee, color="#1F6FB2", lw=2.2, label="KneeLagPenalty (default)", zorder=4)

# mark the knees that fall within the window
for name, k in (("knee₁", KNEE1), ("knee₂", KNEE2)):
    if k > TAU_MAX:
        continue
    v = knee_lag_penalty(np.array([k]))[0]
    ax.plot([k], [v], marker="o", color="#1F6FB2", markersize=6,
            markeredgecolor="white", markeredgewidth=1.0, zorder=5)
    at_edge = k >= TAU_MAX  # no room on the right: label to the left of and below the curve
    ax.annotate(f"λ({k}) = {v:.3f}  ({name})", xy=(k, v),
                xytext=(k, v - 0.13) if at_edge else (k + 0.06 * TAU_MAX, v + 0.10),
                ha="right" if at_edge else "left", fontsize=9, color="#1F6FB2")

ax.text((1 + KNEE1) / 2, 1.10, f"green zone\n(τ 1–{KNEE1})", ha="center", va="bottom",
        fontsize=7.5, color="#3B6D11")
ax.text((KNEE1 + 1 + min(KNEE2, TAU_MAX)) / 2, 1.10,
        f"yellow zone\n(τ {KNEE1 + 1}–{min(KNEE2, TAU_MAX)})" if not has_zone3 else f"yellow\n({KNEE1 + 1}–{KNEE2})",
        ha="center", va="bottom", fontsize=7.5, color="#8A6D00")
if has_zone3:
    ax.text((KNEE2 + TAU_MAX) / 2, 1.10, f"red zone (τ > {KNEE2})", ha="center", va="bottom",
            fontsize=7.5, color="#A33B2E")

ax.set_xlim(0.5, TAU_MAX + 0.5)
ax.set_ylim(-0.03, 1.30)
ax.set_xticks(list(range(1, TAU_MAX + 1, 1)))
ax.tick_params(axis="x", labelsize=9 if TAU_MAX <= 12 else 7.5)
ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
ax.set_ylabel("λ(τ)", fontsize=11)
ax.set_xlabel("lag τ (periods)", fontsize=10)
ax.grid(axis="y", color="#e5e5e5", lw=0.6, zorder=1)
ax.set_axisbelow(True)

ax.set_title(f"Lag-penalization functions λ(τ), lag window τ_max = {TAU_MAX}", fontsize=13, pad=14)
fig.text(0.5, 0.955,
         f"Four variants of λ(τ): default KneeLagPenalty (blue, knee₁={KNEE1}, knee₂={KNEE2}),\n"
         "simple exponential (coral), illustrative linear decay (green), and no penalization (gray).\n"
         f"Lag τ from 1 to {TAU_MAX} periods on the X axis, weight λ from 0 to 1 on the Y axis.",
         ha="center", va="top", fontsize=8.3, color="#444444")

ax.legend(loc="upper right" if has_zone3 else "lower left", fontsize=8.5, frameon=False)
footer = f"knee₁ = {KNEE1}, knee₂ = {KNEE2}  ·  rate₁ = {RATE1}  ·  rate₂ = {RATE2}"
footer += f"  ·  rate₃ = {RATE3}" if has_zone3 else f"  ·  rate₃ = {RATE3} (τ > {KNEE2}, outside this window)"
fig.text(0.5, 0.01, footer + "  (Supplementary Table S4.5)",
         ha="center", va="bottom", fontsize=8, color="#555555")

fig.subplots_adjust(top=0.80, bottom=0.13, left=0.09, right=0.97)
fig.savefig(OUT, format="svg")
print("saved to", OUT)
for k in sorted({KNEE1, min(KNEE2, TAU_MAX), TAU_MAX}):
    print(f"tau={k} -> lambda={knee_lag_penalty(np.array([k]))[0]:.4f}")
