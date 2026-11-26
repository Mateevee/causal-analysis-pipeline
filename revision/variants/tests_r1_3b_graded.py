"""
Unit tests of the graded variant of R1-3b (protocol, section 5), without running the pipeline.

  python revision/variants/tests_r1_3b_graded.py

T1  w_inc = 0.5 at the boundary z = min(atanh(min_rho) - h, h).
T2  w_inc tends to 0 far above the boundary and to 1 far below it.
T3  w_inc rounded (< 0.5 -> 0; >= 0.5 -> 1) reproduces the hard rule with G1 in the 90
    scenarios: regime recomputed by Pipeline.coherence.assess_evidence_coherence
    (overlap='coherent') on the Stage 6 table stored in the paper's runs (rows with
    lag > 0); also checks that the recomputed rho matches the stored one.
T4  w_lean = 0.5 when k = n/2.
T5  interpolation with w = 1 and w = 0 returns the hard decisions; with an intermediate w,
    the floor(w * m + 0.5) with the highest evidence enter.
T6  _combine with w = 1 and w = 0 returns the corresponding design (fields and decision).
T7  install()/uninstall() replaces only the guard and restores the tuple; the graded module
    forwards the attributes of the original dml_engine.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
sys.path.insert(0, REV)
sys.path.insert(0, os.path.dirname(REV))

from variants import r1_3b_graded as g  # noqa: E402

FAILURES = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILURES.append(msg)


def t1_t2():
    from scipy import stats
    for n, min_rho, alpha in ((30, 0.3, 0.05), (200, 0.3, 0.05), (15, 0.1, 0.05), (80, 0.5, 0.05)):
        se = math.sqrt(1.06 / (n - 3))
        h = stats.norm.ppf(1 - alpha) * se
        zf = min(np.arctanh(min_rho) - h, h)
        w = g.weight_inc(float(np.tanh(zf)), n, "coherent", min_rho, alpha)
        check(abs(w - 0.5) < 1e-9, f"T1 w_inc = 0.5 at the boundary (n={n}, min_rho={min_rho}): {w:.12f}")
    check(g.weight_inc(0.95, 200, "coherent") < 1e-6, "T2 w_inc ~ 0 far above (rho=0.95, n=200)")
    check(g.weight_inc(-0.6, 200, "incoherent") > 1 - 1e-6, "T2 w_inc ~ 1 far below (rho=-0.6, n=200)")
    check(g.weight_inc(float("nan"), 5, "undetermined") == 0.0, "T2 no rho and 'undetermined' -> 0")
    check(g.weight_inc(float("nan"), 50, "incoherent") == 1.0, "T2 no rho and 'incoherent' (no TE) -> 1")


def t3():
    import common as c
    from Pipeline import coherence
    n_ok = n_rho = n_scen = 0
    discord, rho_diff = [], []
    for s in c.scenarios():
        d = c.original(s)
        if d.empty:
            continue
        lag = pd.to_numeric(d["lag"], errors="coerce")
        t6 = d[lag > 0].copy()
        if t6.empty:
            continue
        n_scen += 1
        r = coherence.assess_evidence_coherence(t6, min_rho=0.3, alpha=0.05, verbose=False, overlap="coherent")
        w = g.weight_inc(r["rho"], r["n"], r["regime"], 0.3, 0.05)
        if (w >= 0.5) == (r["regime"] == "incoherent"):
            n_ok += 1
        else:
            discord.append((s, r["regime"], round(w, 4)))
        gr = pd.to_numeric(d.get("evidence_coherence_rho"), errors="coerce").dropna() \
            if "evidence_coherence_rho" in d else pd.Series(dtype=float)
        if len(gr) and np.isfinite(r["rho"]):
            if abs(float(gr.iloc[0]) - r["rho"]) < 1e-6:
                n_rho += 1
            else:
                rho_diff.append((s, float(gr.iloc[0]), r["rho"]))
    check(not discord and n_ok == n_scen, f"T3 rounded w_inc = hard rule (G1) in {n_ok} of {n_scen} scenarios "
                                          f"{discord[:5]}")
    print(f"       (recomputed rho = stored in {n_rho} scenarios; different: {len(rho_diff)} {rho_diff[:3]})")


def t4():
    for n in (2, 10, 50, 300):
        check(abs(g.weight_lean(n // 2, n) - 0.5) < 1e-12, f"T4 w_lean = 0.5 with k = n/2 (n={n})")
    check(g.weight_lean(90, 100) > 0.999 and g.weight_lean(10, 100) < 0.001, "T4 w_lean at the extremes")
    check(math.isnan(g.weight_lean(0, 0)), "T4 w_lean without pairs -> NaN (today's rule: blanket)")


def t5():
    rng = np.random.default_rng(1)
    for _ in range(200):
        m = int(rng.integers(1, 40))
        A, B, ev = rng.random(m) < 0.5, rng.random(m) < 0.5, rng.random(m)
        check_q = (np.array_equal(g.interpolate(A, B, 1.0, ev), B)
                   and np.array_equal(g.interpolate(A, B, 0.0, ev), A))
        if not check_q:
            check(False, "T5 interpolation with w = 1 / w = 0")
            return
    check(True, "T5 interpolation with w = 1 returns B and with w = 0 returns A (200 random cases)")
    A = np.array([True, False, False, False, False, True])
    B = np.array([True, True, True, True, False, False])
    ev = np.array([9, 1, 3, 2, 8, 5])
    out = g.interpolate(A, B, 0.5, ev)        # only B: indices 1,2,3 (m=3) -> floor(1.5+0.5)=2 with highest ev: 2,3
    check(out.tolist() == [True, False, True, True, False, True], f"T5 intermediate interpolation {out.tolist()}")


def t6():
    lean = [{"effect": 1.0, "se": 0.5, "ci_lower": 0.0, "ci_upper": 2.0, "pvalue": 0.04, "significant": True,
             "direction": "positive", "cascade_log": ["L"]},
            {"effect": -0.2, "se": 0.5, "ci_lower": -1.2, "ci_upper": 0.8, "pvalue": 0.7, "significant": False,
             "direction": "negative", "cascade_log": ["L"]}]
    blanket = [{"effect": 0.4, "se": 0.4, "ci_lower": -0.4, "ci_upper": 1.2, "pvalue": 0.3, "significant": False,
                "direction": "positive", "cascade_log": ["B"]},
               {"effect": -0.9, "se": 0.3, "ci_lower": -1.5, "ci_upper": -0.3, "pvalue": 0.01, "significant": True,
                "direction": "negative", "cascade_log": ["B"]}]
    for w, ref in ((1.0, lean), (0.0, blanket)):
        out = g._combine(lean, blanket, w)
        equal = all(all((o[k] == r[k]) for k in ("effect", "se", "ci_lower", "ci_upper", "pvalue", "significant",
                                                 "direction", "cascade_log")) for o, r in zip(out, ref))
        check(equal, f"T6 _combine with w = {w:.0f} returns the corresponding design")
    out = g._combine(lean, blanket, 0.75)
    check(abs(out[0]["effect"] - 0.85) < 1e-12 and abs(out[0]["se"] - 0.475) < 1e-12 and out[0]["cascade_log"] == ["L"],
          "T6 _combine with w = 0.75: weighted mean and fields of the dominant design")


def t7():
    import Pipeline.pipeline as pp
    import Pipeline.dml_engine as dml
    orig = pp._STAGES
    o2 = g.install()
    swapped = [(a.__name__, b.__name__) for a, b in zip(o2, pp._STAGES) if a is not b]
    g.uninstall(o2)
    check(o2 is orig and pp._STAGES is orig and swapped == [("_stage8_regime_guards",
                                                             "_stage8_regime_guards_graded")],
          f"T7 install replaces only the guard and uninstall restores it: {swapped}")
    m = g.graded_dml_module()
    check(m.estimate_all_effects is dml.estimate_all_effects and m.REGIME_TRANSITION == "graded"
          and m.estimate_effects_by_control_set is not dml.estimate_effects_by_control_set,
          "T7 graded module forwards the dml_engine and replaces only estimate_effects_by_control_set")


if __name__ == "__main__":
    for f in (t1_t2, t3, t4, t5, t6, t7):
        print(f.__name__)
        f()
    print("\nRESULT:", "ALL TESTS PASSED" if not FAILURES else f"{len(FAILURES)} FAILURE(S)")
    sys.exit(1 if FAILURES else 0)
