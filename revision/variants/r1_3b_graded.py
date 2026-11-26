"""
R1-3b -- variant with graded transitions in the two regime switches.

Protocol: revision/PROTOCOL_graded_transitions_R1-3b.md (frozen), with the
implementation deviation recorded in revision/NOTAS.md ("R1-3b -- DESVIO DO PROTOCOLO",
2026-11-14): the method's code (Pipeline/) is NOT changed; the variant lives here.

How it plugs in, without touching Pipeline/:
  1. Control design (Stage 6): through the existing `dml_engine_module` parameter
     of run_pipeline. `graded_dml_module()` returns a module that forwards everything to
     the original Pipeline.dml_engine and replaces only `estimate_effects_by_control_set`.
     It carries the mark REGIME_TRANSITION = 'graded'.
  2. Coherence guard (Stage 8): `install()` replaces, in Pipeline.pipeline._STAGES,
     `_stage8_regime_guards` with `_stage8_regime_guards_graded`, which calls the
     original unchanged when the run's DML module is not the graded one or
     when w_inc is outside (0.001, 0.999). `uninstall()` undoes the replacement.
  Without the 'graded' mark (any other model), the original code runs.

Weights (protocol, section 3.1):
  w_inc  = Phi((min(atanh(min_rho) - h, h) - atanh(rho)) / se),
           se = sqrt(1.06 / (n - 3)), h = z_{1-alpha} * se;
           without rho: 1 if the regime is 'incoherent' (no candidate with TE), else 0.
  w_lean = P(q > 0.5), q ~ Beta(k + 1, n - k + 1); k = pairs whose best out-of-sample R2
           of the outcome nuisance is below the cut-off, n = pairs with that R2.
Interpolation (section 3.2): pairs kept by both regimes stay; pairs kept by neither
leave; of those kept only by the regime with weight w, the floor(w * m + 0.5) with the
highest evidence enter.

Usage (in the process that runs the pipeline):
  from variants import r1_3b_graded as g
  orig = g.install()
  try:
      m = g.graded_model(ce)          # patch_tau10_graded
      ... main.run_all(..., models=[m]) ...
  finally:
      g.uninstall(orig)
The module and the model are NOT serializable to another process: in a run with one
process per scenario, call install() and graded_model() inside each process.
"""
from __future__ import annotations

import math
import types

import numpy as np
import pandas as pd
from scipy import stats

import Pipeline.pipeline as pp
import Pipeline.dml_engine as dml
from Pipeline import coherence as coherence_mod  # noqa: F401  (documentation: the original guard)
from Pipeline import contemporaneous as contemp_mod

MODEL_NAME = "patch_tau10_graded"
BASE_FINAL = {"time_axis_rule": "block_portmanteau", "evidence_coherence_overlap": "coherent"}
BAND = (0.001, 0.999)
WEIGHTED_FIELDS = ("effect", "se", "ci_lower", "ci_upper", "pvalue")

_ORIGINAL_GUARD = pp._stage8_regime_guards


# ----------------------------------------------------------------------------- weights
def weight_inc(rho, n, regime, min_rho: float = 0.3, alpha: float = 0.05) -> float:
    """w_inc: probability that the hard rule (with G1) declares 'incoherent'."""
    if rho is None or not np.isfinite(rho) or n is None or n <= 3:
        return 1.0 if regime == "incoherent" else 0.0
    se = math.sqrt(1.06 / (n - 3))
    h = float(stats.norm.ppf(1.0 - alpha)) * se
    z = float(np.arctanh(np.clip(rho, -0.999999, 0.999999)))
    boundary = min(float(np.arctanh(min_rho)) - h, h)
    return float(stats.norm.cdf((boundary - z) / se))


def weight_lean(k: int, n: int) -> float:
    """w_lean: P(q > 0.5), q ~ Beta(k + 1, n - k + 1). NaN without pairs."""
    if n <= 0:
        return float("nan")
    return float(stats.beta.sf(0.5, k + 1, n - k + 1))


def interpolate(A, B, w_B: float, evidence) -> np.ndarray:
    """Interpolated decision between regimes A (weight 1 - w_B) and B (weight w_B)."""
    A = np.asarray(A, dtype=bool)
    B = np.asarray(B, dtype=bool)
    ev = np.asarray(pd.to_numeric(pd.Series(evidence), errors="coerce"), dtype=float)
    ev = np.where(np.isfinite(ev), ev, -np.inf)
    out = A & B
    for only, w in ((B & ~A, w_B), (A & ~B, 1.0 - w_B)):
        idx = np.flatnonzero(only)
        m = len(idx)
        if m == 0:
            continue
        q = int(math.floor(w * m + 0.5))
        if q <= 0:
            continue
        order = idx[np.argsort(-ev[idx], kind="mergesort")]
        out[order[:q]] = True
    return out


# ----------------------------------------------------------------------------- graded Stage 6
def _best_r2(est: dict) -> float:
    r2 = [m.get("r2_nuisance_O") for m in (est.get("cascade_log") or [])]
    r2 = [v for v in r2 if v is not None and np.isfinite(v)]
    return max(r2) if r2 else float("nan")


def _combine(lean: list, blanket: list, w: float) -> list:
    """Pairs estimated under both designs: numeric fields weighted; decision interpolated."""
    out = []
    for a, b in zip(lean, blanket):
        dom = a if w >= 0.5 else b
        e = dict(dom)
        for k in WEIGHTED_FIELDS:
            va, vb = a.get(k, np.nan), b.get(k, np.nan)
            fa = va is not None and np.isfinite(va)
            fb = vb is not None and np.isfinite(vb)
            e[k] = (w * va + (1.0 - w) * vb) if (fa and fb) else (va if fa else (vb if fb else np.nan))
        if np.isfinite(e["effect"]) and dom.get("direction") in ("positive", "negative"):
            e["direction"] = "positive" if e["effect"] > 0 else "negative"
        out.append(e)
    sig_a = [bool(x.get("significant")) for x in lean]
    sig_b = [bool(x.get("significant")) for x in blanket]
    t = [abs(e["effect"] / e["se"]) if (np.isfinite(e["effect"]) and np.isfinite(e["se"]) and e["se"] > 0)
         else np.nan for e in out]
    dec = interpolate(sig_b, sig_a, w, t)     # A = blanket (weight 1 - w), B = lean (weight w)
    for e, s in zip(out, dec):
        e["significant"] = bool(s)
    return out


def _estimate_effects_by_control_set_graded(candidates, processed, all_keys, control_set="blanket",
                                            source_block=False, auto_max_outcome_r2=0.5, **kwargs):
    info = {"w_lean": float("nan"), "k": 0, "n": 0, "r2_mediana": float("nan"), "caminho": ""}
    _MOD.ultimo = info
    if control_set != "auto":
        info["caminho"] = "control_set fixo (original)"
        return dml.estimate_effects_by_control_set(candidates, processed, all_keys, control_set=control_set,
                                                   source_block=source_block,
                                                   auto_max_outcome_r2=auto_max_outcome_r2, **kwargs)
    lean = dml.estimate_all_effects(candidates, processed, all_keys, control_set="parents", source_block=True,
                                    **kwargs)
    r2 = np.array([_best_r2(e) for e in lean], dtype=float)
    ok = np.isfinite(r2)
    k, n = int((r2[ok] < auto_max_outcome_r2).sum()), int(ok.sum())
    r2_med = dml.median_outcome_r2(lean)
    w = weight_lean(k, n)
    info.update({"w_lean": w, "k": k, "n": n, "r2_mediana": r2_med, "r2_pares": [float(v) for v in r2[ok]]})
    lean_hard = bool(np.isfinite(r2_med) and r2_med < auto_max_outcome_r2)
    if not (np.isfinite(w) and BAND[0] < w < BAND[1]):
        # outside the band: exactly today's hard rule
        info["caminho"] = "abrupto (" + ("enxuto" if lean_hard else "cobertor") + ")"
        print(f"  [DML controls] graded: w_lean={w:.4f} (k={k}, n={n}) outside the band -> hard rule "
              f"({'parent-based + source block' if lean_hard else 'blanket'})")
        near_det = bool(np.isfinite(r2_med)) and not lean_hard
        if not lean_hard:
            lean = dml.estimate_all_effects(candidates, processed, all_keys, control_set="blanket", **kwargs)
        return lean, lean_hard, near_det
    blanket = dml.estimate_all_effects(candidates, processed, all_keys, control_set="blanket", **kwargs)
    info["caminho"] = "graduado"
    print(f"  [DML controls] graded: w_lean={w:.4f} (k={k}, n={n}, median R2={r2_med:.2f}) -> both designs, "
          f"weighted")
    lean_dominant = w >= 0.5
    return _combine(lean, blanket, w), lean_dominant, bool(np.isfinite(r2_med)) and not lean_dominant


class _GradedDML(types.ModuleType):
    """Forwards everything to the original Pipeline.dml_engine, except what is defined here."""
    REGIME_TRANSITION = "graded"

    def __getattr__(self, name):
        return getattr(dml, name)


_MOD = _GradedDML("dml_engine_graduado")
_MOD.estimate_effects_by_control_set = _estimate_effects_by_control_set_graded
_MOD.ultimo = {}


# ----------------------------------------------------------------------------- logging for the hard version
def _estimate_effects_by_control_set_logging(candidates, processed, all_keys, control_set="blanket",
                                             source_block=False, auto_max_outcome_r2=0.5, **kwargs):
    """Today's hard rule, step by step identical to Pipeline.dml_engine.estimate_effects_by_control_set
    (same calls, in the same order); it only records the per-pair R2 of the lean design, for the savings
    rule of the evaluation (protocol, section 4.1). Identity checked byte by byte in check_r1_3b.py (b2)."""
    info = {"w_lean": float("nan"), "k": 0, "n": 0, "r2_mediana": float("nan"), "caminho": "", "r2_pares": []}
    _MOD_REG.ultimo = info
    if control_set != "auto":
        info["caminho"] = "control_set fixo (original)"
        return dml.estimate_effects_by_control_set(candidates, processed, all_keys, control_set=control_set,
                                                   source_block=source_block,
                                                   auto_max_outcome_r2=auto_max_outcome_r2, **kwargs)
    estimates = dml.estimate_all_effects(candidates, processed, all_keys, control_set="parents",
                                         source_block=True, **kwargs)
    r2 = np.array([_best_r2(e) for e in estimates], dtype=float)
    ok = np.isfinite(r2)
    r2_outcome = dml.median_outcome_r2(estimates)
    lean = bool(np.isfinite(r2_outcome) and r2_outcome < auto_max_outcome_r2)
    info.update({"k": int((r2[ok] < auto_max_outcome_r2).sum()), "n": int(ok.sum()), "r2_mediana": r2_outcome,
                 "r2_pares": [float(v) for v in r2[ok]], "caminho": "abrupto (" + ("enxuto" if lean else "cobertor") + ")"})
    print(f"  [DML controls] auto: median out-of-fold outcome R2 under the "
          f"parent-based design = {r2_outcome:.2f} "
          f"({'<' if lean else '>='} {auto_max_outcome_r2}) -> "
          f"{'parent-based + source block' if lean else 'blanket'}")
    near_deterministic = bool(np.isfinite(r2_outcome)) and not lean
    if not lean:
        estimates = dml.estimate_all_effects(candidates, processed, all_keys, control_set="blanket", **kwargs)
    return estimates, lean, near_deterministic


class _LoggingDML(types.ModuleType):
    """Hard rule (REGIME_TRANSITION = 'hard'), with the diagnostics recorded."""
    REGIME_TRANSITION = "hard"

    def __getattr__(self, name):
        return getattr(dml, name)


_MOD_REG = _LoggingDML("dml_engine_registro")
_MOD_REG.estimate_effects_by_control_set = _estimate_effects_by_control_set_logging
_MOD_REG.ultimo = {}

# record of the coherence guard of the last run (any model; does not change the run)
LAST_COHERENCE = {}


def logging_dml_module():
    return _MOD_REG


def diagnostics() -> dict:
    """Diagnostics of the last run in the process (for the savings rule)."""
    return {"coerencia": dict(LAST_COHERENCE)}


def graded_dml_module():
    return _MOD


# ----------------------------------------------------------------------------- graded Stage 8
def _store_weights(run, w_inc_effective: float, w_inc: float) -> None:
    info = getattr(run.dml_engine, "ultimo", {}) or {}
    run.results["regime_weight_stage6"] = w_inc_effective
    run.results["regime_weight_stage6_raw"] = w_inc
    run.results["regime_weight_lean"] = info.get("w_lean", np.nan)
    run.results["regime_lean_path"] = info.get("caminho", "")


def _record_coherence(run) -> None:
    LAST_COHERENCE.clear()
    if run.coherence is not None:
        LAST_COHERENCE.update({k: run.coherence.get(k) for k in ("regime", "rho", "rho_lower", "rho_upper", "n",
                                                                 "reason")})


def _stage8_regime_guards_graded(run) -> None:
    _record_coherence(run)
    if getattr(run.dml_engine, "REGIME_TRANSITION", "hard") != "graded":
        return _ORIGINAL_GUARD(run)
    if not run.has_decisions():
        return _ORIGINAL_GUARD(run)
    p = run.p
    c = run.coherence
    if c is None:
        _ORIGINAL_GUARD(run)
        if run.has_decisions():
            _store_weights(run, 0.0, 0.0)
        return
    w = weight_inc(c.get("rho"), c.get("n"), c.get("regime"), p["evidence_coherence_min_rho"],
                   p["evidence_coherence_alpha"])
    if not (BAND[0] < w < BAND[1]):
        _ORIGINAL_GUARD(run)                 # exactly the hard rule
        fallback = c.get("regime") == "incoherent" and not (
            "evidence_fallback" in run.results.columns
            and (run.results["evidence_fallback"] == "stage7_8a").any())
        _store_weights(run, float(fallback), w)
        print(f"  [evidence coherence] graded: w_inc={w:.4f} outside the band -> hard rule")
        return

    # --- inside the band: the original guard, with the fallback to Stage 6 interpolated
    if p["evidence_coherence_fallback"] not in ("stage6", "auto"):
        raise ValueError("evidence_coherence_fallback must be 'stage6' or 'auto'.")
    w_ef = w
    if p["evidence_coherence_fallback"] == "auto" and p["contemporaneous_max_past_r2"] is not None:
        past_r2_all = contemp_mod.past_explained_variance(run.processed, list(run.pairs), p["max_lag"])
        past_r2 = past_r2_all
        if p["evidence_fallback_own_past_floor"] and past_r2_all is not None:
            past_r2_own = contemp_mod.own_past_explained_variance(run.processed, list(run.pairs), p["max_lag"])
            if past_r2_own is not None:
                past_r2 = max(past_r2_all, past_r2_own)
            run.results["evidence_past_r2_all"] = past_r2_all
            run.results["evidence_past_r2_own"] = past_r2_own
        skip = past_r2 is not None and past_r2 <= p["contemporaneous_max_past_r2"]
        run.results["evidence_past_r2"] = past_r2
        run.results["evidence_fallback"] = "stage7_8a" if skip else "stage6"
        if skip:
            w_ef = 0.0
    A = (run.results["significant"] == True).to_numpy()  # noqa: E712
    B = run.results["significant_stage6"].fillna(False).astype(bool).to_numpy()
    run.results["significant_before_guard"] = run.results["significant"]
    t = pd.to_numeric(run.results["t_stat"], errors="coerce").abs()
    run.results["significant"] = interpolate(A, B, w_ef, t)
    n_kept = int(run.results["significant"].sum())
    run.counts["8_graded_kept"] = n_kept
    _store_weights(run, w_ef, w)
    print(f"  [evidence coherence] graded: w_inc={w:.4f} (effective {w_ef:.4f}) -> {n_kept} pairs kept "
          f"(Stages 7-8a had kept {int(A.sum())}; Stage 6 {int(B.sum())})")

    # --- time-axis rule: same as the original
    if p["lagged_requires_time_axis"]:
        run.results, n_drop = contemp_mod.drop_lagged_without_time_axis(run.results, run.processed,
                                                                         rule=p["time_axis_rule"])
        if n_drop is not None:
            run.counts["8b_no_time_axis_dropped"] = n_drop
            print(f"  [Stage 8] no time axis detected: {n_drop} lagged edges dropped")


# ----------------------------------------------------------------------------- installation
def install():
    """Replaces the guard in Pipeline.pipeline._STAGES; returns the original tuple."""
    orig = pp._STAGES
    assert _ORIGINAL_GUARD in orig, "Pipeline.pipeline._STAGES does not contain the original guard"
    pp._STAGES = tuple(_stage8_regime_guards_graded if f is _ORIGINAL_GUARD else f for f in orig)
    return orig


def uninstall(orig) -> None:
    pp._STAGES = orig


def graded_model(ce, name: str = MODEL_NAME, **extra) -> dict:
    """patch_tau10_final + the graded module (protocol, section 4)."""
    return ce.model("patch_tau10", name, **BASE_FINAL, dml_engine_module=_MOD, **extra)


def final_model(ce, name: str = "patch_tau10_final", register: bool = False, **extra) -> dict:
    """patch_tau10_final; with register=True, through the logging module (same output, checked in (b2))."""
    if register:
        extra = {**extra, "dml_engine_module": _MOD_REG}
    return ce.model("patch_tau10", name, **BASE_FINAL, **extra)
