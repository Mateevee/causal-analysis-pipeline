"""
R1-6a and R2-1a -- Dempster-Shafer fusion (Stage 7) against a calibrated logistic
stacking with the same 17 pieces of evidence, and ranking against operating point.

Request
  R1: "A calibrated logistic/stacking baseline using the same 17 features,
       trained on development families and tested on held-out ones, is the
       obvious missing comparison."
  R2: "It therefore remains unclear whether Dempster-Shafer fusion offers a
       genuine advantage in edge ranking or serves mainly as a dataset-specific
       selector of the operating point."

Design (fixed before running; not to be changed after seeing the result)
  Unit: each candidate with lag >= 1 that reached Stage 7 (row of
    patch_tau10/original.csv with causal_penalty_score filled in), on the 90
    scenarios. Pairs that did not reach Stage 7 count as negatives for
    every rule, as in Table VI.
  Evidence: the H and non-H masses of the 17 active experts, recomputed by
    Pipeline.SelectorFunc.causal_pen_diagnostics on the saved row, with the
    support floor recalibrated per scenario (generators/decision_trail.py).
    Rows cut by a structural gate get zero masses and
    gate = 1 (34 + 1 variables).
  Label: the entity pair is causal in the ground truth.
  Model: L2 logistic regression (C = 1), standardized variables, balanced
    class weights; pair score = maximum over its rows.
  Train/test:
    (A) trains on the 51 development scenarios, tests on the 39 held-out;
    (B) leaves one family out at a time (17 families): trains on the other 16.
  Threshold on the test set, without test ground truth:
    (i) probability 0.5;
    (ii) the threshold that maximizes the mean per-scenario F1 on the training set.
  Comparison (without Stage 8, as in Table VI): per-scenario F1 of the stacking against
    Stage 7 (causal_penalty_score = 1); two-sided paired Wilcoxon.
  Ranking (R2-1a): per-scenario AUROC and AUPRC, over all pairs of the universe
    (non-candidates tied at zero), of three scores: net support of
    D-S (net_ratio), stacking probability (protocol A/B) and max |t| of the DML.
  Reproducibility: fraction of candidates where the recomputed decision matches
    the saved one (the paper says 99.5% on the 58 scenarios).

Usage:  python revision/offline/r1_6a_stacking.py [--verificar]
      --verificar runs on 6 scenarios, only to check that the script works.
Output: Results_Revisao/r1_6a_stacking/
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as c  # noqa: E402

from Config.execution import ADAPTIVE_CAUSAL_PENALTY as CFG  # noqa: E402
from Pipeline.SelectorFunc import AdaptiveCausalPenalty, causal_pen_diagnostics  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

NAME = "r1_6a_stacking"
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def calibrate_floor(df):
    pen = AdaptiveCausalPenalty(
        percentile=CFG.percentile, min_absolute_floor=CFG.min_absolute_floor,
        max_absolute_floor=CFG.max_absolute_floor, min_n_candidates=CFG.min_n_candidates,
        domain_gate_fn=CFG.domain_gate_fn, fusion_rule=CFG.fusion_rule)
    return pen.calibrate(df.drop(columns=["old_sig"], errors="ignore"))


def evidence(scen: str):
    """One row per Stage 7 candidate: expert masses, decisions and label."""
    df = c.original(scen)
    if df.empty or "causal_penalty_score" not in df.columns:
        return pd.DataFrame()
    lag = pd.to_numeric(df["lag"], errors="coerce")
    df = df[(lag > 0) & df["causal_penalty_score"].notna()].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()
    floor = calibrate_floor(df)
    T, _ = c.truth(scen)
    out = []
    for r in df.to_dict("records"):
        s = causal_pen_diagnostics(r, min_support_threshold=floor)
        row = {"cenario": scen, "src": r["source_entity"], "dst": r["target_entity"],
               "lag": r["lag"], "y": int((r["source_entity"], r["target_entity"]) in T),
               "ds_aceito_gravado": int(float(r["causal_penalty_score"]) > 0),
               "ds_aceito_recalc": int(bool(s["accepted"])),
               "ds_net_ratio": float(s["net_ratio"]), "ds_suporte": float(s["support"]),
               "abs_t": abs(float(r["t_stat"])) if pd.notna(r.get("t_stat")) else 0.0,
               "gate": int(s.get("gate_reason") is not None)}
        for name, m in (s["primary_experts"] + s["active_meta_experts"]
                        + s["active_interaction_experts"]):
            name = name.split(" [")[0]      # 'context_mode [rescue:...]' -> 'context_mode'
            row[f"H__{name}"] = float(m["H"])
            row[f"nH__{name}"] = float(m["notH"])
        out.append(row)
    return pd.DataFrame(out)


def evidence_table(scens, rel):
    cache = os.path.join(rel.dir, "evidencias.csv")
    if os.path.exists(cache):
        ev = pd.read_csv(cache)
        if set(scens) <= set(ev["cenario"]):
            return ev[ev["cenario"].isin(scens)].reset_index(drop=True)
    parts = []
    for i, s in enumerate(scens, 1):
        print(f"  [{i}/{len(scens)}] {s}")
        parts.append(evidence(s))
    ev = pd.concat(parts, ignore_index=True).fillna(0.0)
    ev.to_csv(cache, index=False)
    return ev


def x_columns(ev):
    return [k for k in ev.columns if k.startswith(("H__", "nH__"))] + ["gate"]


def train(ev_train):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    m.fit(ev_train[x_columns(ev_train)].values, ev_train["y"].values)
    return m


def pair_score(ev, prob):
    d = ev[["cenario", "src", "dst"]].copy()
    d["p"] = prob
    return d.groupby(["cenario", "src", "dst"])["p"].max()


def f1_per_scenario(sc_pairs: dict, threshold: float, scens) -> dict:
    out = {}
    for s in scens:
        T, _ = c.truth(s)
        pp = sc_pairs.get(s, pd.Series(dtype=float))
        S = {(a, b) for (a, b), v in pp.items() if v >= threshold}
        out[s] = c.f1_set(S, T)
    return out


def split_by_scenario(series):
    """Series indexed by (scenario, src, dst) -> {scenario: Series indexed by (src, dst)}."""
    out = {}
    for s, g in series.groupby(level=0):
        out[s] = g.droplevel(0)
    return out


def threshold_on_train(ev_tr, model, scens_tr) -> float:
    pr = split_by_scenario(pair_score(ev_tr, model.predict_proba(ev_tr[x_columns(ev_tr)].values)[:, 1]))
    best = [(np.mean(list(f1_per_scenario(pr, t, scens_tr).values())), -t) for t in THRESHOLD_GRID]
    return float(-max(best)[1])


def evaluate(ev, scens_tr, scens_te):
    ev_tr = ev[ev["cenario"].isin(scens_tr)]
    ev_te = ev[ev["cenario"].isin(scens_te)]
    m = train(ev_tr)
    t_tr = threshold_on_train(ev_tr, m, scens_tr)
    pr = split_by_scenario(pair_score(ev_te, m.predict_proba(ev_te[x_columns(ev_te)].values)[:, 1]))
    return pr, t_tr


def f1_ds(ev, scens):
    out = {}
    for s in scens:
        T, _ = c.truth(s)
        d = ev[(ev["cenario"] == s) & (ev["ds_aceito_gravado"] == 1)]
        out[s] = c.f1_set(set(zip(d["src"], d["dst"])), T)
    return out


def ranking(ev, stack_scores: dict, scens):
    """Per-scenario AUROC/AUPRC over the universe of pairs, for the three scores."""
    out = []
    for s in scens:
        T, U = c.truth(s)
        d = ev[ev["cenario"] == s]
        g = d.groupby(["src", "dst"])
        scores = {"D-S net ratio": g["ds_net_ratio"].max(), "max abs(t)": g["abs_t"].max()}
        if s in stack_scores:
            scores["stacking"] = stack_scores[s]
        U = sorted(U | set(zip(d["src"], d["dst"])))
        y = [int(p in T) for p in U]
        for name, v in scores.items():
            base = float(np.nanmin(v.values)) - 1.0 if len(v) else -1.0   # non-candidates below all
            sc = [float(v.get(p, base)) for p in U]
            out.append({"cenario": s, "escore": name, "auroc": c.auroc(y, sc), "auprc": c.auprc(y, sc),
                        "prevalencia": float(np.mean(y))})
    return pd.DataFrame(out)


COMP_HEADER = ["Comparison", "Scenarios", "F1 stacking", "F1 Stage 7 (D-S)", "Difference",
               "Stacking better / tie / worse", "Wilcoxon p"]


def compare(title, f_stack: dict, f_ref: dict, scens):
    a = [f_stack[s] for s in scens]
    b = [f_ref[s] for s in scens]
    w, t, l = c.wins(a, b)
    return [title, len(scens), c.fmt(np.mean(a)), c.fmt(np.mean(b)), f"{np.mean(a) - np.mean(b):+.3f}",
            f"{w} / {t} / {l}", f"{c.wilcoxon_p(a, b):.3g}"]


def main():
    check = "--verificar" in sys.argv
    rel = c.Report(NAME + ("_verificacao" if check else ""),
                   "R1-6a / R2-1a — Stage 7 (Dempster-Shafer) against logistic stacking")
    scens = c.scenarios()
    if check:
        scens = c.scenarios("dev")[:3] + c.scenarios("ho")[:3]
    ev = evidence_table(scens, rel)
    rel.p(f"Stage 7 candidates: {len(ev)} in {ev['cenario'].nunique()} scenarios; "
          f"{len(x_columns(ev)) - 1} masses per candidate (17 experts x H/non-H).")
    same = (ev["ds_aceito_gravado"] == ev["ds_aceito_recalc"]).mean()
    rel.p(f"Recomputed D-S decision = saved one in {100 * same:.1f}% of the candidates.")
    rel.data["reproducao_ds"] = float(same)

    dev = [s for s in scens if not c.is_heldout(s)]
    ho = [s for s in scens if c.is_heldout(s)]
    ref = f1_ds(ev, scens)

    # (A) development -> held-out
    rel.h("## A. Training on the development scenarios, testing on the held-out ones")
    pr_a, t_a = evaluate(ev, dev, ho)
    rel.table(COMP_HEADER, [compare(f"held-out, {name}", f1_per_scenario(pr_a, thr, ho), ref, ho)
                            for name, thr in (("threshold 0.5", 0.5), (f"training threshold ({t_a:.2f})", t_a))])
    rel.data["A_limiar_treino"] = t_a

    # (B) one family left out at a time
    rel.h("## B. One family left out at a time (training on the other 16)")
    fams = sorted({c.family(s) for s in scens})
    f_b05, f_bt, pr_b, body = {}, {}, {}, []
    for fam in fams:
        te = [s for s in scens if c.family(s) == fam]
        tr = [s for s in scens if c.family(s) != fam]
        if not tr or not te:
            continue
        pr, t = evaluate(ev, tr, te)
        pr_b.update(pr)
        a05, at = f1_per_scenario(pr, 0.5, te), f1_per_scenario(pr, t, te)
        f_b05.update(a05); f_bt.update(at)
        body.append([fam, len(te), c.fmt(np.mean([ref[s] for s in te])), c.fmt(np.mean(list(a05.values()))),
                     c.fmt(np.mean(list(at.values()))), f"{t:.2f}"])
    rel.table(["Family left out", "Scenarios", "F1 Stage 7", "F1 stacking (0.5)", "F1 stacking (training threshold)",
               "Training threshold"], body)
    rel.p("")
    rel.table(COMP_HEADER, [compare(f"{part}, {name}", f, ref, sc)
                            for part, sc in (("all", scens), ("development", dev), ("held-out", ho)) if sc
                            for name, f in (("threshold 0.5", f_b05), ("training threshold", f_bt))])

    # Ranking (R2-1a)
    rel.h("## C. Ranking: per-scenario AUROC and AUPRC (R2-1a)")
    rk = ranking(ev, {**pr_b, **pr_a}, scens)
    rel.csv("ranking_por_cenario.csv", rk)
    body = []
    for part, sc in (("all", scens), ("development", dev), ("held-out", ho)):
        for e in ("D-S net ratio", "max abs(t)", "stacking"):
            d = rk[(rk["escore"] == e) & rk["cenario"].isin(sc)]
            body.append([part, e, len(d), c.fmt(d["auroc"].mean()), c.fmt(d["auprc"].mean()),
                         c.fmt(d["prevalencia"].mean())])
    rel.table(["Part", "Score", "Scenarios", "Mean AUROC", "Mean AUPRC", "Mean prevalence"], body)
    rel.p("")
    rel.p("On the held-out set the stacking score is that of protocol A (training on development); "
          "on development, that of protocol B (family left out).")

    pd.DataFrame([{"cenario": s, "familia": c.family(s), "heldout": c.is_heldout(s), "f1_ds": ref[s],
                   "f1_stack_lofo_05": f_b05.get(s), "f1_stack_lofo_treino": f_bt.get(s)} for s in scens]
                 ).to_csv(os.path.join(rel.dir, "f1_por_cenario.csv"), index=False)
    rel.save()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_log import execution
    with execution(NAME, __file__):
        main()
