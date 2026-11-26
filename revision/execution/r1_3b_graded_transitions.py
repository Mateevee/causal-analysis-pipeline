"""
R1-3b -- graded transitions in the two regime switches: evaluation.

Request
  R1: "Only 53% of the cut-off perturbations meet the pre-specified tolerance (target
       90%) ... Graded rather than hard switches ... look necessary for a
       'fixed-configuration' claim."

Design: revision/PROTOCOL_graded_transitions_R1-3b.md (frozen), section 4, with the
implementation deviation of revision/NOTAS.md ("R1-3b -- DESVIO DO PROTOCOLO", 2026-11-14):
the variant is in revision/variants/r1_3b_graded.py; Pipeline/ is not changed.
  Versions: patch_tau10_final (hard switch; through the logging module, same output,
  identity checked in revision/variants/check_r1_3b.py) and patch_tau10_graded.
  4.1  12 configurations (min_rho in {0.10; 0.20; 0.40; 0.50}; outcome R2 in
       {0.35; 0.45; 0.55; 0.65}; 4 corners), 90 scenarios, S20 tolerance against the
       default of the same version: (R1) mean F1 within 0.02 (whole set, development,
       held-out); (R2) scenarios within 0.10 of the best of the six methods drop by at most 3;
       (R3) mean deficit below that of every baseline. Baselines: the paper's runs
       (generators/paper_numbers.py, window 10), the paper's scoring (validation_summary).
       Success: >= 11 of 12. delta = 0.02 (without the between-seed range of S20: here there
       is one run per configuration; declared).
       Savings (section 4.1): a scenario is run again under a configuration only if, by the
       diagnostics of the run at the default, the regime decision can change --
       hard: the regime (incoherent or not; lean or blanket) changes with the new cut-off;
       graded: the effective weight changes (w outside (0.001, 0.999) on the same side at both
       cut-offs counts as equal). For the others, the default output is reused; the list
       goes to economia.csv.
  4.2  non-inferiority at the default: mean F1 graded - final >= -0.01 in
       development and in held-out; paired Wilcoxon alongside.
  4.3  stability: the 10 frequently switching scenarios of E5 x the 20 subsamples of
       R2-m2 (same function, same seed) x 2 versions; Jaccard against the full sample
       of the same version, Jaccard between subsamples, fraction of stable edges (>= 80%) and,
       for the hard version, fraction of subsamples whose regime differs from that of the full sample.

Usage (single process; the pipeline uses the n_jobs of the configuration)
  python revision/execution/r1_3b_graded_transitions.py rodar padrao [--filtro name ...]
  python revision/execution/r1_3b_graded_transitions.py plano            (savings, without running)
  python revision/execution/r1_3b_graded_transitions.py rodar perturbacao [--config name ...]
  python revision/execution/r1_3b_graded_transitions.py rodar estabilidade   [--filtro/--B: testing only]
  python revision/execution/r1_3b_graded_transitions.py avaliar
Output: Results_Revisao/r1_3b_transicoes_graduais/
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
for _p in (HERE, REV, os.path.dirname(REV)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

NAME = "r1_3b_transicoes_graduais"
FROZEN_HASH = "d0473069ebef1978"
DEFAULT = (0.3, 0.5)
CONFIGS = {"padrao": DEFAULT}
for _r in (0.10, 0.20, 0.40, 0.50):
    CONFIGS[f"rho_{_r:.2f}"] = (_r, DEFAULT[1])
for _q in (0.35, 0.45, 0.55, 0.65):
    CONFIGS[f"r2_{_q:.2f}"] = (DEFAULT[0], _q)
for _r, _q in ((0.10, 0.35), (0.10, 0.65), (0.50, 0.35), (0.50, 0.65)):
    CONFIGS[f"canto_{_r:.2f}_{_q:.2f}"] = (_r, _q)
PERTURBED = [k for k in CONFIGS if k != "padrao"]
assert len(PERTURBED) == 12
VERSIONS = {"final": "patch_tau10_final", "graduada": "patch_tau10_graded"}
E5 = ["benchmark_rcaeval_RE1-OB_productcatalogservice_cpu_2", "benchmark_logistic_maps_with_isolated_seed2",
      "benchmark_logistic_maps_chain_seed1", "benchmark_beeline_VSC_VSC-2000-1", "real_easyvista_mom_1",
      "real_easyvista_web_2", "benchmark_synthetic_var_2_seed22", "benchmark_beeline_HSC_HSC-2000-2",
      "benchmark_beeline_VSC_VSC-2000-2", "real_easyvista_antivirus_1"]
B_SUB = 20
TOL, MARK, DROP, EPS = 0.02, 0.10, 3, 1e-9
SUCCESS = 11
NI_MARGIN = -0.01


def arg(name, default=None):
    if name not in sys.argv:
        return default
    vals = []
    for a in sys.argv[sys.argv.index(name) + 1:]:
        if a.startswith("--"):
            break
        vals.append(a)
    return vals


def base():
    import common_exec as ce
    return ce.output_dir(NAME)


# ----------------------------------------------------------------------------- execution
def _model(ce, g, version, cfg):
    mr, q = CONFIGS[cfg]
    extra = {} if cfg == "padrao" else {"evidence_coherence_min_rho": mr, "dml_auto_max_outcome_r2": q}
    if version == "final":
        return g.final_model(ce, register=True, **extra), g.logging_dml_module()
    return g.graded_model(ce, **extra), g.graded_dml_module()


def _read_diagnostics(path):
    done = {}
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            done[(r["cenario"], r["versao"], r["config"])] = r
    return done


def _run_list(ce, g, tasks, folder, diag_path):
    """tasks: [(d, version, cfg)]; resumes; writes one diagnostics record per run."""
    done = _read_diagnostics(diag_path)
    for d, version, cfg in tasks:
        m, mod = _model(ce, g, version, cfg)
        if (d["name"], version, cfg) in done and ce.already_run(folder, d["name"], m["name"]):
            continue
        mod.ultimo = {}
        g.LAST_COHERENCE.clear()
        ce.main_mod.run_all(dataframes=[d], models=[m], config=ce.cfg_exec.PIPELINE_CONFIG, output_dir=folder)
        record = {"cenario": d["name"], "versao": version, "modelo": m["name"], "config": cfg,
                  "coerencia": dict(g.LAST_COHERENCE), "dml": dict(mod.ultimo)}
        with open(diag_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=float) + "\n")
    ce.main_mod.validate_all(output_dir=folder, show_pairs=False, soft_threshold=ce.cfg_exec.SOFT_THRESHOLD,
                             ignore_intra_entity=not ce.cfg_exec.DETECT_INTRA_ENTITY,
                             use_canonical_pair=ce.cfg_exec.USE_CANONICAL_PAIR)


def _check_hash():
    import run_log
    h = run_log.method_code_hash()
    if h != FROZEN_HASH:
        print(f"WARNING: method_code_hash {h} != {FROZEN_HASH}, the hash of the frozen method code (the released files are English translations of the frozen ones; see revision/README.md)", file=sys.stderr)


def run():
    import common_exec as ce
    from variants import r1_3b_graded as g
    _check_hash()
    stage = sys.argv[2] if len(sys.argv) > 2 else ""
    orig = g.install()
    try:
        if stage == "padrao":
            dfs = sorted(ce.paper_scenarios(arg("--filtro")), key=lambda d: d["name"])
            folder = ce.output_dir(os.path.join(NAME, "run", "padrao"))
            _run_list(ce, g, [(d, v, "padrao") for d in dfs for v in VERSIONS], folder,
                      os.path.join(folder, "diag.jsonl"))
        elif stage == "perturbacao":
            dfs = {d["name"]: d for d in ce.paper_scenarios()}
            plan_df = compute_plan()
            cfgs = arg("--config") or PERTURBED
            for cfg in cfgs:
                folder = ce.output_dir(os.path.join(NAME, "run", cfg))
                tasks = [(dfs[r.cenario], r.versao, cfg) for r in plan_df.itertuples()
                         if r.config == cfg and r.rodar]
                _run_list(ce, g, tasks, folder, os.path.join(folder, "diag.jsonl"))
        elif stage == "estabilidade":
            from r2_m2_jaccard import subsamples
            dfs = {d["name"]: d for d in ce.paper_scenarios(E5)}
            missing = [s for s in E5 if s not in dfs]
            assert not missing, missing
            targets = [s for s in E5 if not arg("--filtro") or s in arg("--filtro")]   # --filtro/--B: testing only
            nb = int((arg("--B") or [B_SUB])[0])
            folder = ce.output_dir(os.path.join(NAME, "run", "estabilidade"))
            tasks = []
            for s in targets:
                for sub in subsamples(dfs[s], nb, 7000 + zlib.crc32(s.encode("utf-8"))):
                    tasks += [(sub, v, "padrao") for v in VERSIONS]
            _run_list(ce, g, tasks, folder, os.path.join(folder, "diag.jsonl"))
        else:
            raise SystemExit("stage: padrao | perturbacao | estabilidade")
    finally:
        g.uninstall(orig)


# ----------------------------------------------------------------------------- savings rule
def _hard_incoherent(coh, min_rho):
    rho, lo, hi = coh.get("rho"), coh.get("rho_lower"), coh.get("rho_upper")
    if rho is None or not np.isfinite(rho):
        return coh.get("regime") == "incoherent"
    return bool(hi < min_rho and not (lo > 0))          # G1 (overlap = 'coherent')


def _effective(w):
    if w is None or not np.isfinite(w) or w <= 0.001:
        return "L"
    if w >= 0.999:
        return "H"
    return round(float(w), 12)


def _decision(diag, version, cfg):
    from variants import r1_3b_graded as g
    mr, q = CONFIGS[cfg]
    coh, dm = diag["coerencia"], diag["dml"]
    r2 = np.asarray(dm.get("r2_pares") or [], dtype=float)
    if version == "final":
        lean = bool(len(r2) and np.median(r2) < q)
        return (_hard_incoherent(coh, mr), lean)
    w_lean = g.weight_lean(int((r2 < q).sum()), len(r2)) if len(r2) else float("nan")
    w_inc = g.weight_inc(coh.get("rho"), coh.get("n"), coh.get("regime"), mr, 0.05)
    return (_effective(w_inc), _effective(w_lean))


def compute_plan() -> pd.DataFrame:
    b = base()
    diag = _read_diagnostics(os.path.join(b, "run", "padrao", "diag.jsonl"))
    scens = sorted({k[0] for k in diag})
    recs = []
    for s in scens:
        for v in VERSIONS:
            dg = diag.get((s, v, "padrao"))
            if dg is None:
                raise SystemExit(f"missing the default run of {s} / {v}: run 'rodar padrao' first")
            d0 = _decision(dg, v, "padrao")
            for cfg in PERTURBED:
                d1 = _decision(dg, v, cfg)
                recs.append({"cenario": s, "versao": v, "config": cfg, "rodar": d1 != d0,
                             "decisao_padrao": str(d0), "decisao_config": str(d1)})
    P = pd.DataFrame(recs)
    P.to_csv(os.path.join(b, "economia.csv"), index=False)
    return P


def plan():
    P = compute_plan()
    t = P.groupby(["config", "versao"])["rodar"].sum().unstack()
    print(t.to_string())
    print("runs to execute:", int(P.rodar.sum()), "of", len(P))


# ----------------------------------------------------------------------------- evaluation
def _counts(folder, scen, model):
    p = os.path.join(folder, scen, "validation_summary.csv")
    if not os.path.exists(p):
        return None
    v = pd.read_csv(p)
    r = v[(v["model"] == model) & (v["status"] == "ok")]
    return None if r.empty else (int(r.TP.iloc[0]), int(r.FP.iloc[0]), int(r.FN.iloc[0]))


def _f1(c):
    tp, fp, fn = c
    return 0.0 if tp == 0 else round(2 * tp / (2 * tp + fp + fn), 4)


def _tolerance(F, D, B, scens, ho):
    """S20 criterion (generators/sensitivity_analysis.py), delta = TOL."""
    dev = [s for s in scens if s not in ho]
    mean = lambda X, sc: float(np.mean([X[s] for s in sc]))
    gap = lambda X, s: max([X[s]] + list(B[s].values())) - X[s]
    w = lambda X: sum(gap(X, s) <= MARK + EPS for s in scens)
    mgap = float(np.mean([gap(F, s) for s in scens]))
    bl = sorted({m for s in scens for m in B[s]})
    bgap = min(np.mean([max([F[s]] + list(B[s].values())) - B[s][m] for s in scens if m in B[s]]) for m in bl)
    r1 = all(abs(mean(F, sc) - mean(D, sc)) <= TOL + EPS for sc in (scens, dev, [s for s in scens if s in ho]))
    r2 = (w(D) - w(F)) <= DROP
    r3 = mgap < bgap
    return dict(media=mean(F, scens), media_dev=mean(F, dev), media_ho=mean(F, [s for s in scens if s in ho]),
                dentro_marca=w(F), dentro_marca_padrao=w(D), deficit=mgap, deficit_baseline_min=bgap,
                R1=r1, R2=r2, R3=r3, ok=r1 and r2 and r3)


def evaluate():
    import common as c
    b = base()
    rel = c.Report(NAME, "R1-3b — Graded transitions in the two regime switches")
    rows = c.rows()
    folder0 = os.path.join(b, "run", "padrao")
    diag0 = _read_diagnostics(os.path.join(folder0, "diag.jsonl"))
    scens = sorted({k[0] for k in diag0})
    key = {s: c.scenario_key(s) for s in scens}
    unmatched = [s for s in scens if key[s] is None]
    ho = {s for s in scens if key[s] and c.is_heldout(key[s])}
    B = {s: {m: _f1((rows[key[s]][m]["tp"], rows[key[s]][m]["fp"], rows[key[s]][m]["fn"]))
             for m in c.BASELINES if m in rows[key[s]]} for s in scens if key[s]}
    complete = len(scens) == 90 and not unmatched
    rel.p(f"Scenarios at the default: {len(scens)} (90 expected); without a match in the paper's runs: {unmatched}. "
          + ("" if complete else "**PARTIAL.**"))
    scens = [s for s in scens if key[s]]
    D = {}
    for v, mod in VERSIONS.items():
        D[v] = {}
        for s in scens:
            k = _counts(folder0, s, mod)
            if k is not None:
                D[v][s] = _f1(k)
    common = [s for s in scens if all(s in D[v] for v in VERSIONS)]

    # 4.2 non-inferiority
    rel.p("## 4.2 Non-inferiority at the default (graded − final; margin −0.01)")
    body = []
    for part, sc in (("all", common), ("development", [s for s in common if s not in ho]),
                     ("held-out", [s for s in common if s in ho])):
        a = [D["graduada"][s] for s in sc]; f = [D["final"][s] for s in sc]
        diff = float(np.mean(a) - np.mean(f)) if sc else float("nan")
        body.append([part, len(sc), c.fmt(np.mean(f)), c.fmt(np.mean(a)), f"{diff:+.4f}",
                     f"{c.wilcoxon_p(a, f):.3g}" if sc else "", "yes" if diff >= NI_MARGIN - EPS else "NO"])
    rel.table(["Part", "Scenarios", "F1 final", "F1 graded", "Difference", "Wilcoxon p", "Non-inferior"], body)

    # 4.1 perturbation
    rel.p("")
    rel.p("## 4.1 Perturbation of the two cut-offs (S20 tolerance; success: ≥ 11 of 12)")
    econ_p = os.path.join(b, "economia.csv")
    if not os.path.exists(econ_p):
        rel.p("No economia.csv: run 'plano' and 'rodar perturbacao'.")
        rel.save()
        return
    P = pd.read_csv(econ_p)
    res, missing_runs = [], []
    for v, mod in VERSIONS.items():
        base_v = {s: D[v][s] for s in common}
        for cfg in PERTURBED:
            folder = os.path.join(b, "run", cfg)
            F, missing_here = {}, 0
            for s in common:
                q = P[(P.cenario == s) & (P.versao == v) & (P.config == cfg)]
                if len(q) and bool(q.rodar.iloc[0]):
                    k = _counts(folder, s, mod)
                    if k is None:
                        missing_runs.append((cfg, v, s))
                        missing_here += 1
                        continue
                    F[s] = _f1(k)
                else:
                    F[s] = base_v[s]
            sc = [s for s in common if s in F]
            if missing_here:
                # incomplete configuration: no verdict (a tolerance on a subset would be misleading)
                t = dict(media=np.nan, media_dev=np.nan, media_ho=np.nan, dentro_marca=-1, dentro_marca_padrao=-1,
                         deficit=np.nan, deficit_baseline_min=np.nan, R1=None, R2=None, R3=None, ok=None)
            else:
                t = _tolerance(F, {s: base_v[s] for s in sc}, B, sc, ho)
            t.update(versao=v, config=cfg, n=len(sc), faltam=missing_here,
                     rodados=int(P[(P.versao == v) & (P.config == cfg)].rodar.sum()))
            res.append(t)
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(rel.dir, "perturbacao.csv"), index=False)
    if missing_runs:
        rel.p(f"**PARTIAL: {len(missing_runs)} planned runs without a result** (e.g. {missing_runs[:3]}).")
    for v in VERSIONS:
        g = R[R.versao == v]
        compl = g[g.faltam == 0]
        n_ok = int(sum(bool(x) for x in compl.ok))
        if len(compl) < len(g):
            rel.p(f"**{v}** — INCOMPLETE: {len(g) - len(compl)} configuration(s) without all runs; "
                  f"{n_ok} of {len(compl)} complete ones within the tolerance. No verdict.")
        else:
            rel.p(f"**{v}** — within the tolerance: **{n_ok} of {len(g)}** "
                  f"({'meets' if n_ok >= SUCCESS else 'does not meet'} the criterion of ≥ {SUCCESS})")
        rel.table(["Config", "Runs", "Mean F1", "Dev", "Held-out", "Within 0.10 (default)", "Deficit",
                   "Min. baseline deficit", "R1", "R2", "R3", "Within"],
                  [[r.config, r.rodados, c.fmt(r.media), c.fmt(r.media_dev), c.fmt(r.media_ho),
                    f"{r.dentro_marca} ({r.dentro_marca_padrao})", c.fmt(r.deficit), c.fmt(r.deficit_baseline_min),
                    *(["yes" if r.R1 else "no", "yes" if r.R2 else "no", "yes" if r.R3 else "no",
                       "YES" if r.ok else "no"] if r.faltam == 0 else ["—", "—", "—", f"incomplete ({r.faltam})"])]
                   for r in g.itertuples()])
        rel.p("")

    # 4.3 stability
    rel.p("## 4.3 Stability under subsampling (10 scenarios of E5; post hoc selection, declared)")
    folder_e = os.path.join(b, "run", "estabilidade")
    diag_e = _read_diagnostics(os.path.join(folder_e, "diag.jsonl"))
    if not diag_e:
        rel.p("No stability runs yet.")
        rel.save()
        return

    def graph(p):
        if not os.path.exists(p):
            return None
        r = pd.read_csv(p, skipinitialspace=True)
        return set(zip(r["source_entity"].astype(str).str.strip(), r["target_entity"].astype(str).str.strip())) \
            if len(r) else set()

    jac = lambda a, b_: 1.0 if not a and not b_ else len(a & b_) / len(a | b_)
    recs = []
    for s in E5:
        for v, mod in VERSIONS.items():
            full = graph(os.path.join(folder0, s, mod, "results.csv"))
            subs = [graph(os.path.join(folder_e, f"{s}__sub{i:02d}", mod, "results.csv")) for i in range(B_SUB)]
            subs = [x for x in subs if x is not None]
            if full is None or not subs:
                continue
            jc = float(np.mean([jac(x, full) for x in subs]))
            js = float(np.mean([jac(a, b_) for a, b_ in itertools.combinations(subs, 2)])) if len(subs) > 1 else np.nan
            stable = float(np.mean([sum(e in x for x in subs) / len(subs) >= 0.8 for e in full])) if full else np.nan
            d0 = diag0.get((s, v, "padrao"), {})
            reg0 = (d0.get("coerencia", {}).get("regime"), (d0.get("dml", {}) or {}).get("caminho"))
            differs = [((diag_e.get((f"{s}__sub{i:02d}", v, "padrao"), {}).get("coerencia", {}).get("regime"),
                         (diag_e.get((f"{s}__sub{i:02d}", v, "padrao"), {}).get("dml", {}) or {}).get("caminho")) != reg0)
                       for i in range(B_SUB) if (f"{s}__sub{i:02d}", v, "padrao") in diag_e]
            # the fraction with a different regime only exists for the hard switch (protocol, 4.3)
            frd = float(np.mean(differs)) if (differs and v == "final") else np.nan
            recs.append({"cenario": s, "versao": v, "n_sub": len(subs), "jaccard_cheio": jc, "jaccard_entre": js,
                         "frac_estaveis": stable, "frac_regime_diferente": frd})
    E = pd.DataFrame(recs)
    E.to_csv(os.path.join(rel.dir, "estabilidade.csv"), index=False)
    if len(E):
        W = E.pivot(index="cenario", columns="versao")
        body = []
        for col in ("jaccard_cheio", "jaccard_entre", "frac_estaveis", "frac_regime_diferente"):
            if ("final" in W[col]) and ("graduada" in W[col]):
                x = W[col].dropna(how="all")
                p = c.wilcoxon_p(list(x["graduada"]), list(x["final"])) if col.startswith("jaccard") and len(x) else np.nan
                body.append([col, len(x), c.fmt(x["final"].mean()), c.fmt(x["graduada"].mean()),
                             "" if not np.isfinite(p) else f"{p:.3g}"])
        rel.table(["Measure", "Scenarios", "Final", "Graded", "Wilcoxon p (n = 10, low power)"], body)
        partial = E.n_sub.min() < B_SUB or E.cenario.nunique() < len(E5)
        if partial:
            rel.p("**PARTIAL** (fewer than 20 subsamples or 10 scenarios).")
    rel.save()


if __name__ == "__main__":
    from run_log import execution
    mode = sys.argv[1] if len(sys.argv) > 1 else "avaliar"
    with execution(NAME, __file__):
        if mode == "rodar":
            run()
        elif mode == "plano":
            plan()
        else:
            evaluate()
