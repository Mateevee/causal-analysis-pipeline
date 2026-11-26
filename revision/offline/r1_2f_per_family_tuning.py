"""
R1-2a / R1-2b / R2-3a -- Analyses A1, A2 and A3 on the grid of the baselines.

Request
  R1, point 2: "Please add (a) per-benchmark tuned baselines (or at least a small
       grid), (b) a leave-one-family-out protocol, where cut-offs are frozen using
       other families only ...".
  R2, point 3: "the tuning protocol is asymmetric".

Design (fixed before running)
  That of revision/PROTOCOL_expanded_baseline_grid.md (section 5) and of
  revision/PROTOCOL_per_family_tuning.md (whole), frozen on 2026-11-10.
  Where they differ, that of the widened grid prevails. SHA-256 (content in git, in LF /
  file on disk, in CRLF):
    widened grid        e561cb2ab911ee046d8338c100c1f26249760ef34e2a8a89b39a5ea6f1f94103
                        3ee76e1694f6eafccef9ed7dc7c9101324067638df1c90c5167186a233b64c48
    per-family tuning   9ddc6280b16dfad64d86491217bd97ef592fd2e1cc899a13d6ccf58a9febf8c2
                        eb4238ab17ee534d7ae061febe0afdf31da0f52bcf260a1fed4489dbf9847a7e

  Inputs
    baselines : f1_por_config.csv of the widened grid (r1_2g; 1,019 configurations,
                the default of each method rerun in the grid itself; CUTS+ =
                mean of 3 seeds) or, with --grade-original, that of r1_2a (45
                configurations; "default" = the paper's run);
    pipeline  : the paper's configuration, common.f1_model(scenario, PIPE); grid = the
                216 configurations of the sensitivity run (seed 0), through the function
                `load` of r1_2b (86 scenarios; without Causal Chambers);
    families  : common.family (17); development/held-out: common.is_heldout.
  Metric: F1 per directional pair, tau_max = 10, as in Table III.

  A1  Configuration chosen on development: for each baseline, the one with the highest mean
      F1 on the development scenarios, applied to the held-out ones; per-scenario upper
      bound (best configuration of each scenario), apart.
  A2  Per-family tuning, three conditions for each of the six methods:
      1 default; 2 tuned per family (ceiling): the configuration with the highest mean F1 on
      the scenarios of the family itself; 3 tuned per family with one scenario left out:
      for each scenario, the one with the highest mean F1 on the other scenarios of the family
      (only in families with more than one scenario).
      Comparisons: default pipeline against tuned baselines (conditions 2 and 3);
      tuned pipeline against tuned baselines, in the same condition; default
      against default. Causal Chambers has no pipeline grid: it is left out of the
      comparisons with the tuned pipeline and reported apart.
  A3  Leave-one-family-out for the six methods: for each family, the configuration
      with the highest mean per-scenario F1 on the scenarios of the other families, measured on the
      left-out family, against the default. Pipeline: the output of r1_2b (same rule,
      only complete configurations), checked here.

  Common rules
    - Tie in the selection (difference < 1e-12): the default stays; among tied
      non-defaults, the first in the order of the grid.
    - A configuration only competes if it has a value on every scenario of the selection
      set on which the default of the same method has a value; the selection mean is
      over those scenarios. If the default has no value on any scenario of the
      set, the selection stays with the default (without a value).
    - Empty native graph: the F-test gate variants take the empty graph.
      A1 is also reported by the old rule of r1_2a (those variants without a value;
      only the configurations with the maximum number of scenarios compete; ties among
      non-defaults by the alphabetical order of the name, as in the code of r1_2a).
    - Comparisons: two-sided paired Wilcoxon, Holm over the five baselines;
      mean rank of the six, Friedman and Nemenyi critical difference; on the scenarios
      on which the compared methods have a value (the number is on each row); on
      held-out and in total (development goes beside them).

  Details fixed here, before running on the widened grid (the protocols do not
  define them):
    - Default of the baselines in the widened grid = the default configuration of the grid
      (rerun; for CUTS+, mean of the 3 seeds). The paper's F1 and that of seed
      0 go beside it.
    - Default of the pipeline = the paper's configuration in the main run. The default of the
      sensitivity run goes beside it, and the gain from tuning the pipeline is measured
      against it (same run as the grid).
    - Order of the grid: baselines, that of configuracoes.csv (widened grid) or that of
      first appearance in f1_por_config.csv (original grid); pipeline, that of
      first appearance in the files of the sensitivity run.
    - Original grid, protocol rule: the gate variant missing on a
      scenario on which the default has a value is the empty native graph case (the only
      way r1_2a would not write it) and takes F1 = 0.
    - Result unfavourable to the pipeline = comparison in which the baseline mean is
      greater than or equal to the pipeline's, or in which the Holm p is >= 0.05, or in which the
      baseline has a better mean rank; and every upper bound above the pipeline.

Usage
  python revision/offline/r1_2f_per_family_tuning.py                  (widened grid)
  python revision/offline/r1_2f_per_family_tuning.py --grade-original (grid of r1_2a)
  python revision/offline/r1_2f_per_family_tuning.py --verificar      (test: reads the folder
      r1_2g_grade_ampliada_verificacao and writes to r1_2f_ajuste_por_familia_verificacao; not a result)
Output: Results_Revisao/r1_2f_ajuste_por_familia/ or
        Results_Revisao/r1_2f_ajuste_por_familia_grade_original/
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REVISAO = os.path.dirname(HERE)
for _p in (HERE, REVISAO, os.path.join(REVISAO, "execution")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import common as c  # noqa: E402

NAME = "r1_2f_ajuste_por_familia"
TIE = 1e-12
NAN = float("nan")
WITH_GATE = ("dynotears", "cuts_plus", "varlingam")
PIP = "pipeline"


def canon(s: str) -> str:
    return c.O.canonical(s)


def has(d: dict, s) -> bool:
    v = d.get(s)
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def mean_of(d: dict, sc) -> float:
    xs = [d[s] for s in sc if has(d, s)]
    return float(np.mean(xs)) if xs else NAN


# ---------------------------------------------------------------- inputs
class Method:
    def __init__(self, name, M, default, M_old=None, seed0=None):
        self.name, self.M, self.default = name, M, default
        self.M_old = M if M_old is None else M_old
        self.seed0 = seed0

    @property
    def n_configs(self):
        return int(self.M.notna().any(axis=0).sum())


def load_widened(check: bool = False) -> dict:
    d = os.path.join(c.OUTPUT_ROOT, "r1_2g_grade_ampliada" + ("_verificacao" if check else ""))
    F = pd.read_csv(os.path.join(d, "f1_por_config.csv"), float_precision="round_trip")
    K = pd.read_csv(os.path.join(d, "configuracoes.csv")).sort_values("ordem")
    F["cenario"] = F["cenario"].map(canon)
    met = {}
    for b in c.BASELINES:
        cols = list(K[K.baseline == b]["config"])
        G = F[F.baseline == b]
        piv = lambda v: G.pivot(index="cenario", columns="config", values=v).reindex(columns=cols)  # noqa: E731
        M = piv("f1").astype(float)
        V = piv("nativo_vazio").fillna(False).astype(bool)
        gate = [x for x in cols if "_ftest" in x]
        Ma = M.copy()
        Ma[gate] = Ma[gate].mask(V[gate])
        met[b] = Method(b, M, K[(K.baseline == b) & (K["default"])]["config"].iloc[0], Ma,
                        piv("f1_semente0").astype(float))
    return met


def load_original() -> dict:
    A = pd.read_csv(os.path.join(c.OUTPUT_ROOT, "r1_2a_baselines_ajustados", "f1_por_config.csv"), float_precision="round_trip")
    A["cenario"] = A["cenario"].map(canon)
    met = {}
    for b in c.BASELINES:
        G = A[A.baseline == b]
        cols = list(dict.fromkeys(G["config"]))
        Ma = G.pivot_table(index="cenario", columns="config", values="f1").reindex(columns=cols)
        M = Ma.copy()
        if b in WITH_GATE:
            for col in [x for x in cols if "_ftest" in x]:
                M.loc[M["default"].notna() & M[col].isna(), col] = 0.0
        met[b] = Method(b, M, "default", Ma)
    return met


def load_pipeline(cen: list) -> Method:
    """Grid of the pipeline: sensitivity run (216 configurations, seed 0)."""
    import r1_2b_lofo_cutoffs as r12b
    run = c.O.require("SENSITIVITY_RUN")
    F = r12b.load(run)
    cols = []
    for s in F:
        for k in F[s]:
            if k not in cols:
                cols.append(k)
    M = pd.DataFrame({s: F[s] for s in F if s in set(cen)}).T.reindex(columns=cols).astype(float)
    return Method(PIP, M, "default")


# ---------------------------------------------------------------- selection
def choose(M: pd.DataFrame, default: str, sel: list) -> str:
    """Common rule of the protocols (see docstring)."""
    D = [s for s in sel if s in M.index and not np.isnan(M.at[s, default])]
    if not D:
        return default
    sub = M.loc[D]
    med = sub.loc[:, sub.notna().all(axis=0)].mean(axis=0)
    mx = med.max()
    if med[default] >= mx - TIE:
        return default
    return next(k for k in med.index if med[k] >= mx - TIE)


def choose_old(M: pd.DataFrame, default: str, sel: list):
    """Rule of the code of r1_2a: only the configurations with the maximum number of scenarios with
    a value; highest mean (idxmax over the names in alphabetical order); tie with the default
    -> default. None if no configuration has a value."""
    sub = M.loc[[s for s in sel if s in M.index]]
    cnt = sub.notna().sum(axis=0)
    cnt = cnt[cnt > 0]
    if cnt.empty:
        return None
    med = sub[cnt.index[cnt == cnt.max()]].mean(axis=0).sort_index()
    best = med.idxmax()
    if default in med.index and abs(med[best] - med[default]) < TIE:
        best = default
    return best


# ---------------------------------------------------------------- comparisons
CMP_HEADER = ["Tuned baseline", "Scenarios", "F1 pipeline", "F1 baseline", "Difference",
              "Pipeline higher / equal / lower", "Wilcoxon p", "Holm p"]


def compare(rel, pip: dict, tuned: dict, sc: list, part: str, reg: list, analysis: str, comparison: str,
            header0: str = "Tuned baseline", equal_lines: list = None):
    """Table of the pipeline against each baseline (Wilcoxon, Holm over the five) and the line of the
    ranks, in the formats of r1_2a. Stores each comparison in `reg`."""
    from r1_2a_tuned_baselines import ranks
    pv, body, info = {}, [], {}
    for bl in c.BASELINES:
        ss = [s for s in sc if has(pip, s) and has(tuned[bl], s)]
        x, y = [pip[s] for s in ss], [tuned[bl][s] for s in ss]
        pv[bl] = c.wilcoxon_p(x, y)
        w, t, l = c.wins(x, y)
        mx, my = (float(np.mean(x)), float(np.mean(y))) if ss else (NAN, NAN)
        info[bl] = dict(n=len(ss), f1_pipeline=mx, f1_baseline=my, wtl=f"{w} / {t} / {l}")
        body.append([c.LABEL.get(bl, bl), len(ss), c.fmt(mx), c.fmt(my),
                     f"{mx - my:+.3f}" if ss else "—", f"{w} / {t} / {l}"])
    hp = c.holm(pv)
    for r, bl in zip(body, c.BASELINES):
        r += [f"{pv[bl]:.3g}", f"{hp[bl]:.3g}"]
    rel.p(f"**{part}**")
    rel.table([header0] + CMP_HEADER[1:], body)
    rel.p("")
    rk = None
    ss = [s for s in sc if has(pip, s) and all(has(tuned[bl], s) for bl in c.BASELINES)]
    if len(ss) > 1:
        M = np.array([[pip[s]] + [tuned[bl][s] for bl in c.BASELINES] for s in ss])
        rk, cd, pf = ranks(M)
        line = (f"Mean rank ({len(ss)} scenarios on which all six run; Friedman p = {pf:.2g}; "
                f"Nemenyi critical difference {cd:.2f}): pipeline {rk[0]:.2f}; "
                + "; ".join(f"{c.LABEL.get(bl, bl)} {r:.2f}" for bl, r in zip(c.BASELINES, rk[1:])))
        rel.p(line)
        rel.p("")
        if equal_lines is not None:
            equal_lines.append(line)
    if equal_lines is not None:
        equal_lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in body]
    for i, bl in enumerate(c.BASELINES):
        d = info[bl]
        reasons = []
        if d["n"]:
            if d["f1_baseline"] >= d["f1_pipeline"] - c.EPS:
                reasons.append("média do baseline maior ou igual à do pipeline")
            if hp[bl] >= 0.05:
                reasons.append("diferença não significativa (Holm p >= 0,05)")
        if rk is not None and rk[i + 1] < rk[0] - c.EPS:
            reasons.append(f"posto médio do baseline melhor ({rk[i + 1]:.2f} contra {rk[0]:.2f})")
        reg.append({"analise": analysis, "comparacao": comparison, "parte": part, "baseline": bl,
                    "cenarios": d["n"], "f1_pipeline": d["f1_pipeline"], "f1_baseline": d["f1_baseline"],
                    "diferenca": d["f1_pipeline"] - d["f1_baseline"] if d["n"] else NAN,
                    "pipeline_maior_igual_menor": d["wtl"], "wilcoxon_p": pv[bl], "holm_p": hp[bl],
                    "posto_pipeline": rk[0] if rk is not None else NAN,
                    "posto_baseline": rk[i + 1] if rk is not None else NAN,
                    "desfavoravel": "; ".join(reasons)})


# ---------------------------------------------------------------- A1
def a1(rel, met, pip, parts, reg, rule: str, grid_name: str):
    old = rule == "antiga"
    dev, ho = parts["desenvolvimento"], parts["held-out"]
    choice, body, tuned, lines, csv_rows = {}, [], {}, [], []
    for bl in c.BASELINES:
        m = met[bl]
        M = m.M_old if old else m.M
        cfg = choose_old(M, m.default, dev) if old else choose(M, m.default, dev)
        if cfg is None:
            rel.p(f"- {c.LABEL.get(bl, bl)}: no value on any development scenario.")
            tuned[bl] = {}
            continue
        choice[bl] = cfg
        tuned[bl] = M[cfg].to_dict()
        oracle = M.max(axis=1).to_dict()
        col = lambda k, sc: c.fmt(mean_of(M[k].to_dict(), sc))  # noqa: E731
        body.append([c.LABEL.get(bl, bl), cfg, int(M.notna().any(axis=0).sum()), col(m.default, dev),
                     col(cfg, dev), col(m.default, ho), col(cfg, ho), c.fmt(mean_of(oracle, ho))])
        ss = [s for s in ho if has(pip, s) and has(oracle, s)]
        mp, mo = mean_of(pip, ss), mean_of(oracle, ss)
        csv_rows.append({"regra": rule, "baseline": bl, "config_escolhida": cfg, "configs_na_grade": body[-1][2],
                         "f1_dev_default": mean_of(M[m.default].to_dict(), dev), "f1_dev_ajustado": mean_of(tuned[bl], dev),
                         "f1_heldout_default": mean_of(M[m.default].to_dict(), ho),
                         "f1_heldout_ajustado": mean_of(tuned[bl], ho), "f1_heldout_limite_superior": mean_of(oracle, ho),
                         "cenarios_heldout_limite": len(ss), "f1_pipeline_nos_mesmos": mp})
        if ss and mo >= mp - c.EPS:
            reg.append({"analise": "A1" + (" (regra antiga)" if old else ""),
                        "comparacao": "limite superior por cenário (oráculo; usa o gabarito do próprio cenário)",
                        "parte": "held-out", "baseline": bl, "cenarios": len(ss), "f1_pipeline": mp,
                        "f1_baseline": mo, "diferenca": mp - mo,
                        "desfavoravel": "limite superior do baseline maior ou igual ao pipeline"})
    header = ["Baseline", "Chosen configuration", "Configurations in the grid", "F1 dev (default)",
              "F1 dev (tuned)", "F1 held-out (default)", "F1 held-out (tuned)",
              "F1 held-out (upper bound: best per scenario)"]
    rel.table(header, body)
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in body]
    rel.p("")
    for part in (("desenvolvimento", "held-out") if old else ("desenvolvimento", "held-out", "total")):
        compare(rel, pip, tuned, parts[part], part, reg, "A1" + (" (regra antiga)" if old else ""),
                "pipeline padrão x baselines com a configuração escolhida no desenvolvimento",
                equal_lines=lines if old else None)
    rel.data["A1_" + rule] = choice
    return pd.DataFrame(csv_rows), lines


# ---------------------------------------------------------------- A2
def a2(rel, met, pipg, pip_art, cen, parts, families, fam, reg):
    methods = list(c.BASELINES) + [PIP]
    cond = {m: {1: {}, 2: {}, 3: {}} for m in methods}
    sens0 = {}
    rows = []
    for m in methods:
        obj = pipg if m == PIP else met[m]
        M, d0 = obj.M, obj.default
        for f in families:
            sf = [s for s in cen if fam[s] == f]
            sM = [s for s in sf if s in M.index]
            cfg2 = choose(M, d0, sM) if sM else None
            cfg3 = {}
            for s in sf:
                v1 = float(M.at[s, d0]) if s in M.index else NAN
                if m == PIP:
                    sens0[s] = v1
                    cond[m][1][s] = pip_art.get(s, NAN)
                else:
                    cond[m][1][s] = v1
                cond[m][2][s] = float(M.at[s, cfg2]) if s in M.index else NAN
                if len(sf) > 1 and s in M.index:
                    cfg3[s] = choose(M, d0, [x for x in sM if x != s])
                    cond[m][3][s] = float(M.at[s, cfg3[s]])
            n_grid = int(M.loc[sM].notna().any(axis=0).sum()) if sM else 0
            rows.append({"familia": f, "parte": "held-out" if c.is_heldout(sf[0]) else "desenvolvimento",
                         "cenarios": len(sf), "teto_puro": len(sf) == 1, "metodo": m,
                         "configs_na_grade": obj.n_configs, "configs_com_valor_na_familia": n_grid,
                         "config_teto": cfg2 if cfg2 else "",
                         "cenarios_com_valor_padrao": sum(has(cond[m][1], s) for s in sf),
                         "f1_padrao": mean_of(cond[m][1], sf),
                         "f1_padrao_na_grade": mean_of(sens0, sf) if m == PIP else mean_of(cond[m][1], sf),
                         "f1_teto": mean_of(cond[m][2], sf),
                         "f1_um_cenario_de_fora": mean_of(cond[m][3], sf) if len(sf) > 1 else NAN,
                         "configs_um_cenario_de_fora": "; ".join(sorted(set(cfg3.values())))})
    L = pd.DataFrame(rows)
    rel.csv("a2_por_familia.csv", L)
    rel.csv("a2_por_cenario.csv", pd.DataFrame(
        [{"cenario": s, "familia": fam[s], "heldout": c.is_heldout(s), "metodo": m,
          "f1_padrao": cond[m][1].get(s, NAN), "f1_teto": cond[m][2].get(s, NAN),
          "f1_um_cenario_de_fora": cond[m][3].get(s, NAN),
          "f1_default_run_sensibilidade": sens0.get(s, NAN) if m == PIP else NAN}
         for m in methods for s in cen]))

    rel.p(f"Per family and method (ceiling configuration, F1 in the three conditions, number of configurations): "
          f"`a2_por_familia.csv`; per scenario: `a2_por_cenario.csv`. Single-scenario families (pure ceiling): "
          + ", ".join(f for f in families if sum(fam[s] == f for s in cen) == 1) + ".")
    rel.h("### Summary: mean F1 in the three conditions and gain from tuning")
    rel.p("Condition 2 (ceiling) uses the ground truth of the same scenarios on which it is measured: it is an upper "
          "bound, not a usable result. Gain = mean of the paired differences on the scenarios on which both conditions "
          "have a value. For the pipeline, the default is the paper's configuration (main run), and the gain is "
          "measured against the default of the sensitivity run (the same run as the grid).")
    for part, sc in parts.items():
        body = []
        for m in methods:
            base = sens0 if m == PIP else cond[m][1]

            def gain(k):
                ss = [s for s in sc if has(base, s) and has(cond[m][k], s)]
                return (f"{np.mean([cond[m][k][s] - base[s] for s in ss]):+.3f} ({len(ss)})" if ss else "—")

            n = lambda d: sum(has(d, s) for s in sc)  # noqa: E731
            body.append(["Pipeline" if m == PIP else c.LABEL.get(m, m),
                         (pipg if m == PIP else met[m]).n_configs,
                         f"{c.fmt(mean_of(cond[m][1], sc))} ({n(cond[m][1])})",
                         f"{c.fmt(mean_of(sens0, sc))} ({n(sens0)})" if m == PIP else "—",
                         f"{c.fmt(mean_of(cond[m][2], sc))} ({n(cond[m][2])})", gain(2),
                         f"{c.fmt(mean_of(cond[m][3], sc))} ({n(cond[m][3])})", gain(3)])
        rel.p(f"**{part}** (mean F1; in parentheses, the number of scenarios)")
        rel.table(["Method", "Configurations in the grid", "1 Default", "Default in the sensitivity run",
                   "2 Tuned per family (ceiling)", "Gain 2 − 1", "3 One scenario left out", "Gain 3 − 1"], body)
        rel.p("")

    for title, kp, kb, comparison in (
            ("### Default pipeline against baselines tuned per family (condition 2, ceiling)", 1, 2,
             "pipeline padrão x baselines ajustados por família (teto)"),
            ("### Default pipeline against baselines tuned with one scenario left out (condition 3)", 1, 3,
             "pipeline padrão x baselines ajustados por família com um cenário de fora"),
            ("### Tuned pipeline against tuned baselines, condition 2 (ceiling against ceiling)", 2, 2,
             "pipeline ajustado (teto) x baselines ajustados (teto)"),
            ("### Tuned pipeline against tuned baselines, condition 3 (one scenario left out)", 3, 3,
             "pipeline ajustado x baselines ajustados, um cenário de fora"),
            ("### Default against default (the paper's comparison, on the same scenarios)", 1, 1,
             "pipeline padrão x baselines padrão")):
        rel.h(title)
        for part in ("held-out", "total", "desenvolvimento"):
            compare(rel, cond[PIP][kp], {bl: cond[bl][kb] for bl in c.BASELINES}, parts[part], part, reg,
                    "A2", comparison, header0="Baseline")

    cc = [s for s in cen if fam[s] == "Causal Chambers"]
    rel.h("### Causal Chambers, apart (no pipeline grid)")
    if cc:
        rel.table(["Method", "Scenarios", "1 Default", "2 Ceiling", "3 One scenario left out"],
                  [["Pipeline" if m == PIP else c.LABEL.get(m, m), len(cc), c.fmt(mean_of(cond[m][1], cc)),
                    c.fmt(mean_of(cond[m][2], cc)), c.fmt(mean_of(cond[m][3], cc))] for m in methods])
        rel.p("")
        rel.p("The pipeline only has condition 1 in this family: the 216 configurations were not run on Causal "
              "Chambers (a decision recorded in the protocols).")
    else:
        rel.p("No Causal Chambers scenario in this input.")
    return cond, sens0


# ---------------------------------------------------------------- A3
def a3(rel, met, pipg, pip_art, cen, parts, families, fam, reg):
    lofo, dflt, rows = {}, {}, []
    for bl in c.BASELINES:
        M, d0 = met[bl].M, met[bl].default
        lofo[bl], dflt[bl] = {}, {}
        for f in families:
            sf = [s for s in cen if fam[s] == f]
            left_out = [s for s in sf if s in M.index]
            inside = [s for s in cen if fam[s] != f and s in M.index]
            cfg = choose(M, d0, inside)
            for s in left_out:
                lofo[bl][s], dflt[bl][s] = float(M.at[s, cfg]), float(M.at[s, d0])
            rows.append({"metodo": bl, "familia": f, "cenarios": len(sf), "heldout": c.is_heldout(sf[0]),
                         "config": cfg, "igual_ao_default": cfg == d0,
                         "cenarios_com_valor": sum(has(lofo[bl], s) and has(dflt[bl], s) for s in left_out),
                         "f1_default": mean_of(dflt[bl], [s for s in left_out if has(lofo[bl], s)]),
                         "f1_lofo": mean_of(lofo[bl], [s for s in left_out if has(dflt[bl], s)])})
    # pipeline: the rule of r1_2b (only complete configurations), checked against its output
    Mp = pipg.M.dropna(axis=1, how="any")
    pr = os.path.join(c.OUTPUT_ROOT, "r1_2b_lofo_cutoffs", "por_familia.csv")
    if not os.path.isfile(pr):
        raise SystemExit("[r1_2f] missing Results_Revisao/r1_2b_lofo_cutoffs/por_familia.csv: run first "
                         "python revision/offline/r1_2b_lofo_cutoffs.py")
    R = pd.read_csv(pr, float_precision="round_trip").set_index("familia")
    lofo[PIP], dflt[PIP], checks = {}, {}, []
    for f in families:
        left_out = [s for s in cen if fam[s] == f and s in Mp.index]
        if not left_out:
            continue
        inside = [s for s in Mp.index if fam.get(s, c.family(s)) != f]
        med = Mp.loc[inside].mean(axis=0)
        cfg = med.idxmax()
        if abs(med[cfg] - med["default"]) < TIE:
            cfg = "default"
        if f in R.index:
            same = (R.at[f, "config"] == cfg)
            same_mean = abs(R.at[f, "f1_lofo"] - Mp.loc[left_out, R.at[f, "config"]].mean()) < 1e-9
            checks.append((f, bool(same), bool(same_mean)))
            cfg = R.at[f, "config"]              # the output of r1_2b prevails
        else:
            checks.append((f, False, False))
        for s in left_out:
            lofo[PIP][s], dflt[PIP][s] = float(Mp.at[s, cfg]), float(Mp.at[s, "default"])
        rows.append({"metodo": PIP, "familia": f, "cenarios": len(left_out), "heldout": c.is_heldout(left_out[0]),
                     "config": cfg, "igual_ao_default": cfg == "default", "cenarios_com_valor": len(left_out),
                     "f1_default": mean_of(dflt[PIP], left_out), "f1_lofo": mean_of(lofo[PIP], left_out)})
    L = pd.DataFrame(rows)
    rel.csv("a3_por_familia.csv", L)
    rel.p(f"Pipeline: output of `r1_2b_lofo_cutoffs` ({len(Mp)} scenarios, {Mp.shape[1]} complete configurations, "
          f"{len(checks)} families). Check against the selection recomputed here: same configuration in "
          f"{sum(a for _, a, _ in checks)} of {len(checks)} families; same mean on the left-out family in "
          f"{sum(b for _, _, b in checks)} of {len(checks)}.")
    rel.data["A3_confere_r1_2b"] = checks
    rel.p("")
    rel.p("Per family and method: `a3_por_familia.csv`. Difference = F1 of the configuration chosen on the other "
          "families minus F1 of the default, on the left-out family (mean per family).")
    body = []
    for m in list(c.BASELINES) + [PIP]:
        G = L[(L.metodo == m) & (L.cenarios_com_valor > 0)]
        dif = G["f1_lofo"] - G["f1_default"]
        h = G["heldout"].astype(bool)
        body.append(["Pipeline" if m == PIP else c.LABEL.get(m, m), len(G), int(G["igual_ao_default"].sum()),
                     f"{dif.mean():+.3f}", int((dif > 1e-9).sum()), int((dif < -1e-9).sum()),
                     f"{dif[~h].mean():+.3f} ({int((~h).sum())})", f"{dif[h].mean():+.3f} ({int(h.sum())})",
                     c.fmt(mean_of(dflt[m], parts["held-out"])), c.fmt(mean_of(lofo[m], parts["held-out"]))])
    rel.table(["Method", "Families with a value", "Chosen = default", "Mean difference per family",
               "Families that improve", "Families that worsen", "Difference dev (families)",
               "Difference held-out (families)", "F1 held-out default (mean per scenario)",
               "F1 held-out LOFO (mean per scenario)"], body)
    for title, pp, comparison in (
            ("### Default pipeline (paper) against baselines with the leave-one-family-out configuration", pip_art,
             "pipeline padrão x baselines LOFO"),
            ("### LOFO pipeline (grid of the sensitivity run) against LOFO baselines", lofo[PIP],
             "pipeline LOFO x baselines LOFO")):
        rel.h(title)
        for part in ("held-out", "total", "desenvolvimento"):
            compare(rel, pp, {bl: lofo[bl] for bl in c.BASELINES}, parts[part], part, reg, "A3", comparison,
                    header0="Baseline (LOFO)")
    return lofo, dflt


# ---------------------------------------------------------------- main
def main():
    original = "--grade-original" in sys.argv
    check = "--verificar" in sys.argv and not original
    rel = c.Report(NAME + ("_grade_original" if original else "") + ("_verificacao" if check else ""),
                   "R1-2 — A1, A2 and A3 on the " + ("original grid (r1_2a)" if original else "widened grid (r1_2g)"))
    met = load_original() if original else load_widened(check)
    cen = sorted(set().union(*[set(m.M.index) for m in met.values()]))
    fam = {s: c.family(s) for s in cen}
    families = [f for f in c.FAMILIES_DEV + c.FAMILIES_HO if f in set(fam.values())]
    families += sorted(set(fam.values()) - set(families))
    parts = {"desenvolvimento": [s for s in cen if not c.is_heldout(s)],
             "held-out": [s for s in cen if c.is_heldout(s)], "total": cen}
    pip_art = {}
    for s in cen:
        k = c.scenario_key(s)
        if k is not None:
            pip_art[s] = c.f1_model(k, c.PIPE)
    pipg = load_pipeline(cen)
    reg = []

    rel.h("## Inputs")
    rel.p(f"- Grid of the baselines: {'original, of r1_2a' if original else 'widened (r1_2g)'}; {len(cen)} scenarios "
          f"({len(parts['desenvolvimento'])} development, {len(parts['held-out'])} held-out), "
          f"{len(families)} families. Configurations with some value: "
          + "; ".join(f"{c.LABEL.get(b, b)} {met[b].n_configs}" for b in c.BASELINES) + ".")
    no_pip = [s for s in cen if not has(pip_art, s)]
    rel.p(f"- Pipeline, the paper's configuration: value on {len(cen) - len(no_pip)} of {len(cen)} scenarios"
          + (f" (**no value on: {', '.join(no_pip)}** — paper runs missing on this machine)" if no_pip else "")
          + ".")
    eq = [s for s in pipg.M.index if has(pip_art, s) and abs(pipg.M.at[s, 'default'] - pip_art[s]) < 1e-9]
    rel.p(f"- Pipeline, grid: {pipg.M.shape[1]} configurations of the sensitivity run on {len(pipg.M)} scenarios "
          f"({int(pipg.M.notna().all(axis=0).sum())} complete). The default of that run equals the F1 of the main "
          f"run on {len(eq)} of {len(pipg.M)} scenarios; means on the same scenarios: "
          f"{c.fmt(mean_of(pipg.M['default'].to_dict(), list(pipg.M.index)))} (sensitivity) and "
          f"{c.fmt(mean_of(pip_art, list(pipg.M.index)))} (main run).")
    if not original:
        body = []
        for bl in c.BASELINES:
            m = met[bl]
            d = m.M[m.default].to_dict()
            s0 = m.seed0[m.default].to_dict()
            art = {s: c.f1_model(c.scenario_key(s), bl) for s in cen if c.scenario_key(s) is not None}
            for part in ("desenvolvimento", "held-out"):
                ss = [s for s in parts[part] if has(d, s) and has(art, s)]
                body.append([c.LABEL.get(bl, bl), part, len(ss), c.fmt(mean_of(art, ss)), c.fmt(mean_of(d, ss)),
                             c.fmt(mean_of(s0, ss)), sum(abs(d[s] - art[s]) > 1e-9 for s in ss)])
        rel.p("")
        rel.p("Default of each baseline: the paper's F1 beside the default rerun in the grid (on the scenarios on "
              "which both have a value):")
        rel.table(["Baseline", "Part", "Scenarios", "F1 in the paper", "F1 grid default",
                   "F1 default, seed 0", "Scenarios with F1 different from the paper"], body)

    rel.h("## A1 — Configuration chosen on development and applied to held-out")
    rel.h("### Protocol rule (section 5 of the widened-grid protocol)")
    t1, _ = a1(rel, met, pip_art, parts, reg, "protocolo", "original" if original else "ampliada")
    rel.h("### Old rule of r1_2a (gate variants without a value when the native graph is empty)")
    t2, lines = a1(rel, met, pip_art, parts, reg, "antiga", "original" if original else "ampliada")
    rel.csv("a1_configuracao_escolhida.csv", pd.concat([t1, t2], ignore_index=True))
    if original:
        pr = os.path.join(c.OUTPUT_ROOT, "r1_2a_baselines_ajustados", "relatorio.md")
        ref = {x.rstrip("\n") for x in open(pr, encoding="utf-8")} if os.path.isfile(pr) else set()
        missing = [x for x in lines if x not in ref]
        rel.p(f"**Acceptance criterion of the script:** {len(lines) - len(missing)} of {len(lines)} lines of the "
              f"tables and ranks above are identical to those of the `relatorio.md` of r1_2a.")
        rel.data["aceitacao_linhas"] = len(lines)
        rel.data["aceitacao_iguais"] = len(lines) - len(missing)
        for x in missing:
            rel.p(f"- different: `{x}`")

    rel.h("## A2 — Per-family tuning, for the six methods")
    a2(rel, met, pipg, pip_art, cen, parts, families, fam, reg)

    rel.h("## A3 — Leave-one-family-out for the six methods")
    a3(rel, met, pipg, pip_art, cen, parts, families, fam, reg)

    Rg = pd.DataFrame(reg)
    rel.csv("comparacoes.csv", Rg)
    rel.h("## Results unfavourable to the pipeline")
    D = Rg[Rg["desfavoravel"].fillna("") != ""]
    rel.data["desfavoraveis"] = int(len(D))
    if D.empty:
        rel.p("No comparison in which a baseline is above the pipeline, has a better mean rank or in which the "
              "difference stops being significant.")
    else:
        rel.p(f"{len(D)} of {len(Rg)} comparisons (all in `comparacoes.csv`). Criterion in the script's docstring.")
        rel.p("")
        rel.table(["Analysis", "Comparison", "Part", "Baseline", "Scenarios", "F1 pipeline", "F1 baseline",
                   "Difference", "Pipeline higher / equal / lower", "Holm p", "Why"],
                  [[r.analise, r.comparacao, r.parte, c.LABEL.get(r.baseline, r.baseline), r.cenarios,
                    c.fmt(r.f1_pipeline), c.fmt(r.f1_baseline), f"{r.diferenca:+.3f}",
                    r.pipeline_maior_igual_menor if isinstance(r.pipeline_maior_igual_menor, str) else "—",
                    f"{r.holm_p:.3g}" if not pd.isna(r.holm_p) else "—", r.desfavoravel]
                   for r in D.itertuples()])
    rel.save()


if __name__ == "__main__":
    from run_log import execution
    with execution(NAME, __file__):
        main()
