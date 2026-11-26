"""
Functions shared by the revision experiments (IEEE Access, Access-2026-54916).

The experiments read the same runs as the paper, through the origins in
generators/origins.py (generators/origins_local.py), and score as the paper does:
directional entity pairs, F1 per scenario, development (51) and held-out (39)
reported apart. Loading and the membership of each part come from
generators/paper_numbers.py, so that the reference numbers come out identical to
those of Table III.

Outputs: Results_Revisao/<experiment>/ (ignored by git).
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "generators")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import origins as O  # noqa: E402
import paper_numbers as pn  # noqa: E402

pn.WINDOW = "10"            # main window of the paper
PIPE = pn.PIPELINE_MODEL     # 'patch' (read from patch_tau10 at WINDOW = 10)
PIPE_FOLDER = "patch_tau10"
BASELINES = [m for m, _ in pn.BASELINES]
LABEL = dict(pn.LABEL)
EPS = 1e-9
OUTPUT_ROOT = os.path.join(ROOT, "Results_Revisao")

# Families of Table IV, in the order of the paper.
FAMILIES_DEV = ["VAR synthetic", "tigramite SCM", "Lorenz-96", "Coupled logistic maps",
                "CausalTime", "ALARM", "NetSim"]
FAMILIES_HO = ["Tennessee Eastman", "Walker Circulation", "Sachs", "PeMS-BAY", "PetShop",
               "BEELINE", "DREAM3", "RCAEval", "EasyVista", "Causal Chambers"]
_FAM = [("synthetic", "VAR synthetic"), ("climate_walker", "Walker Circulation"),
        ("tigramite", "tigramite SCM"), ("lorenz", "Lorenz-96"),
        ("logistic", "Coupled logistic maps"), ("causaltime", "CausalTime"),
        ("tep", "Tennessee Eastman"), ("sachs", "Sachs"), ("alarm", "ALARM"),
        ("pems", "PeMS-BAY"), ("netsim", "NetSim"), ("petshop", "PetShop"),
        ("beeline", "BEELINE"), ("dream3", "DREAM3"), ("rcaeval", "RCAEval"),
        ("easyvista", "EasyVista"), ("causal_chamber", "Causal Chambers")]


# ----------------------------------------------------------------------------- scenarios
_ROWS = None


def rows():
    """rows[scenario][model] = dict(tp, fp, fn, n_causal, n_universe, n_pred, run),
    exactly as generators/paper_numbers.py (90 scenarios at tau_max = 10)."""
    global _ROWS
    if _ROWS is None:
        _ROWS, _ = pn.load()
    return _ROWS


def scenarios(part: str = "todos") -> list:
    """'todos' (all), 'dev' (51) or 'ho' (39)."""
    sc = sorted(s for s, mm in rows().items() if PIPE in mm)
    if part == "dev":
        return [s for s in sc if not is_heldout(s)]
    if part == "ho":
        return [s for s in sc if is_heldout(s)]
    return sc


def scenario_key(name: str):
    """Key of rows() for a current or former scenario name (the paper's runs store
    five scenarios under the ID they had before translation, e.g. real_alarm_uti)."""
    if name in rows():
        return name
    mapping = {O.canonical(k): k for k in rows()}
    return mapping.get(O.canonical(name))


def folder_in(run: str, scen: str):
    """Folder of the scenario in any run, accepting the current or the former name; None if absent."""
    names = [scen, O.canonical(scen)] + [old for old, new in getattr(O, "LEGACY_SCENARIOS", {}).items()
                                         if O.canonical(new) == O.canonical(scen) or new == scen]
    for n in names:
        p = os.path.join(run, n)
        if os.path.isdir(p):
            return p
    return None


def is_heldout(scen: str) -> bool:
    return pn.group_of(scen) in pn.HELDOUT


def family(scen: str) -> str:
    g = pn.group_of(scen)
    for k, v in _FAM:
        if k in g:
            return v
    return g


def group(scen: str) -> str:
    return pn.group_of(scen)


def folder(scen: str, model: str = None) -> str:
    """Folder of the scenario in the run the model (by default, the pipeline) was read from."""
    return os.path.join(pn.run_of(rows(), scen, model or PIPE), scen)


def truth(scen: str):
    """(T, U): causal pairs and universe of pairs (directional, entity level)."""
    gt = pd.read_csv(os.path.join(folder(scen), "ground_truth.csv"), skipinitialspace=True)
    for c in ("source_entity", "target_entity"):
        gt[c] = gt[c].astype(str).str.strip()
    pos = gt["causal_gt"].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])
    T = set(map(tuple, gt.loc[pos, ["source_entity", "target_entity"]].values))
    U = set(map(tuple, gt[["source_entity", "target_entity"]].values))
    return T, U


def trivial_floor(scen: str) -> float:
    T, U = truth(scen)
    p = len(T) / len(U) if U else 0.0
    return 2 * p / (1 + p)


def original(scen: str, model_folder: str = PIPE_FOLDER, scenario_folder: str = None) -> pd.DataFrame:
    p = os.path.join(scenario_folder or folder(scen), model_folder, "original.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p, skipinitialspace=True)
    for c in ("source_entity", "target_entity"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def pairs(df: pd.DataFrame) -> pd.Series:
    return df[["source_entity", "target_entity"]].apply(tuple, axis=1)


# ----------------------------------------------------------------------------- metrics
def counts(S: set, T: set):
    return len(S & T), len(S - T), len(T - S)


def f1(tp, fp, fn) -> float:
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def f1_set(S: set, T: set) -> float:
    return f1(*counts(S, T))


def f1_model(scen: str, model: str):
    mm = rows()[scen]
    if model not in mm:
        return None
    r = mm[model]
    return f1(r["tp"], r["fp"], r["fn"])


def wilcoxon_p(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or np.all(np.abs(a - b) < EPS):
        return 1.0
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return 1.0


def holm(pvals: dict) -> dict:
    it = sorted(pvals.items(), key=lambda x: x[1])
    m, run, out = len(it), 0.0, {}
    for i, (k, p) in enumerate(it):
        run = max(run, min(1.0, p * (m - i)))
        out[k] = run
    return out


def wins(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return int((a > b + EPS).sum()), int((np.abs(a - b) <= EPS).sum()), int((a < b - EPS).sum())


def auprc(y, s) -> float:
    """Area under the precision-recall curve (average precision, ties treated as a block)."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    if y.sum() == 0:
        return float("nan")
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s))


def auroc(y, s) -> float:
    y, s = np.asarray(y, int), np.asarray(s, float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


# ----------------------------------------------------------------------------- report
class Report:
    """Markdown and JSON report of an experiment, in Results_Revisao/<name>/."""

    def __init__(self, name: str, title: str):
        self.dir = os.path.join(OUTPUT_ROOT, name)
        os.makedirs(self.dir, exist_ok=True)
        self.lines = [f"# {title}", ""]
        self.data = {}

    def p(self, t: str = ""):
        self.lines.append(t)
        print(t)

    def h(self, t: str):
        self.p("")
        self.p(t)
        self.p("")

    def table(self, header, body):
        self.p("| " + " | ".join(map(str, header)) + " |")
        self.p("|" + "---|" * len(header))
        for r in body:
            self.p("| " + " | ".join(str(x) for x in r) + " |")

    def csv(self, name: str, df: pd.DataFrame):
        df.to_csv(os.path.join(self.dir, name), index=False)

    def save(self):
        with open(os.path.join(self.dir, "relatorio.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")
        with open(os.path.join(self.dir, "dados.json"), "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nok -> {os.path.join(self.dir, 'relatorio.md')}")


def fmt(x, d=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def part_means(values: dict, extra_header=None):
    """Means per part of the suite: {name: {scenario: value}} -> table rows."""
    out = []
    for part, sc in (("all", scenarios()), ("development", scenarios("dev")),
                     ("held-out", scenarios("ho"))):
        for name, v in values.items():
            xs = [v[s] for s in sc if s in v and v[s] is not None]
            out.append([part, name, len(xs), fmt(np.mean(xs)) if xs else "—"])
    return out


def by_family(values: dict) -> dict:
    agg = defaultdict(list)
    for s, x in values.items():
        if x is not None:
            agg[family(s)].append(x)
    return {k: float(np.mean(v)) for k, v in agg.items()}
