"""
majority_vote.py
================
Majority-vote study over an EXISTING run, without re-running any model.

For every scenario and for every lag window (tau24 = plain model names,
tau10 = '<name>_tau10'), the baselines of that window (every model that is
NOT the patch) vote on each directional pair: a model votes "causal" for a
pair when the pair is in its results.csv. The accepted pairs are then scored
against ground_truth.csv as if they were the output of a new model
('majority_vote' / 'majority_vote_tau10'), with the same pair identity and
the same hard metric used by main.validate_dataset() (directional pairs,
intra-entity pairs counted).

SCORING UNIT. The vote that is reported and compared is taken at the unit of
the paper: the ORDERED PAIR OF ENTITIES, whatever the property
and the lag -- a model votes for (A, B) when it has at least one row from an
entity-property series of A to one of B. This is the level of
validation_summary.csv, and every F1, mean and win/tie/loss written by this
script is at that level. A second vote, at the (entity, property) level, is
kept only for the property-level table of validation_summary.txt.

RULE. A pair is accepted by a STRICT majority of the voters (more than half:
3 of 5, 3 of 4) when STRICT_MAJORITY is True; otherwise by at least
VOTE_THRESHOLD of them (which accepts a 2-of-4 split).

Voters are the non-patch models of the window that have a results.csv in
the scenario (a model that ended with status ERROR does not vote, so the
number of voters can change between scenarios).

Written per scenario, in <scenario>/majority_vote/:
    votes_entity_<tau>.csv  every entity pair of the GT universe (plus any
                        voted pair), one 0/1 column per voter, n_votes,
                        vote_fraction, majority, the patch answer, outcome
    votes_<tau>.csv     the same at the (entity, property) level
    results_<tau>.csv   the accepted (entity, property) pairs
    summary.csv / .txt  metrics of the vote next to patch and best baseline
                        (entity-pair level)

Also adds the two new rows to <scenario>/validation_summary.csv (entity
level, same columns as the other models) and to the property-level table of
<scenario>/validation_summary.txt.

Written at the run root:
    majority_vote_summary.csv   one line per scenario x tau
    majority_vote_wtl.txt       mean F1 and win/tie/loss of the patch, first
                                on the scenarios of the paper (without
                                SECONDARY_EXCLUDE), then on every scenario

Re-running is safe: previous majority_vote rows/files are replaced.

Usage:
    python Auxiliary/majority_vote.py [RUN_DIR]
"""

import csv
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# =============================================================================
# CONFIGURATION  <- edit here
# =============================================================================

RUN_DIR = "./Results/Run_20261002_013625"

# True: a pair needs MORE than half of the voters (3 of 5, 3 of 4) -- a
# majority vote proper. False: at least VOTE_THRESHOLD of them (>=, so a
# 2-of-4 split is accepted).
STRICT_MAJORITY = True
VOTE_THRESHOLD = 0.5

# Models whose name starts with this prefix do not vote.
PATCH_PREFIX = "patch"

MV_NAME   = "majority_vote"
OUT_FOLDER = "majority_vote"

# Window of the models without a '_tau<N>' suffix (Config/execution.py).
DEFAULT_TAU = "tau24"

# Order in which the windows are reported.
TAU_ORDER = ["tau10", "tau24"]

# Families reported apart in a second W/T/L block (scenario-name substrings).
SECONDARY_EXCLUDE = ("dream4", "causal_rivers")

# =============================================================================

KEY_COLS = ["source_entity", "source_property", "target_entity", "target_property"]
TRUE_STRINGS = ["true", "1", "1.0", "yes", "t", "y"]

SUMMARY_COLS = [
    "model", "status",
    "precision", "recall", "f1", "accuracy", "fpr",
    "TP", "FP", "FN", "TN",
    "TP_hard", "TP_soft", "FP_hard", "FN_soft", "TN_soft",
    "precision_soft", "recall_soft", "f1_soft",
    "soft_threshold",
    "n_predicted", "n_tested", "n_causal", "n_universe", "n_outside_universe", "duration",
]

TXT_NOTE_TAG = "  [MAJORITY VOTE]"


def _safe_div(num, den):
    return num / den if den > 0 else 0.0


def _tau_of(model_name: str) -> tuple:
    """'pcmciplus_tau10' -> ('pcmciplus', 'tau10'); 'pcmciplus' -> ('pcmciplus', DEFAULT_TAU)."""
    m = re.search(r"_(tau\d+)$", model_name)
    if m:
        return model_name[:m.start()], m.group(1)
    return model_name, DEFAULT_TAU


def _mv_name(tau: str) -> str:
    return MV_NAME if tau == DEFAULT_TAU else f"{MV_NAME}_{tau}"


def _family(scenario: str) -> str:
    return re.sub(r"(_\d+)?_(seed|sample|net)\d+$", "", scenario)


def _status(model_dir: Path) -> str:
    time_path = model_dir / "time.txt"
    if not time_path.exists():
        return "N/A"
    for line in time_path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "status":
            return value.strip()
    return "N/A"


def _keys(df: pd.DataFrame) -> set:
    """Directional (entity, property) pair keys of a results/GT dataframe."""
    if df.empty:
        return set()
    cols = []
    for c in KEY_COLS:
        if c in df.columns:
            cols.append(df[c].astype(str).str.strip())
        else:
            cols.append(pd.Series([""] * len(df), index=df.index))
    return set(zip(*cols))


def _entity(pairs: set) -> set:
    return {(p[0], p[2]) for p in pairs}


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, skipinitialspace=True)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _metrics(predicted: set, causal: set, universe: set) -> dict:
    TP = len(causal & predicted)
    FN = len(causal - predicted)
    FP = len(predicted - causal)
    TN = len(universe - causal - predicted)
    precision = _safe_div(TP, TP + FP)
    recall    = _safe_div(TP, TP + FN)
    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": precision, "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "accuracy": _safe_div(TP + TN, TP + FP + FN + TN),
        "fpr": _safe_div(FP, FP + TN),
        "n_predicted": len(predicted),
    }


def _accepted(n_votes: int, n_voters: int) -> bool:
    """Vote rule (see STRICT_MAJORITY)."""
    if STRICT_MAJORITY:
        return 2 * n_votes > n_voters
    return n_votes >= VOTE_THRESHOLD * n_voters


def _min_votes(n_voters: int) -> int:
    return next(k for k in range(n_voters + 2) if _accepted(k, n_voters))


def _rule_text(n_voters: int) -> str:
    k = _min_votes(n_voters)
    return (f"more than half ({k} of {n_voters})" if STRICT_MAJORITY
            else f">= {VOTE_THRESHOLD:.0%} ({k} of {n_voters})")


def _outcome(a: float, b: float) -> str:
    """W/T/L of a against b, on F1 rounded to 4 decimals (as in the summaries)."""
    a, b = round(a, 4), round(b, 4)
    return "W" if a > b else ("L" if a < b else "T")


# =============================================================================
# validation_summary.csv / .txt
# =============================================================================

def _read_text(path: Path) -> tuple:
    text = path.read_bytes().decode("utf-8")
    eol = "\r\n" if "\r\n" in text else "\n"
    return text, eol


def _update_summary_csv(path: Path, new_rows: list):
    """Replaces/inserts the majority-vote rows, keeping the other rows untouched."""
    text, eol = _read_text(path)
    rows = list(csv.reader(text.splitlines()))
    header, body = rows[0], [r for r in rows[1:] if r and not r[0].startswith(MV_NAME)]
    for new in new_rows:
        line = [str(new.get(c, "")) for c in header]
        pos = next((i for i, r in enumerate(body) if r[0] > line[0]), len(body))
        body.insert(pos, line)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator=eol)
        writer.writerow(header)
        writer.writerows(body)


def _txt_table_rows(lines: list) -> tuple:
    """(first, last+1) line indexes of the model rows of the property-level table."""
    start = next((i + 2 for i, l in enumerate(lines)
                  if l.strip().startswith("MODEL") and "PRED_F" in l), None)
    if start is None:
        return None, None
    end = start
    while end < len(lines) and lines[end].strip() and not lines[end].startswith(TXT_NOTE_TAG):
        end += 1
    return start, end


def _read_txt_table(path: Path) -> dict:
    """{model: {TP, FP, FN, f1}} from the property-level table (sanity check)."""
    if not path.exists():
        return {}
    text, _ = _read_text(path)
    lines = text.splitlines()
    start, end = _txt_table_rows(lines)
    out = {}
    if start is None:
        return out
    for line in lines[start:end]:
        tok = line.split()
        if len(tok) >= 9 and tok[1] != "---":
            out[tok[0]] = {"f1": float(tok[3]), "TP": int(tok[4]),
                           "FP": int(tok[5]), "FN": int(tok[6])}
    return out


def _update_summary_txt(path: Path, new_rows: list, notes: list):
    """Inserts the majority-vote rows in the property-level table and refreshes
    the model counts and the 'Best F1' line."""
    text, eol = _read_text(path)
    lines = [l for l in text.splitlines() if not l.startswith(TXT_NOTE_TAG)]
    while lines and not lines[-1].strip():
        lines.pop()

    start, end = _txt_table_rows(lines)
    if start is not None:
        n_kept = end - start
        table = [l for l in lines[start:end] if not l.split()[0].startswith(MV_NAME)]
        for name, m in new_rows:
            line = (f"  {name:<22} "
                    f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} "
                    f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5} "
                    f"{m['n_predicted']:>7}")
            pos = next((i for i, l in enumerate(table) if l.split()[0] > name), len(table))
            table.insert(pos, line)
        lines[start:end] = table

        scored = [(l.split()[0], float(l.split()[3])) for l in table if l.split()[1] != "---"]
        n_added = len(table) - n_kept   # 0 when the rows were already there (re-run)
        # The vote only takes the 'Best F1' line when it strictly beats every
        # model (first max among the models, as idxmax in main.py).
        models = [s for s in scored if not s[0].startswith(MV_NAME)]
        best = max(models, key=lambda x: x[1]) if models else None
        best_mv = max((s for s in scored if s[0].startswith(MV_NAME)), key=lambda x: x[1])
        if best is None or best_mv[1] > best[1]:
            best = best_mv
        for i, l in enumerate(lines[:start]):
            for label in ("Models evaluated", "With OK results"):
                if l.strip().startswith(label):
                    lines[i] = f"{l.split(':')[0]}: {int(l.split(':')[1]) + n_added}"
            if l.strip().startswith("Best F1") and best:
                lines[i] = f"{l.split(':')[0]}: {best[0]}  (F1={best[1]:.4f})"

    lines.append("")
    lines.extend(notes)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(eol.join(lines))


# =============================================================================
# ONE SCENARIO
# =============================================================================

def process_scenario(folder: Path, mismatches: list) -> list:
    gt = _read_csv(folder / "ground_truth.csv")
    has_causal = "causal_gt" in gt.columns
    has_property = "source_property" in gt.columns and "target_property" in gt.columns
    if not has_property:
        print(f"  [WARNING] {folder.name}: GT without property columns -- skipped.")
        return []

    if has_causal:
        is_causal = gt["causal_gt"] if gt["causal_gt"].dtype == bool else \
            gt["causal_gt"].astype(str).str.strip().str.lower().isin(TRUE_STRINGS)
        causal_f = _keys(gt[is_causal])
        universe_f = _keys(gt)
    else:
        causal_f = _keys(gt)
        universe_f = set(causal_f)
    causal_e, universe_e = _entity(causal_f), _entity(universe_f)

    # ── Predictions of every model already in the scenario ───────────────────
    predicted: dict = {}
    for d in sorted(folder.iterdir()):
        if not d.is_dir() or d.name.startswith(MV_NAME) or not (d / "results.csv").exists():
            continue
        if _status(d).upper() == "ERROR":
            continue
        predicted[d.name] = _keys(_read_csv(d / "results.csv"))

    metrics_f = {m: _metrics(p, causal_f, universe_f) for m, p in predicted.items()}
    # Unit of the paper: ordered pair of entities (the level of validation_summary.csv).
    predicted_e = {m: _entity(p) for m, p in predicted.items()}
    metrics_e = {m: _metrics(p, causal_e, universe_e) for m, p in predicted_e.items()}

    # ── Sanity: reproduce the numbers already in validation_summary ──────────
    csv_path, txt_path = folder / "validation_summary.csv", folder / "validation_summary.txt"
    old = pd.read_csv(csv_path)
    old = old[~old["model"].astype(str).str.startswith(MV_NAME)].set_index("model")
    txt_old = _read_txt_table(txt_path)
    for m, p in predicted.items():
        me = _metrics(_entity(p), causal_e, universe_e)
        if m not in old.index or any(int(old.loc[m, k]) != me[k] for k in ("TP", "FP", "FN")):
            mismatches.append(f"{folder.name}/{m}: entity level differs from validation_summary.csv")
        if m in txt_old and any(txt_old[m][k] != metrics_f[m][k] for k in ("TP", "FP", "FN")):
            mismatches.append(f"{folder.name}/{m}: property level differs from validation_summary.txt")

    soft_threshold = old["soft_threshold"].iloc[0] if "soft_threshold" in old.columns else 1
    soft_count = Counter(p for pairs in predicted.values() for p in _entity(pairs) - causal_e)
    tp_soft = {p for p, c in soft_count.items() if c >= float(soft_threshold) * len(predicted)}

    out_dir = folder / OUT_FOLDER
    out_dir.mkdir(exist_ok=True)

    taus = sorted({_tau_of(m)[1] for m in predicted},
                  key=lambda t: (TAU_ORDER.index(t) if t in TAU_ORDER else len(TAU_ORDER), t))
    rows, csv_rows, txt_rows, notes, report = [], [], [], [], []

    for tau in taus:
        voters = [m for m in predicted
                  if _tau_of(m)[1] == tau and not m.startswith(PATCH_PREFIX)]
        if not voters:
            continue
        patch_name = next((m for m in predicted
                           if _tau_of(m)[1] == tau and m.startswith(PATCH_PREFIX)), None)
        patch_pairs = predicted.get(patch_name, set())
        n_voters = len(voters)
        # property-level vote (only for the table of validation_summary.txt)
        votes = Counter(p for m in voters for p in predicted[m])
        majority = {p for p, c in votes.items() if _accepted(c, n_voters)}
        # entity-pair vote -- the one that is scored and compared
        votes_e = Counter(p for m in voters for p in predicted_e[m])
        maj_e = {p for p, c in votes_e.items() if _accepted(c, n_voters)}
        patch_pairs_e = _entity(patch_pairs)
        name = _mv_name(tau)

        # ── votes_entity_<tau>.csv ───────────────────────────────────────────
        ent_rows = []
        for p in sorted(universe_e | set(votes_e) | patch_pairs_e):
            in_gt, pred = p in causal_e, p in maj_e
            line = {"source_entity": p[0], "target_entity": p[1],
                    "causal_gt": in_gt, "in_gt_universe": p in universe_e}
            for m in voters:
                line[f"vote_{_tau_of(m)[0]}"] = int(p in predicted_e[m])
            line["n_votes"] = votes_e.get(p, 0)
            line["n_voters"] = n_voters
            line["vote_fraction"] = round(votes_e.get(p, 0) / n_voters, 4)
            line["majority"] = pred
            line[PATCH_PREFIX] = int(p in patch_pairs_e)
            line["outcome"] = ("TP" if pred and in_gt else "FP" if pred else
                               "FN" if in_gt else "TN")
            ent_rows.append(line)
        pd.DataFrame(ent_rows).to_csv(out_dir / f"votes_entity_{tau}.csv", index=False)

        # ── votes_<tau>.csv ──────────────────────────────────────────────────
        vote_rows = []
        for p in sorted(universe_f | set(votes) | patch_pairs):
            in_gt, pred = p in causal_f, p in majority
            line = dict(zip(KEY_COLS, p))
            line["causal_gt"] = in_gt
            line["in_gt_universe"] = p in universe_f
            for m in voters:
                line[f"vote_{_tau_of(m)[0]}"] = int(p in predicted[m])
            line["n_votes"] = votes.get(p, 0)
            line["n_voters"] = n_voters
            line["vote_fraction"] = round(votes.get(p, 0) / n_voters, 4)
            line["majority"] = pred
            line[PATCH_PREFIX] = int(p in patch_pairs)
            line["outcome"] = ("TP" if pred and in_gt else "FP" if pred else
                               "FN" if in_gt else "TN")
            vote_rows.append(line)
        df_votes = pd.DataFrame(vote_rows)
        df_votes.to_csv(out_dir / f"votes_{tau}.csv", index=False)

        # ── results_<tau>.csv ────────────────────────────────────────────────
        df_res = df_votes[df_votes["majority"]][KEY_COLS + ["n_votes", "n_voters",
                                                             "vote_fraction", "causal_gt"]].copy()
        df_res["voters"] = [", ".join(m for m in voters if tuple(r) in predicted[m])
                            for r in df_res[KEY_COLS].itertuples(index=False)]
        df_res["significant"] = True
        df_res["method"] = name
        df_res.to_csv(out_dir / f"results_{tau}.csv", index=False)

        # ── Metrics (property level = the table of the .txt; entity = the .csv) ─
        mf = _metrics(majority, causal_f, universe_f)
        me = _metrics(maj_e, causal_e, universe_e)
        raw_fp = maj_e - causal_e
        tp_s, fp_h = len(raw_fp & tp_soft), len(raw_fp - tp_soft)
        prec_s = _safe_div(me["TP"] + tp_s, me["TP"] + tp_s + fp_h)
        rec_s  = _safe_div(me["TP"] + tp_s, me["TP"] + tp_s + me["FN"])

        csv_rows.append({
            "model": name, "status": "ok",
            "precision": round(me["precision"], 4), "recall": round(me["recall"], 4),
            "f1": round(me["f1"], 4), "accuracy": round(me["accuracy"], 4),
            "fpr": round(me["fpr"], 4),
            "TP": me["TP"], "FP": me["FP"], "FN": me["FN"], "TN": me["TN"],
            "TP_hard": me["TP"], "TP_soft": tp_s, "FP_hard": fp_h,
            "FN_soft": me["FN"], "TN_soft": me["TN"],
            "precision_soft": round(prec_s, 4), "recall_soft": round(rec_s, 4),
            "f1_soft": round(_safe_div(2 * prec_s * rec_s, prec_s + rec_s), 4),
            "soft_threshold": soft_threshold,
            "n_predicted": me["n_predicted"],
            "n_tested": len(votes_e),                  # entity pairs with at least one vote
            "n_causal": len(causal_e), "n_universe": len(universe_e),
            "n_outside_universe": len(maj_e - universe_e),
            "duration": "N/A",
        })
        txt_rows.append((name, mf))
        min_votes = _min_votes(n_voters)
        notes.append(f"{TXT_NOTE_TAG} {name}: pairs predicted by {_rule_text(n_voters)} of the "
                     f"non-patch {tau} models ({', '.join(voters)}); this table is the "
                     f"property-level vote, validation_summary.csv the entity-pair vote; "
                     f"details in {OUT_FOLDER}/")

        # ── Comparison line for the run-level summary (entity-pair level) ────
        base_f1 = {_tau_of(m)[0]: metrics_e[m]["f1"] for m in voters}
        best = max(base_f1, key=base_f1.get)
        patch_f1 = metrics_e[patch_name]["f1"] if patch_name else float("nan")
        row = {
            "scenario": folder.name, "family": _family(folder.name), "tau": tau,
            "n_voters": n_voters, "min_votes": min_votes,
            "voters": ", ".join(_tau_of(m)[0] for m in voters),
            "n_causal": len(causal_e), "n_universe": len(universe_e),
            "n_voted_pairs": len(votes_e),
            "mv_n_predicted": me["n_predicted"],
            "mv_TP": me["TP"], "mv_FP": me["FP"], "mv_FN": me["FN"], "mv_TN": me["TN"],
            "mv_precision": round(me["precision"], 4), "mv_recall": round(me["recall"], 4),
            "mv_f1": round(me["f1"], 4), "mv_f1_property": round(mf["f1"], 4),
            "patch_f1": round(patch_f1, 4),
            "patch_f1_property": round(metrics_f[patch_name]["f1"], 4) if patch_name else float("nan"),
            "best_baseline": best, "best_baseline_f1": round(base_f1[best], 4),
            **{f"f1_{b}": round(v, 4) for b, v in base_f1.items()},
        }
        if patch_name:
            row["patch_vs_mv"] = _outcome(patch_f1, me["f1"])
            row["patch_vs_best_baseline"] = _outcome(patch_f1, base_f1[best])
            row["patch_vs_best_with_mv"] = _outcome(patch_f1, max(base_f1[best], me["f1"]))
        rows.append(row)

        # ── summary.txt block ────────────────────────────────────────────────
        report += [
            "-" * 72,
            f"  {name}  ({tau})",
            "-" * 72,
            f"  Voters ({n_voters})            : {', '.join(voters)}",
            f"  Rule                  : entity pair accepted by {_rule_text(n_voters)} of the voters",
            f"  Entity pairs with >= 1 vote : {len(votes_e)}",
            f"  Entity pairs accepted       : {len(maj_e)}",
            "",
            f"  {'MODEL':<22} {'PREC':>7} {'REC':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}",
            "  " + "-" * 68,
        ]
        for label, m in [(name, me)] + ([(patch_name, metrics_e[patch_name])] if patch_name else []) \
                + [(v, metrics_e[v]) for v in voters]:
            report.append(f"  {label:<22} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} "
                          f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}")
        if patch_name:
            report += ["",
                       f"  patch vs majority vote        : {row['patch_vs_mv']}",
                       f"  patch vs best baseline        : {row['patch_vs_best_baseline']}  "
                       f"({best}, F1={base_f1[best]:.4f})",
                       f"  patch vs best baseline + vote : {row['patch_vs_best_with_mv']}"]
        report.append("")

    if not rows:
        return []

    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    header = ["=" * 72, f"  MAJORITY VOTE -- {folder.name}",
              "  Entity-pair level, directional pairs (same metric as validation_summary.csv; "
              "the unit of the paper)",
              "=" * 72]
    (out_dir / "summary.txt").write_text("\n".join(header + report), encoding="utf-8")

    _update_summary_csv(csv_path, csv_rows)
    if txt_path.exists():
        _update_summary_txt(txt_path, txt_rows, notes)
    return rows


# =============================================================================
# RUN LEVEL
# =============================================================================

def _wtl(series: pd.Series) -> str:
    c = series.value_counts()
    return f"{c.get('W', 0)}W / {c.get('T', 0)}T / {c.get('L', 0)}L"


def _wtl_block(df: pd.DataFrame, title: str) -> list:
    lines = ["=" * 72, f"  {title}", "=" * 72]
    for tau, g in df.groupby("tau", sort=False):
        f1_cols = [c for c in g.columns if c.startswith("f1_") and g[c].notna().any()]
        patch = PATCH_PREFIX if tau == DEFAULT_TAU else f"{PATCH_PREFIX}_{tau}"
        lines += ["", f"  [{tau}]  {len(g)} scenario(s)", "",
                  f"  Patch ({patch}) W/T/L",
                  f"    vs best baseline (without the vote) : {_wtl(g['patch_vs_best_baseline'])}",
                  f"    vs majority vote alone              : {_wtl(g['patch_vs_mv'])}",
                  f"    vs best baseline + majority vote    : {_wtl(g['patch_vs_best_with_mv'])}",
                  "",
                  f"  {'MODEL':<22} {'MEAN F1':>8} {'N':>4}   PATCH W/T/L vs model",
                  "  " + "-" * 60]
        suffix = "" if tau == DEFAULT_TAU else f"_{tau}"
        table = [(patch, g["patch_f1"], None), (_mv_name(tau), g["mv_f1"], g["mv_f1"])]
        table += [(c[3:] + suffix, g[c], g[c]) for c in f1_cols]
        for name, f1, rival in sorted(table, key=lambda x: -x[1].mean()):
            if rival is None:
                wtl = ""
            else:
                ok = rival.notna()
                wtl = _wtl(pd.Series([_outcome(a, b) for a, b in
                                      zip(g.loc[ok, "patch_f1"], rival[ok])]))
            lines.append(f"  {name:<22} {f1.mean():>8.4f} {int(f1.notna().sum()):>4}   {wtl}")
    return lines + [""]


def run(run_dir: str):
    root = Path(run_dir)
    scenarios = sorted(d for d in root.iterdir()
                       if d.is_dir() and (d / "ground_truth.csv").exists()
                       and (d / "validation_summary.csv").exists())
    if not scenarios:
        print(f"[WARNING] No validated scenario found in '{root}'.")
        return

    rows, mismatches = [], []
    for folder in scenarios:
        rows += process_scenario(folder, mismatches)
    df = pd.DataFrame(rows)
    df["_t"] = df["tau"].map(lambda t: TAU_ORDER.index(t) if t in TAU_ORDER else len(TAU_ORDER))
    df = df.sort_values(["_t", "scenario"], kind="stable").drop(columns="_t").reset_index(drop=True)
    df.to_csv(root / "majority_vote_summary.csv", index=False)

    rule = "strict majority" if STRICT_MAJORITY else f">= {VOTE_THRESHOLD:.0%} of the voters"
    unit = f"entity-pair F1, directional; {rule}"
    keep = ~df["scenario"].apply(lambda s: any(x in s for x in SECONDARY_EXCLUDE))
    if SECONDARY_EXCLUDE and not keep.all():
        lines = _wtl_block(df[keep], f"MAJORITY VOTE -- {root.name}  ({unit})  "
                                     f"SCENARIOS OF THE PAPER (without: {', '.join(SECONDARY_EXCLUDE)})")
        lines += _wtl_block(df, f"SAME, EVERY SCENARIO OF THE RUN")
    else:
        lines = _wtl_block(df, f"MAJORITY VOTE -- {root.name}  ({unit})")

    lines += ["=" * 72, "  PATCH W/T/L BY FAMILY  (vs best baseline | vs vote | vs best + vote)", "=" * 72]
    for tau, g in df.groupby("tau", sort=False):
        lines += ["", f"  [{tau}]"]
        for fam, gf in g.groupby("family"):
            lines.append(f"  {fam:<40} {len(gf):>2}  {_wtl(gf['patch_vs_best_baseline']):<16} | "
                         f"{_wtl(gf['patch_vs_mv']):<16} | {_wtl(gf['patch_vs_best_with_mv'])}")
    lines += ["", f"  Sanity check against the existing validation summaries: "
                  f"{'OK' if not mismatches else str(len(mismatches)) + ' MISMATCH(ES)'}"]
    lines += [f"    {m}" for m in mismatches]

    text = "\n".join(lines)
    (root / "majority_vote_wtl.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\n  {len(scenarios)} scenario(s) processed -> {root / 'majority_vote_summary.csv'}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else RUN_DIR)
