# Protocol — coherence guard at scale (variant G1; R1-3, R2-4)

**Status: APPROVED and frozen** (authors' decision by delegation, 2026-11-09; see `revision/NOTAS.md`, "AUTHORS'
DECISIONS, by delegation"). Hash recorded in NOTAS.md and in `registro_execucoes.jsonl` (event `congelamento`).

## 1. The current guard

`Pipeline/coherence.py::assess_evidence_coherence`: Spearman's ρ between the transfer entropy and the DML |t|
over the candidates at the end of Stage 6, with a Fisher interval (one-sided, α = 0.05):
- **incoherent** if no candidate has TE evidence or if the **upper bound** of ρ < 0.3;
- **coherent** if the **lower bound** of ρ > 0;
- indeterminate in the other cases.

The code tests "incoherent" before "coherent". In the incoherent regime, the rejections of Stages 7-8a are not
applied and the decision reverts to that of Stage 6.

## 2. The defect

- **The two conditions can hold together.** This happens when the interval lies entirely between 0 and 0.3: the
  agreement is established, but weak. The code classifies this case as "incoherent".
- **This case becomes more frequent with the number of candidates.** With more candidates the interval narrows, and ρ is
  diluted among many weak candidates.
- **Evidence:** on the clean linear VAR with n = 160 entities (`r2_4b_escala`), the guard fired in 2 of 3 seeds
  (ρ = 0.26 [0.23; 0.29]). The F1 fell from ~0.82, without the guard, to 0.15.
- In the 90 scenarios of the paper, 3 fall into this overlap (`r1_3c_guarda_rho_decisao`).

## 3. Variant G1

When the two conditions hold at the same time (lower bound > 0 **and** upper bound < 0.3), the regime is
**coherent**. All other cases stay as they are. There is no new threshold and no new numerical parameter.

**Justification.**
- **What the guard is for.** The docstring of `coherence.py` says: the ranking-based pruning of Stages 7-8a is only better
  than a random cut if the two channels agree. Dempster-Shafer fusion should not combine sources in
  conflict (Zadeh 1984; Shafer 1976). What the guard needs to detect is the **absence of agreement**.
- **What counts as agreement.** An interval that excludes zero establishes positive agreement, however weak it
  may be. Cohen's (1988) value of 0.3 is a magnitude label ("medium"), not a criterion of absence.
- **Three-zone logic.** It is the same as in equivalence tests (Schuirmann 1987; Lakens 2017): "effect
  established" and "effect below a margin" are not mutually exclusive. When both hold, the decision follows the
  question one wants to answer — here, whether there is agreement.
- **It answers R1-3 and R2-4.** The decision no longer depends on the number of candidates through the path that caused the
  collapse.

## 4. Implementation

No default changes.
- `coherence.assess_evidence_coherence(..., overlap='incoherent')`. The default reproduces the current rule;
  `overlap='coherent'` applies G1.
- `run_pipeline(..., evidence_coherence_overlap='incoherent')` passes the parameter on.
- Model variant, built in `revision/execution/common_exec.model`:
  `patch_tau10_final` = `patch_tau10` with `time_axis_rule='block_portmanteau'` and
  `evidence_coherence_overlap='coherent'`. It combines the two approved corrections.

## 5. Criteria fixed before running

- **C1 (implementation).**
  - With the default, the recomputed regime is equal to the recorded one in the 105 scenarios (90 of the paper and 15 of R2-4b).
  - With `overlap='coherent'`, it is equal to the G1 of `r1_3c_guarda_rho_decisao` in the same 105.
- **C2 (re-execution).** `patch_tau10` and `patch_tau10_final`, with the same code and in the same execution, on the 3 scenarios
  of the paper in which G1 changes the decision (`causaltime_medical_sample0`, `rcaeval_RE1-OB_checkoutservice_disk_1`,
  `petshop_temporal_traffic1`) and on seeds 6101 and 6103 of R2-4b with n = 160.
  - In these 5 scenarios the time-axis decision is the same under both rules (none is among the 7), so the
    difference isolates G1.
  - In the other 85 scenarios of the paper, G1 does not change the guard decision (C1).
- **C3 (reporting, whatever the result).**
  - F1, TP, FP and FN of each version in each scenario, and the mean over the 90 (approximation: the paper's F1 for the unchanged ones).
  - Expected, not assumed: at n = 160 the F1 returns close to what was obtained without the guard (~0.8).
  - In the 3 scenarios of the paper, the F1 may rise or fall; the result goes into the text either way.
- **C4 (text).** Describe the corrected guard (III, coherence guard), the collapse at scale and the correction, and the
  before/after numbers. Declare that the correction came from the revision's scale analysis, after the held-out ones had
  already been seen.

## Log
- Draft and approval: 2026-11-09, after `r1_3c_guarda_rho_decisao`, by delegation of the authors.
- Freeze: hash in NOTAS.md and in the jsonl.
