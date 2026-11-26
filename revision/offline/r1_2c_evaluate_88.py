"""
R1-2c -- evaluate with the 88 scenarios (Yeast2 and Yeast3 included), without changing the frozen script.

Origin
  INSTRUCOES_2026-11-21b and NOTAS.md (entry "FILA DE 2026-11-21b"): "Back on the main machine: redo the R1-2c
  avaliar with the 88 (remove Yeast2 and Yeast3 from DECLARED_EXCLUSIONS, with a log entry)". Inclusion rule fixed
  beforehand: the two scenarios enter with whatever was produced, including a time cutoff.

What this wrapper does
  - imports `revision/execution/r1_2c_new_benchmark.py` (frozen; hash checked below) without changing the file;
  - empties DECLARED_EXCLUSIONS in memory only;
  - writes the output to `Results_Revisao/r1_2c_benchmark_novo_88/`, a new folder, so as not to overwrite the
    official 86-scenario report in `Results_Revisao/r1_2c_benchmark_novo/`;
  - roots, in order of precedence per scenario: (1) the `run/` of the main machine (the 9 official scenarios of the
    interim look); (2) the `run/` of the 2026-11-22 run (Yeast2 and Yeast3); (3) the `run/` of the single run on
    the runner of 2026-11-18 (the other 77; in it Yeast2 and Yeast3 only have the skip markers of 11/17, which is why
    it comes after (2)).
  - Time cutoffs: as in the protocol (section 7), a run without a result is left out of that method's mean.

Usage:  python revision/offline/r1_2c_evaluate_88.py
"""
from __future__ import annotations

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
sys.path[:0] = [REV, os.path.join(REV, "execution")]

SCRIPT = os.path.join(REV, "execution", "r1_2c_new_benchmark.py")
FROZEN_HASH = "965c2c828173e3642b091bce9be5765ebee403eca154c72125c704fa580af648"   # disk, CRLF
h = hashlib.sha256(open(SCRIPT, "rb").read()).hexdigest()
if h != FROZEN_HASH:
    print(f"WARNING: r1_2c_new_benchmark.py has SHA-256 {h}, not that of the frozen script (the released files are English translations of the frozen ones; see revision/README.md)", file=sys.stderr)

RES = os.path.join(os.path.dirname(REV), "Results_Revisao")
ROOTS = [
    os.path.join(RES, "r1_2c_benchmark_novo", "run"),
    os.path.join(RES, "_outra_maquina", "execucao_2026-11-22", "execucao_2026-11-22", "r1_2c_benchmark_novo", "run"),
    os.path.join(RES, "_outra_maquina", "r1_2c_completo_2026-11-18", "r1_2c_benchmark_novo", "r1_2c_benchmark_novo",
                 "run"),
]
for r in ROOTS:
    if not os.path.isdir(r):
        raise SystemExit(f"missing root: {r}")

sys.argv = [SCRIPT, "avaliar", "--raizes", *ROOTS]
import r1_2c_new_benchmark as m  # noqa: E402

m.DECLARED_EXCLUSIONS = {}
m.NAME = "r1_2c_benchmark_novo_88"
os.makedirs(os.path.join(m.ce.OUTPUT_ROOT, m.NAME), exist_ok=True)

if __name__ == "__main__":
    from run_log import execution
    with execution("r1_2c_avaliar_88", __file__):
        m.evaluate()
