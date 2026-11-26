"""
"Safe choice" (regret) on block (iii) of R1-2c (TimeGraph and CausalDynamics, 167 scenarios), without changing the
script with the frozen definition. POST HOC ANALYSIS, like the one on the 88.

Origin: block (iii) came back from the runner on 2026-11-24 (`Results_Revisao/r1_2c_benchmark_terceiros/`). The
definition of regret is the one in the docstring of `revision/offline/safe_choice.py` (hash checked below), the same
used on the 88 (`safe_choice_88.py`). This wrapper reuses the same function `table_N`, changing only the folder from
which it reads `T1_por_cenario.csv` (from `r1_2c_benchmark_novo` to `r1_2c_benchmark_terceiros`), and writes the
output to `Results_Revisao/escolha_segura_terceiros/`. The six methods are those of the definition (the pipeline is
`patch_tau10`; the corrected variant is not included). The H set (39 held-out) does not change. Runs without a result
(12 of VARLiNGAM, 1 of PCMCI+) follow versions A and B of the definition.

Usage:  python revision/offline/safe_choice_third_party.py
"""
from __future__ import annotations

import hashlib
import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
sys.path[:0] = [HERE, REV, os.path.join(REV, "execution")]

SCRIPT = os.path.join(HERE, "safe_choice.py")
FROZEN_HASH = "e6217c1b008c32eb0ad137f8d664e73e68d9af84a053568aad61ef60eecc8e1c"   # on disk, CRLF
h = hashlib.sha256(open(SCRIPT, "rb").read()).hexdigest()
if h != FROZEN_HASH:
    print(f"WARNING: safe_choice.py has SHA-256 {h}, not that of the frozen script (the released files are English translations of the frozen ones; see revision/README.md)", file=sys.stderr)

import safe_choice as m  # noqa: E402

source = inspect.getsource(m.table_N)
before = '"r1_2c_benchmark_novo", "T1_por_cenario.csv"'
assert source.count(before) == 1, "the T1 read changed shape"
ns = {}
exec(source.replace(before, '"r1_2c_benchmark_terceiros", "T1_por_cenario.csv"'), m.__dict__, ns)
m.table_N = ns["table_N"]
m.NAME = "escolha_segura_terceiros"

if __name__ == "__main__":
    from run_log import execution
    with execution("escolha_segura_terceiros", __file__):
        m.main()
