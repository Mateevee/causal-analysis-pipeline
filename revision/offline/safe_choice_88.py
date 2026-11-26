"""
"Safe choice" (regret) with the N set of 88 scenarios, without changing the script with the frozen definition.

Origin: R1-2c now has 88 scenarios (Yeast2 and Yeast3 included; `revision/offline/r1_2c_evaluate_88.py`). The
definition of regret is the one in the docstring of `revision/offline/safe_choice.py` (hash recorded in NOTAS.md
before the first run; checked below). This wrapper reuses the same function `table_N`, changing only the folder from
which it reads `T1_por_cenario.csv` (from `r1_2c_benchmark_novo` to `r1_2c_benchmark_novo_88`), and writes the output
to `Results_Revisao/escolha_segura_88/`. The H set (39 held-out) does not change.

Usage:  python revision/offline/safe_choice_88.py
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
exec(source.replace(before, '"r1_2c_benchmark_novo_88", "T1_por_cenario.csv"'), m.__dict__, ns)
m.table_N = ns["table_N"]
m.NAME = "escolha_segura_88"

if __name__ == "__main__":
    from run_log import execution
    with execution("escolha_segura_88", __file__):
        m.main()
