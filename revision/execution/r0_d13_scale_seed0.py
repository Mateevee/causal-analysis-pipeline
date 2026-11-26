"""
D13 — the scaling study of R2-4b (15 scenarios: n = 10, 20, 40, 80 and 160 entities × seeds 6101 to 6103) rerun with
PYTHONHASHSEED = 0, so that the scaling table comes out with the numbers of the reproduction package (decision D15).

Decision of the author on 2026-11-21 (D13: "Yes, together with point 1").

Design (fixed before running):
  - The SAME design and the SAME code as revision/execution/r2_4b_scale.py (generator, seeds, T = 300, patch_tau10,
    per-stage timer only during the script), imported unchanged; only the output folder
    (Results_Revisao/r0_d13_escala_semente0/) and the requirement of PYTHONHASHSEED = 0 change. Frozen method code
    (method_code_hash d0473069ebef1978).
  - The F1 and the number of candidates are valid. THE TIMES ARE NOT VALID AS A MEASUREMENT: the stage runs in
    parallel fronts. The scaling times remain those of the run that measured time.
  - avaliar: the report of r2_4b_escala (means per n) in the new folder. The comparison with the earlier runs without
    a fixed seed is done offline, on the main machine.

Usage (from the repository root, with PYTHONHASHSEED=0):
  python revision/execution/r0_d13_scale_seed0.py rodar | avaliar
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]

import r2_4b_scale as E  # noqa: E402

NAME = "r0_d13_escala_semente0"
FROZEN_HASH = "d0473069ebef1978"
E.NAME = NAME   # only the output folder changes


def _check():
    import run_log
    h = run_log.method_code_hash()
    if h != FROZEN_HASH:
        print(f"WARNING: method_code_hash {h} != {FROZEN_HASH}, the hash of the frozen method code (the released files are English translations of the frozen ones; see revision/README.md)", file=sys.stderr)
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit(f"PYTHONHASHSEED = {os.environ.get('PYTHONHASHSEED')!r}; this script requires 0 (the queue sets it).")


if __name__ == "__main__":
    from run_log import execution
    mode = sys.argv[1] if len(sys.argv) > 1 else "avaliar"
    with execution(NAME, __file__):
        if mode == "rodar":
            _check()
            sys.argv = [sys.argv[0], "rodar"]   # every n of r2_4b_scale.NS, without baselines
            E.run()
        else:
            E.evaluate()
