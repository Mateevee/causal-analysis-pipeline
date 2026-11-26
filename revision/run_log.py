"""
Log of the executions of the revision experiments.

Every script in revision/ opens a `with execution(NAME):` block at its entry
point. The block writes to Results_Revisao/registro_execucoes.jsonl one line
when the execution starts and another when it ends (ok, error or interrupted), with:
  - the experiment, the script, the command and the arguments;
  - the type: 'completo' (complete), 'verificacao' (--verificar), 'piloto'
    (--piloto) or 'parcial' (--filtro, --n, --B different from the default);
  - the commit, the branch, and whether the repository had uncommitted changes;
  - a hash of the method's code (Pipeline/, ModelComparison/, Config/, Data/,
    Auxiliary/, main.py) and one of the script itself, to know exactly with which
    code each result was produced;
  - start and end time, duration, machine and PID;
  - on error, the last line of the exception.
The file only receives new lines; nothing is rewritten. The readable summary is
that of revision/status.py (not distributed).
"""
from __future__ import annotations

import contextlib
import datetime
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import uuid

REVISAO = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(REVISAO)
OUTPUT = os.path.join(ROOT, "Results_Revisao")
LOG_FILE = os.path.join(OUTPUT, "registro_execucoes.jsonl")
METHOD_CODE = ["Pipeline", "ModelComparison", "Config", "Data", "Auxiliary"]


def _git(*args) -> str:
    try:
        return subprocess.run(["git", "-c", "core.useBuiltinFSMonitor=false", *args], cwd=ROOT,
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def hash_files(files) -> str:
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(os.path.relpath(f, ROOT).replace("\\", "/").encode())
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def method_code_hash() -> str:
    files = [os.path.join(ROOT, "main.py")]
    for d in METHOD_CODE:
        files += glob.glob(os.path.join(ROOT, d, "**", "*.py"), recursive=True)
    return hash_files([f for f in files if os.path.isfile(f) and "__pycache__" not in f])


def run_type(argv) -> str:
    if "--verificar" in argv:
        return "verificacao"
    if "--piloto" in argv:
        return "piloto"
    if any(a in argv for a in ("--filtro", "--n")) or ("--B" in argv and argv[argv.index("--B") + 1:][:1] != ["20"]):
        return "parcial"
    return "completo"


def write_line(record: dict) -> None:
    os.makedirs(OUTPUT, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


@contextlib.contextmanager
def execution(experiment: str, script: str = None):
    script = os.path.abspath(script or sys.argv[0])
    argv = sys.argv[1:]
    ident = uuid.uuid4().hex[:12]
    base = {"id": ident, "experimento": experiment}
    write_line({**base, "evento": "inicio", "inicio": now(),
                "script": os.path.relpath(script, ROOT).replace("\\", "/"),
                "modo": argv[0] if argv and not argv[0].startswith("--") else "padrao",
                "argumentos": argv, "tipo": run_type(argv),
                "commit": _git("rev-parse", "--short", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                "mudancas_nao_commitadas": bool(_git("status", "--porcelain", "--", "Pipeline", "ModelComparison",
                                                     "Config", "Data", "Auxiliary", "main.py", "revision")),
                "hash_codigo_metodo": method_code_hash(), "hash_script": hash_files([script]),
                "maquina": f"{platform.node()} | {platform.platform()} | {os.cpu_count()} CPUs", "pid": os.getpid()})
    t0 = time.perf_counter()
    try:
        yield ident
    except KeyboardInterrupt:
        write_line({**base, "evento": "fim", "fim": now(), "status": "interrompido",
                    "duracao_s": round(time.perf_counter() - t0, 1)})
        raise
    except BaseException as e:
        write_line({**base, "evento": "fim", "fim": now(), "status": "erro",
                    "duracao_s": round(time.perf_counter() - t0, 1),
                    "erro": "".join(traceback.format_exception_only(type(e), e)).strip()[-500:]})
        raise
    else:
        write_line({**base, "evento": "fim", "fim": now(), "status": "ok",
                    "duracao_s": round(time.perf_counter() - t0, 1)})


def read_log() -> list:
    if not os.path.exists(LOG_FILE):
        return []
    return [json.loads(x) for x in open(LOG_FILE, encoding="utf-8") if x.strip()]
