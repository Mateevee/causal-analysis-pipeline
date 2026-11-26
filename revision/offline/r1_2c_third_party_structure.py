"""
R1-2c, block (iii) -- checked download, safe extraction and STRUCTURE reading of the coupled TimeGraph and
CausalDynamics, for annex C (revision/PROTOCOL_third_party_benchmark_R1-2c.md, section 6, steps 2 and 3).

It never prints a series value and never runs a method. It prints only: names, counts, shapes, types, attributes, the
ground truth (which is not a result) and the header of the CSVs.

Modes (from the repository root):
  baixar      downloads the two files from the sources pinned in the protocol (skips what already exists), checks the
              SHA-256 of CausalDynamics against the official one and records that of TimeGraph. Exits with code 1 if
              CausalDynamics does not match (in that case the file is renamed to *.HASH_ERRADO and nothing is extracted).
  extrair     inspects the listing of each archive (no absolute path, no '..', link or unexpected type) and only
              then extracts: the TimeGraph zip to external_data/timegraph/extraido/ and the CausalDynamics tar to
              external_data/causaldynamics/extraido_coupled/ (tarfile, filter='data'). No script of the authors runs.
  estrutura   the structure report for annex C, in Results_Revisao/r1_2c_benchmark_terceiros/estrutura.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
EXT = ROOT / "external_data"
OUTPUT = ROOT / "Results_Revisao" / "r1_2c_benchmark_terceiros"

TG_COMMIT = "4d93403b002664fa55607f308ad0e80623472ded"
TG_URL = f"https://codeload.github.com/hferdous/TimeGraph/zip/{TG_COMMIT}"
TG_ZIP = EXT / "timegraph" / "downloads" / f"TimeGraph-{TG_COMMIT[:8]}.zip"
TG_DIR = EXT / "timegraph" / "extraido"
CD_REV = "e42a6eebb4736f982be5332dbba91670b21e861e"
CD_URL = f"https://huggingface.co/datasets/kausable/CausalDynamics/resolve/{CD_REV}/inputs/coupled.tar.gz"
CD_TAR = EXT / "causaldynamics" / "downloads" / "inputs_coupled.tar.gz"
CD_SHA = "92e6a64b34a9908c2c9af5d8b5b753eb057afb7463e4650771ea21c90569f6c0"
CD_SIZE = 6514628417
CD_DIR = EXT / "causaldynamics" / "extraido_coupled"
TG_VARIANTS = ("B2", "C2C", "D2C", "D3", "D3C")
EXPECTED_BASES = {"B2": 36, "C2C": 72, "D2C": 216, "D3": 432, "D3C": 432}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------- baixar
def _curl(url, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".parcial")
    r = subprocess.run(["curl", "-sSL", "--fail", "-A", "research-client", "-C", "-", "-o", str(tmp), url])
    if r.returncode != 0:
        raise SystemExit(f"download failed ({r.returncode}): {url}; the partial file was left in {tmp} and the command resumes")
    tmp.rename(target)


def download():
    out = {}
    if not TG_ZIP.exists():
        print(f"downloading {TG_URL}", flush=True)
        _curl(TG_URL, TG_ZIP)
    out["timegraph_zip_sha256"] = sha256(TG_ZIP)
    out["timegraph_zip_bytes"] = TG_ZIP.stat().st_size
    if not CD_TAR.exists():
        print(f"downloading {CD_URL} (6.5 GB)", flush=True)
        _curl(CD_URL, CD_TAR)
    out["causaldynamics_tar_bytes"] = CD_TAR.stat().st_size
    out["causaldynamics_tar_sha256"] = sha256(CD_TAR)
    out["causaldynamics_confere"] = (out["causaldynamics_tar_sha256"] == CD_SHA and
                                     out["causaldynamics_tar_bytes"] == CD_SIZE)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "download.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    if not out["causaldynamics_confere"]:
        CD_TAR.rename(CD_TAR.with_suffix(".gz.HASH_ERRADO"))
        raise SystemExit("SHA-256 or size of CausalDynamics does NOT match the official one; file renamed; stop.")


# --------------------------------------------------------------------------------------------- extrair
def _bad_path(name: str) -> bool:
    return name.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", name) is not None or ".." in Path(name).parts


def extract():
    # TimeGraph
    with zipfile.ZipFile(TG_ZIP) as z:
        names = z.namelist()
        bad = [n for n in names if _bad_path(n)]
        ext = Counter(Path(n).suffix.lower() for n in names if not n.endswith("/"))
        print("TimeGraph:", len(names), "entries; extensions", dict(ext))
        if bad:
            raise SystemExit(f"TimeGraph: suspicious paths {bad[:5]}; do not extract")
        target = TG_DIR.resolve()
        for n in names:
            dest = (target / n).resolve()
            if target not in dest.parents and dest != target:
                raise SystemExit(f"TimeGraph: {n} would leave {target}; do not extract")
        if not TG_DIR.exists():
            z.extractall(TG_DIR)
    # CausalDynamics
    with tarfile.open(CD_TAR, "r:gz") as t:
        members = t.getmembers()
        bad = [m.name for m in members if _bad_path(m.name) or m.issym() or m.islnk() or m.isdev()]
        ext = Counter(Path(m.name).suffix.lower() for m in members if m.isfile())
        print("CausalDynamics:", len(members), "members; extensions", dict(ext))
        if bad:
            raise SystemExit(f"CausalDynamics: suspicious members {bad[:5]}; do not extract")
        if set(ext) - {".nc", ".json", ".yaml", ".yml", ".txt", ".csv", ".md"}:
            raise SystemExit(f"CausalDynamics: unexpected extensions {set(ext)}; stop and note it before extracting")
        if not CD_DIR.exists():
            t.extractall(CD_DIR, filter="data")
    print("extraction done; no script of the authors was executed")


# --------------------------------------------------------------------------------------------- estrutura
def _tg_root() -> Path:
    cands = list(TG_DIR.glob("TimeGraph-*/Datasets"))
    if len(cands) != 1:
        raise SystemExit(f"expected one TimeGraph-*/Datasets folder in {TG_DIR}; found {cands}")
    return cands[0].parent


def _h5_structure(p: Path, lines: list):
    import h5py

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            lines.append(f"  - dataset `{name}`: shape {obj.shape}, type {obj.dtype}, attributes "
                         f"{ {k: _short(v) for k, v in obj.attrs.items()} }")
            if "adjac" in name.lower() and obj.ndim >= 2 and obj.size <= 10_000:
                a = obj[()]
                import numpy as np
                a2 = a.reshape((-1,) + a.shape[-2:]) if a.ndim > 2 else a[None]
                for i, m in enumerate(a2[:3]):
                    diag = int(np.trace(m != 0))
                    off = int((m != 0).sum() - diag)
                    lines.append(f"      matrix {i}: {m.shape[0]}x{m.shape[1]}, nonzero diagonal {diag}, "
                                 f"nonzero off the diagonal {off}, distinct values {sorted(set(m.ravel().tolist()))[:6]}")
        else:
            lines.append(f"  - group `{name}`: attributes { {k: _short(v) for k, v in obj.attrs.items()} }")

    with h5py.File(p, "r") as f:
        lines.append(f"  - root attributes: { {k: _short(v) for k, v in f.attrs.items()} }")
        f.visititems(visit)


def _short(v):
    s = v.decode() if isinstance(v, bytes) else str(v)
    return s if len(s) <= 120 else s[:117] + "..."


def structure():
    L = ["# Structure of the block (iii) data — for annex C", "",
         "Generated by `revision/offline/r1_2c_third_party_structure.py estrutura`. Only structure, counts and ground truth; "
         "no series value; no method was run.", ""]
    dl = OUTPUT / "download.json"
    if dl.exists():
        L += ["## Download", "", "```", dl.read_text(encoding="utf-8"), "```", ""]
    # ---------------- TimeGraph
    root = _tg_root()
    L += ["## TimeGraph", "", f"Root: `{root.relative_to(ROOT)}`", ""]
    readme = (root / "README.md").read_text(encoding="utf-8", errors="replace")
    prov = [ln.strip() for ln in readme.splitlines() if re.search(r"arxiv|kdd|doi|paper|citation", ln, re.I)]
    L += ["**Provenance (README lines that point to the paper):**", ""] + [f"- {x}" for x in prov[:10]] + [""]
    L += [f"- Licenses present: {[p.name for p in root.glob('LICENSE*')]}", ""]
    all_files = sorted(str(p.relative_to(root)).replace("\\", "/") for p in (root / "Datasets").rglob("*") if p.is_file())
    S = set(all_files)
    bases = defaultdict(list)
    for p in all_files:
        if not p.endswith(".csv"):
            continue
        b = re.sub(r"_(complete|missing)$", "", p[:-4])
        if b + "_structure.txt" in S or b + "_causal_structure.txt" in S:
            if b not in bases[p.split("/")[1]]:
                bases[p.split("/")[1]].append(b)
    count = {v: len(bases[v]) for v in TG_VARIANTS}
    L += [f"**Bases with a machine-readable ground truth:** {count} (expected {EXPECTED_BASES}; "
          f"{'MATCHES' if count == EXPECTED_BASES else 'DOES NOT MATCH — stop and note it'})", ""]
    for v in TG_VARIANTS:
        b = sorted(bases[v])[0]
        truth = b + "_structure.txt" if b + "_structure.txt" in S else b + "_causal_structure.txt"
        csvp = b + "_missing.csv" if b + "_missing.csv" in S else b + ".csv"
        with open(root / csvp, encoding="utf-8", errors="replace") as f:
            header = f.readline().rstrip("\n")
            n_lines = sum(1 for _ in f)
        L += [f"### {v} — first base in order: `{b}`", "",
              f"- CSV `{csvp}`: header `{header}`; {n_lines} data lines", "",
              f"- Ground truth `{truth}` (whole content):", "", "```",
              (root / truth).read_text(encoding="utf-8", errors="replace").strip()[:4000], "```", ""]
        stats = b + "_statistics.txt"
        if stats in S:
            L += [f"- `{stats}` exists (it is not ground truth; not used).", ""]
    # ---------------- CausalDynamics
    L += ["## CausalDynamics, coupled", "", f"Root: `{CD_DIR.relative_to(ROOT)}`", ""]
    ncs = sorted(str(p.relative_to(CD_DIR)).replace("\\", "/") for p in CD_DIR.rglob("*.nc"))
    others = sorted(str(p.relative_to(CD_DIR)).replace("\\", "/") for p in CD_DIR.rglob("*")
                    if p.is_file() and p.suffix != ".nc")
    L += [f"- {len(ncs)} .nc files; other files: {others[:20]}", ""]
    nodes = Counter()
    for n in ncs:
        m = re.search(r"(?:systems|nodes|N)[=_]?(\d+)", n)
        nodes[m.group(1) if m else "?"] += 1
    L += [f"- Count by number of nodes read from the path (pattern `systems=`/`nodes=`/`N`): {dict(nodes)}", "",
          "- First 15 paths:", ""] + [f"  - `{n}`" for n in ncs[:15]] + [""]
    folders = Counter(str(Path(n).parent) for n in ncs)
    L += [f"- {len(folders)} folders; the first 10 with the count: {dict(list(sorted(folders.items()))[:10])}", ""]
    for n in ncs[:3]:
        L += [f"### Structure of `{n}`", ""]
        _h5_structure(CD_DIR / n, L)
        L.append("")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "estrutura.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("baixar", "extrair", "estrutura"):
        raise SystemExit(__doc__)
    {"baixar": download, "extrair": extract, "estrutura": structure}[mode]()
