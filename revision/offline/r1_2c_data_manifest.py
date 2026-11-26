"""
R1-2c -- manifest of the TEST data (SHA-256 of each file that the readers may open).

Used to check, on the other machine, that the R1-2c data are the same as on this one, before running.
It does not open the files as data: it only reads bytes for the hash.

Usage (from the repository root)
  python revision/offline/r1_2c_data_manifest.py gerar      # writes revision/r1_2c_data_manifest.json
  python revision/offline/r1_2c_data_manifest.py conferir   # compares the disk with the manifest; exit code 1 if different
"""
import hashlib
import json
import os
import sys

FOLDERS = ["external_data/causal_chamber/wt_walks_v1", "external_data/causal_chamber/lt_walks_v1",
           "external_data/dream3", "external_data/dream4"]
FILES = ["external_data/causal_rivers/rivers_east_germany.p",
         "external_data/causal_rivers/rivers_ts_east_germany.csv"]
OUTPUT = "revision/r1_2c_data_manifest.json"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def list_files():
    out = list(FILES)
    for d in FOLDERS:
        for root, _, names in os.walk(d):
            out += [os.path.join(root, a).replace("\\", "/") for a in names]
    return sorted(out)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "conferir"
    if mode == "gerar":
        m = {p: {"bytes": os.path.getsize(p), "sha256": sha(p)} for p in list_files()}
        with open(OUTPUT, "w", encoding="utf-8", newline="\r\n") as f:
            json.dump(m, f, indent=1, sort_keys=True)
        print(f"{len(m)} files in {OUTPUT}")
        return
    m = json.load(open(OUTPUT, encoding="utf-8"))
    missing, different = [], []
    for p, v in m.items():
        if not os.path.exists(p):
            missing.append(p)
        elif os.path.getsize(p) != v["bytes"] or sha(p) != v["sha256"]:
            different.append(p)
    extra = sorted(set(list_files()) - set(m))
    print(f"manifest: {len(m)} files; missing {len(missing)}; different {len(different)}; extra on disk {len(extra)}")
    for p in missing[:50]:
        print("  MISSING  ", p)
    for p in different[:50]:
        print("  DIFFERENT", p)
    for p in extra[:20]:
        print("  extra (not counted)", p)
    sys.exit(1 if missing or different else 0)


if __name__ == "__main__":
    main()
