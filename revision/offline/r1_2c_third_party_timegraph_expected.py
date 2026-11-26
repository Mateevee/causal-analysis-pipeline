"""
R1-2c, block (iii) -- the TimeGraph selection computed ONLY from the file listing of the pinned commit (GitHub API),
before any download, by the same rule as the readers (section 3.1 of the protocol). Writes
revision/r1_2c_third_party_timegraph_expected.json; on the runner, the selecionar mode must reproduce this list
(base, CSV and ground truth of each scenario, in the same order), otherwise it stops.

Usage (from the repository root):  python revision/offline/r1_2c_third_party_timegraph_expected.py <github_tree.json>
  <github_tree.json> = response of GET https://api.github.com/repos/hferdous/TimeGraph/git/trees/<commit>?recursive=1
"""
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(REV, "execution")]
import r1_2c_third_party_readers as L  # noqa: E402  (the constants and the draw are those of the readers)

tree = json.load(open(sys.argv[1], encoding="utf-8"))
assert not tree.get("truncated"), "truncated listing"
all_files = sorted(x["path"] for x in tree["tree"] if x["type"] == "blob" and x["path"].startswith("Datasets/"))
S = set(all_files)
bases = defaultdict(set)
for p in all_files:
    if not p.endswith(".csv"):
        continue
    b = re.sub(r"_(complete|missing)$", "", p[:-4])
    if b + "_structure.txt" in S or b + "_causal_structure.txt" in S:
        bases[p.split("/")[1]].add(b)
bases = {v: sorted(bases[v]) for v in L.VARIANTS}
assert {v: len(bases[v]) for v in L.VARIANTS} == L.EXPECTED_BASES
out = []
for k, v in enumerate(L.VARIANTS):
    bs = bases[v]
    for j, i in enumerate(L._draw(len(bs), L.N_TG, L.SEED + k)):
        b = bs[i]
        csv = b + "_missing.csv" if b + "_missing.csv" in S else b + ".csv"
        truth = b + "_structure.txt" if b + "_structure.txt" in S else b + "_causal_structure.txt"
        out.append({"name": f"timegraph_{v}_{j:02d}", "variante": v, "indice_na_variante": i, "base": b,
                    "csv": csv, "verdade": truth})
res = {"commit": "4d93403b002664fa55607f308ad0e80623472ded", "fonte": "GitHub API, recursive git/trees, 2026-11-22",
       "regra": "revision/PROTOCOL_third_party_benchmark_R1-2c.md, section 3.1", "timegraph": out}
target = os.path.join(REV, "r1_2c_third_party_timegraph_expected.json")
with open(target, "w", encoding="utf-8", newline="\r\n") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(len(out), "scenarios; written to", target)
from collections import Counter  # noqa: E402
print(Counter(x["variante"] for x in out))
print(Counter(re.search(r"_n(\d+)_", x["base"]).group(1) for x in out))
