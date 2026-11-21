import argparse
import collections

from _common import ROOT

from router.data import load_swebench_rows, save_jsonl, stratified_split

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=13)
p.add_argument("--fractions", type=float, nargs=3, default=(0.4, 0.2, 0.4), help="train dev test")
args = p.parse_args()

rows = load_swebench_rows()
print("rows:", len(rows))
print("raw difficulty counts:", dict(collections.Counter(r["difficulty_raw"] for r in rows)))
print("mapped label counts:  ", dict(collections.Counter(r["label"] for r in rows)))

splits = stratified_split(rows, tuple(args.fractions), args.seed)
for name, part in splits.items():
    save_jsonl(part, ROOT / "data" / f"{name}.jsonl")
    print(name, len(part), dict(collections.Counter(r["label"] for r in part)))
