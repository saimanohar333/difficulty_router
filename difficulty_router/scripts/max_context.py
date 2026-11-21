import argparse
import collections

import numpy as np
from _common import ROOT, add_model_args

p = argparse.ArgumentParser()
add_model_args(p)
p.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
p.add_argument("--thresholds", type=int, nargs="+", default=[6000, 12000], help="report how many items exceed each")
p.add_argument("--top", type=int, default=10, help="list the N longest items")
args = p.parse_args()

from router.data import load_jsonl, row_to_request
from router.pipeline import DifficultyRouter
from router.tokens import HFTokenCounter

router = DifficultyRouter(None, HFTokenCounter(args.model), max_prompt_tokens=10**9)

records = []
for split in args.splits:
    for row in load_jsonl(ROOT / "data" / f"{split}.jsonl"):
        state, truncated, total = router.build_state(row_to_request(row))
        assert not truncated
        records.append((total, row["instance_id"], row["label"], split))

tokens = np.array([r[0] for r in records])
print(f"items: {len(tokens)}   overhead (rubric + AnyJev wrapper): {router.template_overhead('coding_bugfix')} tokens")
print(f"max: {tokens.max()}   mean: {tokens.mean():.0f}   median: {np.median(tokens):.0f}   "
      f"p90: {np.percentile(tokens, 90):.0f}   p95: {np.percentile(tokens, 95):.0f}   p99: {np.percentile(tokens, 99):.0f}")

print("\nitems over each threshold (these would be truncated by --max-prompt-tokens):")
for th in sorted(args.thresholds):
    over = [r for r in records if r[0] > th]
    by_label = dict(collections.Counter(r[2] for r in over))
    print(f"  > {th:>6}: {len(over):>3} / {len(records)}  ({100 * len(over) / len(records):.1f}%)   by label: {by_label}")

print("\nby label (mean / max tokens):")
for label in ("Easy", "Medium", "Hard"):
    t = np.array([r[0] for r in records if r[2] == label])
    if len(t):
        print(f"  {label:<7} n={len(t):<4} mean={t.mean():.0f}  max={t.max()}")

print(f"\n{args.top} longest items:")
for total, iid, label, split in sorted(records, reverse=True)[: args.top]:
    print(f"  {total:>6}  {label:<6} {split:<5} {iid}")

print("\nRESULT:", "all prompts fit under the largest threshold"
      if tokens.max() <= max(args.thresholds) else f"some prompts exceed {max(args.thresholds)} tokens (see lists above)")
