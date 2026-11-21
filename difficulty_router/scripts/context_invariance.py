import argparse
import json
import random

import numpy as np
from _common import ROOT, add_model_args

p = argparse.ArgumentParser()
add_model_args(p)
p.add_argument("--split", default="test")
p.add_argument("--level", default="L0", choices=["raw", "L0", "L1"])
p.add_argument("--artifacts", default=None)
p.add_argument("--pad-tokens", type=int, nargs="+", default=[0, 1000, 4000])
p.add_argument("--limit", type=int, default=100)
p.add_argument("--seed", type=int, default=7)
args = p.parse_args()

from router.data import load_jsonl, row_to_request
from router.engine import DecisionEngine
from router.pipeline import DifficultyRouter
from router.tokens import HFTokenCounter

rows = load_jsonl(ROOT / "data" / f"{args.split}.jsonl")[: args.limit]
pool = [r for r in load_jsonl(ROOT / "data" / "train.jsonl") if r["hints_text"].strip()]
rng = random.Random(args.seed)

counter = HFTokenCounter(args.model)
engine = DecisionEngine(args.model, level=args.level, dtype=args.dtype, device=args.device,
                        batch_size=args.batch_size, prior=args.prior, artifacts_path=args.artifacts)
if args.level == "L1":
    engine.question("coding_bugfix")
    print("L1 artifacts loaded:", engine.load_artifacts())
router = DifficultyRouter(engine, counter, args.max_prompt_tokens + max(args.pad_tokens))


def make_padding(n_tokens: int, own_id: str) -> str:
    if n_tokens <= 0:
        return ""
    parts, total = [], 0
    while total < n_tokens:
        r = rng.choice(pool)
        if r["instance_id"] == own_id:
            continue
        parts.append(f"[unrelated discussion from {r['repo']}]\n{r['hints_text'].strip()}")
        total += counter.count(parts[-1])
    ids = counter.encode("\n\n".join(parts))[:n_tokens]
    return counter.decode(ids)


results = {pad: [] for pad in args.pad_tokens}
for row in rows:
    for pad in args.pad_tokens:
        req = row_to_request(row, level=args.level, extra_comments=make_padding(pad, row["instance_id"]))
        results[pad].append(router.route(req))

base_pad = args.pad_tokens[0]
base = results[base_pad]
report = {}
for pad in args.pad_tokens:
    cur = results[pad]
    report[pad] = {
        "accuracy": float(np.mean([c["raw_label"] == r["label"] for c, r in zip(cur, rows)])),
        "label_stability_vs_base": float(np.mean([c["raw_label"] == b["raw_label"] for c, b in zip(cur, base)])),
        "mean_abs_delta_p_hard": float(np.mean([abs(c["probabilities"]["Hard"] - b["probabilities"]["Hard"]) for c, b in zip(cur, base)])),
        "mean_p_hard": float(np.mean([c["probabilities"]["Hard"] for c in cur])),
        "mean_prompt_tokens": float(np.mean([c["prompt_tokens"] for c in cur])),
        "mean_latency_ms": float(np.mean([c["latency_ms"] for c in cur])),
    }
print(json.dumps(report, indent=2))
(ROOT / "runs" / f"context_invariance_{args.model.split('/')[-1]}_{args.level}.json").write_text(json.dumps(report, indent=2))
