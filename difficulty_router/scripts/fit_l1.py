import argparse
import json
from collections import Counter

from _common import ROOT, add_model_args

p = argparse.ArgumentParser()
add_model_args(p)
p.add_argument("--out", default="artifacts/l1.json")
p.add_argument("--limit", type=int, default=0)
args = p.parse_args()

from router.calib import calibrate_with_backoff
from router.data import load_jsonl, row_to_request
from router.engine import DecisionEngine
from router.pipeline import DifficultyRouter, _free_cuda
from router.tokens import HFTokenCounter

rows = load_jsonl(ROOT / "data" / "train.jsonl")
if args.limit:
    rows = rows[: args.limit]

engine = DecisionEngine(args.model, dtype=args.dtype, device=args.device,
                        batch_size=args.batch_size, prior=args.prior)
router = DifficultyRouter(engine, HFTokenCounter(args.model), args.max_prompt_tokens)

states, golds, toks = [], [], []
for row in rows:
    state, _, ntok = router.build_state(row_to_request(row))
    states.append(state)
    golds.append(row["label"])
    toks.append(ntok)

print(f"Fitting L1 on {len(states)} labelled states (max prompt tokens {args.max_prompt_tokens}) ...")
art, used = calibrate_with_backoff(engine, states, golds, toks, "coding_bugfix", free_fn=_free_cuda)

out = ROOT / args.out
out.parent.mkdir(exist_ok=True)
engine.save_artifacts(str(out))
meta = {
    "model": args.model, "prior": args.prior, "max_prompt_tokens": args.max_prompt_tokens,
    "n_train": len(states), "n_used": len(used),
    "classes_used": dict(Counter(golds[i] for i in used)),
    "dropped_ids": [rows[i]["instance_id"] for i in range(len(rows)) if i not in set(used)],
}
(out.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2))
print("artifact keys:", list(art.keys()) if isinstance(art, dict) else type(art))
print(f"used {len(used)}/{len(states)} items; classes {meta['classes_used']}")
print("saved:", out)
