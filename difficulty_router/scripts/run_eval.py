import argparse
import json
import time

from _common import ROOT, add_model_args

p = argparse.ArgumentParser()
add_model_args(p)
p.add_argument("--split", default="dev", choices=["train", "dev", "test"])
p.add_argument("--level", default="L0", choices=["raw", "L0", "L1"])
p.add_argument("--tau", type=float, default=0.6, help="escalate one tier if confidence < tau (meaningful at L1)")
p.add_argument("--artifacts", default=None, help="L1 artifact file from scripts/fit_l1.py")
p.add_argument("--flip", action="store_true", help="also run reversed option order; report flip rate (raw/L0 only)")
p.add_argument("--limit", type=int, default=0)
p.add_argument("--resume", action="store_true", help="continue a crashed/disconnected run from its .jsonl log")
p.add_argument("--out", default=None)
args = p.parse_args()

if args.level == "L1" and not args.artifacts:
    raise SystemExit("--level L1 needs --artifacts (run scripts/fit_l1.py first)")
if args.flip and args.level == "L1":
    raise SystemExit("--flip is for raw/L0: an L1 artifact is bound to one option order")

from router.data import load_jsonl, row_to_request
from router.engine import DecisionEngine
from router.metrics import summarize
from router.pipeline import DifficultyRouter
from router.tokens import HFTokenCounter

rows = load_jsonl(ROOT / "data" / f"{args.split}.jsonl")
if args.limit:
    rows = rows[: args.limit]

engine = DecisionEngine(args.model, level=args.level, dtype=args.dtype, device=args.device,
                        batch_size=args.batch_size, prior=args.prior, artifacts_path=args.artifacts)
if args.level == "L1":
    engine.question("coding_bugfix")
    print("L1 artifacts loaded:", engine.load_artifacts())
router = DifficultyRouter(engine, HFTokenCounter(args.model), args.max_prompt_tokens)

tag = args.out or f"{args.model.split('/')[-1]}_{args.level}_{args.split}"
out_path = ROOT / "runs" / f"{tag}.jsonl"
out_path.parent.mkdir(exist_ok=True)

records = []
if args.resume and out_path.exists():
    records = [json.loads(l) for l in open(out_path) if l.strip()]
    print(f"resuming: {len(records)} items already done")
done = {r["request_id"] for r in records}
t_start = time.time()
with open(out_path, "a" if args.resume else "w") as f:
    for i, row in enumerate(rows):
        if row["instance_id"] in done:
            continue
        req = row_to_request(row, level=args.level, tau=args.tau)
        rec = router.route(req)
        rec["gold"] = row["label"]
        rec["difficulty_raw_gold"] = row["difficulty_raw"]
        if args.flip:
            rev = router.route(req, reverse=True)
            rec["raw_label_reversed"] = rev["raw_label"]
        records.append(rec)
        f.write(json.dumps(rec) + "\n")
        if (i + 1) % 10 == 0:
            print(f"[{i + 1}/{len(rows)}] {time.time() - t_start:.0f}s", flush=True)

summary = summarize(records)
summary.update({"model": args.model, "level": args.level, "split": args.split, "prior": args.prior, "tau": args.tau})
(ROOT / "runs" / f"{tag}.summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("logs:", out_path)
