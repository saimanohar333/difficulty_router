import argparse

from _common import add_model_args

p = argparse.ArgumentParser()
add_model_args(p)
args = p.parse_args()

import anyjev
from anyjev import Decider, Question
from anyjev.backends.hf import HFBackend

print("anyjev version:", getattr(anyjev, "__version__", "?"))

backend = HFBackend(args.model, device=args.device, dtype=args.dtype, batch_size=args.batch_size)
q = Question.choice(
    "How difficult is the fix the issue asks for?  Easy: one-line typo-level fix. "
    "Medium: needs tracing across files. Hard: design-level or concurrency issue.",
    ["Easy", "Medium", "Hard"],
    name="difficulty",
)

states = {
    "typo": "Issue: README says 'recieve' instead of 'receive' in the install section.",
    "concurrency": "Issue: intermittent deadlock between the scheduler thread and the I/O reactor under "
    "high load; only reproduces on 64-core machines, no stack trace available.",
}

for level in ("raw", "L0"):
    d = Decider(backend, level="L0")
    print(f"\n== level={level} ==")
    for name, s in states.items():
        dec = d.decide(s, [q], level=level)["difficulty"]
        print(f"{name:12s} -> {dec.argmax:7s}  level={dec.level}  dist={ {k: round(v, 3) for k, v in dec.distribution.items()} }")

print("Smoke test finished")
