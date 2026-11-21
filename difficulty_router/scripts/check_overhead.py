import argparse

from _common import ROOT, add_model_args

p = argparse.ArgumentParser()
add_model_args(p)
p.add_argument("--split", default="dev")
p.add_argument("--n", type=int, default=30)
p.add_argument("--show", action="store_true", help="print the first full rendered prompt")
args = p.parse_args()

from anyjev import Question
from anyjev.readout import build_prompt, render_chat_parts
from transformers import AutoTokenizer

from router.data import load_jsonl, row_to_request
from router.pipeline import DifficultyRouter
from router.prompt import build_question_text
from router.schema import LABELS
from router.tokens import HFTokenCounter

tok = AutoTokenizer.from_pretrained(args.model)
router = DifficultyRouter(None, HFTokenCounter(args.model), args.max_prompt_tokens)
q = Question.choice(build_question_text("coding_bugfix"), LABELS, name="difficulty_coding_bugfix")

rows = load_jsonl(ROOT / "data" / f"{args.split}.jsonl")[: args.n]
diffs, over = [], 0
for i, row in enumerate(rows):
    state, truncated, predicted = router.build_state(row_to_request(row))
    spec = build_prompt(state, q, list(range(len(LABELS))))
    pre, suf = render_chat_parts(tok, spec)
    full = pre + suf
    assert "Difficulty criteria" in full and "Question:" in full, "rubric missing from the real prompt"
    actual = len(tok.encode(full, add_special_tokens=False))
    diffs.append(actual - predicted)
    over += actual > args.max_prompt_tokens
    if i == 0 and args.show:
        print("=" * 30, "FIRST RENDERED PROMPT (truncated view)", "=" * 30)
        print(full[:1500], "\n[...]\n", full[-900:])
print(f"items={len(diffs)}  actual - predicted tokens: min={min(diffs)} max={max(diffs)} mean={sum(diffs)/len(diffs):.1f}")
print(f"items whose REAL prompt exceeds --max-prompt-tokens ({args.max_prompt_tokens}): {over}")
print("OK" if over == 0 and max(abs(d) for d in diffs) <= 16 else "CHECK: accounting is off, see numbers above")
