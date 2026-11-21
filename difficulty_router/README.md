# Difficulty Router: AnyJev + SWE-bench Verified

Easy / Medium / Hard difficulty decision for semantic routing, following the design report:
Normalizer -> Budget Guard -> Prompt Builder (rubric) -> AnyJev Decision Engine -> Post-processor (tau escalation) -> JSONL logs.

Label mapping (SWE-bench Verified `difficulty`):

| SWE-bench value | Label |
|---|---|
| `<15 min fix` | Easy |
| `15 min - 1 hour` | Medium |
| `1-4 hours` | Hard |
| `>4 hours` | Hard |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```
