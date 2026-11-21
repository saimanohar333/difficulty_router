from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from .normalizer import Section

TRIM_ORDER = {
    "coding_bugfix": ["issue_comments", "docs", "repo", "tests", "stack_trace", "code"],
    "insurance_claims": ["correspondence", "policy", "clinical_notes", "claim_form"],
}

PER_SECTION_OVERHEAD = 16  # delimiter tokens


def _shrink(text: str, keep: int, counter, head_frac: float = 0.6) -> str:
    ids = counter.encode(text)
    if len(ids) <= keep:
        return text
    head = int(keep * head_frac)
    tail = keep - head
    tail_ids = ids[len(ids) - tail:] if tail > 0 else []
    return counter.decode(ids[:head]) + "\n[... truncated ...]\n" + counter.decode(tail_ids)


def apply_budget(
    sections: List[Section],
    domain: str,
    max_tokens: int,
    overhead_tokens: int,
    counter,
    min_keep_context: int = 64,
    min_keep_task: int = 512,
    compressor: Optional[Callable[[str, int], str]] = None,
) -> Tuple[List[Section], bool, int]:
    budget = max_tokens - overhead_tokens - PER_SECTION_OVERHEAD * len(sections)
    toks = [counter.count(s.text) for s in sections]
    excess = sum(toks) - budget
    if excess <= 0:
        return sections, False, sum(toks)

    order = TRIM_ORDER.get(domain, [])
    rank = lambda i: order.index(sections[i].kind) if sections[i].kind in order else -1
    ctx_idx = sorted((i for i, s in enumerate(sections) if not s.is_task), key=rank)
    task_idx = [i for i, s in enumerate(sections) if s.is_task]

    new_text = {i: s.text for i, s in enumerate(sections)}
    for i in ctx_idx + task_idx:
        if excess <= 0:
            break
        floor = min_keep_task if sections[i].is_task else min_keep_context
        cut = min(excess, max(0, toks[i] - floor))
        if cut <= 0:
            continue
        keep = toks[i] - cut
        fn = compressor or (lambda t, k: _shrink(t, k, counter))
        new_text[i] = fn(sections[i].text, keep)
        toks[i] = keep
        excess -= cut

    out = [Section(s.kind, new_text[i], s.is_task) for i, s in enumerate(sections)]
    return out, True, sum(toks)
