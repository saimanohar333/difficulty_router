from __future__ import annotations

from collections import Counter
from typing import Callable, List, Tuple


def is_oom(e: BaseException) -> bool:
    return "OutOfMemory" in type(e).__name__


def calibrate_with_backoff(
    engine,
    states: List[str],
    golds: List[str],
    tokens: List[int],
    domain: str,
    free_fn: Callable[[], None] = lambda: None,
    drop_frac: float = 0.10,
    max_attempts: int = 15,
    min_items: int = 50,
) -> Tuple[dict, List[int]]:
    keep = list(range(len(states)))
    for attempt in range(1, max_attempts + 1):
        oom = False
        try:
            art = engine.calibrate([states[i] for i in keep], [golds[i] for i in keep], domain)
            return art, keep
        except Exception as e:
            if not is_oom(e):
                raise
            oom = True
        if oom:
            free_fn()
            n_drop = max(2, int(len(keep) * drop_frac))
            by_len = sorted(keep, key=lambda i: tokens[i])
            new_keep = sorted(by_len[: len(by_len) - n_drop])
            cutoff = tokens[by_len[len(by_len) - n_drop - 1]] if len(by_len) > n_drop else 0
            print(f"[attempt {attempt}] out of memory: dropping the {n_drop} longest prompts "
                  f"(kept <= {cutoff} tokens; {len(keep)} -> {len(new_keep)} items; "
                  f"classes {dict(Counter(golds[i] for i in new_keep))})")
            if len(new_keep) < min_items or len(new_keep) == len(keep):
                raise RuntimeError(
                    "Calibration does not fit in GPU memory even after dropping long prompts. "
                    "Use a smaller --max-prompt-tokens, a smaller model, or a bigger GPU."
                )
            keep = new_keep
    raise RuntimeError("calibration kept running out of memory")
