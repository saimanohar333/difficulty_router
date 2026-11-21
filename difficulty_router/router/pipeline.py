from __future__ import annotations

import time
from typing import Optional, Tuple

from .budget import apply_budget
from .normalizer import normalize, render_state
from .postprocess import postprocess
from .prompt import build_question_text
from .schema import Request

TEMPLATE_OVERHEAD = 80  # fallback


class DifficultyRouter:
    def __init__(self, engine, counter, max_prompt_tokens: int = 12000):
        self.engine = engine
        self.counter = counter
        self.max_prompt_tokens = max_prompt_tokens
        self._overhead_cache = {}

    def template_overhead(self, domain: str) -> int:
        if domain not in self._overhead_cache:
            question = build_question_text(domain)
            if hasattr(self.counter, "render_overhead"):
                n = self.counter.render_overhead(question)
            else:
                n = self.counter.count(question) + TEMPLATE_OVERHEAD
            self._overhead_cache[domain] = n
        return self._overhead_cache[domain]

    def build_state(self, req: Request, max_tokens: Optional[int] = None) -> Tuple[str, bool, int]:
        limit = max_tokens or self.max_prompt_tokens
        overhead = self.template_overhead(req.domain)
        sections = normalize(req)
        effective = limit
        for _ in range(4):
            secs, truncated, _ = apply_budget(sections, req.domain, effective, overhead, self.counter)
            state = render_state(secs)
            total = self.counter.count(state) + overhead
            if total <= limit:
                break
            effective -= total - limit
        return state, truncated, total

    def route(self, req: Request, level: Optional[str] = None, reverse: bool = False, max_oom_retries: int = 3) -> dict:
        """On CUDA out-of-memory, free the cache and retry with half the token budget."""
        budget, retries = self.max_prompt_tokens, 0
        while True:
            state, truncated, prompt_tokens = self.build_state(req, budget)
            t0 = time.perf_counter()
            try:
                probs, level_used = self.engine.decide(state, req.domain, level=level or req.level, reverse=reverse)
                break
            except Exception as e:  # torch.OutOfMemoryError
                if "OutOfMemory" not in type(e).__name__ or retries >= max_oom_retries:
                    raise
                retries += 1
                budget = max(1024, budget // 2)
                _free_cuda()
        latency_ms = (time.perf_counter() - t0) * 1000
        out = postprocess(probs, req.tau)
        out.update(
            {
                "request_id": req.request_id,
                "anyjev_level": level_used,
                "truncated": truncated,
                "prompt_tokens": prompt_tokens,
                "latency_ms": round(latency_ms, 1),
                "oom_retries": retries,
                "over_budget": prompt_tokens > budget,
            }
        )
        _free_cuda()
        return out


def _free_cuda() -> None:
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
