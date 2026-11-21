from __future__ import annotations

from typing import Dict, List, Optional

from .prompt import build_question_text
from .schema import LABELS


class DecisionEngine:
    def __init__(
        self,
        model_name: str,
        level: str = "L0",
        dtype: str = "bfloat16",
        device: str = "cuda",
        batch_size: int = 4,
        prior: str = "batch",
        artifacts_path: Optional[str] = None,
    ):
        from anyjev import Decider
        from anyjev.backends.hf import HFBackend

        self.model_name = model_name
        self.level = level
        backend = HFBackend(model_name, device=device, dtype=dtype, batch_size=batch_size)
        self.decider = Decider(backend, level="L0", prior=prior)
        self._questions: Dict[tuple, object] = {}
        self._artifacts_path = artifacts_path

    def load_artifacts(self) -> int:
        if not self._artifacts_path:
            return 0
        n = self.decider.load_artifacts(self._artifacts_path)
        if not n:
            raise RuntimeError(
                f"No L1 artifact matched in {self._artifacts_path}. It must be fitted with the same model, "
                "rubric text and option order (re-run scripts/fit_l1.py after any rubric change)."
            )
        return n

    def question(self, domain: str, reverse: bool = False):
        from anyjev import Question

        key = (domain, reverse)
        if key not in self._questions:
            options = list(reversed(LABELS)) if reverse else list(LABELS)
            self._questions[key] = Question.choice(
                build_question_text(domain),
                options,
                name=f"difficulty_{domain}" + ("_rev" if reverse else ""),
            )
        return self._questions[key]

    def decide(self, state: str, domain: str, level: Optional[str] = None, reverse: bool = False):
        q = self.question(domain, reverse)
        level = level or self.level
        ds = self.decider.decide(state, [q], level=level, require=None)
        dec = ds[q.name]
        dist = dec.distribution
        return {l: float(dist[l]) for l in LABELS}, dec.level

    def calibrate(self, states: List[str], gold_labels: List[str], domain: str) -> dict:
        q = self.question(domain)
        idx = [list(q.options).index(g) for g in gold_labels]
        return self.decider.calibrate(q, states, idx, level="L1")

    def save_artifacts(self, path: str) -> None:
        self.decider.save_artifacts(path)
