from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

LABELS: List[str] = ["Easy", "Medium", "Hard"]  # ordinal, low -> high
TIER = {label: i for i, label in enumerate(LABELS)}


@dataclass
class ContextItem:
    type: str  # code|stack_trace|tests|repo|docs|history|issue_comments|claim_form|policy|clinical_notes|correspondence
    text: str


@dataclass
class Request:
    request_id: str
    domain: str  # "coding_bugfix" | "insurance_claims"
    task: str
    context: List[ContextItem] = field(default_factory=list)
    level: str = "L0"  # "raw" | "L0" | "L1"
    tau: float = 0.6

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Request":
        opts = d.get("options", {})
        return cls(
            request_id=d["request_id"],
            domain=d["domain"],
            task=d["task"],
            context=[ContextItem(**c) for c in d.get("context", [])],
            level=opts.get("level", "L0"),
            tau=opts.get("tau", 0.6),
        )
