from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

import yaml

from .schema import LABELS

RUBRIC_DIR = Path(__file__).resolve().parents[1] / "rubrics"


@lru_cache(maxsize=None)
def load_rubric(domain: str) -> dict:
    path = RUBRIC_DIR / f"{domain}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No rubric for domain '{domain}' at {path}")
    return yaml.safe_load(path.read_text())


def build_question_text(domain: str) -> str:
    r = load_rubric(domain)
    lines: List[str] = [r["preamble"].strip(), "", "Difficulty criteria:"]
    for label in LABELS:
        lines.append(f"- {label}: {r['levels'][label].strip()}")
    lines += ["", r["question"].strip()]
    return "\n".join(lines)
