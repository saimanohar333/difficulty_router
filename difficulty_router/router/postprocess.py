from __future__ import annotations

from typing import Dict

from .schema import LABELS


def postprocess(probs: Dict[str, float], tau: float) -> dict:
    raw_label = max(LABELS, key=lambda l: probs[l])
    confidence = float(probs[raw_label])
    difficulty, escalated = raw_label, False
    if confidence < tau and raw_label != "Hard":
        difficulty = LABELS[LABELS.index(raw_label) + 1]  # escalate
        escalated = True
    return {
        "difficulty": difficulty,
        "raw_label": raw_label,
        "probabilities": {l: float(probs[l]) for l in LABELS},
        "confidence": confidence,
        "escalated": escalated,
    }
