from __future__ import annotations

from typing import Dict, List

import numpy as np

from .schema import LABELS, TIER


def confusion(gold: List[str], pred: List[str]) -> np.ndarray:
    m = np.zeros((3, 3), dtype=int)
    for g, p in zip(gold, pred):
        m[TIER[g], TIER[p]] += 1
    return m


def macro_f1(cm: np.ndarray) -> float:
    f1s = []
    for k in range(3):
        tp = cm[k, k]
        prec = tp / cm[:, k].sum() if cm[:, k].sum() else 0.0
        rec = tp / cm[k, :].sum() if cm[k, :].sum() else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence >= lo) & (confidence <= hi)
        if mask.any():
            total += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(total)


def summarize(records: List[dict]) -> Dict:
    gold = [r["gold"] for r in records]
    out: Dict = {"n": len(records)}
    for key in ("raw_label", "difficulty"):
        pred = [r[key] for r in records]
        cm = confusion(gold, pred)
        out[key] = {
            "accuracy": float(np.mean([g == p for g, p in zip(gold, pred)])),
            "macro_f1": macro_f1(cm),
            "ordinal_mae": float(np.mean([abs(TIER[g] - TIER[p]) for g, p in zip(gold, pred)])),
            "confusion_rows_gold_cols_pred": cm.tolist(),
        }
    conf = np.array([r["confidence"] for r in records])
    correct = np.array([r["raw_label"] == r["gold"] for r in records])
    out["ece"] = ece(conf, correct)
    out["escalated_rate"] = float(np.mean([r["escalated"] for r in records]))
    out["truncated_rate"] = float(np.mean([r["truncated"] for r in records]))
    lat = np.array([r["latency_ms"] for r in records])
    out["latency_ms"] = {"mean": float(lat.mean()), "p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}
    out["prompt_tokens_mean"] = float(np.mean([r["prompt_tokens"] for r in records]))
    if all("raw_label_reversed" in r for r in records):
        out["flip_rate_option_reversal"] = float(np.mean([r["raw_label"] != r["raw_label_reversed"] for r in records]))
    out["gold_distribution"] = {l: int(sum(g == l for g in gold)) for l in LABELS}
    return out
