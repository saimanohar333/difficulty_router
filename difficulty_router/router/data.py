from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

from .schema import ContextItem, Request

DIFFICULTY_MAP = {
    "<15 min fix": "Easy",
    "15 min - 1 hour": "Medium",
    "1-4 hours": "Hard",
    ">4 hours": "Hard",
}


def map_difficulty(raw: str) -> str:
    key = raw.strip()
    if key in DIFFICULTY_MAP:
        return DIFFICULTY_MAP[key]
    raise ValueError(f"Unknown SWE-bench difficulty value: {raw!r}")


def load_swebench_rows() -> List[dict]:
    from datasets import load_dataset

    ds = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
    rows = []
    for r in ds:
        rows.append(
            {
                "instance_id": r["instance_id"],
                "repo": r["repo"],
                "version": r["version"],
                "problem_statement": r["problem_statement"],
                "hints_text": r["hints_text"] or "",
                "difficulty_raw": r["difficulty"],
                "label": map_difficulty(r["difficulty"]),
            }
        )
    return rows


def stratified_split(rows: List[dict], fractions=(0.4, 0.2, 0.4), seed: int = 13) -> Dict[str, List[dict]]:
    rng = random.Random(seed)
    out = {"train": [], "dev": [], "test": []}
    for label in sorted({r["label"] for r in rows}):
        group = sorted((r for r in rows if r["label"] == label), key=lambda r: r["instance_id"])
        rng.shuffle(group)
        n_tr = round(len(group) * fractions[0])
        n_dev = round(len(group) * fractions[1])
        out["train"] += group[:n_tr]
        out["dev"] += group[n_tr : n_tr + n_dev]
        out["test"] += group[n_tr + n_dev :]
    for v in out.values():
        v.sort(key=lambda r: r["instance_id"])
    return out


def save_jsonl(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path: Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def row_to_request(row: dict, level: str = "L0", tau: float = 0.6, extra_comments: str = "") -> Request:
    ctx = [ContextItem("repo", f"Repository: {row['repo']} (version {row['version']})")]
    comments = row["hints_text"]
    if extra_comments:
        comments = (comments + "\n\n" + extra_comments).strip()
    if comments.strip():
        ctx.append(ContextItem("issue_comments", comments))
    return Request(
        request_id=row["instance_id"],
        domain="coding_bugfix",
        task=row["problem_statement"],
        context=ctx,
        level=level,
        tau=tau,
    )
