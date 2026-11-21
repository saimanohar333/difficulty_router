from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .schema import Request


@dataclass
class Section:
    kind: str
    text: str
    is_task: bool = False


def normalize(req: Request) -> List[Section]:
    sections = [Section("task", req.task.strip(), is_task=True)]
    for c in req.context:
        text = (c.text or "").strip()
        if text:
            sections.append(Section(c.type, text))
    return sections


def render_state(sections: List[Section]) -> str:
    parts = []
    for s in sections:
        if s.is_task:
            parts.append(f"<TASK>\n{s.text}\n</TASK>")
        else:
            parts.append(f'<CONTEXT type="{s.kind}">\n{s.text}\n</CONTEXT>')
    return "\n\n".join(parts)
