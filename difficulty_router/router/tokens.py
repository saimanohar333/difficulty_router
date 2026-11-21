from __future__ import annotations

from typing import List


class HFTokenCounter:
    def __init__(self, model_name: str):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: List[int]) -> str:
        return self.tok.decode(ids)

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def render_overhead(self, question_text: str, options=("Easy", "Medium", "Hard")) -> int:
        try:
            from anyjev.readout import DEFAULT_SYSTEM as system
        except Exception:
            system = "You are a decision function.\nReply with the answer label only: no words, no punctuation, no explanation."
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        opts = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options))
        user = f"State:\n\n\nQuestion: {question_text}\nOptions:\n{opts}\nAnswer with the letter only."
        if getattr(self.tok, "chat_template", None):
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            try:
                text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            text = f"{system}\n\n{user}\nAnswer:"
        return len(self.tok.encode(text, add_special_tokens=False))


class WhitespaceCounter:
    def encode(self, text: str) -> List[str]:
        return text.split()

    def decode(self, ids: List[str]) -> str:
        return " ".join(ids)

    def count(self, text: str) -> int:
        return len(text.split())
