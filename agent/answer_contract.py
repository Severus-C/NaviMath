from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict

from .answer_normalizer import DEFAULT_NORMALIZER, is_placeholder_answer


@dataclass(frozen=True)
class AnswerContract:
    """A lightweight description of the quantity the problem asks to return."""

    kind: str = "expression"
    exact_required: bool = True
    integer_required: bool = False
    unit_hint: str = ""
    rationale: str = "default exact mathematical expression"

    @classmethod
    def infer(
        cls,
        problem: str,
        *,
        integer_only: bool = False,
        proof_mode: bool = False,
    ) -> "AnswerContract":
        text = " ".join(str(problem or "").lower().split())
        approximate = bool(
            re.search(
                r"\b(?:approximate|approximately|nearest|decimal|to \d+ decimal|estimate)\b",
                text,
            )
        )
        unit_hint = ""
        if re.search(r"\b(?:dollar|usd|cents?)\b|\\textdollar|\$\s*\d", text):
            unit_hint = "currency"
        elif re.search(r"\b(?:square|area)\b", text):
            unit_hint = "area"
        elif re.search(r"\b(?:length|distance|yards?|feet|inches?|miles?)\b", text):
            unit_hint = "length"

        if proof_mode:
            return cls("proof", False, False, unit_hint, "metadata requests a proof")
        if integer_only:
            return cls("integer", True, True, unit_hint, "benchmark schema requires one integer")
        if re.search(r"\b(?:number of solutions|how many|number of ways|number of pairs)\b", text):
            return cls("count", not approximate, True, unit_hint, "question asks for a discrete count")
        if re.search(r"\bnumber of revolutions\b", text):
            return cls("count", not approximate, False, unit_hint, "question asks for a revolution count")
        if re.search(r"\b(?:under what condition|condition.*(?:is|holds)|holds? if)\b", text) or text.endswith(" if:"):
            return cls("condition", not approximate, False, unit_hint, "question asks for a condition")
        if re.search(r"\b(?:ratio|is to .* as)\b", text):
            return cls("ratio", not approximate, False, unit_hint, "question asks for a ratio")
        if re.search(r"\b(?:another point|coordinates? of|ordered pair)\b", text):
            return cls("point", not approximate, False, unit_hint, "question asks for a point")
        if re.search(r"\b(?:all solutions|solution set|all values|values of .+ are)\b", text):
            return cls("set", not approximate, False, unit_hint, "question asks for all admissible values")
        if re.search(r"\barea of\b", text):
            return cls("area", not approximate, False, "area", "question asks for an area")
        return cls("expression", not approximate, False, unit_hint)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def instruction(self) -> str:
        exactness = "exact" if self.exact_required else "requested numerical"
        details = {
            "integer": "one integer",
            "count": "the requested count, not the objects being counted",
            "condition": "the complete mathematical condition",
            "ratio": "the requested ratio, not either component quantity",
            "point": "one coordinate point",
            "set": "the complete solution set",
            "area": "the requested area, not an auxiliary point, arc, sector, or whole figure",
            "proof": "a complete proof",
            "expression": "the requested mathematical expression",
        }
        return f"Return {details.get(self.kind, details['expression'])} in {exactness} form."

    def accepts(self, answer: object, reasoning: str = "") -> bool:
        raw = str(answer or "").strip()
        if not raw or is_placeholder_answer(raw):
            return False
        if self.kind == "proof":
            return bool(raw)

        compact = re.sub(r"\s+", "", raw.lower())
        if self.exact_required:
            if re.search(r"\\approx|≈|\b(?:approximately|approx\.?|about|nearest)\b", raw, re.I):
                return False
            context = f"{raw}\n{reasoning}"
            if re.search(
                r"(?:pi|π)\s*(?:=|≈|\\approx)\s*(?:22\s*/\s*7|3\.14)|"
                r"using\s+(?:the\s+)?approximation\s+(?:pi|π)",
                context,
                re.I,
            ):
                return False

        if self.integer_required:
            cleaned = DEFAULT_NORMALIZER.canonicalize(raw)
            if not re.fullmatch(r"[-+]?\d+", cleaned):
                return False

        if self.kind == "condition":
            return bool(re.search(r"=|<=|>=|<|>|\\(?:in|leq|geq|mid)\b|\b(?:divisible|odd|even)\b", raw))
        if self.kind == "ratio":
            if re.search(r"^\s*[\[(].*,.*[\])]\s*$", raw) or "\\cap" in raw:
                return False
            return bool(re.search(r"[:/]", raw) or re.search(r"\\(?:frac|dfrac|tfrac)", raw))
        if self.kind == "point":
            return bool(re.search(r"[\[(]\s*[^,]+\s*,\s*[^,]+\s*[\])]", raw))
        if self.kind == "set":
            return bool(re.search(r"\\?[{].*\\?[}]|,|\bor\b", raw, re.I))
        if self.kind == "area":
            if "\\cap" in raw or re.fullmatch(r"[A-Za-z0-9_]+\s*=\s*[A-Za-z0-9_]+", raw):
                return False
        if self.kind == "count" and re.search(r"\\?[{].*\\?[}]", raw):
            return False
        return True


def has_terminal_answer(text: object) -> bool:
    return bool(
        re.search(
            r"(?im)^\s*(?:\*\*)?FINAL\s*_?\s*ANSWER(?:\*\*)?\s*[:：]",
            str(text or ""),
        )
    )


__all__ = ["AnswerContract", "has_terminal_answer"]
