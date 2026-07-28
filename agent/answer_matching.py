from __future__ import annotations

import math
import re
from typing import List, Sequence

from .agent_utils import extract_final_answer
from .answer_normalizer import AnswerContext, DEFAULT_NORMALIZER, expand_single_letter_products
from .tool_verify import VERIFIED, ToolVerify


_VERIFY = ToolVerify()
_EXPRESSION_TYPES = {"expression", "exact", "formula", "equation", "symbolic"}
_INTEGER_TYPES = {"integer", "integer_mod_1000"}
_TRAILING_UNITS_RE = re.compile(
    r"(?:%|percent|mph|miles?perhour|yds?\.?|yards?\.?|ft\.?|feet|inches?|"
    r"cm|mm|km|meters?|metres?|degrees?|deg|radians?|rad|usd|dollars?|cents?)$",
    re.IGNORECASE,
)


def answer_key(value: object, answer_type: str = "", contest: str = "") -> str:
    """Extract and canonicalize an answer using the benchmark schema."""

    text = str(value or "")
    extracted = extract_final_answer(text)
    context = AnswerContext(answer_type=answer_type, contest=contest)
    return DEFAULT_NORMALIZER.canonicalize(extracted or text, context)


def _simple_assignment_rhs(key: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"([a-z][a-z0-9_]*(?:\([^=()]+\))?)=([^=]+)", key)
    return (match.group(1), match.group(2)) if match else None


def _has_wrapping_parentheses(text: str) -> bool:
    if len(text) < 2 or text[0] != "(" or text[-1] != ")":
        return False
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and index != len(text) - 1:
                return False
    return depth == 0


def _split_top_level_items(key: str) -> tuple[List[str], bool]:
    text = key
    if _has_wrapping_parentheses(text) and "," in text:
        text = text[1:-1]

    parts: List[str] = []
    start = 0
    depth = 0
    unordered = False
    index = 0
    while index < len(text):
        char = text[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0 and char in ",;":
            parts.append(text[start:index])
            start = index + 1
        elif depth == 0 and text[index : index + 2] == "or":
            parts.append(text[start:index])
            start = index + 2
            index += 1
            unordered = True
        index += 1
    parts.append(text[start:])
    cleaned = [part for part in parts if part]
    return cleaned, unordered


def _strip_trailing_unit(key: str) -> tuple[str, str]:
    match = _TRAILING_UNITS_RE.search(key)
    if not match:
        return key, ""
    return key[: match.start()], match.group(0).lower()


def _unit_family(unit: str) -> str:
    compact = unit.rstrip(".")
    aliases = {
        "yard": "yd", "yards": "yd", "yds": "yd",
        "feet": "ft", "foot": "ft",
        "inch": "in", "inches": "in",
        "meter": "m", "meters": "m", "metre": "m", "metres": "m",
        "degree": "deg", "degrees": "deg",
        "radian": "rad", "radians": "rad",
        "percent": "%",
        "dollar": "usd", "dollars": "usd",
        "cent": "cent", "cents": "cent",
    }
    return aliases.get(compact, compact)


def _percent_of_expression(key: str) -> str:
    match = re.fullmatch(r"(.+)%of(.+)", key)
    if match:
        return f"(({match.group(1)})/100)*({match.group(2)})"
    return key


def _transaction_value(key: str) -> tuple[str, float] | None:
    compact = key.replace("usd", "")
    patterns = (
        r"(loss|gain|profit)(?:of)?([-+]?\d+(?:\.\d+)?)",
        r"([-+]?\d+(?:\.\d+)?)(loss|gain|profit)",
    )
    for index, pattern in enumerate(patterns):
        match = re.fullmatch(pattern, compact)
        if not match:
            continue
        kind, value = match.groups() if index == 0 else (match.group(2), match.group(1))
        return kind, float(value)
    return None


def _symbolic_form(key: str) -> str:
    text, _ = _strip_trailing_unit(key)
    text = _percent_of_expression(text)
    # HARP canonical answers encode a mixed coefficient such as
    # ``21 1/3 pi`` as ``21+(1)/(3)pi``. Restore the intended grouping.
    text = re.sub(
        r"(?<![\d.])(\d+)\+\((\d+)\)/\((\d+)\)(pi)\b",
        r"(\1+(\2)/(\3))*\4",
        text,
    )
    # Human contest notation commonly writes 5280/6pi for 5280/(6*pi).
    text = re.sub(r"(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)pi\b", r"\1/(\2*pi)", text)
    return expand_single_letter_products(text)


def _numeric_value(key: str) -> float | None:
    expression = _symbolic_form(key)
    parsed = _VERIFY.parse(expression)
    if parsed is None or getattr(parsed, "free_symbols", None):
        return None
    try:
        value = complex(parsed.evalf(30))
    except Exception:  # noqa: BLE001 - malformed answers fail closed.
        return None
    if abs(value.imag) > 1e-12 or not math.isfinite(value.real):
        return None
    return float(value.real)


def _scalar_match(left: str, right: str, answer_type: str) -> bool:
    if left == right:
        return True

    left_transaction = _transaction_value(left)
    right_transaction = _transaction_value(right)
    if left_transaction and right_transaction:
        return left_transaction == right_transaction

    left_assignment = _simple_assignment_rhs(left)
    right_assignment = _simple_assignment_rhs(right)
    if left_assignment and right_assignment:
        if left_assignment[0] != right_assignment[0]:
            return False
        left, right = left_assignment[1], right_assignment[1]
    elif left_assignment:
        left = left_assignment[1]
    elif right_assignment:
        right = right_assignment[1]
    if left == right:
        return True

    _, left_unit = _strip_trailing_unit(left)
    _, right_unit = _strip_trailing_unit(right)
    if left_unit and right_unit and _unit_family(left_unit) != _unit_family(right_unit):
        return False

    left_value, right_value = _numeric_value(left), _numeric_value(right)
    if left_value is not None and right_value is not None:
        scale = max(1.0, abs(left_value), abs(right_value))
        if abs(left_value - right_value) <= 1e-10 * scale:
            return True

    if answer_type not in _EXPRESSION_TYPES:
        return False
    result = _VERIFY.check_equivalence(_symbolic_form(left), _symbolic_form(right))
    return result.verdict == VERIFIED


def _items_match(
    left: Sequence[str],
    right: Sequence[str],
    *,
    unordered: bool,
    answer_type: str,
) -> bool:
    if len(left) != len(right):
        return False
    if not unordered:
        return all(_scalar_match(a, b, answer_type) for a, b in zip(left, right))

    remaining = list(right)
    for item in left:
        match_index = next(
            (index for index, candidate in enumerate(remaining) if _scalar_match(item, candidate, answer_type)),
            None,
        )
        if match_index is None:
            return False
        remaining.pop(match_index)
    return not remaining


def answers_match(
    prediction: object,
    expected: object,
    answer_type: str = "",
    contest: str = "",
) -> bool:
    """Compare mathematical answers while tolerating harmless surface syntax.

    The matcher remains conservative about prose and symbolic claims: it only
    relaxes assignment wrappers, tuple delimiters, answer ordering, standard
    units/percent signs, explicit approximation, and verified equivalence.
    """

    kind = answer_type.strip().lower()
    pred_key = answer_key(prediction, kind, contest)
    expected_key = answer_key(expected, kind, contest)
    if not pred_key or not expected_key:
        return False
    if pred_key == expected_key:
        return True
    if kind in _INTEGER_TYPES:
        return False

    pred_items, pred_unordered = _split_top_level_items(pred_key)
    expected_items, expected_unordered = _split_top_level_items(expected_key)
    if len(pred_items) > 1 or len(expected_items) > 1:
        schema_unordered = kind in {"multiple", "multi_answer", "answers", "roots", "solutions", "set"}
        return _items_match(
            pred_items,
            expected_items,
            unordered=schema_unordered or pred_unordered or expected_unordered,
            answer_type=kind,
        )

    if _scalar_match(pred_key, expected_key, kind):
        return True

    approximate = re.search(r"(?:approximately|approx\.?|about|nearest)", str(prediction), re.I)
    if approximate:
        pred_value, expected_value = _numeric_value(pred_key), _numeric_value(expected_key)
        if pred_value is not None and expected_value is not None and expected_value.is_integer():
            return round(pred_value) == int(expected_value)
    return False


__all__ = ["answer_key", "answers_match"]
