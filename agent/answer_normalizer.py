from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import List, Sequence


_MATRIX_RE = re.compile(
    r"\\begin\s*\{(?:p|b|B|v|V)?matrix\}(.*?)\\end\s*\{(?:p|b|B|v|V)?matrix\}",
    re.DOTALL,
)

_PLACEHOLDER_WORDS = {
    "answer",
    "content",
    "contenthere",
    "actualanswer",
    "finalanswer",
    "youranswer",
    "value",
    "result",
    "expression",
    "formula",
    "response",
    "output",
    "solution",
    "proof",
    "integer",
    "number",
    "percentage",
    "constant",
    "same",
    "condition",
    "completion",
    "thecondition",
    "thecompletion",
    "justthenumber",
    "justtheexpression",
    "justthenumberexpression",
    "justthevalues",
    "actualfinalanswer",
    "text",
    "placeholder",
    "writethecorrectedactualfinalanswer",
    "writetheactualfinalanswer",
    "写修正后的实际最终答案",
    "写实际最终答案",
}

_SINGLE_LETTER_PRODUCT_CHARS = frozenset("abcdefghjklmnopqrstuvwxyz") - {"i", "o"}
_PRODUCT_TOKEN_EXCLUSIONS = {
    "abs", "cos", "deg", "exp", "log", "max", "min", "mph", "rad", "sin",
    "tan", "usd", "yd", "yds",
}


def expand_single_letter_products(text: str) -> str:
    """Expand short contest-style monomials such as ``abc`` and ``ca``.

    SymPy otherwise treats each run as one multi-letter symbol, so ``ca`` and
    ``ac`` incorrectly compare as different variables.  The deliberately
    narrow alphabet/length guard leaves function names, constants, and units
    such as ``sqrt``, ``pi``, and ``mph`` untouched.
    """

    def expand(match: re.Match[str]) -> str:
        token = match.group(0)
        if (
            2 <= len(token) <= 3
            and token.lower() not in _PRODUCT_TOKEN_EXCLUSIONS
            and all(char.lower() in _SINGLE_LETTER_PRODUCT_CHARS for char in token)
        ):
            return "*".join(token)
        return token

    return re.sub(r"(?<![A-Za-z_])[A-Za-z]{2,4}(?![A-Za-z_])", expand, str(text))


def is_placeholder_answer(answer: object) -> bool:
    """Detect schema placeholders without rejecting mathematical brackets."""
    text = str(answer or "").strip().lower().strip(" .,:;!?`*_\"'")
    text = re.sub(r"^(?:final[_\s-]*)?answer\s*[:=]\s*", "", text)
    if len(text) >= 2 and (text[0], text[-1]) in {
        ("[", "]"),
        ("{", "}"),
        ("<", ">"),
        ("(", ")"),
    }:
        text = text[1:-1].strip()
    compact = re.sub(r"[\s_/\-]+", "", text)
    compact = re.sub(r"^(?:the|an|a)(?=[a-z])", "", compact)
    return compact in _PLACEHOLDER_WORDS


def _balanced_group(text: str, start: int) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif text[index] == "}" and (index == 0 or text[index - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    return None


def _unwrap_command(text: str, commands: Sequence[str]) -> str:
    """Return the last complete command argument, favoring the final answer."""
    matches: List[str] = []
    for command in commands:
        cursor = 0
        while True:
            start = text.find(command, cursor)
            if start < 0:
                break
            brace = text.find("{", start + len(command))
            group = _balanced_group(text, brace)
            if group:
                matches.append(group[0])
                cursor = group[1]
            else:
                cursor = start + len(command)
    return matches[-1] if matches else text


def _replace_fractions(text: str) -> str:
    cursor = 0
    output = ""
    commands = ("\\frac", "\\dfrac", "\\tfrac")
    while cursor < len(text):
        found = [(text.find(command, cursor), command) for command in commands]
        found = [(position, command) for position, command in found if position >= 0]
        if not found:
            output += text[cursor:]
            break
        position, command = min(found)
        output += text[cursor:position]
        first_start = position + len(command)
        while first_start < len(text) and text[first_start].isspace():
            first_start += 1
        first = _balanced_group(text, first_start)
        if not first:
            output += command
            cursor = position + len(command)
            continue
        second_start = first[1]
        while second_start < len(text) and text[second_start].isspace():
            second_start += 1
        second = _balanced_group(text, second_start)
        if not second:
            output += text[position:first[1]]
            cursor = first[1]
            continue
        output += f"({_replace_fractions(first[0])})/({_replace_fractions(second[0])})"
        cursor = second[1]
    return output


def _split_top_level(text: str, delimiters: str = ",;") -> List[str]:
    parts: List[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char in delimiters and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return [part for part in parts if part != ""]


def _numeric_key(text: str) -> str | None:
    compact = text.strip()
    if re.fullmatch(r"[-+]?\d+/[-+]?\d+", compact):
        numerator, denominator = compact.split("/", 1)
        if int(denominator) == 0:
            return None
        value = Fraction(int(numerator), int(denominator))
        return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", compact):
        return None
    try:
        value = Fraction(Decimal(compact))
    except (InvalidOperation, ValueError):
        return None
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _numeric_expression_key(text: str) -> str | None:
    """Evaluate a small, numeric-only arithmetic grammar exactly."""
    if not re.fullmatch(r"[\d+\-*/().]+", text):
        return None
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return None

    def evaluate(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        raise ValueError("unsupported numeric expression")

    try:
        value = evaluate(tree)
    except (ValueError, ZeroDivisionError):
        return None
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


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


@dataclass(frozen=True)
class AnswerContext:
    answer_type: str = ""
    contest: str = ""
    aime_width: int = 3


class AnswerNormalizer:
    """Canonicalize answer syntax without attempting symbolic mathematics."""

    def clean(self, answer: object) -> str:
        original = str(answer or "").strip()
        text = original.replace("\u2212", "-").replace("\uff0c", ",").replace("\uff1b", ";")
        text = re.sub(r"^\s*\$+|\$+\s*$", "", text)
        # A dollar sign embedded immediately before a number is a currency
        # marker, while leading/trailing dollar signs above are math delimiters.
        text = re.sub(r"\$(?=\s*\d)", "usd", text)
        text = text.replace("\\displaystyle", "").replace("\\left", "").replace("\\right", "")
        text = text.replace("\\%", "%")
        text = text.replace("\\textdollar", "usd").replace("\\$", "usd")
        text = _unwrap_command(text, ("\\boxed", "\\fbox"))
        text = re.sub(r"\\(?:operatorname|mathrm|text)\s*\{([^{}]*)\}", r"\1", text)
        # In answer notation an integer immediately followed by a numeric
        # LaTeX fraction is a mixed number, not implicit multiplication.
        text = re.sub(
            r"(?<![\w.)}])([-+]?\d+)\s*(?=\\(?:d?frac|tfrac)\s*\{\d+\}\s*\{\d+\})",
            r"\1+",
            text,
        )
        text = _replace_fractions(text)
        text = re.sub(r"(?<![\w.)])([-+]?\d+)\((\d+)\)/\((\d+)\)", r"\1+(\2)/(\3)", text)
        text = re.sub(r"(?:\\sqrt|sqrt)\s*\{([^{}]+)\}", r"sqrt(\1)", text)
        text = re.sub(r"\^\{([-+]?[A-Za-z0-9.]+)\}", r"^\1", text)
        text = re.sub(r"\^\{([^{}]+)\}", r"^(\1)", text)
        replacements = {
            "\\cdot": "*", "\\times": "*", "\\div": "/", "\\pi": "pi",
            "\\infty": "infinity", "\u221e": "infinity", "\\leq": "<=", "\\le": "<=",
            "\\geq": ">=", "\\ge": ">=", "\\neq": "!=", "\\pm": "+-",
            "\\colon": ":", "\\,": "", "\\!": "", "\\;": "",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
        text = text.replace("$", "")
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"^(?:a|an|the)(?=(?:loss|gain|profit))", "", text, flags=re.IGNORECASE)
        text = text.strip(" .\u3002;,")
        return text or original

    def canonicalize(self, answer: object, context: AnswerContext | None = None) -> str:
        context = context or AnswerContext()
        text = self.clean(answer).lower().strip()
        text = re.sub(r"^(?:final[_\s-]*)?answer[:\uff1a]?", "", text).strip()

        matrix = _MATRIX_RE.fullmatch(text)
        if matrix:
            rows = re.split(r"\\\\", matrix.group(1))
            canonical_rows = []
            for row in rows:
                cells = [self._canonical_atom(cell) for cell in row.split("&")]
                canonical_rows.append(",".join(cells))
            return "matrix[" + ";".join(canonical_rows) + "]"

        text = text.replace("\\{", "{").replace("\\}", "}")
        if len(text) >= 2 and text[0] == "{" and text[-1] == "}":
            members = _split_top_level(text[1:-1])
            if len(members) > 1 or context.answer_type.lower() in {"set", "roots", "solutions"}:
                keys = sorted({self.canonicalize(member, context) for member in members})
                return "{" + ",".join(keys) + "}"

        if len(text) >= 3 and text[0] in "([" and text[-1] in ")]":
            members = _split_top_level(text[1:-1])
            if len(members) == 2 and (
                text[0] == "[" or text[-1] == "]" or context.answer_type.lower() == "interval"
            ):
                endpoints = [self.canonicalize(member, context) for member in members]
                return text[0] + ",".join(endpoints) + text[-1]
            if context.answer_type.lower() in {"vector", "tuple", "point"} or len(members) > 1:
                return "(" + ",".join(self.canonicalize(member, context) for member in members) + ")"

        if context.answer_type.lower() in {"multiple", "multi_answer", "answers"}:
            text = re.sub(r"\b(?:and|or)\b", ";", text)
            members = _split_top_level(text, delimiters=",;")
            if len(members) > 1:
                return ";".join(sorted(self.canonicalize(member, context) for member in members))

        return self._canonical_atom(text)

    @staticmethod
    def _canonical_atom(text: str) -> str:
        atom = text.strip()
        word_numbers = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
            "ten": "10", "eleven": "11", "twelve": "12",
        }
        if atom in word_numbers:
            return word_numbers[atom]
        numeric_ratio = re.fullmatch(
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)):([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
            atom,
        )
        if numeric_ratio:
            ratio_key = _numeric_expression_key(
                f"({numeric_ratio.group(1)})/({numeric_ratio.group(2)})"
            )
            if ratio_key is not None:
                return ratio_key
        numeric = _numeric_expression_key(atom) or _numeric_key(atom)
        if numeric is not None:
            return numeric
        while _has_wrapping_parentheses(atom):
            atom = atom[1:-1]
        numeric = _numeric_key(atom)
        return numeric if numeric is not None else atom

    def equivalent(self, left: object, right: object, context: AnswerContext | None = None) -> bool:
        left_key = self.canonicalize(left, context)
        right_key = self.canonicalize(right, context)
        return bool(left_key and right_key and left_key == right_key)

    def format_aime(self, answer: object, width: int = 3) -> str:
        key = self.canonicalize(answer, AnswerContext(contest="AIME", aime_width=width))
        if not re.fullmatch(r"\d+", key):
            raise ValueError(f"AIME answer must be a non-negative integer, got {answer!r}")
        value = int(key)
        if value >= 10**width:
            raise ValueError(f"AIME answer must fit in {width} digits, got {value}")
        return str(value).zfill(width)


DEFAULT_NORMALIZER = AnswerNormalizer()
