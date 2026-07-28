from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Sequence

from .answer_normalizer import DEFAULT_NORMALIZER, expand_single_letter_products

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )
except ImportError:  # The competition runtime may omit optional local tools.
    sp = None
    parse_expr = None


VERIFIED = "VERIFIED"
REJECTED = "REJECTED"
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class VerificationResult:
    verdict: str
    check: str
    confidence: float
    message: str
    expected: str = ""
    observed: str = ""
    samples: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ToolVerify:
    """Bounded, fail-open mathematical verification backed by SymPy.

    `UNKNOWN` is the normal result when a problem cannot be parsed reliably.
    Only deterministic contradictions produce `REJECTED`.
    """

    _FUNCTIONS = {
        "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
        "exp", "log", "sqrt", "abs", "factorial", "gamma", "floor", "ceiling",
    }

    def __init__(
        self,
        max_expression_chars: int = 400,
        max_operations: int = 160,
        max_symbols: int = 6,
        random_samples: int = 6,
        random_seed: int = 1729,
    ) -> None:
        self.max_expression_chars = max_expression_chars
        self.max_operations = max_operations
        self.max_symbols = max_symbols
        self.random_samples = random_samples
        self.random_seed = random_seed

    @property
    def available(self) -> bool:
        return sp is not None and parse_expr is not None

    def parse(self, expression: object) -> Any | None:
        if not self.available:
            return None
        text = self._normalize_expression(expression)
        if not text or len(text) > self.max_expression_chars:
            return None
        if "__" in text or not re.fullmatch(r"[A-Za-z0-9_+\-*/^()., ]+", text):
            return None

        identifiers = set(re.findall(r"[A-Za-z_]\w*", text))
        if any(name.startswith("_") for name in identifiers):
            return None
        symbol_names = identifiers - self._FUNCTIONS - {"pi", "e", "E", "oo", "infinity", "I"}
        if len(symbol_names) > self.max_symbols:
            return None

        local_dict: Dict[str, Any] = {name: sp.Symbol(name, finite=True) for name in symbol_names}
        local_dict.update({name: getattr(sp, name) for name in self._FUNCTIONS if hasattr(sp, name)})
        local_dict["abs"] = sp.Abs
        local_dict.update({"pi": sp.pi, "e": sp.E, "E": sp.E, "oo": sp.oo, "infinity": sp.oo, "I": sp.I})
        global_dict = {
            "Integer": sp.Integer,
            "Float": sp.Float,
            "Rational": sp.Rational,
            "Symbol": sp.Symbol,
            "Function": sp.Function,
        }
        transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
        try:
            parsed = parse_expr(
                text,
                local_dict=local_dict,
                global_dict=global_dict,
                transformations=transformations,
                evaluate=True,
            )
        except Exception:  # noqa: BLE001 - malformed model output must fail open.
            return None
        try:
            if int(sp.count_ops(parsed)) > self.max_operations:
                return None
        except Exception:  # noqa: BLE001
            return None
        return parsed

    def check_equivalence(self, left: object, right: object) -> VerificationResult:
        if not self.available:
            return self._unknown("equivalence", "SymPy is not installed")
        left_expr, right_expr = self.parse(left), self.parse(right)
        if left_expr is None or right_expr is None:
            return self._unknown("equivalence", "one or both expressions could not be parsed")
        try:
            difference = sp.cancel(sp.together(left_expr - right_expr))
            if difference == 0 or sp.simplify(difference) == 0:
                return VerificationResult(
                    VERIFIED, "equivalence", 1.0, "symbolic difference simplifies to zero",
                    expected=str(right_expr), observed=str(left_expr),
                )
        except Exception:  # noqa: BLE001
            try:
                difference = left_expr - right_expr
            except Exception:  # noqa: BLE001 - tuples/matrices are not scalar expressions.
                return self._unknown(
                    "equivalence",
                    "parsed values do not support scalar symbolic comparison",
                )

        numeric = self.random_numeric_equivalence(left_expr, right_expr)
        if numeric.verdict != UNKNOWN:
            return numeric
        return self._unknown("equivalence", "symbolic simplification was inconclusive")

    def random_numeric_equivalence(self, left: object, right: object) -> VerificationResult:
        left_expr = left if self._is_sympy_expr(left) else self.parse(left)
        right_expr = right if self._is_sympy_expr(right) else self.parse(right)
        if left_expr is None or right_expr is None:
            return self._unknown("numeric_substitution", "expressions could not be parsed")
        symbols = sorted(left_expr.free_symbols | right_expr.free_symbols, key=str)
        if not symbols:
            try:
                equal = abs(complex(sp.N(left_expr - right_expr, 30))) <= 1e-12
            except Exception:  # noqa: BLE001
                return self._unknown("numeric_substitution", "constant expression is not numerically comparable")
            verdict = VERIFIED if equal else REJECTED
            return VerificationResult(verdict, "numeric_substitution", 1.0, "exact constant comparison", samples=1)

        rng = random.Random(self.random_seed)
        completed = 0
        for _ in range(self.random_samples * 3):
            substitutions = {symbol: sp.Rational(rng.choice([-7, -5, -3, -2, -1, 1, 2, 3, 5, 7]), rng.choice([1, 1, 1, 2, 3])) for symbol in symbols}
            try:
                left_value = complex(sp.N(left_expr.subs(substitutions), 30))
                right_value = complex(sp.N(right_expr.subs(substitutions), 30))
            except Exception:  # noqa: BLE001
                continue
            if not all(map(self._finite_number, [left_value.real, left_value.imag, right_value.real, right_value.imag])):
                continue
            completed += 1
            scale = max(1.0, abs(left_value), abs(right_value))
            if abs(left_value - right_value) > 1e-9 * scale:
                return VerificationResult(
                    REJECTED, "numeric_substitution", 0.98,
                    f"expressions disagree at {substitutions}", samples=completed,
                )
            if completed >= self.random_samples:
                return VerificationResult(
                    VERIFIED, "numeric_substitution", 0.9,
                    "expressions agree on deterministic random substitutions", samples=completed,
                )
        return self._unknown("numeric_substitution", "not enough nonsingular sample points", samples=completed)

    def verify_equation_roots(
        self,
        equation: str,
        candidate: object,
        variable: str | None = None,
        require_complete: bool = True,
    ) -> VerificationResult:
        sides = self._split_equation(equation)
        if sides is None:
            return self._unknown("equation_roots", "equation could not be parsed")
        lhs, rhs = self.parse(sides[0]), self.parse(sides[1])
        if lhs is None or rhs is None:
            return self._unknown("equation_roots", "equation sides could not be parsed")
        expression = lhs - rhs
        symbols = sorted(expression.free_symbols, key=str)
        symbol = self._resolve_symbol(expression, variable) if variable else (symbols[0] if len(symbols) == 1 else None)
        if symbol is None:
            return self._unknown("equation_roots", "could not identify a unique equation variable")

        candidates = self._parse_answer_items(candidate)
        if not candidates:
            return self._unknown("equation_roots", "candidate roots could not be parsed")
        parsed_candidates = [self.parse(item) for item in candidates]
        if any(item is None for item in parsed_candidates):
            return self._unknown("equation_roots", "a candidate root could not be parsed")
        for item in parsed_candidates:
            try:
                if sp.simplify(expression.subs(symbol, item)) != 0:
                    return VerificationResult(
                        REJECTED, "equation_roots", 1.0,
                        f"{item} does not satisfy the equation", observed=str(candidate),
                    )
            except Exception:  # noqa: BLE001
                return self._unknown("equation_roots", "root substitution was inconclusive")

        if require_complete:
            try:
                solved = sp.solve(expression, symbol)
            except Exception:  # noqa: BLE001
                solved = []
            if solved:
                unmatched = list(solved)
                for item in parsed_candidates:
                    match_index = next((i for i, root in enumerate(unmatched) if sp.simplify(item - root) == 0), None)
                    if match_index is not None:
                        unmatched.pop(match_index)
                if unmatched or len(solved) != len(parsed_candidates):
                    return VerificationResult(
                        REJECTED, "equation_roots", 0.99,
                        "candidate root set is incomplete or contains duplicates",
                        expected=str(solved), observed=str(candidate),
                    )
        return VerificationResult(VERIFIED, "equation_roots", 1.0, "all candidate roots satisfy the equation", observed=str(candidate))

    def verify_derivative(self, expression: object, variable: str, candidate: object) -> VerificationResult:
        expr, answer = self.parse(expression), self.parse(candidate)
        if expr is None or answer is None:
            return self._unknown("derivative", "expression or answer could not be parsed")
        symbol = self._resolve_symbol(expr, variable)
        try:
            expected = sp.diff(expr, symbol)
        except Exception:  # noqa: BLE001
            return self._unknown("derivative", "SymPy could not differentiate the expression")
        return self._compare_expected("derivative", answer, expected)

    def verify_integral(
        self,
        integrand: object,
        variable: str,
        candidate: object,
        lower: object | None = None,
        upper: object | None = None,
    ) -> VerificationResult:
        expr, answer = self.parse(integrand), self.parse(candidate)
        if expr is None or answer is None:
            return self._unknown("integral", "integrand or answer could not be parsed")
        symbol = self._resolve_symbol(expr, variable)
        try:
            if lower is None or upper is None:
                difference = sp.simplify(sp.diff(answer, symbol) - expr)
                if difference == 0:
                    return VerificationResult(VERIFIED, "integral", 1.0, "candidate derivative equals integrand", observed=str(answer))
                return VerificationResult(REJECTED, "integral", 1.0, "candidate derivative does not equal integrand", observed=str(answer))
            lower_expr, upper_expr = self.parse(lower), self.parse(upper)
            if lower_expr is None or upper_expr is None:
                return self._unknown("integral", "integration bounds could not be parsed")
            expected = sp.integrate(expr, (symbol, lower_expr, upper_expr))
        except Exception:  # noqa: BLE001
            return self._unknown("integral", "SymPy could not evaluate the integral")
        return self._compare_expected("integral", answer, expected)

    def verify_limit(self, expression: object, variable: str, point: object, candidate: object, direction: str = "+-") -> VerificationResult:
        expr, at, answer = self.parse(expression), self.parse(point), self.parse(candidate)
        if expr is None or at is None or answer is None:
            return self._unknown("limit", "limit expression, point, or answer could not be parsed")
        try:
            expected = sp.limit(expr, self._resolve_symbol(expr, variable), at, dir=direction)
        except Exception:  # noqa: BLE001
            return self._unknown("limit", "SymPy could not evaluate the limit")
        return self._compare_expected("limit", answer, expected)

    def verify_residue(self, expression: object, variable: str, point: object, candidate: object) -> VerificationResult:
        expr, at, answer = self.parse(expression), self.parse(point), self.parse(candidate)
        if expr is None or at is None or answer is None:
            return self._unknown("residue", "residue expression, point, or answer could not be parsed")
        try:
            expected = sp.residue(expr, self._resolve_symbol(expr, variable), at)
        except Exception:  # noqa: BLE001
            return self._unknown("residue", "SymPy could not evaluate the residue")
        return self._compare_expected("residue", answer, expected)

    def verify_candidate(self, problem: str, candidate: object, answer_type: str = "") -> VerificationResult:
        if not self.available:
            return self._unknown("auto", "SymPy is not installed")
        if answer_type.lower() == "proof":
            return self._unknown("auto", "proof answers are outside ToolVerify scope")
        compact = " ".join(str(problem).replace("\n", " ").split())
        lowered = compact.lower()

        match = re.search(r"(?:differentiate|derivative\s+of)\s+(.+?)\s+(?:with\s+respect\s+to|w\.?r\.?t\.?)\s+([a-z])\b", compact, re.I)
        if match:
            return self.verify_derivative(match.group(1), match.group(2), candidate)

        match = re.search(r"integral\s+of\s+(.+?)\s+from\s+(.+?)\s+to\s+(.+?)\s+(?:with\s+respect\s+to|d)\s*([a-z])\b", compact, re.I)
        if match:
            return self.verify_integral(match.group(1), match.group(4), candidate, match.group(2), match.group(3))
        match = re.search(r"(?:integrate|integral\s+of)\s+(.+?)\s+(?:with\s+respect\s+to|d)\s*([a-z])\b", compact, re.I)
        if match:
            return self.verify_integral(match.group(1), match.group(2), candidate)

        match = re.search(r"limit\s+of\s+(.+?)\s+as\s+([a-z])\s+(?:approaches|tends\s+to|->)\s+([^ ?.,]+)", compact, re.I)
        if match:
            return self.verify_limit(match.group(1), match.group(2), match.group(3), candidate)

        match = re.search(r"residue\s+of\s+(.+?)\s+at\s+([a-z])\s*=\s*([^ ?.,]+)", compact, re.I)
        if match:
            return self.verify_residue(match.group(1), match.group(2), match.group(3), candidate)

        equation_match = re.search(r"solve(?:\s+the)?\s+equation\s+(.+?=.+?)(?:\s+for\s+([a-z]))?(?:[?.]|$)", compact, re.I)
        if equation_match:
            return self.verify_equation_roots(equation_match.group(1), candidate, equation_match.group(2))

        # LaTeX-heavy prompts often expose the relevant expression only inside math spans.
        spans = re.findall(r"\$([^$]+)\$", str(problem))
        for span in spans:
            derivative_match = re.search(r"\\frac\s*\{d\}\s*\{d([a-z])\}\s*(.+)", span, re.I)
            if derivative_match:
                return self.verify_derivative(derivative_match.group(2), derivative_match.group(1), candidate)

            integral_match = re.search(
                r"\\int\s*_\s*\{([^{}]+)\}\s*\^\s*\{([^{}]+)\}\s*(.*?)"
                r"\s*(?:\\[,;!])?\s*d([a-z])\b",
                span,
                re.I,
            )
            if integral_match:
                return self.verify_integral(
                    integral_match.group(3), integral_match.group(4), candidate,
                    integral_match.group(1), integral_match.group(2),
                )
            integral_match = re.search(r"\\int\s+(.*?)\s*(?:\\[,;!])?\s*d([a-z])\b", span, re.I)
            if integral_match:
                return self.verify_integral(integral_match.group(1), integral_match.group(2), candidate)

            limit_match = re.search(
                r"\\lim\s*_\s*\{\s*([a-z])\s*\\to\s*([^{}]+)\}\s*(.+)",
                span,
                re.I,
            )
            if limit_match:
                return self.verify_limit(limit_match.group(3), limit_match.group(1), limit_match.group(2), candidate)

            residue_match = re.search(
                r"(?:\\operatorname\s*\{res\}|\\mathrm\s*\{res\}|res)\s*_\s*"
                r"\{\s*([a-z])\s*=\s*([^{}]+)\}\s*(.+)",
                span,
                re.I,
            )
            if residue_match:
                return self.verify_residue(
                    residue_match.group(3), residue_match.group(1), residue_match.group(2), candidate
                )

        if "derivative" in lowered and spans:
            expression = next((span for span in spans if not re.fullmatch(r"[a-z]", span.strip(), re.I)), "")
            variable_match = re.search(r"with\s+respect\s+to\s+\$?([a-z])", compact, re.I)
            if expression and variable_match:
                return self.verify_derivative(expression, variable_match.group(1), candidate)
        asks_for_roots = bool(
            "solve" in lowered
            or re.search(r"\b(?:find|determine)(?:\s+all)?\s+(?:the\s+)?roots?\b", lowered)
            or re.search(r"\broots?\s+(?:is|are|of)\b", lowered)
        )
        derived_root_quantity = bool(
            re.search(r"\b(?:difference|sum|product|distance)\s+(?:of|between)\s+the\s+roots?\b", lowered)
        )
        if asks_for_roots and not derived_root_quantity and spans:
            equation = next((span for span in spans if self._split_equation(span)), "")
            if equation:
                return self.verify_equation_roots(equation, candidate)
        return self._unknown("auto", "no safely parseable verification target was detected")

    def equivalent_groups(self, answers: Sequence[object]) -> List[List[int]]:
        groups: List[List[int]] = []
        for index, answer in enumerate(answers):
            for group in groups:
                if self.check_equivalence(answer, answers[group[0]]).verdict == VERIFIED:
                    group.append(index)
                    break
            else:
                groups.append([index])
        return groups

    def _compare_expected(self, check: str, observed: Any, expected: Any) -> VerificationResult:
        result = self.check_equivalence(observed, expected)
        return VerificationResult(
            result.verdict,
            check,
            result.confidence,
            result.message,
            expected=str(expected),
            observed=str(observed),
            samples=result.samples,
        )

    def _normalize_expression(self, expression: object) -> str:
        text = DEFAULT_NORMALIZER.clean(expression)
        text = text.strip().strip("$")
        replacements = {
            "\\sin": "sin", "\\cos": "cos", "\\tan": "tan", "\\ln": "log",
            "\\log": "log", "\\exp": "exp", "\\arcsin": "asin", "\\arccos": "acos",
            "\\arctan": "atan", "\\mathrm{e}": "E", "\\operatorname{abs}": "abs",
            "\u00b7": "*", "\u00d7": "*", "\u2212": "-",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = re.sub(r"\|([^|]+)\|", r"abs(\1)", text)
        text = text.replace("{", "(").replace("}", ")")
        text = re.sub(r"\s+", "", text)
        return expand_single_letter_products(text)

    @staticmethod
    def _split_equation(equation: str) -> tuple[str, str] | None:
        parts = re.split(r"(?<![<>!])=(?!=)", str(equation), maxsplit=1)
        return (parts[0], parts[1]) if len(parts) == 2 and all(part.strip() for part in parts) else None

    def _parse_answer_items(self, answer: object) -> List[str]:
        text = DEFAULT_NORMALIZER.clean(answer).replace("\\{", "{").replace("\\}", "}")
        if len(text) >= 2 and text[0] in "{[" and text[-1] in "}]":
            text = text[1:-1]
        return [item.strip() for item in re.split(r"[,;]", text) if item.strip()]

    @staticmethod
    def _resolve_symbol(expression: Any, name: str) -> Any:
        for symbol in expression.free_symbols:
            if str(symbol) == name:
                return symbol
        return sp.Symbol(name, finite=True)

    @staticmethod
    def _finite_number(value: float) -> bool:
        return value == value and value not in {float("inf"), float("-inf")}

    @staticmethod
    def _is_sympy_expr(value: Any) -> bool:
        return sp is not None and isinstance(value, sp.Basic)

    @staticmethod
    def _unknown(check: str, message: str, samples: int = 0) -> VerificationResult:
        return VerificationResult(UNKNOWN, check, 0.0, message, samples=samples)
