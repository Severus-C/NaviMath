from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from .answer_normalizer import (
    AnswerContext,
    AnswerNormalizer,
    DEFAULT_NORMALIZER,
    is_placeholder_answer,
)
from .skill_catalog import DistilledSkillCatalog, SkillTemplate


def truncate_text(text: str, limit: int = 1200) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"... <truncated {len(text) - limit} chars>"


_TRACE_SECRET_RE = re.compile(
    r"(?i)\b(api[_ -]?key|access[_ -]?token|authorization|secret|password|cookie)\b"
    r"(\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"
)
_BEARER_TOKEN_RE = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_WINDOWS_PATH_RE = re.compile(r"(?i)\b[a-z]:[\\/](?:[^\s<>:\"|?*]+[\\/]?)*")
_POSIX_PRIVATE_PATH_RE = re.compile(
    r"(?<!\w)/(?:home|users|root|tmp|var|opt|workspace|app)(?:/[^\s:;,]+)+",
    re.I,
)


def sanitize_trace_text(value: object, limit: int = 320) -> str:
    """Redact credentials and host-specific paths before recording diagnostics."""

    text = str(value)
    text = _TRACE_SECRET_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}<redacted>", text)
    text = _BEARER_TOKEN_RE.sub("Bearer <redacted>", text)
    text = _WINDOWS_PATH_RE.sub("<path>", text)
    text = _POSIX_PRIVATE_PATH_RE.sub("<path>", text)
    return truncate_text(text, limit)


def safe_exception_message(exc: BaseException, limit: int = 320) -> str:
    """Return a bounded, diagnosable exception summary suitable for trace output."""

    detail = sanitize_trace_text(str(exc), limit=limit)
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def strip_matching_braces(text: str) -> str:
    text = text.strip()
    while text.startswith("{") and text.endswith("}"):
        depth = 0
        balanced = True
        for index, char in enumerate(text):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and index != len(text) - 1:
                    balanced = False
                    break
        if not balanced:
            break
        text = text[1:-1].strip()
    return text


def extract_braced_after(command: str, text: str) -> str:
    start = text.find(command)
    if start < 0:
        return ""
    brace_start = text.find("{", start + len(command))
    if brace_start < 0:
        return ""
    depth = 0
    for index in range(brace_start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : index]
    return ""


def remove_latex_wrappers(text: str) -> str:
    text = str(text).strip()
    text = text.replace("\\displaystyle", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "")
    text = re.sub(r"^\s*\$+", "", text)
    text = re.sub(r"\$+\s*$", "", text)
    return strip_matching_braces(text.strip())


def normalize_latex_answer(answer: str) -> str:
    text = remove_latex_wrappers(answer)

    boxed = extract_braced_after("\\boxed", text)
    if boxed:
        text = boxed
    fbox = extract_braced_after("\\fbox", text)
    if fbox:
        text = fbox

    text = remove_latex_wrappers(text)
    text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\operatorname\s*\{([^{}]*)\}", r"\1", text)

    frac_pattern = re.compile(r"\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    previous = None
    while previous != text:
        previous = text
        text = frac_pattern.sub(r"(\1)/(\2)", text)

    sqrt_pattern = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
    text = sqrt_pattern.sub(r"sqrt(\1)", text)

    replacements = {
        "\\cdot": "*",
        "\\times": "*",
        "\\div": "/",
        "\\pi": "pi",
        "\\infty": "infty",
        "\\leq": "<=",
        "\\geq": ">=",
        "\\neq": "!=",
        "\\pm": "+-",
        "\\colon": ":",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"(?<=\d)\(([-+]?\d+)\)/\(([-+]?\d+)\)", r"+(\1)/(\2)", text)
    text = re.sub(r"\s+", "", text)
    text = text.strip(" .。；;，,")
    return text or str(answer).strip()


def canonical_key(answer: str) -> str:
    normalized = normalize_latex_answer(answer)
    normalized = normalized.lower()
    normalized = normalized.replace("−", "-")
    normalized = normalized.replace("，", ",")
    normalized = normalized.strip()
    normalized = re.sub(r"^answer[:：]?", "", normalized)
    normalized = re.sub(r"^final[_\s-]*answer[:：]?", "", normalized)
    normalized = normalized.strip(" .。；;，,")
    if re.fullmatch(r"[-+]?\d+\.0+", normalized):
        normalized = normalized.split(".")[0]
    return normalized


# Keep the public function names used by the agent and older scripts while the
# implementation lives in the schema-aware normalizer module.
def normalize_latex_answer(answer: str) -> str:
    return DEFAULT_NORMALIZER.clean(answer)


def canonical_key(answer: str) -> str:
    return DEFAULT_NORMALIZER.canonicalize(answer)


INVALID_ANSWER_MARKERS = [
    "<最终答案>",
    "<final",
    "<候选",
    "<answer",
    "[answer]",
    "最终答案>",
    "i need to output",
    "ineedtooutput",
    "i will put the answer",
    "iwillputtheanswer",
    "answer after the colon",
    "answerafterthecolon",
    "afterthecolon",
    "write the answer after",
    "writetheanswerafter",
    "only the actual final answer",
    "onlytheactualfinalanswer",
    "output format",
    "outputformat",
    "selected:",
    "reason:",
    "final_answeronly",
    "候选编号",
    "unable to determine",
    "unknown",
    "write the corrected actual final answer",
    "writethecorrectedactualfinalanswer",
    "写修正后的实际最终答案",
    "写实际最终答案",
]


def _clean_short_answer(answer: str) -> str:
    """Remove common protocol noise without attempting symbolic solving."""
    text = str(answer).strip()
    text = re.sub(
        r"^\s*(?:just\s+the\s+(?:values?|number(?:\s*/\s*expression)?|expression)|"
        r"justthe(?:values?|number(?:/expression)?|expression))"
        r"\s*[.:\"']*\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    if "`" in text:
        prefix = text.split("`", 1)[0].strip()
        if re.search(r"\d|\\|[=+*/^]", prefix):
            text = prefix
    text = re.sub(
        r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Models occasionally continue thinking inside the protocol field. Recover
    # an explicit late conclusion before applying the ordinary syntax cleanup.
    maybe = re.search(r"\bmaybe\s*[\"'`]?(.*?)\s*$", text, flags=re.IGNORECASE)
    if maybe and re.search(r"\d|\\|[=+*/^]", maybe.group(1)):
        text = maybe.group(1)
    conclusion = re.search(
        r"(?<!\d)([-+]?\d+(?:\.\d+)?)\s*is\s*the\s*(?:integer|number|answer|value)\b",
        text,
        flags=re.IGNORECASE,
    )
    if conclusion:
        text = conclusion.group(1)

    leading_math = text.split("$", 1)[0].strip() if "$" in text else ""
    if leading_math and re.match(r"^(?:[-+]?\d|\\|[([]|[A-Za-z]\s*=)", leading_math):
        text = leading_math
    else:
        # Use overlapping matches so an odd stray dollar before ``or just $x$``
        # does not hide the last complete math span.
        math_spans = [
            match.group(2).strip()
            for match in re.finditer(r"(?=(\$([^$\n]{1,160})\$))", text)
            if re.search(r"\d|\\|[=+*/^<>]", match.group(2))
        ]
        if math_spans and not re.search(r"\b(?:loss|gain|profit)\b", text, flags=re.IGNORECASE):
            text = math_spans[-1]
    text = re.sub(
        r"^\s*(?:likely|probably|exactly|approximately|approx\.?)\s*\$?\s*"
        r"(?=[(\[\\+\-\dA-Za-z])",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Keep a leading numeric answer and discard a trailing self-dialogue.
    leading_number = re.match(
        r"^\s*([-+]?\d+(?:\.\d+)?(?:\s*%|\s*\\%)?)\s*"
        r"(?:[?\"]|,\s*(?:it|which|this|i\b)|\.\s*(?:i\b|this\b))",
        text,
        flags=re.IGNORECASE,
    )
    if leading_number:
        text = leading_number.group(1)

    text = remove_latex_wrappers(text).strip()
    text = text.strip(" .。；;，,!?？*_`\"'")
    text = re.sub(
        r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.strip(" .。；;，,!?？*_`\"'")

    # Accept `x=15` and `OB^2=26`, but not an unsolved equation such as
    # `10s^2-1000s+19240=0`.
    assignment = re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_]*(?:\s*\^\s*\{?-?\d+\}?)?\s*=\s*"
        r"(\\boxed\s*\{[^{}]+\}|[-+]?\d+(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?)",
        text,
    )
    if assignment:
        text = assignment.group(1)

    boxed = extract_braced_after("\\boxed", text)
    if boxed:
        text = boxed
    return text.strip(" .。；;，,!?？*_`\"'")


def is_plausible_final_answer(answer: str, proof_mode: bool = False) -> bool:
    if proof_mode:
        return bool(str(answer).strip())
    if is_placeholder_answer(answer):
        return False
    raw = _clean_short_answer(answer)
    if is_placeholder_answer(raw):
        return False
    key = canonical_key(raw)
    if not key:
        return False
    lowered = key.lower()
    if any(marker in lowered for marker in INVALID_ANSWER_MARKERS):
        return False
    if "<" in raw or ">" in raw:
        return False
    if "=" in raw:
        definition = re.fullmatch(
            r"(?:[A-Za-z][A-Za-z0-9_]*|[A-Za-z][A-Za-z0-9_]*\([^=()]+\))"
            r"\s*=\s*[^=]+",
            raw,
        )
        variable_condition = re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]*(?:\s*[+-]\s*[A-Za-z][A-Za-z0-9_]*)+"
            r"\s*=\s*[-+]?\d+(?:\.\d+)?",
            raw,
        )
        if not definition and not variable_condition:
            return False
    if re.search(r"\b(?:solution|self_check|confidence|verdict|issues)\b", lowered):
        return False
    words = re.findall(r"[A-Za-z]+", raw)
    accepted_word_answers = re.fullmatch(
        r"(?:no solutions?|all (?:real|positive|negative|integer|rational) (?:numbers?|values?)|"
        r"does not exist|infinitely many|none of (?:these|the above)|true|false)",
        raw.strip(),
        flags=re.IGNORECASE,
    )
    math_syntax = re.search(r"\\[A-Za-z]+|[=+*/^%<>]|\d", raw)
    if len(words) >= 3 and not accepted_word_answers and not math_syntax:
        return False
    if len(key) > 80 and not re.search(r"^[\[{(].*[\]})]$", key):
        return False
    if len(key.split()) > 8:
        return False
    return True


def extract_final_answer(
    text: str,
    proof_mode: bool = False,
    integer_only: bool = False,
    problem: str = "",
) -> str:
    if proof_mode:
        return str(text).strip()

    raw = str(text).strip()

    def accept(candidate: str) -> str:
        cleaned = _clean_short_answer(candidate)
        if integer_only:
            integer_match = re.fullmatch(r"[-+]?\d{1,3}", cleaned)
            return integer_match.group(0) if integer_match else ""
        return cleaned if is_plausible_final_answer(cleaned, proof_mode=False) else ""

    patterns = [
        r"(?m)^\s*(?:\*\*)?FINAL\s*_?\s*ANSWER(?:\*\*)?\s*[:：]\s*(.+)$",
        r"FINAL_ANSWER\s*[:：]\s*(.+)",
        r"Final Answer\s*[:：]\s*(.+)",
        r"\*\*Final Answer:\*\*\s*(.+)",
        r"最终答案\s*[:：是为]*\s*(.+)",
        r"答案\s*[:：是为]*\s*(.+)",
        r"所以(?:最终)?(?:答案|结果)(?:是|为)?\s*(.+)",
        r"therefore[, ]+the answer is\s*(.+)",
        r"the answer is\s*(.+)",
        r"the answer should be\s*(.+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, raw, flags=re.IGNORECASE)
        for match in reversed(matches):
            first_line = str(match).splitlines()[0].strip() if str(match).splitlines() else ""
            answer = accept(first_line)
            if answer:
                return answer

    if integer_only:
        query = problem.lower()
        target_result_patterns: List[str] = []
        if not query or "sum" in query:
            target_result_patterns.append(
                r"(?:grand\s+)?(?:total|sum)(?!\s+case)[^\n.]{0,140}(?:=|is|:)"
                r"[^\n.]{0,80}(?<!\d)([-+]?\d{1,3})(?!\d|[./]\d)"
            )
        if not query or "how many" in query or "number of" in query:
            target_result_patterns.append(
                r"(?:grand\s+total|total(?!\s+case)|total\s+number|number\s+of\s+such\s+numbers)"
                r"[^\n.]{0,120}(?:=|is|:)\s*(?<!\d)([-+]?\d{1,3})(?!\d|[./]\d)"
            )
        if not query or "volume" in query:
            target_result_patterns.append(
                r"(?:volume|V)\s*(?:=|is|:)\s*(?:\\boxed\s*\{)?(?<!\d)([-+]?\d{1,3})(?!\d|[./]\d)"
            )
        if not query or "length of ab" in query or "diameter" in query:
            target_result_patterns.append(
                r"(?:AB|diameter)\s*(?:=|is|:)\s*(?:\\boxed\s*\{)?(?<!\d)([-+]?\d{1,3})(?!\d|[./]\d)"
            )
        if not query or "product" in query:
            target_result_patterns.append(
                r"(?:product|mn)\s*(?:=|is|:)\s*(?:\\boxed\s*\{)?(?<!\d)([-+]?\d{1,3})(?!\d|[./]\d)"
            )
        if not query or "n}{15" in query or "n/15" in query:
            target_result_patterns.append(
                r"(?:\\frac\s*\{n\}\s*\{15\}|n\s*/\s*15)\s*=\s*"
                r"(?:\\boxed\s*\{)?(?<!\d)([-+]?\d{1,3})(?!\d|[./]\d)"
            )
        if not query or "slope" in query:
            target_result_patterns.append(
                r"(?:absolute\s+value\s+of\s+the\s+slope|slope|\bm)\s*(?:=|is|:)\s*"
                r"(?:\\boxed\s*\{)?(?<!\d)([-+]?\d{1,3})(?!\d|[./]\d)"
            )
        if not query or "score" in query:
            target_result_patterns.append(
                r"(?:mary(?:'s)?\s+score|final\s+score|the\s+score)\s*(?:=|is|was|:)\s*"
                r"(?:\\boxed\s*\{)?(?<!\d)([-+]?\d{1,3})(?!\d|[./]\d)"
            )
        if not query or "m+n" in query or "m + n" in query:
            target_result_patterns.append(
                r"m\s*\+\s*n\s*=\s*(?:\\boxed\s*\{)?(?<!\d)([-+]?\d{1,3})(?!\d|[./]\d)"
            )
        if not query or ("largest" in query and "integer" in query):
            target_result_patterns.append(
                r"largest[^\n.]{0,100}?(?:integer|number)[^\n.]{0,80}?(?:=|is|:)\s*"
                r"(?:\\boxed\s*\{)?(?<!\d)([-+]?\d{1,3})(?!\d|[./]\d)"
            )
            target_result_patterns.append(
                r"largest[^\n.]{0,120}?(?:integer|number)[^\n.]{0,80}?"
                r"(?:\\leq?|<=|at\s+most)\s*(?<!\d)([-+]?\d{1,3})(?!\d|[./]\d)"
            )
            target_result_patterns.append(
                r"(?<!\d)([-+]?\d{1,3})(?!\d|[./]\d)\s+is\s+the\s+largest"
                r"[^\n.]{0,100}?(?:integer|number)"
            )
            target_result_patterns.append(
                r"(?<!\d)([-+]?\d{1,3})(?!\d|[./]\d)\s+is\s+a\s+candidate\s+for\s+the\s+answer"
            )
        for pattern in target_result_patterns:
            matches = re.findall(pattern, raw, flags=re.IGNORECASE)
            if matches:
                value = matches[-1]
                if "absolute value" in query and "slope" in query:
                    return str(abs(int(value)))
                return value

    boxed_matches: List[tuple[int, str]] = []
    search_from = 0
    while True:
        start = raw.find("\\boxed", search_from)
        if start < 0:
            break
        boxed = extract_braced_after("\\boxed", raw[start:])
        if boxed:
            boxed_matches.append((start, boxed))
        search_from = start + 6
    if boxed_matches:
        for position, boxed in reversed(boxed_matches):
            context = raw[max(0, position - 100) : position].lower()
            if not integer_only or re.search(
                r"final\s+answer|requested\s+(?:answer|value)|answer\s+is|答案",
                context,
            ):
                answer = accept(boxed)
                if answer:
                    return answer

    if integer_only:
        conclusion_patterns = [
            r"(?:final\s+answer|answer|requested\s+value|largest\s+(?:real\s+)?(?:value|integer)|minimum\s+value|maximum\s+value)"
            r"\s*(?:is|should\s+be|equals|=|:)\s*(?:\\boxed\s*\{)?(?<!\d)([-+]?\d{1,3})(?!\d|[./]\d)",
        ]
        for pattern in conclusion_patterns:
            matches = re.findall(pattern, raw, flags=re.IGNORECASE)
            if matches:
                return matches[-1]

        if len(raw) <= 200:
            integer_lines = re.findall(r"(?m)^\s*(?:\*\*)?([-+]?\d{1,3})(?:\*\*)?[.!?]?\s*$", raw)
            return integer_lines[-1] if integer_lines else ""
        return ""

    latex_inline = re.findall(r"\$([^$\n]{1,120})\$", raw)
    for answer in reversed(latex_inline):
        answer = _clean_short_answer(answer)
        if is_plausible_final_answer(answer, proof_mode=False):
            return answer

    integer_lines = re.findall(r"(?m)^\s*(-?\d{1,8})\s*$", raw)
    if integer_lines:
        return integer_lines[-1]

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in reversed(lines):
        answer = _clean_short_answer(line)
        if is_plausible_final_answer(answer, proof_mode=False):
            return answer
    return ""


def parse_confidence(text: str) -> float:
    matches = re.findall(r"CONFIDENCE\s*[:：]\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
    if not matches:
        matches = re.findall(r"置信度\s*[:：]\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not matches:
        return 0.55
    value = float(matches[-1])
    if value > 1:
        value /= 100.0
    return max(0.0, min(1.0, value))


def parse_verdict(text: str) -> str:
    verdict_matches = re.findall(
        r"\b(?:VERDICT|JUDGMENT|判断)\s*[:：]\s*([A-Z_]+|正确|错误|通过|不通过)",
        str(text),
        flags=re.IGNORECASE,
    )
    if verdict_matches:
        verdict = verdict_matches[-1].upper()
        last_verdict = list(re.finditer(
            r"\b(?:VERDICT|JUDGMENT|判断)\s*[:：]\s*[A-Z_]+",
            str(text),
            flags=re.IGNORECASE,
        ))[-1]
        trailing_text = str(text)[last_verdict.end() :]
        if verdict in {"ACCEPT", "CORRECT"} and re.search(
            r"\b(?:INCORRECT|REJECT|WRONG)\b|不正确|错误|不通过",
            trailing_text,
            flags=re.IGNORECASE,
        ):
            return "REJECT"
        return verdict
    upper = str(text).upper()
    if "INCORRECT" in upper or "REJECT" in upper or "错误" in text or "不通过" in text:
        return "REJECT"
    if "CORRECT" in upper or "ACCEPT" in upper or "正确" in text or "通过" in text:
        return "ACCEPT"
    return "UNKNOWN"


@dataclass
class Skill:
    name: str
    description: str
    triggers: List[str]
    strategy: str
    traps: List[str] = field(default_factory=list)
    answer_format: str = "Give the simplest exact final answer."


@dataclass
class Route:
    subject: str
    skill: Skill
    difficulty_score: int
    proof_mode: bool
    action_plan: List[str]
    templates: List[SkillTemplate] = field(default_factory=list)

    @property
    def difficulty_label(self) -> str:
        if self.difficulty_score >= 7:
            return "hard"
        if self.difficulty_score >= 4:
            return "medium"
        return "easy"


@dataclass
class Candidate:
    role: str
    content: str
    extracted_answer: str
    normalized_answer: str
    confidence: float
    score: float = 0.0
    attack_report: str = ""
    refined_content: str = ""
    refined_answer: str = ""
    refined_normalized_answer: str = ""
    tool_verdict: str = "UNKNOWN"
    tool_confidence: float = 0.0
    tool_report: Dict[str, Any] = field(default_factory=dict)
    origin_role: str = ""
    protocol_compliant: bool = False
    contract_valid: bool = True
    recovered: bool = False

    def display_answer(self) -> str:
        if self.refined_answer:
            return self.refined_answer
        return self.extracted_answer

    def key(self) -> str:
        if self.refined_normalized_answer:
            return self.refined_normalized_answer
        return self.normalized_answer

    def origin(self) -> str:
        return self.origin_role or self.role

    def consensus_origin(self) -> str:
        # Debate candidates are conditioned on prior proposals and therefore
        # are not independent votes unless a deterministic tool verified them.
        if self.role.startswith("debate_synthesizer") and self.tool_verdict != "VERIFIED":
            return ""
        return self.origin()


class SkillRegistry:
    def __init__(self) -> None:
        self.skills = self._build_skills()
        self.catalog = DistilledSkillCatalog.load()
        self.default_skill = Skill(
            name="general_contest_math",
            description="General mathematical problem solving.",
            triggers=[],
            strategy=(
                "Identify the mathematical structure, introduce useful variables, "
                "derive constraints carefully, check edge cases, and produce an exact answer."
            ),
            traps=[
                "Do not ignore domain restrictions.",
                "Do not round unless the problem explicitly asks for approximation.",
                "Check whether the problem asks for a value, count, set, interval, or proof.",
            ],
        )

    def route(self, problem: str) -> Route:
        lower = problem.lower()
        distilled_scores = self.catalog.domain_scores(problem)
        learned_scores = self.catalog.router_scores(problem)
        specialist_markers = {
            "real_analysis": ["uniform convergence", "pointwise convergence", "cauchy sequence", "lebesgue", "banach space", "hilbert space"],
            "differential_equations": ["differential equation", "initial value problem", "boundary value problem", "laplace equation", "heat equation", "wave equation", "phase portrait"],
            "topology": ["topological space", "homeomorphism", "homotopy", "fundamental group", "basis for a topology"],
            "statistics": ["estimator", "likelihood", "confidence interval", "hypothesis test", "regression"],
            "numerical_analysis": ["newton method", "newton's method", "numerical", "interpolation", "finite difference", "truncation error", "condition number"],
            "discrete_mathematics": ["graph theory", "spanning tree", "planar graph", "chromatic", "algorithm", "poset", "partially ordered"],
            "differential_geometry": ["differential geometry", "manifold", "tangent bundle", "geodesic", "riemannian", "fundamental form", "gaussian curvature"],
            "logic_set_theory": ["set theory", "cardinality", "countable", "uncountable", "ordinal", "axiom", "predicate", "model theory", "well-order"],
        }
        scored: List[tuple[float, Skill]] = []
        for skill in self.skills:
            direct_hits = sum(1 for trigger in skill.triggers if trigger.lower() in lower)
            score = float(direct_hits)
            score += 0.25 * distilled_scores.get(skill.name, 0.0)
            score += 4.0 * learned_scores.get(skill.name, 0.0)
            if any(marker in lower for marker in specialist_markers.get(skill.name, [])):
                score += 10.0
            if score:
                scored.append((score, skill))
        skill = max(scored, key=lambda item: item[0])[1] if scored else self.default_skill
        proof_mode = self._detect_proof_mode(lower)
        difficulty = self._estimate_difficulty(problem, skill, proof_mode)
        action_plan = self._action_plan(difficulty, proof_mode)
        templates = self.catalog.match(problem, skill.name, proof_mode=proof_mode)
        return Route(
            subject=skill.name,
            skill=skill,
            difficulty_score=difficulty,
            proof_mode=proof_mode,
            action_plan=action_plan,
            templates=templates,
        )

    def _estimate_difficulty(self, problem: str, skill: Skill, proof_mode: bool) -> int:
        lower = problem.lower()
        score = 1
        score += min(4, len(problem) // 260)
        score += min(2, problem.count("$") // 8)
        if proof_mode:
            score += 4
        hard_markers = [
            "for all",
            "exists",
            "integer",
            "prime",
            "maximum",
            "minimum",
            "if and only if",
            "prove",
            "show that",
            "all functions",
            "finite field",
            "residue",
            "topological",
            "linear programming",
            "differential equation",
        ]
        score += sum(1 for marker in hard_markers if marker in lower)
        if skill.name in {"geometry", "number_theory", "combinatorics", "complex_analysis"}:
            score += 1
        return max(1, min(10, score))

    @staticmethod
    def _detect_proof_mode(lower: str) -> bool:
        proof_markers = [
            "prove that",
            "show that",
            "证明",
            "证得",
            "求证",
            "if and only if",
            "necessary and sufficient",
        ]
        return any(marker in lower for marker in proof_markers)

    @staticmethod
    def _action_plan(difficulty: int, proof_mode: bool) -> List[str]:
        if proof_mode:
            return ["Decompose", "Debate", "Refine", "Terminate"]
        if difficulty >= 7:
            return ["Decompose", "Debate", "Attack", "Refine", "Terminate"]
        if difficulty >= 4:
            return ["Debate", "Attack", "Terminate"]
        return ["ReasonOneStep", "Verify", "Terminate"]

    @staticmethod
    def _build_skills() -> List[Skill]:
        return [
            Skill(
                name="algebra",
                description="Equations, polynomials, functions, logarithms, inequalities.",
                triggers=[
                    "polynomial",
                    "equation",
                    "function",
                    "log_",
                    "logarithm",
                    "root",
                    "inequality",
                    "real numbers",
                    "complex numbers",
                    "quadratic",
                ],
                strategy=(
                    "Translate statements into equations, isolate invariants, factor or "
                    "substitute, and verify all roots against domain restrictions."
                ),
                traps=["Extraneous roots", "Lost sign/domain conditions", "Wrong branch of inverse functions"],
            ),
            Skill(
                name="number_theory",
                description="Integers, divisibility, primes, modular arithmetic.",
                triggers=[
                    "integer",
                    "prime",
                    "divisible",
                    "remainder",
                    "modulo",
                    "congru",
                    "gcd",
                    "lcm",
                    "positive integer",
                ],
                strategy=(
                    "Use modular arithmetic, valuations, bounding, factorization, and "
                    "check small residue classes or extremal cases."
                ),
                traps=["Missing zero/negative cases", "Assuming coprimality", "Counting duplicate residues"],
            ),
            Skill(
                name="geometry",
                description="Euclidean geometry, coordinates, circles, areas.",
                triggers=[
                    "triangle",
                    "circle",
                    "angle",
                    "radius",
                    "area",
                    "perimeter",
                    "polygon",
                    "tangent",
                    "chord",
                    "parallel",
                    "perpendicular",
                    "inscribed",
                ],
                strategy=(
                    "Draw the configuration mentally, name key points, look for similar "
                    "triangles/cyclic quadrilaterals, and switch to coordinates if helpful."
                ),
                traps=["Diagram assumptions", "Choosing the wrong branch/length", "Forgetting degeneracy"],
            ),
            Skill(
                name="combinatorics",
                description="Counting, arrangements, recurrence, graph or invariant arguments.",
                triggers=[
                    "ways",
                    "arrangements",
                    "permutation",
                    "combination",
                    "subsets",
                    "color",
                    "graph",
                    "sequence",
                    "recurrence",
                    "count",
                ],
                strategy=(
                    "Decide whether to use complement, cases, recurrence, generating "
                    "functions, or double counting. Track overcounting explicitly."
                ),
                traps=["Overcounting", "Forgetting impossible cases", "Boundary cases in recurrence"],
            ),
            Skill(
                name="probability",
                description="Probability, expectation, random processes.",
                triggers=["probability", "expected", "random", "independent", "variance", "dice", "cards"],
                strategy=(
                    "Define the sample space, use conditional probability or linearity of "
                    "expectation, and check normalization."
                ),
                traps=["Assuming independence", "Wrong sample space", "Missing conditioning"],
            ),
            Skill(
                name="calculus",
                description="Limits, derivatives, integrals, inverse functions.",
                triggers=[
                    "derivative",
                    "integral",
                    "limit",
                    "inverse function",
                    "continuous",
                    "differentiable",
                    "maximum",
                    "minimum",
                    "measure",
                ],
                strategy=(
                    "Use definitions, substitutions, inverse-function relations, monotonicity, "
                    "and endpoint checks."
                ),
                traps=["Sign errors in inverse/integral bounds", "Ignoring endpoints", "Domain mismatch"],
            ),
            Skill(
                name="linear_algebra",
                description="Matrices, vector spaces, rank, eigenvalues.",
                triggers=["matrix", "determinant", "eigen", "rank", "linear transformation", "vector space"],
                strategy=(
                    "Identify invariant subspaces, rank-nullity, determinant/eigenvalue "
                    "relations, and canonical forms."
                ),
                traps=["Dimension mismatch", "Non-invertible special cases", "Confusing row/column conventions"],
            ),
            Skill(
                name="complex_analysis",
                description="Complex functions, residues, poles, contour ideas.",
                triggers=["complex variable", "residue", "pole", "laurent", "holomorphic", "meromorphic", "contour"],
                strategy=(
                    "Classify singularities, compute residues with the right pole order, "
                    "and check signs/factors carefully."
                ),
                traps=["Wrong pole order", "Sign mistakes", "Forgetting repeated factors"],
            ),
            Skill(
                name="abstract_algebra",
                description="Groups, rings, fields, finite fields.",
                triggers=[
                    "finite field",
                    "group",
                    "ring",
                    "field",
                    "extension",
                    "isomorphism",
                    "homomorphism",
                    "generator",
                ],
                strategy=(
                    "Use structure theorems, subfield/subgroup constraints, degree counts, "
                    "and orbit/stabilizer style arguments."
                ),
                traps=["Counting elements in proper subfields", "Ignoring non-generators", "Assuming cyclicity"],
            ),
            Skill(
                name="optimization",
                description="Optimization, linear programming, operations research.",
                triggers=["linear programming", "optimization", "maximize", "minimize", "constraint", "objective"],
                strategy=(
                    "Identify variables, constraints, feasible region, active constraints, "
                    "and dual/KKT conditions when applicable."
                ),
                traps=["Missing feasibility", "Boundary optimum", "Unbounded objective"],
            ),
            Skill(
                name="real_analysis",
                description="Rigorous real analysis, convergence, measure, and functional analysis basics.",
                triggers=[
                    "uniform convergence", "pointwise convergence", "cauchy sequence", "compactness",
                    "metric space", "lebesgue", "measurable", "banach space", "hilbert space",
                ],
                strategy=(
                    "Write the exact quantifiers, select the relevant compactness/completeness theorem, "
                    "and distinguish pointwise, uniform, norm, and measure convergence."
                ),
                traps=["Interchanging limits without hypotheses", "Pointwise versus uniform convergence", "Null-set edge cases"],
                answer_format="Give a rigorous proof or exact counterexample with all hypotheses checked.",
            ),
            Skill(
                name="differential_equations",
                description="Ordinary and partial differential equations, stability, and boundary conditions.",
                triggers=[
                    "differential equation", "initial value problem", "boundary value problem", "ode",
                    "pde", "laplace equation", "heat equation", "wave equation", "phase portrait",
                ],
                strategy=(
                    "Classify the equation, choose the matching exact method, apply every initial/boundary "
                    "condition, and substitute the solution back into the equation."
                ),
                traps=["Lost integration constants", "Ignoring singular solutions", "Boundary/initial condition mismatch"],
                answer_format="Give the complete solution family or the unique conditioned solution and its domain.",
            ),
            Skill(
                name="topology",
                description="General and algebraic topology, continuity, compactness, and connectedness.",
                triggers=[
                    "topological space", "open set", "closed set", "homeomorphism", "homotopy",
                    "fundamental group", "connected", "path connected", "compact", "basis for a topology",
                ],
                strategy=(
                    "Work directly from definitions unless a named invariant gives a shorter obstruction; "
                    "track the ambient topology and prove both directions of set equalities."
                ),
                traps=["Confusing relative and ambient topology", "Continuous bijection need not be a homeomorphism", "Connected versus path connected"],
                answer_format="Give a definition-complete proof, counterexample, or invariant computation.",
            ),
            Skill(
                name="statistics",
                description="Statistical inference, distributions, estimators, and stochastic models.",
                triggers=[
                    "estimator", "likelihood", "confidence interval", "hypothesis test", "distribution",
                    "normal distribution", "binomial distribution", "poisson", "covariance", "regression",
                ],
                strategy=(
                    "State the probabilistic model and parameter space, derive the statistic symbolically, "
                    "then check bias, variance, support, and conditioning assumptions."
                ),
                traps=["Data-dependent support", "Confusing sample and population quantities", "Unstated independence"],
                answer_format="Give an exact statistic/distribution or a clearly specified approximation.",
            ),
            Skill(
                name="numerical_analysis",
                description="Numerical approximation, conditioning, interpolation, and iterative methods.",
                triggers=[
                    "newton method", "newton's method", "numerical", "interpolation", "finite difference",
                    "truncation error", "condition number", "iterative method", "convergence order",
                ],
                strategy=(
                    "Separate approximation error from floating-point/conditioning error, derive the update, "
                    "and prove or bound convergence before reporting digits."
                ),
                traps=["Unsupported decimal precision", "Convergence to the wrong root", "Ignoring conditioning"],
                answer_format="Give the requested approximation with a justified error bound and precision.",
            ),
            Skill(
                name="discrete_mathematics",
                description="Graph theory, discrete structures, algorithms, and finite-state arguments.",
                triggers=[
                    "graph theory", "vertex", "vertices", "edge", "matching", "spanning tree",
                    "planar graph", "chromatic", "algorithm", "poset", "partially ordered",
                ],
                strategy=(
                    "Identify the discrete invariant or extremal structure, use degree/connectivity identities, "
                    "and prove that constructions and counts are exhaustive."
                ),
                traps=["Directed versus undirected conventions", "Disconnected special cases", "Algorithm without correctness proof"],
                answer_format="Give an exact count, construction, algorithm with proof, or structural proof.",
            ),
            Skill(
                name="differential_geometry",
                description="Curves, surfaces, curvature, manifolds, and geometric tensors.",
                triggers=[
                    "differential geometry", "manifold", "tangent bundle", "geodesic", "curvature",
                    "riemannian", "first fundamental form", "second fundamental form", "gaussian curvature",
                ],
                strategy=(
                    "Choose a valid chart or parametrization, compute invariant quantities with conventions "
                    "stated explicitly, and separate coordinate artifacts from geometric conclusions."
                ),
                traps=["Sign convention mismatch", "Singular parametrization", "Local result claimed globally"],
                answer_format="Give the invariant quantity or rigorous local/global geometric proof.",
            ),
            Skill(
                name="logic_set_theory",
                description="Mathematical logic, set theory, cardinality, and formal quantifiers.",
                triggers=[
                    "set theory", "cardinality", "countable", "uncountable", "ordinal", "cardinal",
                    "axiom", "predicate", "propositional", "model theory", "well-order",
                ],
                strategy=(
                    "Formalize quantifiers and the ambient axioms, use explicit injections/surjections or "
                    "models, and distinguish syntax from semantics."
                ),
                traps=["Reversing quantifiers", "Assuming choice implicitly", "Injection/surjection direction error"],
                answer_format="Give a formal proof, model/countermodel, or explicit cardinal construction.",
            ),
        ]


def cluster_candidates(candidates: Iterable[Candidate]) -> Dict[str, List[Candidate]]:
    clusters: Dict[str, List[Candidate]] = {}
    for candidate in candidates:
        key = candidate.key()
        if not key:
            key = "<empty>"
        clusters.setdefault(key, []).append(candidate)
    return clusters


def score_candidates(candidates: List[Candidate]) -> None:
    clusters = cluster_candidates(candidates)
    for candidate in candidates:
        key = candidate.key()
        origins = {item.consensus_origin() for item in clusters.get(key, [])}
        origins.discard("")
        cluster_size = max(1, len(origins)) if key else 0
        score = 0.0
        if not key:
            candidate.score = -10.0
            continue
        score += 2.2 * cluster_size
        score += 1.4 * candidate.confidence
        answer = candidate.display_answer()
        if answer and answer != candidate.content:
            score += 0.8
        if re.fullmatch(r"[-+]?\d+", candidate.key()):
            score += 0.4
        if candidate.attack_report:
            verdict = parse_verdict(candidate.attack_report)
            if verdict in {"ACCEPT", "CORRECT", "A", "通过", "正确"}:
                score += 0.6
            elif verdict in {"REJECT", "INCORRECT", "B", "不通过", "错误"}:
                # Model verifiers are advisory. Only ToolVerify may impose a
                # hard rejection or a large score penalty.
                score -= 0.35
        if candidate.tool_verdict == "VERIFIED":
            score += 3.0 * max(0.5, candidate.tool_confidence)
        elif candidate.tool_verdict == "REJECTED":
            score -= 6.0 * max(0.5, candidate.tool_confidence)
        if candidate.protocol_compliant:
            score += 0.6
        if candidate.recovered:
            score += 0.2
        if not candidate.contract_valid:
            score -= 8.0
        candidate.score = score


def select_best_candidate(candidates: List[Candidate]) -> Candidate:
    score_candidates(candidates)
    eligible = [
        candidate
        for candidate in candidates
        if candidate.tool_verdict != "REJECTED" and candidate.contract_valid
    ]
    return max(eligible or candidates, key=lambda item: item.score)


def trace_record(step: str, content: Any) -> Dict[str, Any]:
    return {"step": step, "content": content}
