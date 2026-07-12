"""Build a small public math-contest dataset as JSONL.

The first source adapter targets AoPS Wiki AIME pages because AIME problems
have short canonical answers and are useful for objective agent evaluation.

Please verify each source's terms before redistributing downloaded content.
The script is intended for local research/evaluation dataset preparation.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import quote
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


AOPS_BASE = "https://artofproblemsolving.com/wiki/index.php"
HARP_ZIP_URL = "https://github.com/aadityasingh/HARP/raw/main/HARP.jsonl.zip"
HARP_PROOF_ZIP_URL = (
    "https://github.com/aadityasingh/HARP/raw/main/HARP_proof-based.jsonl.zip"
)
HARP_REPO_URL = "https://github.com/aadityasingh/HARP"
DEFAULT_USER_AGENT = (
    "NaviMathDatasetBuilder/0.1 "
    "(local research dataset preparation; contact: local-user)"
)


@dataclass(frozen=True)
class AimeSource:
    year: int
    contest: str

    @property
    def page_prefix(self) -> str:
        return f"{self.year}_AIME_{self.contest}"

    @property
    def problem_title(self) -> str:
        return f"{self.page_prefix}_Problems"

    @property
    def answer_title(self) -> str:
        return f"{self.page_prefix}_Answer_Key"

    @property
    def source_name(self) -> str:
        return f"AIME {self.year} {self.contest}"


class TextExtractingHTMLParser(HTMLParser):
    """Small HTML-to-text extractor that preserves math image alt text."""

    BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self.BLOCK_TAGS:
            self._parts.append("\n")
        if tag == "img":
            attr_map = {key.lower(): value for key, value in attrs}
            alt = attr_map.get("alt") or ""
            if alt:
                self._parts.append(f" {html.unescape(alt)} ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        raw = html.unescape(raw)
        raw = raw.replace("\xa0", " ")
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        raw = re.sub(r"\n[ \t]+", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


class CachedHttpClient:
    def __init__(self, cache_dir: Path, delay_seconds: float, timeout: int) -> None:
        self.cache_dir = cache_dir
        self.delay_seconds = delay_seconds
        self.timeout = timeout
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_at = 0.0

    def get_text(self, url: str, refresh: bool = False) -> str:
        raw = self.get_bytes(url, refresh=refresh)
        return raw.decode("utf-8", errors="replace")

    def get_bytes(self, url: str, refresh: bool = False) -> bytes:
        cache_path = self._cache_path(url)
        if cache_path.exists() and not refresh:
            return cache_path.read_bytes()

        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)

        request = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
        self._last_request_at = time.monotonic()
        with urlopen(request, timeout=self.timeout) as response:
            raw = response.read()
        cache_path.write_bytes(raw)
        return raw

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{digest}.html"


def aops_page_url(title: str) -> str:
    return f"{AOPS_BASE}/{quote(title, safe='')}"


def html_to_text(page_html: str) -> str:
    parser = TextExtractingHTMLParser()
    parser.feed(page_html)
    parser.close()
    return parser.text()


def clean_problem_text(text: str) -> str:
    text = re.sub(r"\bSolution\s*(?:1|2|3|[A-Z])?\b.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAnswer\b.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bSee Also\b.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_aime_problems(text: str) -> dict[int, str]:
    """Extract Problem N sections from an AoPS AIME problem page."""

    marker_pattern = re.compile(r"(?m)^\s*Problem\s+([1-9]|1[0-5])\s*$")
    matches = list(marker_pattern.finditer(text))
    problems: dict[int, str] = {}

    for index, match in enumerate(matches):
        number = int(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = clean_problem_text(text[start:end])
        body = strip_aops_noise(body)
        if len(body) > 20:
            problems[number] = body

    return problems


def strip_aops_noise(text: str) -> str:
    noise_patterns = [
        r"^\s*The following problem is from.*$",
        r"^\s*This problem is from.*$",
        r"^\s*Problem\s+\d+\s*$",
        r"^\s*\[.*?\]\s*$",
        r"^\s*Retrieved from .*$",
    ]
    lines = []
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            lines.append("")
            continue
        if any(re.search(pattern, cleaned, flags=re.IGNORECASE) for pattern in noise_patterns):
            continue
        lines.append(cleaned)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def extract_aime_answers(text: str) -> dict[int, str]:
    """Extract numeric AIME answer keys from a text-rendered answer page."""

    answers: dict[int, str] = {}

    for number, answer in re.findall(
        r"(?im)^\s*(?:Problem\s*)?([1-9]|1[0-5])\s*[:.\-\)]\s*([0-9]{1,3})\s*$",
        text,
    ):
        answers[int(number)] = answer.zfill(3)

    if len(answers) >= 10:
        return answers

    # Fallback for pages whose rendered text is a compact 15-number answer key.
    candidates = re.findall(r"\b([0-9]{1,3})\b", text)
    compact = [value.zfill(3) for value in candidates if 0 <= int(value) <= 999]
    if len(compact) >= 15:
        return {idx + 1: value for idx, value in enumerate(compact[:15])}

    return answers


def normalize_subject_tags(problem: str) -> list[str]:
    lower = problem.lower()
    tags: list[str] = []
    keyword_map = [
        ("number_theory", ["integer", "prime", "divisible", "modulo", "remainder"]),
        ("algebra", ["polynomial", "equation", "real numbers", "complex numbers"]),
        ("geometry", ["triangle", "circle", "angle", "area", "radius", "perimeter"]),
        ("combinatorics", ["ways", "arrangements", "subsets", "permutations", "combinations"]),
        ("probability", ["probability", "expected", "random"]),
        ("calculus", ["function", "maximum", "minimum", "derivative", "integral"]),
    ]
    for tag, keywords in keyword_map:
        if any(keyword in lower for keyword in keywords):
            tags.append(tag)
    return tags or ["unknown"]


def infer_answer_type(answer: str) -> str:
    if re.fullmatch(r"\d{3}", answer):
        return "integer_mod_1000"
    if re.fullmatch(r"-?\d+", answer):
        return "integer"
    if re.fullmatch(r"-?\d+/\d+", answer):
        return "rational"
    return "expression"


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


def normalize_latex_answer(answer: str) -> str:
    """Normalize common contest-answer LaTeX without trying to prove equivalence."""

    text = str(answer).strip()
    text = re.sub(r"^\$+|\$+$", "", text).strip()
    text = re.sub(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1", text)
    text = strip_matching_braces(text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "")
    text = re.sub(r"\s+", "", text)

    # Convert simple \frac{a}{b} patterns. This intentionally stays conservative.
    frac_pattern = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    previous = None
    while previous != text:
        previous = text
        text = frac_pattern.sub(r"(\1)/(\2)", text)

    # Mixed number shorthand: 10(2)/(3) -> 10+(2)/(3).
    text = re.sub(r"(?<=\d)\(([-+]?\d+)\)/\(([-+]?\d+)\)", r"+(\1)/(\2)", text)

    replacements = {
        "\\cdot": "*",
        "\\times": "*",
        "\\div": "/",
        "\\pi": "pi",
        "\\sqrt": "sqrt",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text or str(answer).strip()


def build_aime_records(
    client: CachedHttpClient,
    source: AimeSource,
    refresh: bool,
) -> list[dict]:
    problem_url = aops_page_url(source.problem_title)
    answer_url = aops_page_url(source.answer_title)

    problem_text = html_to_text(client.get_text(problem_url, refresh=refresh))
    answer_text = html_to_text(client.get_text(answer_url, refresh=refresh))

    problems = extract_aime_problems(problem_text)
    answers = extract_aime_answers(answer_text)

    records: list[dict] = []
    for number in sorted(problems):
        if number not in answers:
            continue
        answer = answers[number]
        problem = problems[number]
        record_id = f"aime_{source.year}_{source.contest.lower()}_{number:02d}"
        records.append(
            {
                "id": record_id,
                "source": "AoPS Wiki",
                "source_name": source.source_name,
                "source_url": problem_url,
                "answer_source_url": answer_url,
                "license_hint": "Public webpage; verify source terms before redistribution.",
                "year": source.year,
                "contest": f"AIME {source.contest}",
                "problem_no": number,
                "language": "en",
                "problem": problem,
                "answer": answer,
                "canonical_answer": answer,
                "equivalent_answers": [str(int(answer))],
                "answer_type": infer_answer_type(answer),
                "subject": normalize_subject_tags(problem),
                "skills": [],
                "difficulty": "medium",
                "requires_proof": False,
                "tool_checkable": True,
                "judge_type": "exact_string_or_integer_mod_1000",
                "metadata": {
                    "adapter": "aops_aime",
                    "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            }
        )
    return records


def safe_id_part(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def normalize_harp_record(raw: dict, row_index: int) -> dict:
    contest = raw.get("contest", "unknown")
    year = raw.get("year", "unknown")
    number = raw.get("number", raw.get("problem_no", row_index + 1))
    source_name = f"HARP {contest} {year} #{number}"
    answer = str(raw.get("answer", "")).strip()
    canonical_answer = normalize_latex_answer(answer)
    subject = raw.get("subject", "unknown")
    if isinstance(subject, str):
        subjects = [subject]
    elif isinstance(subject, list):
        subjects = [str(item) for item in subject]
    else:
        subjects = ["unknown"]

    solutions = {
        key: value
        for key, value in raw.items()
        if re.fullmatch(r"solution_\d+", str(key)) and str(value).strip()
    }

    return {
        "id": "harp_"
        + "_".join(
            [
                safe_id_part(year),
                safe_id_part(contest),
                safe_id_part(number),
                str(row_index),
            ]
        ),
        "source": "HARP",
        "source_name": source_name,
        "source_url": HARP_REPO_URL,
        "answer_source_url": HARP_REPO_URL,
        "license_hint": "HARP repository is MIT licensed; verify upstream terms before redistribution.",
        "year": year,
        "contest": contest,
        "problem_no": number,
        "language": "en",
        "problem": str(raw.get("problem", "")).strip(),
        "answer": answer,
        "canonical_answer": canonical_answer,
        "raw_answer": answer,
        "equivalent_answers": [],
        "answer_type": infer_answer_type(canonical_answer),
        "subject": subjects,
        "skills": [],
        "difficulty": str(raw.get("level", "unknown")),
        "requires_proof": False,
        "tool_checkable": bool(answer),
        "judge_type": "exact_or_symbolic",
        "solution": next(iter(solutions.values()), ""),
        "all_solutions": solutions,
        "metadata": {
            "adapter": "harp",
            "raw_keys": sorted(str(key) for key in raw.keys()),
            "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }


def normalize_harp_proof_record(raw: dict, row_index: int) -> dict:
    contest = raw.get("contest", "unknown")
    year = raw.get("year", "unknown")
    number = raw.get("number", raw.get("problem_no", row_index + 1))
    source_name = f"HARP proof-based {contest} {year} #{number}"
    subject = raw.get("subject", "unknown")
    if isinstance(subject, str):
        subjects = [subject]
    elif isinstance(subject, list):
        subjects = [str(item) for item in subject]
    else:
        subjects = ["unknown"]

    solutions = {
        key: value
        for key, value in raw.items()
        if re.fullmatch(r"solution_\d+", str(key)) and str(value).strip()
    }
    reference_solution = next(iter(solutions.values()), "")
    answer = str(raw.get("answer") or raw.get("final_answer") or "").strip()

    return {
        "id": "harp_proof_"
        + "_".join(
            [
                safe_id_part(year),
                safe_id_part(contest),
                safe_id_part(number),
                str(row_index),
            ]
        ),
        "source": "HARP proof-based",
        "source_name": source_name,
        "source_url": HARP_REPO_URL,
        "answer_source_url": HARP_REPO_URL,
        "license_hint": "HARP repository is MIT licensed; verify upstream terms before redistribution.",
        "year": year,
        "contest": contest,
        "problem_no": number,
        "language": "en",
        "problem": str(raw.get("problem", "")).strip(),
        "answer": answer,
        "canonical_answer": answer or "PROOF_REQUIRED",
        "raw_answer": answer,
        "equivalent_answers": [],
        "answer_type": "proof",
        "subject": subjects,
        "skills": [],
        "difficulty": str(raw.get("level", "unknown")),
        "requires_proof": True,
        "tool_checkable": False,
        "judge_type": "proof_judge_with_reference_solution",
        "solution": reference_solution,
        "all_solutions": solutions,
        "metadata": {
            "adapter": "harp_proof",
            "raw_keys": sorted(str(key) for key in raw.keys()),
            "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }


def build_harp_records(
    client: CachedHttpClient,
    refresh: bool,
    max_records: int,
    contest_filter: set[str] | None,
) -> list[dict]:
    data = client.get_bytes(HARP_ZIP_URL, refresh=refresh)
    records: list[dict] = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        jsonl_names = [name for name in archive.namelist() if name.endswith(".jsonl")]
        if not jsonl_names:
            raise ValueError("HARP zip did not contain a .jsonl file.")
        with archive.open(jsonl_names[0]) as file:
            for row_index, raw_line in enumerate(file):
                if len(records) >= max_records:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                raw = json.loads(line)
                contest = str(raw.get("contest", "")).upper()
                if contest_filter and contest not in contest_filter:
                    continue
                record = normalize_harp_record(raw, row_index)
                if record["problem"] and record["canonical_answer"]:
                    records.append(record)
    return records


def build_harp_proof_records(
    client: CachedHttpClient,
    refresh: bool,
    max_records: int,
    contest_filter: set[str] | None,
) -> list[dict]:
    data = client.get_bytes(HARP_PROOF_ZIP_URL, refresh=refresh)
    records: list[dict] = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        jsonl_names = [name for name in archive.namelist() if name.endswith(".jsonl")]
        if not jsonl_names:
            raise ValueError("HARP proof zip did not contain a .jsonl file.")
        with archive.open(jsonl_names[0]) as file:
            for row_index, raw_line in enumerate(file):
                if len(records) >= max_records:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                raw = json.loads(line)
                contest = str(raw.get("contest", "")).upper()
                if contest_filter and contest not in contest_filter:
                    continue
                record = normalize_harp_proof_record(raw, row_index)
                if record["problem"] and record["solution"]:
                    records.append(record)
    return records


def parse_years(value: str) -> list[int]:
    years: set[int] = set()
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_raw, end_raw = chunk.split("-", 1)
            start, end = int(start_raw), int(end_raw)
            years.update(range(min(start, end), max(start, end) + 1))
        else:
            years.add(int(chunk))
    return sorted(years)


def dedupe_records(records: Iterable[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []
    for record in records:
        key = record["id"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def validate_record(record: dict) -> list[str]:
    errors: list[str] = []
    required = ["id", "problem", "canonical_answer", "answer_type", "source_url"]
    for field in required:
        if not record.get(field):
            errors.append(f"{record.get('id', '<missing-id>')}: missing {field}")
    if not isinstance(record.get("subject"), list):
        errors.append(f"{record.get('id', '<missing-id>')}: subject must be a list")
    return errors


def write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            file.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch a small public math-contest dataset and write normalized JSONL."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/public_math_small.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/public_math_pages"),
        help="HTML cache directory.",
    )
    parser.add_argument(
        "--sources",
        default="harp",
        help="Comma-separated source adapters: harp,harp_proof,aops_aime.",
    )
    parser.add_argument(
        "--harp-contests",
        default="",
        help=(
            "Optional comma-separated contest filter for HARP, e.g. "
            "'AIME,AIME_I,AIME_II'. Leave empty for all HARP contests."
        ),
    )
    parser.add_argument(
        "--aime-years",
        default="2022-2024",
        help="AIME years to fetch, e.g. '2024' or '2020-2024'.",
    )
    parser.add_argument(
        "--aime-contests",
        default="I,II",
        help="Comma-separated AIME contest labels, usually I,II.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=60,
        help="Maximum records to write after fetching and deduplication.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.0,
        help="Delay between network requests.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached HTML and re-fetch pages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and validate records without writing output.",
    )
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = build_arg_parser().parse_args()
    client = CachedHttpClient(args.cache_dir, args.delay_seconds, args.timeout)

    records: list[dict] = []
    sources = [source.strip().lower() for source in args.sources.split(",") if source.strip()]

    if "harp" in sources:
        try:
            harp_contests = {
                value.strip().upper()
                for value in args.harp_contests.split(",")
                if value.strip()
            }
            harp_records = build_harp_records(
                client,
                args.refresh_cache,
                args.limit,
                harp_contests or None,
            )
            print(f"[info] HARP: {len(harp_records)} records")
            records.extend(harp_records)
        except (HTTPError, URLError, TimeoutError, zipfile.BadZipFile, ValueError) as exc:
            print(f"[warn] failed to fetch HARP: {exc}")

    if "harp_proof" in sources:
        try:
            harp_contests = {
                value.strip().upper()
                for value in args.harp_contests.split(",")
                if value.strip()
            }
            proof_records = build_harp_proof_records(
                client,
                args.refresh_cache,
                args.limit,
                harp_contests or None,
            )
            print(f"[info] HARP proof-based: {len(proof_records)} records")
            records.extend(proof_records)
        except (HTTPError, URLError, TimeoutError, zipfile.BadZipFile, ValueError) as exc:
            print(f"[warn] failed to fetch HARP proof-based: {exc}")

    if "aops_aime" in sources:
        contests = [contest.strip().upper() for contest in args.aime_contests.split(",")]
        for year in parse_years(args.aime_years):
            for contest in contests:
                if contest not in {"I", "II"}:
                    raise ValueError(f"Unsupported AIME contest label: {contest}")
                source = AimeSource(year=year, contest=contest)
                try:
                    source_records = build_aime_records(client, source, args.refresh_cache)
                except (HTTPError, URLError, TimeoutError) as exc:
                    print(f"[warn] failed to fetch {source.source_name}: {exc}")
                    continue
                print(f"[info] {source.source_name}: {len(source_records)} records")
                records.extend(source_records)

    unknown_sources = sorted(set(sources) - {"harp", "harp_proof", "aops_aime"})
    if unknown_sources:
        raise ValueError(f"Unsupported source adapters: {', '.join(unknown_sources)}")

    records = dedupe_records(records)
    records = records[: args.limit]
    validation_errors = [error for record in records for error in validate_record(record)]
    if validation_errors:
        for error in validation_errors:
            print(f"[error] {error}")
        raise SystemExit(2)

    if args.dry_run:
        print(f"[info] dry run complete: {len(records)} valid records")
        return

    write_jsonl(records, args.output)
    print(f"[info] wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
