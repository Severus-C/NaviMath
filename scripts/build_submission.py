from __future__ import annotations

import argparse
import ast
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dist" / "navimath-submission.zip"

RUNTIME_FILES = (
    "user_agent.py",
    "requirements.txt",
    "agent/__init__.py",
    "agent/reasoning_agent.py",
    "agent/answer_selection_agent.py",
    "agent/answer_contract.py",
    "agent/answer_normalizer.py",
    "agent/agent_utils.py",
    "agent/skill_catalog.py",
    "agent/skill_catalog.json",
    "agent/tool_verify.py",
    "agent/rlot_navigator.py",
    "agent/rlot_policy.json",
)

FORBIDDEN_PARTS = {
    ".agents",
    ".cache",
    ".codex",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "data",
    "eval_outputs",
    "outputs",
    "tests",
}


def validate_sources() -> None:
    missing = [name for name in RUNTIME_FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing submission files: {', '.join(missing)}")

    for name in RUNTIME_FILES:
        path = ROOT / name
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=name)
        elif path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))


def validate_archive(path: Path) -> None:
    expected = set(RUNTIME_FILES)
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if names != expected:
            missing = sorted(expected - names)
            extra = sorted(names - expected)
            raise ValueError(f"Archive manifest mismatch; missing={missing}, extra={extra}")
        for name in names:
            parts = set(PurePosixPath(name).parts)
            if parts & FORBIDDEN_PARTS or name.endswith((".pyc", ".pyo")):
                raise ValueError(f"Forbidden submission entry: {name}")


def build(output: Path) -> Path:
    validate_sources()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in RUNTIME_FILES:
            archive.write(ROOT / name, arcname=name)
    validate_archive(output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the minimal competition submission archive.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    output = build(parse_args().output.resolve())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"Built {output} ({output.stat().st_size} bytes, sha256={digest})")


if __name__ == "__main__":
    main()
