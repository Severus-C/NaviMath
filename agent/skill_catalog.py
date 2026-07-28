from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class SkillTemplate:
    id: str
    name: str
    domains: List[str]
    triggers: List[str]
    strategy: str
    traps: List[str] = field(default_factory=list)
    verifier_checklist: List[str] = field(default_factory=list)
    answer_schema: str = "exact"
    proof_methods: List[str] = field(default_factory=list)
    solver_role: str = "specialist"
    support: int = 0
    proof_support: int = 0
    solution_support: int = 0
    answer_types: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillTemplate":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: data[key] for key in allowed if key in data})


class DistilledSkillCatalog:
    """Runtime view of aggregate HARP-derived strategy templates."""

    def __init__(self, templates: List[SkillTemplate] | None = None, router: Dict[str, Any] | None = None) -> None:
        self.templates = templates or []
        self.router = router or {}

    @classmethod
    def load(cls, path: Path | None = None) -> "DistilledSkillCatalog":
        catalog_path = path or Path(__file__).with_name("skill_catalog.json")
        try:
            payload = json.loads(catalog_path.read_text(encoding="utf-8"))
            templates = [SkillTemplate.from_dict(item) for item in payload.get("templates", [])]
            router = payload.get("router") or {}
        except (OSError, ValueError, TypeError):
            templates = []
            router = {}
        return cls(templates, router)

    def match(
        self,
        problem: str,
        domain: str,
        proof_mode: bool = False,
        limit: int = 3,
    ) -> List[SkillTemplate]:
        scored: List[tuple[float, SkillTemplate]] = []
        lower = problem.lower()
        for template in self.templates:
            if domain not in template.domains and "general" not in template.domains:
                continue
            trigger_score = self._trigger_score(lower, template.triggers)
            if trigger_score <= 0:
                continue
            score = trigger_score
            score += min(2.0, template.support / 400.0)
            if proof_mode:
                score += min(1.5, template.proof_support / 20.0)
                if template.proof_methods:
                    score += 0.5
            scored.append((score, template))
        scored.sort(key=lambda item: (-item[0], -item[1].support, item[1].id))
        return [template for _, template in scored[:limit]]

    def domain_scores(self, problem: str) -> Dict[str, float]:
        lower = problem.lower()
        scores: Dict[str, float] = {}
        for template in self.templates:
            score = self._trigger_score(lower, template.triggers)
            if score <= 0:
                continue
            for domain in template.domains:
                if domain != "general":
                    scores[domain] = scores.get(domain, 0.0) + score
        return scores

    def router_scores(self, problem: str) -> Dict[str, float]:
        labels = self.router.get("labels") or []
        priors = self.router.get("log_priors") or {}
        likelihoods = self.router.get("token_log_likelihoods") or {}
        if not labels or not priors or not likelihoods:
            return {}
        features = self._tokenize(problem)
        raw_scores = {
            label: float(priors.get(label, -20.0))
            + sum(float(likelihoods.get(label, {}).get(token, 0.0)) for token in features)
            for label in labels
        }
        maximum = max(raw_scores.values())
        exponentials = {label: math.exp(max(-60.0, score - maximum)) for label, score in raw_scores.items()}
        total = sum(exponentials.values()) or 1.0
        return {label: value / total for label, value in exponentials.items()}

    @staticmethod
    def _trigger_score(problem: str, triggers: List[str]) -> float:
        score = 0.0
        for trigger in triggers:
            token = trigger.lower().strip()
            if not token or not DistilledSkillCatalog._contains(problem, token):
                continue
            words = len(token.split())
            score += 1.0 + min(2.0, (words - 1) * 0.75) + min(1.0, len(token) / 30.0)
        return score

    @staticmethod
    def _contains(text: str, token: str) -> bool:
        if re.fullmatch(r"[a-z0-9_]+", token):
            return re.search(rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])", text) is not None
        return token in text

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        words = re.findall(r"\\[a-z]+|[a-z]{2,}|\d+", text.lower())
        stopwords = {
            "the", "and", "that", "with", "from", "this", "then", "are", "for", "has",
            "have", "its", "let", "find", "what", "which", "when", "where", "given", "such",
        }
        words = [word for word in words if word not in stopwords]
        bigrams = [f"{left}__{right}" for left, right in zip(words, words[1:])]
        return list(dict.fromkeys(words + bigrams))
