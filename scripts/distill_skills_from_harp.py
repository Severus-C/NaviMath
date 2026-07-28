from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TEMPLATES: List[Dict[str, Any]] = [
    {
        "id": "algebra_substitution_system", "name": "Equation substitution and systems",
        "domains": ["algebra"], "triggers": ["equation", "system", "when x", "when $x", "satisfies", "simultaneous"],
        "solution_signals": ["substitut", "eliminat", "solve for"],
        "strategy": "Name variables, translate every condition, eliminate or substitute, then plug the result into the exact requested target.",
        "traps": ["Solving for an intermediate variable instead of the requested quantity", "Dropping a branch", "Arithmetic after substitution"],
        "verifier_checklist": ["Substitute the final values into every original equation", "Confirm the output is the requested expression, not merely a variable"],
        "answer_schema": "exact scalar or finite solution set", "solver_role": "equation_modeler",
    },
    {
        "id": "algebra_polynomial_vieta", "name": "Polynomial roots, factors, and Vieta",
        "domains": ["algebra"], "triggers": ["polynomial", "roots", "zeros", "quadratic", "factor", "remainder theorem", "vieta"],
        "solution_signals": ["vieta", "factor theorem", "sum of the roots", "product of the roots", "discriminant"],
        "strategy": "Use factor/remainder theorems and symmetric root relations before expanding; track multiplicity and evaluate only the requested symmetric expression.",
        "traps": ["Ignoring multiplicity", "Assuming roots are real", "Sign errors in Vieta"],
        "verifier_checklist": ["Check degree and leading coefficient", "Verify reconstructed coefficients or substitute candidate roots"],
        "answer_schema": "number, polynomial, or sorted root set", "solver_role": "polynomial_specialist",
    },
    {
        "id": "algebra_inequality_extremum", "name": "Inequalities and equality cases",
        "domains": ["algebra", "optimization"], "triggers": ["inequality", "maximum", "minimum", "least possible", "greatest possible", "at most", "at least"],
        "solution_signals": ["am-gm", "cauchy", "equality holds", "completing the square", "discriminant"],
        "strategy": "Derive a sharp bound, state the equality condition, and prove that the equality configuration satisfies every domain constraint.",
        "traps": ["A bound that is not attainable", "Ignoring endpoints or integrality", "Reversing an inequality"],
        "verifier_checklist": ["Check equality attainability", "Test boundary and integer-neighbor cases", "Confirm max versus supremum"],
        "answer_schema": "extremal value plus equality case when requested", "proof_methods": ["sharp inequality", "extremal argument"],
        "solver_role": "inequality_optimizer",
    },
    {
        "id": "algebra_functional_equation", "name": "Functional equations",
        "domains": ["algebra"], "triggers": ["f(x)", "f(", "functional equation", "all functions", "function satisfies"],
        "solution_signals": ["set x=", "set y=", "injective", "surjective", "cauchy equation"],
        "strategy": "Probe neutral and repeated inputs, derive injectivity/surjectivity only when justified, then prove the candidate family satisfies the equation.",
        "traps": ["Finding candidates without proving completeness", "Using continuity not given", "Dividing by a possibly zero expression"],
        "verifier_checklist": ["Substitute the complete claimed family", "Audit every regularity assumption", "Prove no other functions exist"],
        "answer_schema": "complete function family with domain restrictions", "proof_methods": ["strategic substitution", "injectivity/surjectivity"],
        "solver_role": "functional_equation_solver",
    },
    {
        "id": "algebra_sequences_recurrence", "name": "Sequences and recurrences",
        "domains": ["algebra", "combinatorics"], "triggers": ["sequence", "recurrence", "arithmetic progression", "geometric progression", "a_n", "term of"],
        "solution_signals": ["characteristic equation", "telescop", "induction", "generating function"],
        "strategy": "Compute small terms, identify invariant or closed form, prove it by recurrence/induction, and check the initial index.",
        "traps": ["Off-by-one indexing", "Wrong initial condition", "Guessing a pattern without proof"],
        "verifier_checklist": ["Check the first two valid indices", "Substitute the closed form into the recurrence"],
        "answer_schema": "exact term, recurrence, or proved closed form", "proof_methods": ["induction", "telescoping"],
        "solver_role": "recurrence_solver",
    },
    {
        "id": "algebra_rational_simplification", "name": "Rational expression simplification",
        "domains": ["algebra"], "triggers": ["simplify", "lowest terms", "rational expression", "reduced to", "equivalent to"],
        "solution_signals": ["factor", "common denominator", "cancel"],
        "strategy": "Factor completely before cancellation, record excluded values, and verify the simplified form symbolically or at safe sample points.",
        "traps": ["Cancelling additive terms", "Losing excluded domain values", "Sign error in factorization"],
        "verifier_checklist": ["Compare domains before and after simplification", "Cross-multiply or use symbolic equivalence"],
        "answer_schema": "simplified exact expression with exclusions if requested", "solver_role": "symbolic_simplifier",
    },
    {
        "id": "prealgebra_ratio_percent", "name": "Ratios, proportions, and percentages",
        "domains": ["algebra"], "triggers": ["ratio", "proportion", "percent", "proportional", "mixture", "divided into"],
        "solution_signals": ["parts", "cross multiply", "percent"],
        "strategy": "Represent ratio parts with one scale variable, distinguish percent change from percentage points, and preserve the requested units.",
        "traps": ["Using the wrong base quantity", "Confusing part-to-part with part-to-whole", "Premature rounding"],
        "verifier_checklist": ["Reconstruct the original total", "Check ratio order and units"],
        "answer_schema": "exact rational, percentage, or quantity with units", "solver_role": "ratio_modeler",
    },
    {
        "id": "prealgebra_rate_work", "name": "Rates, work, motion, and mixtures",
        "domains": ["algebra"], "triggers": ["rate", "speed", "miles per", "kilometers per", "work together", "fills", "mixture", "average speed"],
        "solution_signals": ["distance =", "rate =", "work rate", "relative speed"],
        "strategy": "Choose a conserved quantity, build a rate-time table, align time intervals, and solve with consistent units.",
        "traps": ["Averaging speeds arithmetically", "Unit mismatch", "Adding times when rates should add"],
        "verifier_checklist": ["Check dimensions", "Recompute total distance/work from the candidate"],
        "answer_schema": "nonnegative scalar with units", "solver_role": "rate_table_solver",
    },
    {
        "id": "number_theory_modular", "name": "Modular arithmetic and periodicity",
        "domains": ["number_theory"], "triggers": ["remainder", "modulo", "congruent", "divisible by", "last digit", "units digit", "residue"],
        "solution_signals": ["mod ", "modulo", "period", "congruence"],
        "strategy": "Reduce early, identify periods or invertible factors, solve residue classes, then enforce original bounds and sign conventions.",
        "traps": ["Dividing by a nonunit modulo n", "Reporting a nonleast residue", "Missing compatible residue classes"],
        "verifier_checklist": ["Substitute into every congruence", "Normalize the requested remainder range", "Check CRT compatibility"],
        "answer_schema": "least nonnegative residue or complete congruence class", "proof_methods": ["residue classes", "Chinese remainder theorem"],
        "solver_role": "modular_arithmetic_solver",
    },
    {
        "id": "number_theory_valuation_gcd", "name": "Prime valuations, gcd, and lcm",
        "domains": ["number_theory"], "triggers": ["gcd", "lcm", "greatest common divisor", "least common multiple", "prime factor", "highest power", "valuation"],
        "solution_signals": ["p-adic", "valuation", "prime factorization", "minimum exponent", "maximum exponent"],
        "strategy": "Work prime by prime, translate gcd/lcm into min/max exponents, and rebuild the integer only after all valuations are settled.",
        "traps": ["Forgetting the zero valuation", "Confusing min with max", "Assuming pairwise coprimality"],
        "verifier_checklist": ["Compare both sides at an arbitrary prime", "Check primes absent from some factors"],
        "answer_schema": "integer, factorization, or valuation identity", "proof_methods": ["prime-by-prime valuation"],
        "solver_role": "valuation_solver",
    },
    {
        "id": "number_theory_diophantine", "name": "Diophantine equations",
        "domains": ["number_theory", "algebra"], "triggers": ["integer solutions", "positive integers", "diophantine", "ordered pairs", "integer pairs", "whole numbers"],
        "solution_signals": ["factorization", "modulo", "bounds", "integer solutions"],
        "strategy": "Use congruences, gcd constraints and factorization to bound the integer search; prove the resulting case list is exhaustive.",
        "traps": ["Missing negative or zero solutions", "Counting ordered versus unordered pairs incorrectly", "Unproved search cutoff"],
        "verifier_checklist": ["Substitute every listed solution", "Prove exhaustiveness from bounds or factor pairs", "Apply positivity and ordering"],
        "answer_schema": "sorted finite set or requested count", "proof_methods": ["factor pairs", "modular obstruction", "descent"],
        "solver_role": "diophantine_solver",
    },
    {
        "id": "number_theory_digits_bases", "name": "Digits and numeral bases",
        "domains": ["number_theory", "algebra"], "triggers": ["digit", "decimal representation", "base ", "three-digit", "four-digit", "units digit", "tens digit"],
        "solution_signals": ["place value", "mod 9", "digit sum", "base"],
        "strategy": "Expand place values explicitly, impose digit ranges and leading-digit constraints, then use congruences to reduce cases.",
        "traps": ["Allowing a leading zero", "Forgetting digit bounds", "Treating concatenation as multiplication"],
        "verifier_checklist": ["Reconstruct the numeral", "Check every digit lies in the base range", "Verify leading digit is nonzero"],
        "answer_schema": "integer, digit tuple, or count", "solver_role": "digit_constraint_solver",
    },
    {
        "id": "geometry_similarity", "name": "Similarity and congruence",
        "domains": ["geometry"], "triggers": ["similar", "congruent", "parallel", "midpoint", "angle bisector", "triangle"],
        "solution_signals": ["similar triangles", "sss", "sas", "aa similarity", "angle bisector theorem"],
        "strategy": "Write an explicit correspondence before using ratios; propagate directed angles and verify all lengths are attached to matching sides.",
        "traps": ["Wrong vertex correspondence", "Assuming the diagram is to scale", "Using similarity before proving it"],
        "verifier_checklist": ["State the similarity/congruence criterion", "Check ratio orientation", "Audit degenerate configurations"],
        "answer_schema": "positive exact length, ratio, angle, or proof", "proof_methods": ["similarity", "congruence"],
        "solver_role": "similarity_solver",
    },
    {
        "id": "geometry_circle_power", "name": "Circles, tangency, and power of a point",
        "domains": ["geometry"], "triggers": ["circle", "tangent", "chord", "secant", "cyclic", "diameter", "arc", "inscribed angle"],
        "solution_signals": ["power of a point", "cyclic quadrilateral", "tangent-chord", "radical axis"],
        "strategy": "Identify cyclic quadrilaterals and equal powers, use directed angles, and choose the correct secant/tangent segment convention.",
        "traps": ["Using whole versus external secant length incorrectly", "Wrong supplementary angle", "Assuming tangency"],
        "verifier_checklist": ["Prove concyclicity/tangency", "Check power products use the correct segments", "Reject negative lengths"],
        "answer_schema": "positive exact length, angle, power, or proof", "proof_methods": ["power of a point", "directed angles"],
        "solver_role": "circle_geometry_solver",
    },
    {
        "id": "geometry_coordinate_vector", "name": "Coordinate and vector geometry",
        "domains": ["geometry", "linear_algebra"], "triggers": ["coordinate", "slope", "line", "vector", "dot product", "plane", "distance between"],
        "solution_signals": ["coordinates", "dot product", "distance formula", "equation of the line"],
        "strategy": "Choose symmetry-aware coordinates, encode incidence/perpendicularity algebraically, solve, then discard branches violating geometry.",
        "traps": ["Coordinate placement losing generality", "Squared-distance extraneous branch", "Slope failure for vertical lines"],
        "verifier_checklist": ["Substitute coordinates into every incidence condition", "Check signs, orientation and nondegeneracy"],
        "answer_schema": "coordinate tuple, exact length/slope, or proof", "proof_methods": ["coordinate bash", "vector dot products"],
        "solver_role": "coordinate_geometry_solver",
    },
    {
        "id": "geometry_area_volume", "name": "Area, volume, and decomposition",
        "domains": ["geometry"], "triggers": ["area", "volume", "surface area", "perimeter", "tetrahedron", "prism", "pyramid"],
        "solution_signals": ["area ratio", "shoelace", "base times height", "heron"],
        "strategy": "Decompose into nonoverlapping pieces or use invariant area ratios; keep length, area and volume scaling powers distinct.",
        "traps": ["Double-counting overlap", "Using slant instead of perpendicular height", "Wrong scale-factor power"],
        "verifier_checklist": ["Check dimensions", "Sum component measures", "Test positivity and geometric feasibility"],
        "answer_schema": "nonnegative exact measure with square/cubic units", "solver_role": "area_decomposition_solver",
    },
    {
        "id": "combinatorics_casework_bijection", "name": "Casework, bijections, and symmetry",
        "domains": ["combinatorics"], "triggers": ["number of ways", "how many", "arrangements", "permutation", "combination", "subsets", "colorings"],
        "solution_signals": ["case", "bijection", "choose", "symmetry", "burnside"],
        "strategy": "Define the counted object, split into disjoint exhaustive cases or construct a bijection, and divide by symmetry only after stabilizers are understood.",
        "traps": ["Overlapping cases", "Treating identical objects as distinct", "Unjustified division by symmetry"],
        "verifier_checklist": ["Prove cases are exhaustive and disjoint", "Check labeled versus unlabeled objects", "Verify on a small instance"],
        "answer_schema": "nonnegative integer count", "proof_methods": ["bijection", "case partition", "Burnside"],
        "solver_role": "bijective_counter",
    },
    {
        "id": "combinatorics_inclusion_exclusion", "name": "Complement and inclusion-exclusion",
        "domains": ["combinatorics", "probability"], "triggers": ["at least one", "none", "no two", "avoid", "not adjacent", "derangement", "excluded"],
        "solution_signals": ["complement", "inclusion-exclusion", "subtract", "derangement"],
        "strategy": "Count the clean universe, define bad events precisely, and apply complement or inclusion-exclusion with intersections indexed consistently.",
        "traps": ["Nonuniform universe", "Missing higher intersections", "Complementing the wrong event"],
        "verifier_checklist": ["Check universe size", "List bad events and intersections", "Ensure result lies in valid count/probability range"],
        "answer_schema": "nonnegative integer or reduced probability", "proof_methods": ["complement", "inclusion-exclusion"],
        "solver_role": "inclusion_exclusion_solver",
    },
    {
        "id": "combinatorics_recurrence_dp", "name": "Combinatorial recurrence and state DP",
        "domains": ["combinatorics"], "triggers": ["recurrence", "tiling", "steps", "consecutive", "binary string", "sequence of length", "ways to form"],
        "solution_signals": ["recurrence", "state", "dynamic programming", "generating function"],
        "strategy": "Choose a minimal state that records future-relevant information, derive transitions, and establish all base cases before evaluating.",
        "traps": ["State omits necessary history", "Off-by-one base case", "Transitions overlap"],
        "verifier_checklist": ["Enumerate the smallest sizes directly", "Check every object has one predecessor transition"],
        "answer_schema": "nonnegative integer sequence value", "proof_methods": ["recurrence", "state decomposition"],
        "solver_role": "state_recurrence_solver",
    },
    {
        "id": "combinatorics_double_counting", "name": "Double counting and invariants",
        "domains": ["combinatorics", "number_theory"], "triggers": ["prove that", "show that", "divisible", "at least", "graph", "degree", "pairs"],
        "solution_signals": ["count in two ways", "double count", "invariant", "handshaking"],
        "strategy": "Define one finite incidence set and count it in two ways; for processes, identify a quantity preserved or monotone under every move.",
        "traps": ["Counting different universes on two sides", "Hidden multiplicity", "Claimed invariant changes in an edge case"],
        "verifier_checklist": ["Name the counted set explicitly", "Check the contribution of one object from each side", "Test every allowed move"],
        "answer_schema": "integer identity, divisibility statement, or proof", "proof_methods": ["double counting", "invariant"],
        "solver_role": "double_counting_solver",
    },
    {
        "id": "probability_sample_space", "name": "Finite probability and conditional counting",
        "domains": ["probability"], "triggers": ["probability", "randomly", "equally likely", "conditional", "given that", "without replacement", "with replacement"],
        "solution_signals": ["sample space", "favorable", "conditional probability", "bayes"],
        "strategy": "Specify equally likely atomic outcomes, count numerator and denominator under the same labeling convention, then reduce the fraction.",
        "traps": ["Mixing labeled and unlabeled outcomes", "Assuming independence", "Conditioning only the numerator"],
        "verifier_checklist": ["Confirm equiprobability", "Check probability is in [0,1]", "Reduce and perform requested numerator/denominator postprocessing"],
        "answer_schema": "reduced rational probability or requested derived integer", "solver_role": "probability_modeler",
    },
    {
        "id": "probability_expectation", "name": "Expectation and indicator variables",
        "domains": ["probability"], "triggers": ["expected", "expectation", "average number", "mean", "expected value"],
        "solution_signals": ["indicator", "linearity of expectation", "expected value"],
        "strategy": "Define indicator variables for atomic contributions and use linearity before attempting a full distribution; condition only when it simplifies dependencies.",
        "traps": ["Assuming indicators are independent", "Wrong contribution size", "Confusing expected ratio with ratio of expectations"],
        "verifier_checklist": ["Bound the expectation by the possible range", "Sum one-object probabilities", "Check symmetry assumptions"],
        "answer_schema": "exact rational or real expectation", "proof_methods": ["indicator variables", "linearity of expectation"],
        "solver_role": "expectation_solver",
    },
    {
        "id": "precalculus_trigonometry", "name": "Trigonometric identities and equations",
        "domains": ["algebra", "calculus"], "triggers": ["sin", "cos", "tan", "trigonometric", "angle measure", "radian", "period"],
        "solution_signals": ["identity", "unit circle", "law of sines", "law of cosines"],
        "strategy": "Normalize angles, reduce to a small identity set, preserve quadrant/sign information, and enumerate all solutions in the requested interval.",
        "traps": ["Missing periodic solutions", "Degree/radian mismatch", "Squaring introduces extraneous solutions"],
        "verifier_checklist": ["Substitute solutions into the original equation", "Check interval endpoints and quadrants", "State units"],
        "answer_schema": "exact trig value or sorted solution set", "solver_role": "trigonometry_solver",
    },
    {
        "id": "precalculus_complex_numbers", "name": "Complex numbers and polar form",
        "domains": ["algebra", "complex_analysis"], "triggers": ["complex number", "imaginary", "real part", "imaginary part", "modulus", "argument", "complex plane"],
        "solution_signals": ["polar form", "de moivre", "conjugate", "modulus"],
        "strategy": "Choose rectangular or polar form based on the operation, track all arguments/roots, and translate the requested component at the end.",
        "traps": ["Returning only one nth root", "Argument branch error", "Confusing modulus with squared modulus"],
        "verifier_checklist": ["Substitute into the original polynomial", "Check root count and angular spacing", "Verify real/imaginary extraction"],
        "answer_schema": "exact complex expression, component, or complete root set", "solver_role": "complex_number_solver",
    },
    {
        "id": "proof_contradiction", "name": "Contradiction and minimal counterexample",
        "domains": ["general"], "triggers": ["prove", "show that", "cannot", "impossible", "no such", "at least", "must"],
        "solution_signals": ["contradiction", "suppose not", "minimal counterexample", "assume for the sake"],
        "strategy": "Negate the exact conclusion, preserve all quantifiers, derive a concrete violation, and identify the line that closes the contradiction.",
        "traps": ["Assuming a stronger negation", "Circular contradiction", "Failing to handle equality"],
        "verifier_checklist": ["Check the negation is logically exact", "Identify where each hypothesis is used", "Confirm contradiction excludes every case"],
        "answer_schema": "complete proof", "proof_methods": ["contradiction", "minimal counterexample"],
        "solver_role": "contradiction_prover",
    },
    {
        "id": "proof_induction", "name": "Induction and strengthening",
        "domains": ["general"], "triggers": ["for every positive integer", "for all n", "for all positive integers", "sequence", "divisible for all"],
        "solution_signals": ["induction", "base case", "inductive hypothesis", "strong induction"],
        "strategy": "Choose the correct induction parameter, prove all required base cases, and strengthen the statement if the step needs additional structure.",
        "traps": ["Missing base cases", "Using the desired n+1 statement", "Induction cannot reach every residue class"],
        "verifier_checklist": ["Check reachability from base cases", "Mark the exact use of the inductive hypothesis", "Verify quantifiers"],
        "answer_schema": "complete proof", "proof_methods": ["induction", "strong induction"],
        "solver_role": "induction_prover",
    },
    {
        "id": "proof_extremal_invariant", "name": "Extremal principle and invariants",
        "domains": ["general"], "triggers": ["operation", "move", "process", "always", "eventually", "minimum", "maximum", "finite set"],
        "solution_signals": ["invariant", "monovariant", "extremal", "smallest", "largest"],
        "strategy": "Select an extremal object or a discrete invariant/monovariant, prove it behaves under every allowed operation, then derive termination or impossibility.",
        "traps": ["Quantity is not actually invariant", "No well-founded descent", "Extremal object need not exist"],
        "verifier_checklist": ["Test every move type", "Prove boundedness/well-ordering", "Connect the invariant to the stated conclusion"],
        "answer_schema": "complete proof", "proof_methods": ["extremal principle", "invariant", "infinite descent"],
        "solver_role": "invariant_prover",
    },
    {
        "id": "proof_pigeonhole", "name": "Pigeonhole and averaging",
        "domains": ["combinatorics", "number_theory", "general"], "triggers": ["at least two", "there exist", "some pair", "among", "selected", "subset"],
        "solution_signals": ["pigeonhole", "average", "boxes", "by averaging"],
        "strategy": "Define pigeons and holes so that a collision has exactly the desired meaning; use averaging when contributions have weights.",
        "traps": ["Too many holes", "Collision does not imply conclusion", "Floor/ceiling error"],
        "verifier_checklist": ["Count pigeons and holes explicitly", "Translate one collision back to the original objects", "Check strict versus weak bound"],
        "answer_schema": "existence proof or extremal integer", "proof_methods": ["pigeonhole principle", "averaging"],
        "solver_role": "pigeonhole_prover",
    },
]


SUBJECT_ALIASES = {
    "prealgebra": "algebra",
    "precalculus": "algebra",
    "counting_and_probability": "combinatorics",
}

ROUTER_STOPWORDS = {
    "the", "and", "that", "with", "from", "this", "then", "are", "for", "has",
    "have", "its", "let", "find", "what", "which", "when", "where", "given", "such",
}


def load_jsonl(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            records.extend(json.loads(line) for line in file if line.strip())
    return records


def matches_any(text: str, signals: Iterable[str]) -> bool:
    lower = text.lower()
    for signal in signals:
        token = signal.lower().strip()
        if re.fullmatch(r"[a-z0-9_]+", token):
            if re.search(rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])", lower):
                return True
        elif token and token in lower:
            return True
    return False


def distill(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for spec in TEMPLATES:
        problem_hits = []
        solution_hits = []
        for record in records:
            problem = str(record.get("problem") or "")
            solution = str(record.get("solution") or "")
            if matches_any(problem, spec["triggers"]):
                problem_hits.append(record)
            if matches_any(solution, spec.get("solution_signals", [])):
                solution_hits.append(record)
        answer_types = Counter(str(row.get("answer_type") or "unknown") for row in problem_hits)
        subjects = Counter(
            SUBJECT_ALIASES.get(str(subject).lower(), str(subject).lower())
            for row in problem_hits
            for subject in (row.get("subject") or ["unknown"])
        )
        item = {key: value for key, value in spec.items() if key != "solution_signals"}
        item.update(
            {
                "support": len(problem_hits),
                "solution_support": len(solution_hits),
                "proof_support": sum(1 for row in problem_hits if row.get("answer_type") == "proof"),
                "answer_types": dict(answer_types.most_common()),
                "subject_support": dict(subjects.most_common()),
                "example_ids": [str(row.get("id")) for row in problem_hits[:5]],
            }
        )
        output.append(item)
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def router_label(record: Dict[str, Any]) -> str:
    subjects = [str(subject).lower() for subject in (record.get("subject") or ["unknown"])]
    subject = subjects[0]
    problem = str(record.get("problem") or "").lower()
    if subject == "counting_and_probability":
        probability_markers = ["probability", "random", "expected", "dice", "card", "chance"]
        return "probability" if any(marker in problem for marker in probability_markers) else "combinatorics"
    if subject == "precalculus":
        calculus_markers = ["limit", "derivative", "integral", "continuous", "differentiable"]
        return "calculus" if any(marker in problem for marker in calculus_markers) else "algebra"
    return SUBJECT_ALIASES.get(subject, subject)


def router_tokens(text: str) -> List[str]:
    words = re.findall(r"\\[a-z]+|[a-z]{2,}|\d+", text.lower())
    words = [word for word in words if word not in ROUTER_STOPWORDS]
    bigrams = [f"{left}__{right}" for left, right in zip(words, words[1:])]
    return list(dict.fromkeys(words + bigrams))


def train_router(records: List[Dict[str, Any]], max_features: int = 3000) -> Dict[str, Any]:
    label_docs: Counter[str] = Counter()
    token_docs: Dict[str, Counter[str]] = {}
    global_docs: Counter[str] = Counter()
    for record in records:
        label = router_label(record)
        if label not in {"algebra", "calculus", "combinatorics", "geometry", "number_theory", "probability"}:
            continue
        label_docs[label] += 1
        tokens = router_tokens(str(record.get("problem") or ""))
        token_docs.setdefault(label, Counter()).update(tokens)
        global_docs.update(tokens)

    labels = sorted(label_docs)
    total_docs = sum(label_docs.values())
    candidates = [token for token, count in global_docs.items() if count >= 3]
    def informativeness(token: str) -> float:
        distribution = [token_docs.get(label, Counter()).get(token, 0) for label in labels]
        dominant = max(distribution) if distribution else 0
        return dominant * (dominant / max(1, sum(distribution)))
    vocabulary = sorted(candidates, key=lambda token: (-informativeness(token), -global_docs[token], token))[:max_features]

    alpha = 1.0
    log_priors = {label: math.log((label_docs[label] + alpha) / (total_docs + alpha * len(labels))) for label in labels}
    token_log_likelihoods: Dict[str, Dict[str, float]] = {}
    for label in labels:
        denominator = sum(token_docs[label].get(token, 0) for token in vocabulary) + alpha * len(vocabulary)
        token_log_likelihoods[label] = {
            token: math.log((token_docs[label].get(token, 0) + alpha) / denominator)
            for token in vocabulary
        }
    return {
        "model": "multinomial presence naive bayes",
        "labels": labels,
        "documents": dict(label_docs),
        "vocabulary_size": len(vocabulary),
        "log_priors": log_priors,
        "token_log_likelihoods": token_log_likelihoods,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill aggregate skill templates from local HARP JSONL data.")
    parser.add_argument(
        "--inputs", nargs="+", type=Path,
        default=[ROOT / "data" / "public_math_harp_all_short.jsonl", ROOT / "data" / "public_math_proof_harp.jsonl"],
    )
    parser.add_argument("--output", type=Path, default=ROOT / "agent" / "skill_catalog.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = load_jsonl(args.inputs)
    payload = {
        "schema_version": 1,
        "method": "curated template signatures with aggregate HARP problem/solution support counts",
        "privacy": "No problem text, reference solution, or answer is stored in this runtime artifact.",
        "sources": [
            {"path": str(path.relative_to(ROOT)), "records": sum(1 for line in path.open("r", encoding="utf-8") if line.strip()), "sha256": sha256(path)}
            for path in args.inputs
        ],
        "router": train_router(records),
        "templates": distill(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Distilled {len(payload['templates'])} templates from {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
