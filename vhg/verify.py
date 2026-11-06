from __future__ import annotations

import contextlib
import os
import random
import re
import string
import warnings
from collections import Counter
from functools import lru_cache
from typing import Any

import sympy as sp
from math_verify import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    parse as math_parse,
    verify as math_verify_compare,
)
from sympy.parsing.latex import parse_latex

from .prompts import question_format_prompt


QUESTION_BLOCK_RE = re.compile(
    r"<question>\s*(.*?)\s*</question>", re.DOTALL | re.IGNORECASE
)
INTEGRATION_LATEX_QUESTION_CONTENT_RE = re.compile(
    r"^\s*Integrate\s+the\s+expression\s+\$(.+?)\$\s+with\s+respect\s+to\s+\$(.+?)\$\.\s*$",
    re.DOTALL | re.IGNORECASE,
)
INTEGRATION_PLAIN_QUESTION_CONTENT_RES = (
    re.compile(
        r"^\s*Integrate\s+the\s+expression\s+(.+?)\s+with\s+respect\s+to\s+([A-Za-z]+)\.?\s*$",
        re.DOTALL | re.IGNORECASE,
    ),
    re.compile(
        r"^\s*Compute\s+the\s+indefinite\s+integral\s+of\s+(.+?)\s+with\s+respect\s+to\s+([A-Za-z]+)\.?\s*$",
        re.DOTALL | re.IGNORECASE,
    ),
    re.compile(
        r"^\s*Compute\s+the\s+indefinite\s+integral\s*:?\s*\u222b\s*(.+?)\s*d\s*([A-Za-z]+)\.?\s*$",
        re.DOTALL | re.IGNORECASE,
    ),
    re.compile(
        r"^\s*\u222b\s*(.+?)\s*d\s*([A-Za-z]+)\.?\s*$",
        re.DOTALL | re.IGNORECASE,
    ),
    re.compile(
        r"^\s*\\int\s*(.+?)\s*d\s*([A-Za-z]+)\.?\s*$",
        re.DOTALL | re.IGNORECASE,
    ),
)

MATH_VERIFY_EXTRACTION_CONFIG = (
    LatexExtractionConfig(),
    ExprExtractionConfig(),
)


@lru_cache(maxsize=1)
def latex_parser_health_error() -> str | None:
    try:
        parsed = parse_latex("x")
    except Exception as exc:
        return str(exc)
    if str(parsed) != "x":
        return f"parsed 'x' as {parsed!r}"
    return None


def ensure_latex_parser_available() -> None:
    error = latex_parser_health_error()
    if error is not None:
        raise RuntimeError(
            "SymPy LaTeX parsing is unavailable. Hard-verifier integration checks require "
            "antlr4-python3-runtime==4.11.0 with a compatible SymPy install. "
            f"Parser smoke error: {error}"
        )


def _find_expr(text: str, prefix: str = "\\boxed", find_last: bool = True) -> str | None:
    if not isinstance(text, str):
        return None
    prefix = prefix + "{"
    if prefix not in text:
        return None
    start = text.rfind(prefix) if find_last else text.find(prefix)
    ind = start + len(prefix)
    left = 1
    while ind < len(text):
        if text[ind] == "{":
            left += 1
        elif text[ind] == "}":
            left -= 1
            if left == 0:
                break
        ind += 1
    if left != 0:
        return None
    return text[start + len(prefix) : ind].strip()


def find_expr(text: str, prefix: str = "\\boxed") -> str | None:
    return _find_expr(text, prefix=prefix, find_last=True)


def find_expr_first(text: str, prefix: str = "\\boxed") -> str | None:
    return _find_expr(text, prefix=prefix, find_last=False)


def extract_first_tagged_triplet(text: str) -> tuple[str, str, str] | None:
    if not isinstance(text, str):
        return None
    original = find_expr_first(text, "\\original")
    integrand = find_expr_first(text, "\\integrand")
    variable = find_expr_first(text, "\\variable")
    if original is None or integrand is None or variable is None:
        return None
    return original, integrand, variable


def canonicalize_tagged_question(text: str) -> str:
    triplet = extract_first_tagged_triplet(text)
    if triplet is None:
        return text
    original, integrand, variable = triplet
    return (
        f"\\original{{{original}}} "
        f"\\integrand{{{integrand}}} "
        f"\\variable{{{variable}}}"
    )


def extract_question_block(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    match = QUESTION_BLOCK_RE.search(text)
    if not match:
        return None
    block = match.group(1).strip()
    return block or None


def extract_plain_integration_expr_var(surface: str) -> tuple[str | None, str | None]:
    if not isinstance(surface, str):
        return None, None
    stripped = surface.strip()
    if not stripped:
        return None, None
    for pattern in INTEGRATION_PLAIN_QUESTION_CONTENT_RES:
        match = pattern.match(stripped)
        if match:
            expr, var = match.groups()
            return expr.strip(), var.strip()
    return None, None


def extract_question_content_surface(
    text: str,
    *,
    question_type: str = "integration",
) -> str | None:
    question_block = extract_question_block(text) if isinstance(text, str) else None
    surface = question_block or text
    if not isinstance(surface, str):
        return None
    surface = surface.strip()
    if not surface:
        return None

    if question_type == "integration":
        expr = find_expr_first(surface, "\\expression")
        var = find_expr_first(surface, "\\variable")
        if expr is not None and var is not None:
            expr = expr.strip()
            var = var.strip()
            return expr if var == "x" else f"{expr} d{var}"
        latex_match = INTEGRATION_LATEX_QUESTION_CONTENT_RE.match(surface)
        if latex_match:
            expr, var = latex_match.groups()
            expr = expr.strip()
            var = var.strip()
            return expr if var == "x" else f"{expr} d{var}"
        plain_expr, plain_var = extract_plain_integration_expr_var(surface)
        if plain_expr is not None and plain_var is not None:
            return plain_expr if plain_var == "x" else f"{plain_expr} d{plain_var}"

    return re.sub(r"\s+", " ", surface).strip()


def make_symbols_real(expr):
    symbols_map = {sym: sp.symbols(sym.name, real=True) for sym in expr.free_symbols}
    return expr.subs(symbols_map)


def is_valid_variable(var_str: str, _whitelist: set[str] | None = None) -> bool:
    if _whitelist is None:
        latin_vars = set(string.ascii_letters)
        greek_vars = {
            "\\alpha",
            "\\beta",
            "\\gamma",
            "\\delta",
            "\\epsilon",
            "\\varepsilon",
            "\\zeta",
            "\\eta",
            "\\theta",
            "\\vartheta",
            "\\iota",
            "\\kappa",
            "\\lambda",
            "\\mu",
            "\\nu",
            "\\xi",
            "\\rho",
            "\\varrho",
            "\\sigma",
            "\\varsigma",
            "\\tau",
            "\\upsilon",
            "\\phi",
            "\\varphi",
            "\\chi",
            "\\psi",
            "\\omega",
            "\\Gamma",
            "\\Delta",
            "\\Theta",
            "\\Lambda",
            "\\Xi",
            "\\Upsilon",
            "\\Phi",
            "\\Psi",
            "\\Omega",
        }
        _whitelist = latin_vars.union(greek_vars)
    return isinstance(var_str, str) and var_str.strip() in _whitelist


def parse_latex_strict(latex_expr: str | None):
    if latex_expr is None:
        return None
    try:
        latex_expr = re.sub(r"\\frac\\pi(\d)", r"\\frac{\\pi}{\1}", latex_expr)
        latex_expr = re.sub(r"\\\(", "(", latex_expr)
        latex_expr = re.sub(r"\\\)", ")", latex_expr)
        parsed = parse_latex(latex_expr)
        parsed = parsed.subs("e", sp.E).subs("pi", sp.pi)
        parsed = make_symbols_real(parsed)
    except Exception:
        parsed = None
    return parsed


def normalize_expression_text_for_exact_dedup(expression_text: str | None) -> str | None:
    if not isinstance(expression_text, str):
        return None
    normalized = (
        expression_text.replace("\\right", "")
        .replace("\\left", "")
        .replace("\\displaystyle", "")
        .replace("\\,", " ")
        .replace("\\ ", " ")
        .replace("\\;", " ")
    )
    normalized = re.sub(r"\s+", "", normalized).strip()
    return normalized or None


def parse_symbolic_generation_question(
    text: str,
    *,
    require_original: bool = False,
    require_boxed: bool = False,
    question_type: str = "integration",
) -> dict[str, Any]:
    def result(
        parseable,
        normalized_question,
        expr,
        var,
        original,
        question_surface,
        question_content_surface,
        structural_valid,
    ):
        return {
            "parseable": parseable,
            "normalized_question": normalized_question,
            "expr": expr,
            "var": var,
            "original": original,
            "question_surface": question_surface,
            "question_content_surface": question_content_surface,
            "structural_valid": structural_valid,
        }

    if not isinstance(text, str):
        return result(False, None, None, None, None, None, None, False)

    normalized = canonicalize_tagged_question(text).replace(
        "\\integrand", "\\expression"
    )
    expr = find_expr_first(normalized, "\\expression")
    var = find_expr_first(normalized, "\\variable")
    original = find_expr_first(normalized, "\\original")
    boxed_answer = find_expr(text, "\\boxed")
    if original is None and boxed_answer is not None:
        original = boxed_answer
    question_surface = extract_question_block(text)
    question_content_surface = extract_question_content_surface(
        text,
        question_type=question_type,
    )
    structural_valid = True
    if require_boxed and (question_surface is None or boxed_answer is None):
        structural_valid = False
    if require_original and original is None:
        structural_valid = False
    if question_type == "integration" and (
        expr is None or var is None or not is_valid_variable(var)
    ):
        fallback_surface = question_surface or text
        plain_expr, plain_var = extract_plain_integration_expr_var(fallback_surface)
        if plain_expr is not None and plain_var is not None:
            expr, var = plain_expr, plain_var

    if expr is None or var is None or not is_valid_variable(var):
        return result(
            False,
            None,
            expr,
            var,
            original,
            question_surface,
            question_content_surface,
            structural_valid,
        )

    if parse_latex_strict(expr) is None:
        return result(
            False,
            None,
            expr,
            var,
            original,
            question_surface,
            question_content_surface,
            structural_valid,
        )

    if original is not None and parse_latex_strict(original) is None:
        return result(
            False,
            None,
            expr,
            var,
            original,
            question_surface,
            question_content_surface,
            structural_valid,
        )

    if require_original and original is None:
        return result(
            False,
            None,
            expr,
            var,
            original,
            question_surface,
            question_content_surface,
            structural_valid,
        )

    if require_boxed and boxed_answer is None:
        return result(
            False,
            None,
            expr,
            var,
            original,
            question_surface,
            question_content_surface,
            structural_valid,
        )

    if question_surface is None:
        question_surface = question_format_prompt(question_type, expr, var, latex=True)

    canonical_parts = []
    if original is not None:
        canonical_parts.append(f"\\original{{{original}}}")
    canonical_parts.append(f"\\expression{{{expr}}}")
    canonical_parts.append(f"\\variable{{{var}}}")
    return result(
        True,
        " ".join(canonical_parts),
        expr,
        var,
        original,
        question_surface,
        question_content_surface,
        structural_valid,
    )


def _normalize_boxed_candidate(solution_text: str) -> str | None:
    boxed = find_expr(solution_text, "\\boxed")
    if boxed is None:
        return None
    candidate = boxed.split("=")[-1].strip()
    candidate = (
        candidate.replace("\\mathrm{constant}", "")
        .replace("\\text{constant}", "")
        .replace("\\mathrm{C}", "")
        .replace("\\text{C}", "")
        .replace("+ C", "")
        .replace("- C", "")
        .replace("+C", "")
        .replace("-C", "")
        .replace("+ constant", "")
        .replace("- constant", "")
        .replace("constant", "")
    )
    candidate = (
        candidate.replace("\\right", "")
        .replace("\\left", "")
        .replace("\\displaystyle", "")
        .replace("\\,", " ")
        .replace("\\ ", " ")
        .replace("\\;", " ")
    ).strip()
    candidate = re.sub(r" +", " ", candidate)
    return candidate or None


def correctness_integral_without_timeout(question: str, solution: str) -> bool:
    var = find_expr(question, "\\variable")
    if var is None or not is_valid_variable(var):
        return False
    parsed_var = parse_latex_strict(var)
    if parsed_var is None:
        return False

    llm_answer = _normalize_boxed_candidate(solution)
    if llm_answer is None:
        return False
    if "int" in llm_answer.lower() or "integral" in llm_answer.lower():
        return False
    parsed_llm_answer = parse_latex_strict(llm_answer)
    if parsed_llm_answer is None:
        return False

    gt = find_expr(question, "\\expression")
    parsed_gt = parse_latex_strict(gt)
    if parsed_gt is None:
        return False

    try:
        diffed = sp.diff(parsed_llm_answer, parsed_var)
        difference = diffed - parsed_gt
        if difference == 0:
            return True

        test_points = []
        for i in range(0, 1000, 50):
            test_points.extend(
                [i + 1e-5, -i - 1e-5, 1 / (i + 1 + 1e-5), -1 / (1 + i + 1e-5)]
            )

        free_symbols = list(difference.free_symbols - {parsed_var})
        valid_evaluations = 0
        max_rel_error = 0.0

        for point in test_points:
            subs_dict = {parsed_var: point}
            for sym in free_symbols:
                subs_dict[sym] = random.uniform(1, 5)
            try:
                val_diff = difference.evalf(n=50, subs=subs_dict)
                val_gt = parsed_gt.evalf(n=50, subs=subs_dict)
                if (
                    not val_diff.is_finite
                    or not val_gt.is_finite
                    or sp.Abs(val_diff) > sp.Float("1e50", precision=50)
                    or sp.Abs(val_gt) > sp.Float("1e50", precision=50)
                ):
                    continue
                if abs(sp.im(val_gt)) > 1e-20:
                    continue
                abs_diff = sp.Abs(val_diff)
                abs_gt = sp.Abs(val_gt)
                rel_error = float(abs_diff / (abs_gt + sp.Float("1e-100", precision=50)))
                max_rel_error = max(max_rel_error, rel_error)
                valid_evaluations += 1
            except Exception:
                continue

        if valid_evaluations < len(test_points) // 10:
            return False
        return max_rel_error < 1e-20
    except Exception:
        return False


def correctness(question: str, question_type: str, solution: str) -> bool:
    if question_type == "integration":
        return correctness_integral_without_timeout(question, solution)
    raise ValueError(f"Unknown question type: {question_type}")


def check_batch(items: list[dict[str, Any]]) -> list[bool]:
    return [
        bool(
            correctness(
                item["question"],
                item.get("question_type", "integration"),
                item["solution"],
            )
        )
        for item in items
    ]


def extract_boxed_answers(text: str) -> tuple[str, ...]:
    if not text:
        return ()
    token = "\\boxed{"
    idx = 0
    answers: list[str] = []
    while True:
        start = text.find(token, idx)
        if start == -1:
            break
        i = start + len(token)
        depth = 1
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    boxed = text[start + len(token) : i]
                    normalized = normalize_boxed_text(boxed)
                    if normalized:
                        answers.append(normalized)
                    idx = i + 1
                    break
            i += 1
        else:
            break
    return tuple(answers)


def extract_last_boxed(text: str) -> str:
    answers = extract_boxed_answers(text)
    return answers[-1] if answers else ""


def extract_single_boxed_answer(text: str) -> str:
    answers = extract_boxed_answers(text)
    return answers[0] if len(answers) == 1 else ""


def normalize_boxed_text(text: str) -> str:
    if not text:
        return ""
    out = re.sub(r"\s+", "", text.strip())
    out = out.strip("$")
    out = out.replace("\\left", "").replace("\\right", "")
    out = out.replace("\\,", "")
    while "\\\\" in out:
        out = out.replace("\\\\", "\\")
    return out


@lru_cache(maxsize=4096)
def parse_math_text(text: str) -> tuple[Any, ...]:
    if not text:
        return ()
    try:
        return tuple(
            math_parse(
                text,
                extraction_config=MATH_VERIFY_EXTRACTION_CONFIG,
                fallback_mode="first_match",
                extraction_mode="any_match",
                raise_on_error=False,
            )
        )
    except Exception:
        return ()


@lru_cache(maxsize=4096)
def parse_boxed_answer(text: str) -> tuple[Any, ...]:
    normalized = normalize_boxed_text(text)
    if not normalized:
        return ()
    return parse_math_text(f"\\boxed{{{normalized}}}")


def extract_single_parseable_boxed_answer(text: str) -> tuple[str, tuple[Any, ...]]:
    boxed = extract_single_boxed_answer(text)
    if not boxed:
        return "", ()
    parsed = parse_boxed_answer(boxed)
    if not parsed:
        return "", ()
    return boxed, parsed


def parsed_answers_equal(left: tuple[Any, ...] | list[Any], right: tuple[Any, ...] | list[Any]) -> bool:
    if not left or not right:
        return False
    try:
        return bool(
            math_verify_compare(
                list(left),
                list(right),
                allow_set_relation_comp=True,
                raise_on_error=False,
            )
        )
    except Exception:
        return False


def solver_output_targets(solution_text: str) -> tuple[Any, ...]:
    return parse_boxed_answer(extract_single_boxed_answer(solution_text))


def compute_math_pass_rate(outputs: list[str], reference_targets: tuple[Any, ...]) -> float:
    if not outputs or not reference_targets:
        return 0.0
    return sum(
        parsed_answers_equal(solver_output_targets(text), reference_targets)
        for text in outputs
    ) / max(len(outputs), 1)


def parse_math_setter_response(text: str) -> dict[str, str] | None:
    if not text:
        return None
    pattern = re.compile(
        r"^\s*<reasoning>\s*(.*?)\s*</reasoning>\s*"
        r"<derived_problem>\s*(.*?)\s*</derived_problem>\s*"
        r"<modified_solution>\s*(.*?)\s*</modified_solution>\s*$",
        flags=re.DOTALL | re.IGNORECASE,
    )
    match = pattern.match(text)
    if not match:
        return None
    reasoning, derived_problem, modified_solution = (
        group.strip() for group in match.groups()
    )
    if not reasoning or not derived_problem or not modified_solution:
        return None
    return {
        "reasoning": reasoning,
        "derived_problem": derived_problem,
        "modified_solution": modified_solution,
    }


def format_math_setter_response(
    *,
    reasoning: str,
    derived_problem: str,
    modified_solution: str,
) -> str:
    return (
        f"<reasoning>\n{str(reasoning).strip()}\n</reasoning>\n"
        f"<derived_problem>\n{str(derived_problem).strip()}\n</derived_problem>\n"
        f"<modified_solution>\n{str(modified_solution).strip()}\n</modified_solution>"
    )


def _parse_bool_tag(value: str) -> bool | None:
    lowered = str(value or "").strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def parse_soft_verifier_response(text: str) -> tuple[str, dict[str, Any]] | None:
    if not text:
        return None
    pattern = re.compile(
        r"^\s*<reasoning>\s*(.*?)\s*</reasoning>\s*"
        r"<valid_problem>\s*(.*?)\s*</valid_problem>\s*"
        r"<valid_solution>\s*(.*?)\s*</valid_solution>\s*"
        r"<seed_anchored>\s*(.*?)\s*</seed_anchored>\s*"
        r"<not_trivial_copy>\s*(.*?)\s*</not_trivial_copy>\s*"
        r"<complete_final_answer>\s*(.*?)\s*</complete_final_answer>\s*$",
        flags=re.DOTALL | re.IGNORECASE,
    )
    match = pattern.match(text)
    if not match:
        return None
    (
        reasoning,
        valid_problem_text,
        valid_solution_text,
        seed_anchored_text,
        not_trivial_copy_text,
        complete_final_answer_text,
    ) = (group.strip() for group in match.groups())
    if not reasoning:
        return None
    values = {
        "valid_problem": _parse_bool_tag(valid_problem_text),
        "valid_solution": _parse_bool_tag(valid_solution_text),
        "seed_anchored": _parse_bool_tag(seed_anchored_text),
        "not_trivial_copy": _parse_bool_tag(not_trivial_copy_text),
        "complete_final_answer": _parse_bool_tag(complete_final_answer_text),
    }
    if any(value is None for value in values.values()):
        return None
    return reasoning, values


WORDLIKE_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
TEXTUAL_ANSWER_COMMAND_RE = re.compile(
    r"\\(?:text|mathrm|textrm|mbox|operatorname|textbf|textit)\b",
    re.IGNORECASE,
)
MALFORMED_BOXED_ANSWER_RE = re.compile(r"[#@]")
BARE_WORD_ANSWER_RE = re.compile(r"^[A-Za-z]{4,}$")

MATH_PAIR_SELF_CORRECTION_PATTERNS = (
    ("wait_discourse", re.compile(r"\bwait(?:\s+a\s+second)?\b", re.IGNORECASE)),
    ("actually_correction", re.compile(r"\bactually\s*,", re.IGNORECASE)),
    ("self_correction", re.compile(r"\b(?:i|we)\s+(?:was|were)\s+wrong\b", re.IGNORECASE)),
    ("misread", re.compile(r"\b(?:i|we)\s+(?:misread|misinterpreted)\b", re.IGNORECASE)),
    ("careful_preface", re.compile(r"\b(?:but\s+)?careful\s*:", re.IGNORECASE)),
    (
        "restart_cue",
        re.compile(r"\b(?:let me|we should)\s+(?:correct|revise|restart|start over)\b", re.IGNORECASE),
    ),
)


def _self_correction_reason(text: str) -> str | None:
    text = str(text or "").lower()
    for label, pattern in MATH_PAIR_SELF_CORRECTION_PATTERNS:
        if pattern.search(text):
            return f"self_correction_phrase:{label}"
    return None


def math_text_word_count(text: str) -> int:
    return len(WORDLIKE_TOKEN_RE.findall(str(text or "")))


def _boxed_answer_local_reject_reason(boxed_answer: str) -> str | None:
    boxed_answer = str(boxed_answer or "").strip()
    if not boxed_answer:
        return "missing_boxed_answer"

    collapsed_letters = re.sub(r"[^a-z]+", "", boxed_answer.lower())
    for needle in (
        "nosuch",
        "nosolution",
        "doesnotexist",
        "emptyset",
        "undefined",
        "impossible",
        "inconsistent",
    ):
        if needle in collapsed_letters:
            return f"degenerate_boxed_answer:{needle}"

    if TEXTUAL_ANSWER_COMMAND_RE.search(boxed_answer):
        return "textual_boxed_answer"
    if MALFORMED_BOXED_ANSWER_RE.search(boxed_answer):
        return "malformed_boxed_answer"
    if BARE_WORD_ANSWER_RE.fullmatch(boxed_answer):
        return "textual_boxed_answer"
    if not parse_boxed_answer(boxed_answer):
        return "unparseable_boxed_answer"
    return None


def math_repeated_bigram_ratio(text: str) -> float:
    tokens = [token.lower() for token in WORDLIKE_TOKEN_RE.findall(str(text or ""))]
    if len(tokens) < 2:
        return 0.0
    bigrams = list(zip(tokens, tokens[1:]))
    repeated = sum(count - 1 for count in Counter(bigrams).values() if count > 1)
    return repeated / len(bigrams)


def math_setter_high_repetition(
    text: str,
    *,
    min_word_count: int = 160,
    min_repeated_bigram_ratio: float = 0.18,
) -> bool:
    return (
        math_text_word_count(text) >= min_word_count
        and math_repeated_bigram_ratio(text) >= min_repeated_bigram_ratio
    )


def math_cold_sft_priority_key(
    *,
    reasoning: str,
    derived_problem: str,
    modified_solution: str,
) -> tuple[Any, ...]:
    reasoning_words = math_text_word_count(reasoning)
    problem_words = math_text_word_count(derived_problem)
    solution_words = math_text_word_count(modified_solution)
    response = format_math_setter_response(
        reasoning=reasoning,
        derived_problem=derived_problem,
        modified_solution=modified_solution,
    )
    repetition = math_repeated_bigram_ratio(response)
    return (
        int(math_setter_high_repetition(response)),
        repetition,
        reasoning_words + problem_words + solution_words,
        solution_words,
        reasoning_words,
    )


def math_cold_sft_reject_reason(
    *,
    reasoning: str,
    derived_problem: str,
    modified_solution: str,
) -> str | None:
    correction = _self_correction_reason(
        f"{reasoning}\n{derived_problem}\n{modified_solution}"
    )
    if correction is not None:
        return correction
    final_answer = extract_single_boxed_answer(modified_solution) or ""
    return _boxed_answer_local_reject_reason(final_answer)


def math_setter_local_reject_reason(
    *,
    derived_problem: str,
    modified_solution: str,
) -> str | None:
    self_correction_reason = _self_correction_reason(
        f"{derived_problem}\n{modified_solution}"
    )
    if self_correction_reason is not None:
        return self_correction_reason
    final_answer = extract_single_boxed_answer(modified_solution) or ""
    return _boxed_answer_local_reject_reason(final_answer)


def normalize_problem_identity(problem: str) -> str:
    text = str(problem or "").lower()
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[{}$]", "", text)
    return text.strip()


def normalize_problem_template_identity(problem: str) -> str:
    text = normalize_problem_identity(problem)
    text = re.sub(r"\d+(?:\.\d+)?", "0", text)
    return re.sub(r"\s+", " ", text).strip()


def has_invalid_unicode_surrogate(text: str) -> bool:
    return any(0xD800 <= ord(ch) <= 0xDFFF for ch in str(text or ""))


def replace_invalid_unicode_surrogates(text: str) -> str:
    return str(text or "").encode("utf-8", errors="backslashreplace").decode("utf-8")


class NumericalFingerprinter:
    def __init__(self, num_points: int = 20, precision: int = 50, rounding: int = 18):
        num_points = max(9, num_points)
        self.test_points = []
        for i in range(0, num_points * 10, 10):
            self.test_points.extend(
                [i + 1e-5, -i - 1e-5, 1 / (i + 1 + 1e-5), -1 / (1 + i + 1e-5)]
            )
        self.test_points = self.test_points[:num_points]
        self.precision = precision
        self.rounding = rounding
        self.seen_hashes: set[str] = set()

    def _evaluate(self, expr):
        import hashlib

        symbols = sorted(expr.free_symbols, key=lambda s: s.name)
        base_symbol = symbols[0] if symbols else None
        fixed_values = {
            sym: sp.Float(f"0.{i + 2}", self.precision)
            for i, sym in enumerate(symbols[1:])
        }
        rounded_values = []
        for point in self.test_points:
            subs = dict(fixed_values)
            if base_symbol is not None:
                subs[base_symbol] = sp.Float(point, self.precision)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(
                        devnull
                    ), contextlib.redirect_stderr(devnull):
                        val = sp.N(expr.subs(subs), self.precision)
                if not getattr(val, "is_finite", False) or val.has(sp.I):
                    continue
                rounded_values.append(format(sp.N(val, self.rounding), f".{self.rounding}g"))
            except Exception:
                continue
        if len(rounded_values) < max(3, len(self.test_points) // 4):
            return None
        return hashlib.md5("|".join(rounded_values).encode()).hexdigest()

    def is_new(self, expr) -> bool:
        fp = self._evaluate(expr)
        if fp is None:
            return True
        if fp in self.seen_hashes:
            return False
        self.seen_hashes.add(fp)
        return True


def fingerprint_sympy_expr(expr, num_points: int = 5, precision: int = 50, rounding: int = 18) -> str | None:
    return NumericalFingerprinter(
        num_points=num_points,
        precision=precision,
        rounding=rounding,
    )._evaluate(expr)


def analyze_integrand_fingerprint(question_text: str, num_points: int = 5) -> dict[str, Any]:
    parsed_question = parse_symbolic_generation_question(
        question_text,
        require_original=False,
        question_type="integration",
    )
    if not parsed_question.get("parseable", False):
        return {
            "parseable": False,
            "integrand_fingerprint": None,
            "integrand_normalized_text": None,
            "expr": parsed_question.get("expr"),
            "var": parsed_question.get("var"),
            "original": parsed_question.get("original"),
        }
    parsed_expr = parse_latex_strict(parsed_question["expr"])
    return {
        "parseable": True,
        "integrand_fingerprint": fingerprint_sympy_expr(parsed_expr, num_points=num_points),
        "integrand_normalized_text": normalize_expression_text_for_exact_dedup(
            parsed_question["expr"]
        ),
        "expr": parsed_question["expr"],
        "var": parsed_question["var"],
        "original": parsed_question.get("original"),
    }


def analyze_expression_fingerprint(expression_text: str | None, num_points: int = 5) -> dict[str, Any]:
    if not expression_text:
        return {
            "parseable": False,
            "fingerprint": None,
            "expr": expression_text,
        }
    parsed_expr = parse_latex_strict(expression_text)
    if parsed_expr is None:
        return {
            "parseable": False,
            "fingerprint": None,
            "expr": expression_text,
        }
    return {
        "parseable": True,
        "fingerprint": fingerprint_sympy_expr(parsed_expr, num_points=num_points),
        "expr": expression_text,
    }


def analyze_service_item(item: dict[str, Any]) -> dict[str, Any]:
    operation = item.get("operation")
    if operation == "integrand_fingerprint":
        return analyze_integrand_fingerprint(
            item.get("question", ""),
            num_points=int(item.get("num_points", 5)),
        )
    if operation == "expression_fingerprint":
        return analyze_expression_fingerprint(
            item.get("expression"),
            num_points=int(item.get("num_points", 5)),
        )
    raise ValueError(f"Unknown analysis operation: {operation}")
