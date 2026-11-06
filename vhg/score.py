from __future__ import annotations

import concurrent.futures as cfuts
import os
import re
from typing import Any, Callable

from .prompts import question_format_prompt, question_solution_prompt
from .utils import compute_pass_metrics
from .verify import (
    canonicalize_tagged_question,
    check_batch,
    compute_math_pass_rate,
    extract_boxed_answers,
    extract_last_boxed,
    extract_single_boxed_answer,
    extract_single_parseable_boxed_answer,
    find_expr,
    has_invalid_unicode_surrogate,
    is_valid_variable,
    math_setter_local_reject_reason,
    normalize_boxed_text,
    normalize_problem_identity,
    parse_boxed_answer,
    parse_latex_strict,
    parse_math_text,
    parse_math_setter_response,
    parse_symbolic_generation_question,
    parsed_answers_equal,
    replace_invalid_unicode_surrogates,
    solver_output_targets,
)


def _default_correctness_service_url() -> str | None:
    import os

    explicit = os.environ.get("CORRECTNESS_SERVICE_URL", "").strip()
    if explicit:
        return explicit
    port = os.environ.get("STANDALONE_SERVICE_PORT", "").strip()
    if port:
        return f"http://localhost:{port}"
    return None


def _check_items(items: list[dict[str, Any]]) -> list[bool]:
    service_url = _default_correctness_service_url()
    if service_url:
        from .service import CorrectnessClient

        with CorrectnessClient(service_url=service_url) as client:
            return client.check_batch(items)
    return check_batch(items)


def objective_score(pass_rate: float, objective: str) -> float:
    if objective == "hard":
        return max(0.0, 1.0 - pass_rate)
    raise ValueError(f"Unknown objective: {objective}")


def _hard_verifier_setter_result() -> dict[str, Any]:
    return {
        "score": -1.0,
        "format_valid": False,
        "reference_symbolic_valid": False,
        "is_matching": False,
        "accuracy": 0.0,
        "solver_hard_verifier_pass_rate": 0.0,
        "solver_hard_verifier_hardness": 0.0,
        "objective": "",
        "active_reward_source": "hard_verifier",
        "active_validity_metric": "reference_symbolic_valid",
        "active_difficulty_metric": "solver_hard_verifier_pass_rate",
    }


def _hard_verifier_setter_objective(extra_info: dict[str, Any] | None) -> str:
    extra_info = extra_info or {}
    if extra_info.get("objective"):
        return str(extra_info["objective"]).lower()
    return "hard"


def _score_hard_verifier_reference_valid(pass_rate: float, extra_info: dict[str, Any] | None) -> float:
    return objective_score(pass_rate, _hard_verifier_setter_objective(extra_info))


def _is_valid_solver_solution(solution_str: str) -> bool:
    boxed_solution = find_expr(solution_str, "\\boxed")
    if boxed_solution is None:
        return False
    if "int" in boxed_solution.lower() or "integral" in boxed_solution.lower():
        return False
    candidate = boxed_solution.split("=")[-1].strip()
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
    return parse_latex_strict(candidate) is not None


def _solver_validation_task(args: tuple[int, str, dict[str, Any]]) -> dict[str, Any]:
    idx, solution_str, extra_info = args
    if not _is_valid_solver_solution(solution_str):
        return {"idx": idx, "is_valid": False}
    return {
        "idx": idx,
        "is_valid": True,
        "question": extra_info["question"],
        "question_type": extra_info.get("question_type", "integration"),
        "solution": solution_str,
    }


def compute_score_solver_batched(
    solution_strs: list[str],
    extra_infos: list[dict[str, Any]],
) -> list[float]:
    if len(solution_strs) != len(extra_infos):
        raise ValueError("solution_strs and extra_infos must have the same length")
    if not solution_strs:
        return []

    scores = [-1.0] * len(solution_strs)
    validation_args = [
        (idx, solution_strs[idx], extra_infos[idx] or {})
        for idx in range(len(solution_strs))
    ]

    max_workers = max(1, (os.cpu_count() or 1) // 2)
    with cfuts.ProcessPoolExecutor(max_workers=max_workers) as executor:
        validation_results = list(executor.map(_solver_validation_task, validation_args))

    requests_batch = []
    request_indices = []
    for result in validation_results:
        if result["is_valid"]:
            requests_batch.append(
                {
                    "question": result["question"],
                    "question_type": result["question_type"],
                    "solution": result["solution"],
                }
            )
            request_indices.append(result["idx"])

    if not requests_batch:
        return scores

    correctness_flags = _check_items(requests_batch)
    for idx, flag in zip(request_indices, correctness_flags):
        scores[idx] = float(flag)
    return scores


def _parse_hard_verifier_setter_solution(
    solution_str: str,
) -> dict[str, Any]:
    normalized_solution = canonicalize_tagged_question(solution_str).replace(
        "\\integrand", "\\expression"
    )
    expr = find_expr(normalized_solution, "\\expression")
    var = find_expr(normalized_solution, "\\variable")
    if (
        expr is None
        or var is None
        or parse_latex_strict(expr) is None
        or not is_valid_variable(var)
    ):
        return {"is_valid": False}

    original_func = find_expr(normalized_solution, "\\original")
    if original_func is None or parse_latex_strict(original_func) is None:
        return {"is_valid": False}

    return {
        "is_valid": True,
        "normalized_solution": normalized_solution,
        "expr": expr,
        "var": var,
        "original_func": original_func,
    }


def _hard_verifier_solver_sampling_from_info(
    extra_info: dict[str, Any],
) -> tuple[int, float, float | None, int | None, int]:
    top_p = extra_info.get("top_p")
    top_k = extra_info.get("top_k")
    return (
        int(extra_info.get("sample_num", 1)),
        float(extra_info.get("temperature", 1.0)),
        None if top_p in (None, "") else float(top_p),
        None if top_k in (None, "") else int(top_k),
        int(extra_info.get("max_tokens", 4096)),
    )


def score_hard_verifier_setter_candidates_batched(
    solution_strs: list[str],
    extra_infos: list[dict[str, Any]],
    *,
    solver_sampler: Callable[..., dict[str, list[str]]] | None = None,
) -> list[dict[str, Any]]:
    results = []
    for extra_info in extra_infos:
        result = _hard_verifier_setter_result()
        result["objective"] = _hard_verifier_setter_objective(extra_info)
        results.append(result)
    valid_entries = []
    for idx, (solution_str, extra_info) in enumerate(zip(solution_strs, extra_infos)):
        extra_info = extra_info or {"question_type": "integration"}
        parsed = _parse_hard_verifier_setter_solution(solution_str)
        if not parsed["is_valid"]:
            continue
        results[idx]["objective"] = _hard_verifier_setter_objective(extra_info)
        results[idx]["format_valid"] = True
        question = question_format_prompt(
            extra_info.get("question_type", "integration"),
            parsed["expr"],
            parsed["var"],
            latex=True,
        )
        valid_entries.append(
            {
                "idx": idx,
                "prompt": question_solution_prompt(question),
                "question": parsed["normalized_solution"],
                "question_type": extra_info.get("question_type", "integration"),
                "original_func": parsed["original_func"],
                "extra_info": extra_info,
            }
        )

    if not valid_entries:
        return results

    verify_requests = [
        {
            "question": entry["question"],
            "question_type": entry["question_type"],
            "solution": f"\\boxed{{{entry['original_func']}}}",
        }
        for entry in valid_entries
    ]
    for entry, match_flag in zip(valid_entries, _check_items(verify_requests)):
        results[entry["idx"]]["is_matching"] = bool(match_flag)
        results[entry["idx"]]["reference_symbolic_valid"] = bool(match_flag)
    matching_entries = []
    for entry in valid_entries:
        if results[entry["idx"]]["reference_symbolic_valid"]:
            matching_entries.append(entry)
        else:
            results[entry["idx"]]["score"] = 0.0
    valid_entries = matching_entries
    if not valid_entries:
        return results

    from .generate import VHGRuntimeConfig, config_from_env, sample_local_solver_prompts

    base_config = config_from_env()
    if not base_config.solver_ckpt_path:
        raise ValueError("Set SOLVER_CKPT_PATH before calling hard verifier setter reward.")
    if not base_config.solver_visible_devices:
        raise ValueError(
            "SOLVER_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES must be set before "
            "calling hard verifier setter reward."
        )

    grouped_entries: dict[tuple[int, float, float | None, int | None, int], list[dict[str, Any]]] = {}
    for entry in valid_entries:
        key = _hard_verifier_solver_sampling_from_info(entry["extra_info"])
        grouped_entries.setdefault(key, []).append(entry)

    for key, entries in grouped_entries.items():
        sample_num, temperature, top_p, top_k, max_tokens = key
        indexed_prompts = [
            {"idx": str(entry["idx"]), "prompt": entry["prompt"]} for entry in entries
        ]
        if solver_sampler is not None:
            solutions_map = solver_sampler(
                indexed_prompts=indexed_prompts,
                config=base_config,
                sample_num=sample_num,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
            )
        else:
            solutions_map = sample_local_solver_prompts(
                indexed_prompts,
                config=VHGRuntimeConfig(
                    solver_ckpt_path=base_config.solver_ckpt_path,
                    solver_visible_devices=base_config.solver_visible_devices,
                    solver_gpu_memory_utilization=base_config.solver_gpu_memory_utilization,
                ),
                sample_num=sample_num,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
            )

        correctness_requests = []
        offsets = {}
        for entry in entries:
            solutions = solutions_map.get(str(entry["idx"]), [])
            offsets[entry["idx"]] = (len(correctness_requests), len(solutions))
            for solution in solutions:
                correctness_requests.append(
                    {
                        "question": entry["question"],
                        "question_type": entry["question_type"],
                        "solution": solution,
                    }
                )
        correctness_flags = _check_items(correctness_requests) if correctness_requests else []
        for entry in entries:
            start, count = offsets[entry["idx"]]
            flags = correctness_flags[start : start + count]
            acc = float(sum(flags)) / len(flags) if flags else 0.0
            results[entry["idx"]]["accuracy"] = acc
            results[entry["idx"]]["solver_hard_verifier_pass_rate"] = acc
            results[entry["idx"]]["solver_hard_verifier_hardness"] = 1.0 - acc
            results[entry["idx"]]["score"] = _score_hard_verifier_reference_valid(
                acc,
                entry["extra_info"],
            )

    return results


def enrich_hard_verifier_solver_metrics_batched(
    *,
    solution_strs: list[str],
    extra_infos: list[dict[str, Any]],
    validity_results: list[dict[str, Any]],
    solver_sampler: Callable[..., dict[str, list[str]]] | None = None,
) -> list[dict[str, Any]]:
    results = [dict(row) for row in validity_results]
    solver_entries = []
    for idx, (solution_str, extra_info) in enumerate(zip(solution_strs, extra_infos)):
        if idx >= len(results) or not results[idx].get("reference_symbolic_valid"):
            continue
        parsed = parse_symbolic_generation_question(
            solution_str,
            require_original=True,
            require_boxed=False,
            question_type=extra_info.get("question_type", "integration"),
        )
        if not parsed.get("parseable"):
            continue
        question = question_format_prompt(
            extra_info.get("question_type", "integration"),
            parsed["expr"],
            parsed["var"],
            latex=True,
        )
        solver_entries.append(
            {
                "idx": idx,
                "prompt": question_solution_prompt(question),
                "question": parsed["normalized_question"],
                "question_type": extra_info.get("question_type", "integration"),
                "extra_info": extra_info,
            }
        )

    if not solver_entries:
        return results

    from .generate import VHGRuntimeConfig, config_from_env, sample_local_solver_prompts

    base_config = config_from_env()
    if not base_config.solver_ckpt_path:
        raise ValueError("Set SOLVER_CKPT_PATH before enriching hard verifier solver metrics.")
    if not base_config.solver_visible_devices:
        raise ValueError(
            "SOLVER_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES must be set before "
            "enriching hard verifier solver metrics."
        )

    grouped_entries: dict[tuple[int, float, float | None, int | None, int], list[dict[str, Any]]] = {}
    for entry in solver_entries:
        key = _hard_verifier_solver_sampling_from_info(entry["extra_info"])
        grouped_entries.setdefault(key, []).append(entry)

    for key, entries in grouped_entries.items():
        sample_num, temperature, top_p, top_k, max_tokens = key
        indexed_prompts = [
            {"idx": str(entry["idx"]), "prompt": entry["prompt"]} for entry in entries
        ]
        if solver_sampler is not None:
            solutions_map = solver_sampler(
                indexed_prompts=indexed_prompts,
                config=base_config,
                sample_num=sample_num,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
            )
        else:
            solutions_map = sample_local_solver_prompts(
                indexed_prompts,
                config=VHGRuntimeConfig(
                    solver_ckpt_path=base_config.solver_ckpt_path,
                    solver_visible_devices=base_config.solver_visible_devices,
                    solver_gpu_memory_utilization=base_config.solver_gpu_memory_utilization,
                ),
                sample_num=sample_num,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
            )

        hard_verifier_requests = []
        hard_verifier_offsets = {}
        for entry in entries:
            solutions = solutions_map.get(str(entry["idx"]), [])
            hard_verifier_offsets[entry["idx"]] = (len(hard_verifier_requests), len(solutions))
            for solution in solutions:
                hard_verifier_requests.append(
                    {
                        "question": entry["question"],
                        "question_type": entry["question_type"],
                        "solution": solution,
                    }
                )

        hard_verifier_flags = _check_items(hard_verifier_requests) if hard_verifier_requests else []
        for entry in entries:
            start, count = hard_verifier_offsets[entry["idx"]]
            flags = hard_verifier_flags[start : start + count]
            hard_verifier_acc = float(sum(flags)) / len(flags) if flags else 0.0
            result = results[entry["idx"]]
            result["solver_hard_verifier_pass_rate"] = hard_verifier_acc
            result["solver_hard_verifier_hardness"] = 1.0 - hard_verifier_acc

    return results


def score_hard_verifier_analysis_result(
    result: dict[str, Any],
    *,
    objective: str = "hard",
) -> dict[str, Any]:
    scored = dict(result)
    scored["objective"] = objective
    scored["active_reward_source"] = "hard_verifier"
    scored["active_validity_metric"] = "reference_symbolic_valid"
    scored["active_difficulty_metric"] = "solver_hard_verifier_pass_rate"
    if not scored["format_valid"]:
        scored["score"] = -1.0
    elif not scored["reference_symbolic_valid"]:
        scored["score"] = 0.0
    else:
        scored["score"] = objective_score(
            float(scored.get("solver_hard_verifier_pass_rate", 0.0)),
            objective,
        )
    return scored


def evaluate_math_candidate_record(
    *,
    seed_problem: str,
    seed_solution: str,
    reasoning: str,
    derived_problem: str,
    modified_solution: str,
    verifier_payload: dict[str, Any] | None,
    original_solver_outputs: list[str] | None = None,
    modified_solver_outputs: list[str] | None = None,
    objective: str = "hard",
    metadata: dict[str, Any] | None = None,
    generation_trace: dict[str, Any] | None = None,
    record_id: int = 0,
) -> dict[str, Any]:
    reasoning = str(reasoning or "").strip()
    derived_problem = str(derived_problem or "").strip()
    modified_solution = str(modified_solution or "").strip()
    invalid_unicode = (
        has_invalid_unicode_surrogate(reasoning)
        or has_invalid_unicode_surrogate(derived_problem)
        or has_invalid_unicode_surrogate(modified_solution)
    )
    if invalid_unicode:
        reasoning = replace_invalid_unicode_surrogates(reasoning)
        derived_problem = replace_invalid_unicode_surrogates(derived_problem)
        modified_solution = replace_invalid_unicode_surrogates(modified_solution)

    modified_boxed_answers = extract_boxed_answers(modified_solution)
    modified_solution_single_boxed = extract_single_boxed_answer(modified_solution)
    modified_solution_boxed, modified_targets = extract_single_parseable_boxed_answer(
        modified_solution
    )
    original_reference_boxed, original_targets = extract_single_parseable_boxed_answer(
        seed_solution
    )
    if not original_reference_boxed or not original_targets:
        raise ValueError("Seed solution must contain exactly one parseable boxed answer.")

    structural_valid = bool(
        reasoning
        and derived_problem
        and modified_solution
        and not invalid_unicode
        and len(modified_boxed_answers) == 1
        and modified_targets
    )
    local_reject_reason = math_setter_local_reject_reason(
        derived_problem=derived_problem,
        modified_solution=modified_solution,
    )
    if not reasoning or not derived_problem or not modified_solution:
        failure_reason = "invalid_format"
    elif invalid_unicode:
        failure_reason = "invalid_unicode_surrogate"
    elif len(modified_boxed_answers) != 1:
        failure_reason = "expected_exactly_one_boxed_answer"
    elif not modified_targets:
        failure_reason = "unparseable_boxed_answer"
    elif normalize_problem_identity(derived_problem) == normalize_problem_identity(seed_problem):
        failure_reason = "same_as_seed_problem"
    elif local_reject_reason is not None:
        failure_reason = f"local_reject:{local_reject_reason}"
    else:
        failure_reason = ""

    record = {
        "id": record_id,
        "baseline": "VHG (Soft)",
        "baseline_key": "soft-verifier",
        "domain": "math",
        "objective": objective,
        "verification_backend": "soft_verifier",
        "original_problem": seed_problem,
        "original_solution": seed_solution,
        "metadata": metadata or {},
        "generation_reasoning": reasoning,
        "derived_problem": derived_problem,
        "modified_solution": modified_solution,
        "modified_solution_boxed": modified_solution_boxed
        or modified_solution_single_boxed,
        "format_valid": structural_valid,
        "generation_trace": generation_trace or {},
        "soft_verifier_called": bool(verifier_payload),
        "soft_verifier_valid_problem": None,
        "soft_verifier_valid_solution": None,
        "soft_verifier_seed_anchored": None,
        "soft_verifier_not_trivial_copy": None,
        "soft_verifier_complete_final_answer": None,
        "verified_valid": False,
        "seed_anchored": None,
        "not_trivial_copy": None,
        "complete_final_answer": None,
        "objective_score": 0.0,
        "modified_pass_rate": None,
        "original_pass_rate": None,
        "difficulty_gap": None,
        "harder_than_seed": False,
        "hardness_score": None,
    }
    if failure_reason:
        record["verification"] = {"reason": failure_reason}
        if failure_reason == "same_as_seed_problem":
            record["seed_anchored"] = True
            record["not_trivial_copy"] = False
            record["complete_final_answer"] = True
        return record

    verifier_payload = verifier_payload or {
        "valid_problem": False,
        "valid_solution": False,
        "seed_anchored": False,
        "not_trivial_copy": False,
        "complete_final_answer": False,
        "reason": "missing_verifier_payload",
    }
    verified_valid = bool(
        bool(verifier_payload.get("valid_problem"))
        and bool(verifier_payload.get("valid_solution"))
        and bool(verifier_payload.get("seed_anchored"))
        and bool(verifier_payload.get("not_trivial_copy"))
        and bool(verifier_payload.get("complete_final_answer"))
    )
    record.update(
        {
            "verification": verifier_payload,
            "soft_verifier_valid_problem": verifier_payload.get("valid_problem"),
            "soft_verifier_valid_solution": verifier_payload.get("valid_solution"),
            "soft_verifier_seed_anchored": verifier_payload.get("seed_anchored"),
            "soft_verifier_not_trivial_copy": verifier_payload.get("not_trivial_copy"),
            "soft_verifier_complete_final_answer": verifier_payload.get("complete_final_answer"),
            "verified_valid": verified_valid,
            "seed_anchored": bool(verifier_payload.get("seed_anchored")),
            "not_trivial_copy": bool(verifier_payload.get("not_trivial_copy")),
            "complete_final_answer": bool(verifier_payload.get("complete_final_answer")),
        }
    )
    if not verified_valid:
        return record

    if original_solver_outputs is None or modified_solver_outputs is None:
        return record

    original_pass_rate = compute_math_pass_rate(original_solver_outputs, original_targets)
    modified_pass_rate = compute_math_pass_rate(modified_solver_outputs, modified_targets)
    difficulty_gap = original_pass_rate - modified_pass_rate
    record.update(
        {
            "original_reference_boxed": original_reference_boxed,
            "modified_reference_boxed": modified_solution_boxed,
            "original_pass_rate": original_pass_rate,
            "modified_pass_rate": modified_pass_rate,
            "difficulty_gap": difficulty_gap,
            "harder_than_seed": difficulty_gap > 1e-9,
            "hardness_score": 1.0 - modified_pass_rate,
            "objective_score": objective_score(modified_pass_rate, objective),
        }
    )
    return record


def score_math_solver_solution(solution_str: str, extra_info: dict[str, Any] | None) -> float:
    reference_answer = str((extra_info or {}).get("reference_answer", "")).strip()
    if not reference_answer:
        return -1.0
    candidate = extract_single_boxed_answer(solution_str)
    if not candidate:
        return -1.0
    reference_targets = math_reference_targets(reference_answer)
    if not reference_targets:
        return -1.0
    return 1.0 if parsed_answers_equal(parse_boxed_answer(candidate), reference_targets) else 0.0


def math_reference_targets(answer: str) -> tuple[Any, ...]:
    answer = str(answer or "").strip()
    if not answer:
        return ()
    boxed = extract_last_boxed(answer)
    if boxed:
        targets = parse_boxed_answer(boxed)
        if targets:
            return targets
    targets = parse_math_text(answer)
    if targets:
        return targets
    return parse_boxed_answer(normalize_boxed_text(answer))


def solver_output_last_boxed_targets(solution_text: str) -> tuple[Any, ...]:
    return parse_boxed_answer(extract_last_boxed(solution_text))


def summarize_solution_samples(
    problem: str,
    reference_answer: str,
    samples: list[str],
    *,
    scoring_rule: str = "strict_single_boxed_answer",
) -> dict[str, Any]:
    reference_targets = math_reference_targets(reference_answer)
    if scoring_rule == "last_boxed_answer":
        target_fn = solver_output_last_boxed_targets
    elif scoring_rule == "strict_single_boxed_answer":
        target_fn = solver_output_targets
    else:
        raise ValueError(f"Unknown math scoring rule: {scoring_rule}")
    correct_flags = [
        bool(parsed_answers_equal(target_fn(sample), reference_targets))
        for sample in samples
    ]
    correct_count = sum(correct_flags)
    return {
        "problem": problem,
        "reference_answer": reference_answer,
        "samples": samples,
        "correct_flags": correct_flags,
        "correct_count": correct_count,
        "total_samples": len(samples),
        "pass_at_k": compute_pass_metrics(correct_count, len(samples)),
        "scoring_rule": scoring_rule,
    }


def parse_math_candidate_text(solution_str: str) -> dict[str, str]:
    return parse_math_setter_response(solution_str) or {
        "reasoning": "",
        "derived_problem": "",
        "modified_solution": "",
    }
