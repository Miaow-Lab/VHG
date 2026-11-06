from __future__ import annotations

import asyncio
import os
from typing import Any

try:
    from .generate import (
        MathCandidateEvaluationInput,
        MathSeedRow,
        build_backend,
        config_from_env,
        ensure_config,
        evaluate_generated_candidates_batched,
    )
    from .score import (
        compute_score_solver_batched,
        parse_math_candidate_text,
        score_hard_verifier_setter_candidates_batched,
        score_math_solver_solution,
    )
except ImportError:
    from vhg.generate import (
        MathCandidateEvaluationInput,
        MathSeedRow,
        build_backend,
        config_from_env,
        ensure_config,
        evaluate_generated_candidates_batched,
    )
    from vhg.score import (
        compute_score_solver_batched,
        parse_math_candidate_text,
        score_hard_verifier_setter_candidates_batched,
        score_math_solver_solution,
    )


_BACKEND = None
_CONFIG = None


def _get_backend_and_config():
    global _BACKEND, _CONFIG
    if _BACKEND is None or _CONFIG is None:
        _CONFIG = config_from_env()
        ensure_config(_CONFIG)
        _BACKEND = build_backend(_CONFIG)
    return _BACKEND, _CONFIG


def _seed_from_extra_info(extra_info: dict[str, Any] | None) -> MathSeedRow:
    extra_info = extra_info or {}
    metadata = extra_info.get("metadata", {})
    return MathSeedRow(
        idx=int(extra_info.get("index", 0)),
        seed_problem=str(extra_info.get("seed_problem", "")).strip(),
        seed_solution=str(extra_info.get("seed_solution", "")).strip(),
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def _solver_sampling_from_extra_info(
    extra_info: dict[str, Any] | None,
    *,
    config,
) -> tuple[int, float, float, int | None, int]:
    extra_info = extra_info or {}
    required = ("sample_num", "temperature", "top_p", "top_k", "max_tokens")
    missing = [field for field in required if field not in extra_info]
    if missing:
        raise ValueError(
            "Setter reward extra_info is missing saved solver sampling metadata: "
            + ", ".join(missing)
        )
    top_k_raw = extra_info.get("top_k")
    top_k = None if top_k_raw in (None, "") else int(top_k_raw)
    return (
        int(extra_info.get("sample_num", config.solver_eval_samples)),
        float(extra_info.get("temperature", config.solver_temperature)),
        float(extra_info.get("top_p", config.solver_top_p)),
        top_k,
        int(extra_info.get("max_tokens", config.solver_max_tokens)),
    )


def _optional_bool_metric(value: Any) -> float:
    if value is True:
        return 1.0
    if value is False:
        return 0.0
    return -1.0


def _record_to_reward_dict(record: dict[str, Any]) -> dict[str, float]:
    objective = float(record.get("objective_score") or 0.0)
    return {
        "score": objective,
        "format_valid": 1.0 if record.get("format_valid") else 0.0,
        "verified_valid": 1.0 if record.get("verified_valid") else 0.0,
        "harder_than_seed": 1.0 if record.get("harder_than_seed") else 0.0,
        "seed_anchored": 1.0 if record.get("seed_anchored") else 0.0,
        "not_trivial_copy": 1.0 if record.get("not_trivial_copy") else 0.0,
        "soft_verifier_called": 1.0 if record.get("soft_verifier_called") else 0.0,
        "soft_verifier_valid_problem": _optional_bool_metric(record.get("soft_verifier_valid_problem")),
        "soft_verifier_valid_solution": _optional_bool_metric(record.get("soft_verifier_valid_solution")),
        "soft_verifier_seed_anchored": _optional_bool_metric(record.get("soft_verifier_seed_anchored")),
        "soft_verifier_not_trivial_copy": _optional_bool_metric(
            record.get("soft_verifier_not_trivial_copy")
        ),
        "soft_verifier_complete_final_answer": _optional_bool_metric(
            record.get("soft_verifier_complete_final_answer")
        ),
        "original_pass_rate": float(record.get("original_pass_rate") or 0.0),
        "modified_pass_rate": float(record.get("modified_pass_rate") or 0.0),
        "difficulty_gap": float(record.get("difficulty_gap") or 0.0),
        "hardness_score": float(record.get("hardness_score") or 0.0),
        "objective_score": objective,
    }


async def _score_soft_verifier_records_async(solution_strs, extra_infos):
    backend, config = _get_backend_and_config()
    candidates = []
    for solution_str, extra_info in zip(solution_strs, extra_infos):
        parsed = parse_math_candidate_text(solution_str)
        (
            solver_eval_samples,
            solver_temperature,
            solver_top_p,
            solver_top_k,
            solver_max_tokens,
        ) = _solver_sampling_from_extra_info(extra_info, config=config)
        candidates.append(
            MathCandidateEvaluationInput(
                seed=_seed_from_extra_info(extra_info),
                reasoning=parsed["reasoning"],
                derived_problem=parsed["derived_problem"],
                modified_solution=parsed["modified_solution"],
                generation_trace={
                    "source": "math_setter_reward",
                    "response_format": "tagged_v1",
                },
                objective=str((extra_info or {}).get("objective", "hard")),
                solver_eval_samples=solver_eval_samples,
                solver_temperature=solver_temperature,
                solver_top_p=solver_top_p,
                solver_top_k=solver_top_k,
                solver_max_tokens=solver_max_tokens,
            )
        )
    soft_verifier_concurrency = max(
        1,
        int(os.getenv("SETTER_SOFT_VERIFIER_CONCURRENCY", "512")),
    )
    return await evaluate_generated_candidates_batched(
        candidates,
        backend=backend,
        soft_verifier_concurrency=soft_verifier_concurrency,
        run_solver_eval=True,
    )


def _batch_is_empty(values: Any) -> bool:
    if values is None:
        return True
    try:
        return len(values) == 0
    except TypeError:
        return False


def _first_batch_value(values: Any) -> Any:
    if values is None:
        return ""
    if isinstance(values, str):
        return values
    try:
        return values[0] if len(values) else ""
    except (KeyError, TypeError):
        return values


def compute_score_batched(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    del ground_truths, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    if _batch_is_empty(solution_strs):
        return []
    source = str(_first_batch_value(data_sources)).lower()
    if "math_solver" in source:
        return [
            float(score_math_solver_solution(solution_str, extra_info))
            for solution_str, extra_info in zip(solution_strs, extra_infos)
        ]
    if "solver" in source:
        return compute_score_solver_batched(solution_strs, extra_infos)
    if "soft_verifier" in source or "math_setter" in source:
        records = asyncio.run(_score_soft_verifier_records_async(solution_strs, extra_infos))
        return [_record_to_reward_dict(record) for record in records]
    if (
        "setter" in source
        or "generator" in source
        or "hard_verifier" in source
        or "hard-verifier" in source
    ):
        return score_hard_verifier_setter_candidates_batched(
            solution_strs,
            extra_infos,
        )
    raise NotImplementedError(f"Reward function is not implemented for {source=}")
