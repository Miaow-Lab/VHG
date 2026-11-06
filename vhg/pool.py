from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .evaluate import score_math_pool_difficulty
from .generate import (
    MathCandidateEvaluationInput,
    MathSeedRow,
    RealAPIBackend,
    VHGRuntimeConfig,
    evaluate_generated_candidates_batched,
)
from .score import (
    enrich_hard_verifier_solver_metrics_batched,
    evaluate_math_candidate_record,
    score_hard_verifier_analysis_result,
)
from .service import CorrectnessClient
from .utils import iter_jsonl, save_json, save_jsonl
from .verify import (
    normalize_expression_text_for_exact_dedup,
    normalize_problem_identity,
    normalize_problem_template_identity,
    parse_math_setter_response,
)


def filter_integration_pool(
    input_file: str | Path,
    output_dir: str | Path,
    *,
    service_url: str,
    previous_archive: str | Path | None = None,
    round_number: int = 1,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed_file = output_dir / "parsed_unique_records.jsonl"
    verified_file = output_dir / "verified_records.jsonl"
    seen_exact = set()
    seen_fingerprints = set()
    parsed_unique = []
    input_count = 0

    chunk = []

    def process_chunk(rows: list[dict[str, Any]]) -> None:
        analyses = client.analyze_batch(
            [
                {
                    "operation": "integrand_fingerprint",
                    "question": str(row.get("response", "")),
                    "num_points": 5,
                }
                for row in rows
            ]
        )
        for row, analysis in zip(rows, analyses):
            if not analysis.get("parseable") or not analysis.get("original"):
                continue
            exact = analysis.get("integrand_normalized_text")
            fingerprint = analysis.get("integrand_fingerprint")
            if exact and exact in seen_exact:
                continue
            if fingerprint and fingerprint in seen_fingerprints:
                continue
            if exact:
                seen_exact.add(exact)
            if fingerprint:
                seen_fingerprints.add(fingerprint)
            parsed_unique.append(
                {
                    **row,
                    "expr": analysis["expr"],
                    "var": analysis["var"],
                    "original": analysis["original"],
                    "expression_fingerprint": fingerprint,
                    "integrand_normalized_text": exact,
                }
            )

    with CorrectnessClient(service_url) as client:
        for row in iter_jsonl(input_file):
            input_count += 1
            chunk.append(row)
            if len(chunk) == 512:
                process_chunk(chunk)
                chunk = []
        if chunk:
            process_chunk(chunk)
    save_jsonl(parsed_file, parsed_unique)

    valid_flags = []
    with CorrectnessClient(service_url) as client:
        for start in range(0, len(parsed_unique), 512):
            valid_flags.extend(
                client.check_batch(
                    [
                        {
                            "question": (
                                f"\\original{{{row['original']}}} "
                                f"\\expression{{{row['expr']}}} "
                                f"\\variable{{{row['var']}}}"
                            ),
                            "question_type": "integration",
                            "solution": f"\\boxed{{{row['original']}}}",
                        }
                        for row in parsed_unique[start : start + 512]
                    ]
                )
            )
    verified = [{**row, "archive_first_round": round_number}
                for row, flag in zip(parsed_unique, valid_flags) if flag]
    new_verified_count = len(verified)
    save_jsonl(output_dir / "new_verified_records.jsonl", verified)
    if previous_archive is not None:
        # Keep every verified problem, including earlier 0/10 and 10/10 outcomes.
        # Only validity is inherited: all difficulty estimates are recomputed below.
        from .utils import load_jsonl

        with CorrectnessClient(service_url) as client:
            verified = merge_verified_archive(load_jsonl(previous_archive), verified, client)
    save_jsonl(verified_file, verified)

    extra_infos = [
        {
            "question_type": "integration",
            "sample_num": 10,
            "temperature": 1.0,
            "max_tokens": 4096,
            "objective": "hard",
        }
        for _ in verified
    ]
    validity_results = [
        {
            "score": 0.0,
            "format_valid": True,
            "reference_symbolic_valid": True,
            "solver_hard_verifier_pass_rate": 0.0,
            "solver_hard_verifier_hardness": 0.0,
            "objective": "hard",
        }
        for _ in verified
    ]
    difficulty_results = enrich_hard_verifier_solver_metrics_batched(
        solution_strs=[row["response"] for row in verified],
        extra_infos=extra_infos,
        validity_results=validity_results,
    )
    audited = [
        {**row, **score_hard_verifier_analysis_result(result)}
        for row, result in zip(verified, difficulty_results)
    ]
    selected = [row for row in audited if float(row.get("solver_hard_verifier_hardness", 0)) > 0]
    if previous_archive is None:
        # The original first-round candidate export sorted by hardness score.
        selected.sort(key=lambda row: float(row["solver_hard_verifier_hardness"]), reverse=True)
    save_jsonl(output_dir / "audited_records.jsonl", audited)
    save_jsonl(output_dir / "solver_records.jsonl", selected)

    training_extra_infos = [
        {
            "question_type": "integration",
            "sample_num": 10,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 4096,
            "objective": "hard",
        }
        for _ in selected
    ]
    training_metrics = enrich_hard_verifier_solver_metrics_batched(
        solution_strs=[row["response"] for row in selected],
        extra_infos=training_extra_infos,
        validity_results=[
            {"format_valid": True, "reference_symbolic_valid": True}
            for _ in selected
        ],
    )
    pass_band = []
    for row, metric in zip(selected, training_metrics):
        pass_rate = float(metric.get("solver_hard_verifier_pass_rate", 0.0))
        if 0.1 <= pass_rate <= 0.9:
            pass_band.append(
                {
                    **row,
                    "solver_training_pass_rate": pass_rate,
                    "solver_training_hardness": 1.0 - pass_rate,
                }
            )
    if not pass_band and previous_archive is None:
        raise ValueError("No generated integration problems survived the solver pass band")
    save_jsonl(output_dir / "solver_pass_band_records.jsonl", pass_band)
    summary = {
        "input_rows": input_count,
        "parsed_deduplicated_rows": len(parsed_unique),
        "reference_valid_rows": len(verified),
        "new_reference_valid_rows": new_verified_count,
        "previous_archive": str(previous_archive) if previous_archive else None,
        "round": round_number,
        "solver_data_rows": len(selected),
        "solver_pass_band_rows": len(pass_band),
        "solver_training_rows": len(pass_band),
        "solver_samples_per_problem": 10,
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def merge_verified_archive(previous, current, client):
    """Keep the first integrand under exact/20-point identity, as in the paper.

    Remove previous solver metrics so each round assesses the archive with
    its current frozen solver before selection.
    """
    rows = [*previous, *current]
    analyses = []
    for start in range(0, len(rows), 512):
        analyses.extend(client.analyze_batch([
            {"operation": "expression_fingerprint", "expression": row["expr"], "num_points": 20}
            for row in rows[start:start + 512]
        ]))
    if len(analyses) != len(rows):
        raise RuntimeError("Incomplete archive fingerprint response")
    exact_seen, fingerprint_seen, retained = set(), set(), []
    for row, analysis in zip(rows, analyses):
        exact = normalize_expression_text_for_exact_dedup(row["expr"])
        if exact and exact in exact_seen:
            continue
        if exact:
            exact_seen.add(exact)
        fingerprint = analysis.get("fingerprint")
        if fingerprint and fingerprint in fingerprint_seen:
            continue
        if fingerprint:
            fingerprint_seen.add(fingerprint)
        # Archive files contain verified problem records, never solver eligibility.
        retained.append({key: value for key, value in row.items()
                         if not key.startswith("solver_") and key not in {"score", "eligible"}})
    return retained


def prefilter_math_pool(
    input_file: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    local_file = output_dir / "locally_filtered_records.jsonl"
    template_file = output_dir / "template_deduplicated_records.jsonl"
    seen_problems = set()
    seen_templates = set()
    input_count = 0
    local_count = 0
    template_count = 0

    with local_file.open("w", encoding="utf-8") as local_handle, template_file.open(
        "w", encoding="utf-8"
    ) as template_handle:
        for input_index, row in enumerate(iter_jsonl(input_file)):
            input_count += 1
            parsed = parse_math_setter_response(str(row.get("response", "")))
            if not parsed:
                continue
            local_record = evaluate_math_candidate_record(
                seed_problem=str(row.get("seed_problem", "")),
                seed_solution=str(row.get("seed_solution", "")),
                reasoning=parsed["reasoning"],
                derived_problem=parsed["derived_problem"],
                modified_solution=parsed["modified_solution"],
                verifier_payload=None,
                metadata=row.get("metadata") or {},
                generation_trace={
                    "checkpoint": row.get("checkpoint"),
                    "checkpoint_tag": row.get("checkpoint_tag"),
                    "sample_idx": row.get("sample_idx"),
                },
                record_id=input_index,
            )
            if local_record.get("verification", {}).get("reason") != "missing_verifier_payload":
                continue
            problem_key = normalize_problem_identity(local_record["derived_problem"])
            if problem_key in seen_problems:
                continue
            seen_problems.add(problem_key)
            merged = {**row, **local_record, "id": input_index}
            local_handle.write(json.dumps(merged, ensure_ascii=False) + "\n")
            local_count += 1

            template_key = normalize_problem_template_identity(local_record["derived_problem"])
            if template_key in seen_templates:
                continue
            seen_templates.add(template_key)
            template_handle.write(json.dumps(merged, ensure_ascii=False) + "\n")
            template_count += 1

    summary = {
        "input_rows": input_count,
        "locally_filtered_unique_rows": local_count,
        "template_deduplicated_rows": template_count,
    }
    save_json(output_dir / "prefilter_summary.json", summary)
    return summary


def score_math_pool_file(
    input_file: str | Path,
    output_file: str | Path,
    *,
    config: VHGRuntimeConfig,
    min_pass_rate: float = 0.1,
    max_pass_rate: float = 0.9,
) -> dict[str, Any]:
    slim_records = [
        {
            "derived_problem": row.get("derived_problem"),
            "modified_solution_boxed": row.get("modified_solution_boxed"),
        }
        for row in iter_jsonl(input_file)
    ]
    metrics = score_math_pool_difficulty(
        slim_records,
        config=config,
        samples_per_problem=10,
    )
    metric_by_position = {row["position"]: row for row in metrics}
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with output_file.open("w", encoding="utf-8") as handle:
        for position, row in enumerate(iter_jsonl(input_file)):
            metric = metric_by_position.get(position)
            if metric is None:
                continue
            pass_rate = float(metric["modified_pass_rate"])
            if min_pass_rate <= pass_rate <= max_pass_rate:
                merged = {
                    **row,
                    **metric,
                    "hardness_score": 1.0 - pass_rate,
                    "objective_score": 1.0 - pass_rate,
                }
                handle.write(json.dumps(merged, ensure_ascii=False) + "\n")
                kept += 1
    return {
        "difficulty_scored_rows": len(metrics),
        "pass_band_rows": kept,
        "pass_band": [min_pass_rate, max_pass_rate],
        "solver_samples_per_problem": 10,
    }


async def judge_math_pool_file(
    input_file: str | Path,
    output_dir: str | Path,
    *,
    verifier_config: VHGRuntimeConfig,
    target_valid_records: int = 20000,
    shuffle_seed: int = 42,
    concurrency: int = 256,
    batch_size: int = 1024,
) -> dict[str, Any]:
    """Keep complete judge batches until the valid target or input exhaustion."""
    if target_valid_records < 1 or batch_size < 1:
        raise ValueError("The valid-record target and judge batch size must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(iter_jsonl(input_file))
    total_rows = len(rows)
    random.Random(shuffle_seed).shuffle(rows)
    backend = RealAPIBackend(verifier_config)
    all_file = output_dir / "judge_records.jsonl"
    accepted_file = output_dir / "accepted_records.jsonl"

    async def judge_batch(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates = []
        for batch_index, row in enumerate(rows):
            seed = MathSeedRow(
                idx=batch_index,
                seed_problem=str(row["original_problem"]),
                seed_solution=str(row["original_solution"]),
                metadata=row.get("metadata") or {},
            )
            candidates.append(
                MathCandidateEvaluationInput(
                    seed=seed,
                    reasoning=str(row["generation_reasoning"]),
                    derived_problem=str(row["derived_problem"]),
                    modified_solution=str(row["modified_solution"]),
                    generation_trace=row.get("generation_trace") or {},
                    solver_eval_samples=0,
                )
            )
        judged = await evaluate_generated_candidates_batched(
            candidates,
            backend=backend,
            soft_verifier_concurrency=concurrency,
            run_solver_eval=False,
        )
        output = []
        for source, result in zip(rows, judged):
            pass_rate = source.get("modified_pass_rate")
            output.append(
                {
                    **source,
                    **result,
                    "id": source.get("id"),
                    "modified_pass_rate": pass_rate,
                    "hardness_score": 1.0 - float(pass_rate),
                    "objective_score": 1.0 - float(pass_rate),
                }
            )
        return output

    judged_records = []
    accepted_records = []
    try:
        for start in range(0, len(rows), batch_size):
            batch_results = await judge_batch(rows[start : start + batch_size])
            judged_records.extend(batch_results)
            accepted_records.extend(
                row for row in batch_results if row.get("verified_valid")
            )
            if len(accepted_records) >= target_valid_records:
                break
    finally:
        await backend.close()

    def record_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
        trace = row.get("generation_trace") or {}
        return (
            int(row.get("id", 0)),
            int(trace.get("sample_idx", 0) or 0),
            str(trace.get("checkpoint", "") or row.get("checkpoint", "")),
        )

    judged_records.sort(key=record_sort_key)
    accepted_records.sort(key=record_sort_key)
    save_jsonl(all_file, judged_records)
    save_jsonl(accepted_file, accepted_records)
    return {
        "pass_band_rows": total_rows,
        "judge_evaluated_rows": len(judged_records),
        "judge_accepted_rows": len(accepted_records),
        "target_valid_records": target_valid_records,
        "target_reached": len(accepted_records) >= target_valid_records,
        "batch_size": batch_size,
        "shuffle_seed": shuffle_seed,
    }
