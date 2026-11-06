from __future__ import annotations

import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any

import httpx

from .data import write_sft_dataset
from .generate import (
    MathCandidateEvaluationInput,
    MathSeedRow,
    RealAPIBackend,
    VHGRuntimeConfig,
    evaluate_generated_candidates_batched,
)
from .prompts import (
    INTEGRATION_COLD_SFT_SYSTEM_PROMPT,
    MATH_COLD_SFT_SYSTEM_PROMPT,
    make_math_setter_collection_prompt,
    make_math_setter_prompt,
    setter_prompt,
)
from .service import CorrectnessClient
from .utils import load_jsonl, save_json, save_jsonl
from .verify import (
    format_math_setter_response,
    math_cold_sft_priority_key,
    math_cold_sft_reject_reason,
    normalize_problem_identity,
    parse_math_setter_response,
    parse_symbolic_generation_question,
)


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _sanitize_teacher_response(text: str) -> str:
    text = THINK_BLOCK_RE.sub("", str(text or ""))
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def _usable_api_record(row: dict[str, Any]) -> bool:
    response = str(row.get("response", "")).strip()
    if not response:
        return False
    if row.get("response_format") == "math_setter_tagged":
        return parse_math_setter_response(response) is not None
    return True


async def collect_api_candidates(
    jobs: list[dict[str, Any]],
    output_file: str | Path,
    *,
    base_url: str,
    api_key: str,
    model: str,
    concurrency: int,
    temperature: float,
    max_tokens: int,
    max_retries: int = 5,
) -> list[dict[str, Any]]:
    import openai

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    existing = load_jsonl(output_file) if output_file.exists() else []
    job_ids = {str(job["job_id"]) for job in jobs}
    if any(str(row["job_id"]) not in job_ids for row in existing):
        raise ValueError("Cached teacher records exceed the requested jobs; use a new output directory.")
    successful = {
        str(row["job_id"]): row
        for row in existing
        if _usable_api_record(row)
    }
    save_jsonl(output_file, [successful[key] for key in sorted(successful)])
    completed = set(successful)
    pending = [job for job in jobs if str(job["job_id"]) not in completed]
    client = openai.AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=180.0,
        http_client=httpx.AsyncClient(timeout=180.0, trust_env=False),
    )
    semaphore = asyncio.Semaphore(max(1, concurrency))
    write_lock = asyncio.Lock()

    async def run_job(job: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            last_error: Exception | None = None
            for attempt in range(max_retries):
                try:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=job["messages"],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    text = _sanitize_teacher_response(
                        response.choices[0].message.content or ""
                    )
                    if not text:
                        raise ValueError("empty API response")
                    if (
                        job.get("response_format") == "math_setter_tagged"
                        and parse_math_setter_response(text) is None
                    ):
                        raise ValueError("invalid tagged math-setter response")
                    record = {**job, "response": text, "api_model": model}
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 < max_retries:
                        await asyncio.sleep(2**attempt)
            else:
                record = {
                    **job,
                    "response": "",
                    "api_model": model,
                    "error": f"{type(last_error).__name__}: {last_error}",
                }
            async with write_lock:
                with output_file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            return record

    try:
        if pending:
            await asyncio.gather(*(run_job(job) for job in pending))
    finally:
        await client.close()
    rows = load_jsonl(output_file)
    rows.sort(key=lambda row: str(row["job_id"]))
    save_jsonl(output_file, rows)
    return rows


def _shuffle_prefix(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return rows
    from datasets import Dataset

    indices = Dataset.from_dict({"index": range(len(rows))}).shuffle(seed=42)["index"][:limit]
    return [rows[index] for index in indices]


def integration_cold_sft_jobs(
    seed_file: str | Path,
    *,
    icl_examples: str,
    samples_per_seed: int = 10,
) -> list[dict[str, Any]]:
    jobs = []
    seeds = json.loads(Path(seed_file).read_text(encoding="utf-8"))
    for seed_idx, seed in enumerate(seeds):
        if seed.get("question_type") != "integration":
            continue
        original = str(seed.get("latex answer") or seed.get("answer") or "").strip()
        if not original:
            continue
        runtime_prompt = setter_prompt(original)
        collection_prompt = (
            f"{icl_examples.rstrip()}\n\n{runtime_prompt}" if icl_examples else runtime_prompt
        )
        for sample_idx in range(samples_per_seed):
            jobs.append(
                {
                    "job_id": f"{seed_idx:06d}:{sample_idx:03d}",
                    "seed_id": seed.get("seed_id", seed_idx),
                    "sample_idx": sample_idx,
                    "question_type": "integration",
                    "response_format": "integration_text",
                    "prompt": runtime_prompt,
                    "messages": [
                        {"role": "system", "content": INTEGRATION_COLD_SFT_SYSTEM_PROMPT},
                        {"role": "user", "content": collection_prompt},
                    ],
                }
            )
    return jobs


def process_integration_cold_sft(
    raw_file: str | Path,
    output_dir: str | Path,
    *,
    service_url: str,
    max_examples: int = 400,
) -> dict[str, Any]:
    parsed_rows = []
    for row in load_jsonl(raw_file):
        parsed = parse_symbolic_generation_question(
            str(row.get("response", "")),
            require_original=True,
            require_boxed=False,
            question_type="integration",
        )
        if parsed.get("parseable"):
            parsed_rows.append({**row, **parsed})

    analyses = []
    with CorrectnessClient(service_url) as client:
        for start in range(0, len(parsed_rows), 512):
            analyses.extend(
                client.analyze_batch(
                    [
                        {
                            "operation": "integrand_fingerprint",
                            "question": row["normalized_question"],
                            "num_points": 5,
                        }
                        for row in parsed_rows[start : start + 512]
                    ]
                )
            )

    unique_rows = []
    seen_exact = set()
    seen_fingerprints = set()
    for row, analysis in zip(parsed_rows, analyses):
        exact_key = analysis.get("integrand_normalized_text")
        fingerprint = analysis.get("integrand_fingerprint")
        if exact_key and exact_key in seen_exact:
            continue
        if fingerprint and fingerprint in seen_fingerprints:
            continue
        if exact_key:
            seen_exact.add(exact_key)
        if fingerprint:
            seen_fingerprints.add(fingerprint)
        unique_rows.append(
            {
                **row,
                "expression_fingerprint": fingerprint,
                "integrand_normalized_text": exact_key,
            }
        )

    output_dir = Path(output_dir)
    save_jsonl(output_dir / "parsed_unique_records.jsonl", unique_rows)
    selected = _shuffle_prefix(unique_rows, max_examples)
    save_jsonl(output_dir / "selected_records.jsonl", selected)
    sft_summary = write_sft_dataset(
        selected,
        output_dir,
    )
    summary = {
        **sft_summary,
        "raw_candidate_count": len(load_jsonl(raw_file)),
        "parsed_unique_count": len(unique_rows),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def math_cold_sft_jobs(
    seed_file: str | Path,
    *,
    samples_per_seed: int = 1,
) -> list[dict[str, Any]]:
    jobs = []
    for seed_idx, seed in enumerate(load_jsonl(seed_file)):
        problem = str(seed.get("seed_problem", "")).strip()
        solution = str(seed.get("seed_solution", "")).strip()
        if not problem or not solution:
            continue
        for sample_idx in range(samples_per_seed):
            jobs.append(
                {
                    "job_id": f"{seed_idx:06d}:{sample_idx:03d}",
                    "seed_id": int(seed.get("id", seed_idx)),
                    "sample_idx": sample_idx,
                    "seed_problem": problem,
                    "seed_solution": solution,
                    "metadata": seed.get("metadata") or {},
                    "response_format": "math_setter_tagged",
                    "messages": [
                        {"role": "system", "content": MATH_COLD_SFT_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": make_math_setter_collection_prompt(
                                problem,
                                solution,
                                objective="hard",
                            ),
                        },
                    ],
                }
            )
    return jobs


async def process_math_cold_sft(
    raw_file: str | Path,
    output_dir: str | Path,
    *,
    verifier_config: VHGRuntimeConfig,
    tokenizer,
    verifier_concurrency: int = 256,
) -> dict[str, Any]:
    candidates = []
    seen_problems = set()
    for pool_id, row in enumerate(load_jsonl(raw_file)):
        parsed = parse_math_setter_response(str(row.get("response", "")))
        if not parsed:
            continue
        if math_cold_sft_reject_reason(**parsed) is not None:
            continue
        problem_key = normalize_problem_identity(parsed["derived_problem"])
        if not problem_key or problem_key in seen_problems:
            continue
        seen_problems.add(problem_key)
        seed = MathSeedRow(
            idx=int(row.get("seed_id", pool_id)),
            seed_problem=str(row["seed_problem"]),
            seed_solution=str(row["seed_solution"]),
            metadata=row.get("metadata") or {},
        )
        candidates.append(
            (
                pool_id,
                row,
                MathCandidateEvaluationInput(
                    seed=seed,
                    reasoning=parsed["reasoning"],
                    derived_problem=parsed["derived_problem"],
                    modified_solution=parsed["modified_solution"],
                    generation_trace={"stage": "cold_sft", "job_id": row["job_id"]},
                    solver_eval_samples=0,
                ),
            )
        )

    backend = RealAPIBackend(verifier_config)
    try:
        records = await evaluate_generated_candidates_batched(
            [item[2] for item in candidates],
            backend=backend,
            soft_verifier_concurrency=verifier_concurrency,
            run_solver_eval=False,
        )
    finally:
        await backend.close()
    all_records = []
    for (pool_id, raw, _), record in zip(candidates, records):
        all_records.append({**raw, **record, "id": pool_id})
    verified = [row for row in all_records if row.get("verified_valid")]
    selected = sorted(
        verified,
        key=lambda row: (
            math_cold_sft_priority_key(
                reasoning=row["generation_reasoning"],
                derived_problem=row["derived_problem"],
                modified_solution=row["modified_solution"],
            ),
            int(row.get("seed_id", 0)),
            int(row.get("sample_idx", 0)),
        ),
    )
    sft_rows = [
        {
            **row,
            "prompt": make_math_setter_prompt(
                row["original_problem"],
                row["original_solution"],
                objective="hard",
            ),
            "response": format_math_setter_response(
                reasoning=row["generation_reasoning"],
                derived_problem=row["derived_problem"],
                modified_solution=row["modified_solution"],
            ),
        }
        for row in selected
    ]
    sft_rows, length_stats = filter_math_sft_lengths(sft_rows, tokenizer)
    # Final-prompt corpus preparation shuffled once before the seed-42 split.
    random.Random(42).shuffle(sft_rows)
    output_dir = Path(output_dir)
    save_jsonl(output_dir / "all_records.jsonl", all_records)
    save_jsonl(output_dir / "verified_records.jsonl", verified)
    save_jsonl(output_dir / "selected_records.jsonl", sft_rows)
    sft_summary = write_sft_dataset(
        sft_rows,
        output_dir,
    )
    summary = {
        **sft_summary,
        "raw_candidate_count": len(load_jsonl(raw_file)),
        "locally_unique_count": len(candidates),
        "verified_count": len(verified),
        "selected_count": len(sft_rows),
        **length_stats,
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def filter_math_sft_lengths(rows, tokenizer, max_length=4096, max_solution_ratio=4.0):
    """Filter by solution-length ratio and total prompt-plus-response tokens."""
    kept = []
    overlength = expanded = 0
    def tokens(text):
        return len(tokenizer.encode(str(text or ""), add_special_tokens=False))
    for row in rows:
        total = tokens(row["prompt"]) + tokens(row["response"])
        seed_solution = row.get("original_solution") or row.get("seed_solution", "")
        ratio = tokens(row["modified_solution"]) / max(1, tokens(seed_solution))
        overlength += total > max_length
        expanded += ratio > max_solution_ratio
        if total <= max_length and ratio <= max_solution_ratio:
            kept.append({**row, "prompt_plus_response_tokens": total, "solution_token_ratio": ratio})
    return kept, {"dropped_overlength": overlength, "dropped_solution_ratio": expanded,
                  "max_length": max_length, "max_solution_ratio": max_solution_ratio}
