from __future__ import annotations

import asyncio
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from .prompts import (
    make_math_solver_prompt,
    make_math_setter_messages,
    make_soft_verifier_prompt,
    setter_messages,
)
from .utils import load_jsonl, render_generation_prompt
from .verify import (
    extract_single_parseable_boxed_answer,
    parse_soft_verifier_response,
)


@dataclass(frozen=True)
class MathSeedRow:
    idx: int
    seed_problem: str
    seed_solution: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MathCandidateEvaluationInput:
    seed: MathSeedRow
    reasoning: str
    derived_problem: str
    modified_solution: str
    generation_trace: dict[str, Any]
    objective: str = "hard"
    solver_eval_samples: int = 4
    solver_temperature: float | None = None
    solver_top_p: float | None = None
    solver_top_k: int | None = None
    solver_max_tokens: int | None = None


@dataclass(frozen=True)
class VHGRuntimeConfig:
    soft_verifier_base_url: str = ""
    soft_verifier_api_key: str = ""
    soft_verifier_model: str = ""
    soft_verifier_max_tokens: int = 512
    solver_ckpt_path: str = ""
    solver_visible_devices: str = ""
    solver_gpu_memory_utilization: float = 0.9
    solver_max_tokens: int = 2048
    solver_temperature: float = 0.7
    solver_top_p: float = 0.95
    solver_top_k: int | None = None
    solver_eval_samples: int = 4
    max_retries: int = 3


def normalize_visible_devices(visible_devices: str) -> str:
    return ",".join(
        device.strip()
        for device in str(visible_devices or "").split(",")
        if device.strip()
    )


def is_local_openai_base_url(base_url: str) -> bool:
    if not base_url:
        return False
    try:
        parsed = urlparse(base_url)
    except Exception:
        return False
    host = (parsed.hostname or "").strip().lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def config_from_env() -> VHGRuntimeConfig:
    soft_verifier_base_url = os.getenv("SOFT_VERIFIER_BASE_URL", "")
    soft_verifier_api_key = os.getenv("SOFT_VERIFIER_API_KEY", "")
    soft_verifier_model = os.getenv("SOFT_VERIFIER_MODEL", "")
    solver_ckpt_path = os.getenv("SOLVER_CKPT_PATH", os.getenv("BASE_MODEL_PATH", ""))
    solver_visible_devices = normalize_visible_devices(
        os.getenv("SOLVER_VISIBLE_DEVICES", os.getenv("CUDA_VISIBLE_DEVICES", ""))
    )
    if not soft_verifier_api_key and is_local_openai_base_url(soft_verifier_base_url):
        soft_verifier_api_key = "EMPTY"
    return VHGRuntimeConfig(
        soft_verifier_base_url=soft_verifier_base_url,
        soft_verifier_api_key=soft_verifier_api_key,
        soft_verifier_model=soft_verifier_model,
        soft_verifier_max_tokens=int(os.getenv("SOFT_VERIFIER_MAX_TOKENS", "512")),
        solver_ckpt_path=solver_ckpt_path,
        solver_visible_devices=solver_visible_devices,
        solver_gpu_memory_utilization=float(os.getenv("SOLVER_GPU_MEMORY_UTIL", "0.9")),
        solver_max_tokens=int(os.getenv("SOLVER_MAX_TOKENS", "2048")),
        solver_temperature=float(os.getenv("SOLVER_TEMPERATURE", "0.7")),
        solver_top_p=float(os.getenv("SOLVER_TOP_P", "0.95")),
        solver_top_k=(
            int(os.getenv("SOLVER_TOP_K"))
            if os.getenv("SOLVER_TOP_K", "").strip()
            else None
        ),
        solver_eval_samples=int(os.getenv("SOLVER_EVAL_SAMPLES", "4")),
        max_retries=int(
            os.getenv(
                "SOFT_VERIFIER_MAX_RETRIES",
                os.getenv("MAX_RETRIES", "3"),
            )
        ),
    )


def ensure_config(
    config: VHGRuntimeConfig,
    *,
    need_solver: bool = True,
) -> None:
    missing = []
    if not config.soft_verifier_api_key and not is_local_openai_base_url(config.soft_verifier_base_url):
        missing.append("SOFT_VERIFIER_API_KEY")
    if not config.soft_verifier_base_url:
        missing.append("SOFT_VERIFIER_BASE_URL")
    if not config.soft_verifier_model:
        missing.append("SOFT_VERIFIER_MODEL")
    if need_solver:
        if not config.solver_ckpt_path:
            missing.append("SOLVER_CKPT_PATH or BASE_MODEL_PATH")
        if not normalize_visible_devices(config.solver_visible_devices):
            missing.append("SOLVER_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES")
    if missing:
        raise ValueError("Missing VHG runtime configuration: " + ", ".join(missing))


def build_verifier_config(base_url: str, api_key: str, model: str) -> VHGRuntimeConfig:
    config = VHGRuntimeConfig(
        soft_verifier_base_url=base_url,
        soft_verifier_api_key=api_key or ("EMPTY" if is_local_openai_base_url(base_url) else ""),
        soft_verifier_model=model,
    )
    ensure_config(config, need_solver=False)
    return config


def resolve_solver_sampling(
    config: VHGRuntimeConfig,
    *,
    n: int,
) -> tuple[float, float | None, int | None]:
    if n <= 1:
        return 0.0, None, None
    return config.solver_temperature, config.solver_top_p, config.solver_top_k


def build_openai_http_client(
    *,
    max_connections: int,
    timeout_seconds: float = 120.0,
) -> httpx.AsyncClient:
    connection_limit = max(32, int(max_connections))
    keepalive_limit = max(20, min(connection_limit, 256))
    timeout = httpx.Timeout(
        connect=min(30.0, timeout_seconds),
        read=timeout_seconds,
        write=timeout_seconds,
        pool=timeout_seconds,
    )
    limits = httpx.Limits(
        max_connections=connection_limit,
        max_keepalive_connections=keepalive_limit,
    )
    return httpx.AsyncClient(timeout=timeout, trust_env=False, limits=limits)


class RealAPIBackend:
    def __init__(self, config: VHGRuntimeConfig):
        import openai

        self.config = config
        soft_verifier_connections = int(os.getenv("SOFT_VERIFIER_HTTP_MAX_CONNECTIONS", "512"))
        timeout_seconds = float(os.getenv("SOFT_VERIFIER_HTTP_TIMEOUT_SECONDS", "120"))
        self.soft_verifier_client = openai.AsyncOpenAI(
            base_url=config.soft_verifier_base_url,
            api_key=config.soft_verifier_api_key,
            timeout=timeout_seconds,
            http_client=build_openai_http_client(
                max_connections=soft_verifier_connections,
                timeout_seconds=timeout_seconds,
            ),
        )

    async def close(self) -> None:
        await self.soft_verifier_client.close()

    async def _chat_text(
        self,
        client,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                resp = await client.chat.completions.create(**kwargs)
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                last_error = exc
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2**attempt)
        raise RuntimeError("OpenAI-compatible text call failed.") from last_error

    async def verify_pair(
        self,
        seed_problem: str,
        seed_solution: str,
        derived_problem: str,
        modified_solution: str,
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": "Follow the requested tagged format exactly."},
            {
                "role": "user",
                "content": make_soft_verifier_prompt(
                    seed_problem=seed_problem,
                    seed_solution=seed_solution,
                    derived_problem=derived_problem,
                    modified_solution=modified_solution,
                ),
            },
        ]
        for _ in range(self.config.max_retries):
            response = await self._chat_text(
                self.soft_verifier_client,
                self.config.soft_verifier_model,
                messages,
                max_tokens=self.config.soft_verifier_max_tokens,
                temperature=0.0,
            )
            parsed = parse_soft_verifier_response(response)
            if parsed is not None:
                _, payload = parsed
                return payload
        return {
            "valid_problem": False,
            "valid_solution": False,
            "seed_anchored": False,
            "not_trivial_copy": False,
            "complete_final_answer": False,
            "reason": "soft_verifier_malformed_tagged_response",
        }


def build_backend(config: VHGRuntimeConfig):
    return RealAPIBackend(config)


def _effective_solver_config(
    config: VHGRuntimeConfig,
    candidate: MathCandidateEvaluationInput,
) -> VHGRuntimeConfig:
    return replace(
        config,
        solver_temperature=(
            config.solver_temperature
            if candidate.solver_temperature is None
            else float(candidate.solver_temperature)
        ),
        solver_top_p=(
            config.solver_top_p
            if candidate.solver_top_p is None
            else float(candidate.solver_top_p)
        ),
        solver_top_k=(
            config.solver_top_k
            if candidate.solver_top_k is None
            else int(candidate.solver_top_k)
        ),
        solver_max_tokens=(
            config.solver_max_tokens
            if candidate.solver_max_tokens is None
            else int(candidate.solver_max_tokens)
        ),
    )


def sample_local_solver_prompts(
    indexed_prompts: list[dict[str, str]],
    *,
    config: VHGRuntimeConfig,
    sample_num: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    max_tokens: int,
    sampling_seed: int | None = None,
) -> dict[str, list[str]]:
    if not indexed_prompts:
        return {}
    import ray
    from vllm import LLM, SamplingParams

    visible_devices = normalize_visible_devices(config.solver_visible_devices)
    if not visible_devices:
        raise ValueError(
            "SOLVER_VISIBLE_DEVICES is empty after parsing; expected comma-separated GPU ids."
        )
    if not config.solver_ckpt_path:
        raise ValueError("Missing solver checkpoint path for local solver sampling.")

    @ray.remote(num_gpus=0)
    class SolverSampler:
        def __init__(self, model_path: str, gpu_memory_utilization: float, cuda_device: str, rank: int):
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
            python_bin = str(Path(sys.executable).resolve().parent)
            path_parts = os.environ.get("PATH", "").split(os.pathsep)
            if python_bin and python_bin not in path_parts:
                os.environ["PATH"] = os.pathsep.join([python_bin, *path_parts])
            base_port = int(os.environ.get("SOLVER_MASTER_PORT_BASE", "30000"))
            port_stride = int(os.environ.get("SOLVER_MASTER_PORT_STRIDE", "53"))
            init_retries = int(os.environ.get("SOLVER_LLM_INIT_RETRIES", "5"))
            last_error = None
            for attempt in range(init_retries):
                vllm_port = base_port + rank * port_stride + attempt
                if vllm_port > 65535:
                    vllm_port = 30000 + (vllm_port - 30000) % 30000
                os.environ["MASTER_ADDR"] = "127.0.0.1"
                os.environ["VLLM_PORT"] = str(vllm_port)
                try:
                    self.llm = LLM(
                        model=model_path,
                        tensor_parallel_size=1,
                        gpu_memory_utilization=gpu_memory_utilization,
                        trust_remote_code=True,
                    )
                    return
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 < init_retries:
                        time.sleep(0.2 * (attempt + 1))
                        continue
                    raise
            raise RuntimeError("Failed to initialize solver sampler") from last_error

        def generate(self, batch: list[dict[str, str]], sampling_kwargs: dict[str, Any]):
            if "seed" in sampling_kwargs:
                from hashlib import sha256
                # Independent, repeatable draws even when a prompt is repeated.
                sampling_params = [
                    SamplingParams(**{
                        **sampling_kwargs,
                        "seed": int.from_bytes(sha256(
                            f"{sampling_kwargs['seed']}:{entry['idx']}".encode()
                        ).digest()[:4], "big") % (2**31),
                    })
                    for entry in batch
                ]
            else:
                sampling_params = SamplingParams(**sampling_kwargs)
            outputs = self.llm.generate([entry["prompt"] for entry in batch], sampling_params)
            return {
                entry["idx"]: [completion.text for completion in output.outputs]
                for entry, output in zip(batch, outputs)
            }

    gpu_list = [device.strip() for device in visible_devices.split(",") if device.strip()]
    ray.init(ignore_reinit_error=True)
    actors = [
        SolverSampler.remote(
            config.solver_ckpt_path,
            float(config.solver_gpu_memory_utilization),
            gpu,
            rank,
        )
        for rank, gpu in enumerate(gpu_list)
    ]
    sampling_kwargs: dict[str, Any] = {
        "n": int(sample_num),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    if top_p is not None:
        sampling_kwargs["top_p"] = top_p
    if top_k is not None:
        sampling_kwargs["top_k"] = top_k
    if sampling_seed is not None:
        sampling_kwargs["seed"] = int(sampling_seed)
    def start(rank: int) -> int:
        floor = len(indexed_prompts) // len(actors)
        remainder = len(indexed_prompts) % len(actors)
        return rank * floor + min(rank, remainder)

    try:
        shards = [
            indexed_prompts[start(rank) : start(rank + 1)]
            for rank in range(len(actors))
        ]
        futures = [
            actor.generate.remote(shard, sampling_kwargs)
            for actor, shard in zip(actors, shards)
            if shard
        ]
        results: dict[str, list[str]] = {}
        for shard_result in ray.get(futures, timeout=float(os.getenv("SOLVER_RAY_GET_TIMEOUT_S", "3600"))):
            results.update(shard_result)
        expected = {entry["idx"] for entry in indexed_prompts}
        if len(expected) != len(indexed_prompts) or set(results) != expected or any(
            len(values) != sample_num for values in results.values()
        ):
            raise RuntimeError("Incomplete solver sampling; every prompt needs its full attempt count.")
        return results
    finally:
        for actor in actors:
            ray.kill(actor)
        # The reward runs inside VERL's Ray driver, so only remove our actors.


def _sample_local_solver_outputs(
    indexed_problems: list[dict[str, str]],
    *,
    config: VHGRuntimeConfig,
    sample_num: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    max_tokens: int,
) -> dict[str, list[str]]:
    prompt_entries = [
        {
            "idx": item["idx"],
            "prompt": render_generation_prompt(
                [{"role": "user", "content": make_math_solver_prompt(item["problem"])}],
            ),
        }
        for item in indexed_problems
    ]
    return sample_local_solver_prompts(
        prompt_entries,
        config=config,
        sample_num=sample_num,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )


async def evaluate_generated_candidates_batched(
    candidates: list[MathCandidateEvaluationInput],
    *,
    backend,
    soft_verifier_concurrency: int = 16,
    run_solver_eval: bool = True,
    solver_sampler=None,
) -> list[dict[str, Any]]:
    from .score import evaluate_math_candidate_record

    if not candidates:
        return []

    config = getattr(backend, "config", config_from_env())
    results: list[dict[str, Any] | None] = [None] * len(candidates)
    verifier_entries = []

    for position, candidate in enumerate(candidates):
        local_record = evaluate_math_candidate_record(
            seed_problem=candidate.seed.seed_problem,
            seed_solution=candidate.seed.seed_solution,
            reasoning=candidate.reasoning,
            derived_problem=candidate.derived_problem,
            modified_solution=candidate.modified_solution,
            verifier_payload=None,
            objective=candidate.objective,
            metadata=candidate.seed.metadata,
            generation_trace=candidate.generation_trace,
            record_id=candidate.seed.idx,
        )
        if local_record.get("verification", {}).get("reason") != "missing_verifier_payload":
            results[position] = local_record
            continue
        verifier_entries.append({"position": position, "candidate": candidate})

    semaphore = asyncio.Semaphore(max(1, int(soft_verifier_concurrency)))

    async def verify_one(entry: dict[str, Any]):
        candidate = entry["candidate"]
        async with semaphore:
            try:
                payload = await backend.verify_pair(
                    candidate.seed.seed_problem,
                    candidate.seed.seed_solution,
                    candidate.derived_problem,
                    candidate.modified_solution,
                )
            except Exception as exc:
                payload = {
                    "valid_problem": False,
                    "valid_solution": False,
                    "seed_anchored": False,
                    "not_trivial_copy": False,
                    "complete_final_answer": False,
                    "reason": f"soft_verifier_api_error: {type(exc).__name__}: {str(exc)[:240]}",
                }
        return entry, payload

    verifier_results = await asyncio.gather(*(verify_one(entry) for entry in verifier_entries))
    verified_entries = []
    for entry, verifier_payload in verifier_results:
        position = entry["position"]
        candidate = entry["candidate"]
        record = evaluate_math_candidate_record(
            seed_problem=candidate.seed.seed_problem,
            seed_solution=candidate.seed.seed_solution,
            reasoning=candidate.reasoning,
            derived_problem=candidate.derived_problem,
            modified_solution=candidate.modified_solution,
            verifier_payload=verifier_payload,
            objective=candidate.objective,
            metadata=candidate.seed.metadata,
            generation_trace=candidate.generation_trace,
            record_id=candidate.seed.idx,
        )
        results[position] = record
        if record.get("verified_valid"):
            verified_entries.append(
                {
                    "position": position,
                    "candidate": candidate,
                    "record": record,
                    "original_reference_boxed": extract_single_parseable_boxed_answer(
                        candidate.seed.seed_solution
                    )[0],
                }
            )
    if not verified_entries or not run_solver_eval:
        return [record for record in results if record is not None]

    grouped_entries: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    effective_configs: dict[tuple[Any, ...], VHGRuntimeConfig] = {}
    for entry in verified_entries:
        candidate = entry["candidate"]
        effective_config = _effective_solver_config(config, candidate)
        eval_temperature, eval_top_p, eval_top_k = resolve_solver_sampling(
            effective_config,
            n=int(candidate.solver_eval_samples),
        )
        key = (
            int(candidate.solver_eval_samples),
            eval_temperature,
            eval_top_p,
            eval_top_k,
            int(effective_config.solver_max_tokens),
        )
        grouped_entries[key].append(entry)
        effective_configs[key] = effective_config

    for key, entries in grouped_entries.items():
        sample_num, eval_temperature, eval_top_p, eval_top_k, max_tokens = key
        effective_config = effective_configs[key]
        indexed_problems = []
        original_idx_by_seed: dict[tuple[int, str, str], str] = {}
        for entry in entries:
            candidate = entry["candidate"]
            seed_key = (
                int(candidate.seed.idx),
                candidate.seed.seed_problem,
                entry["original_reference_boxed"],
            )
            if seed_key not in original_idx_by_seed:
                original_idx_by_seed[seed_key] = f"seed:{candidate.seed.idx}:original"
                indexed_problems.append(
                    {
                        "idx": original_idx_by_seed[seed_key],
                        "variant": "original",
                        "problem": candidate.seed.seed_problem,
                    }
                )
            entry["original_eval_idx"] = original_idx_by_seed[seed_key]
            indexed_problems.append(
                {
                    "idx": f"{entry['position']}:modified",
                    "variant": "modified",
                    "problem": candidate.derived_problem,
                }
            )

        if solver_sampler is not None:
            solutions_map = solver_sampler(
                indexed_problems=indexed_problems,
                config=effective_config,
                sample_num=sample_num,
                temperature=eval_temperature,
                top_p=eval_top_p,
                top_k=eval_top_k,
                max_tokens=max_tokens,
            )
        else:
            solutions_map = _sample_local_solver_outputs(
                indexed_problems,
                config=effective_config,
                sample_num=sample_num,
                temperature=eval_temperature,
                top_p=eval_top_p,
                top_k=eval_top_k,
                max_tokens=max_tokens,
            )

        for entry in entries:
            position = entry["position"]
            candidate = entry["candidate"]
            record = evaluate_math_candidate_record(
                seed_problem=candidate.seed.seed_problem,
                seed_solution=candidate.seed.seed_solution,
                reasoning=candidate.reasoning,
                derived_problem=candidate.derived_problem,
                modified_solution=candidate.modified_solution,
                verifier_payload=results[position].get("verification", {}),
                original_solver_outputs=solutions_map.get(entry["original_eval_idx"], []),
                modified_solver_outputs=solutions_map.get(f"{position}:modified", []),
                objective=candidate.objective,
                metadata=candidate.seed.metadata,
                generation_trace=candidate.generation_trace,
                record_id=candidate.seed.idx,
            )
            results[position] = record

    return [record for record in results if record is not None]


def load_setter_sampling_entries(
    task: str,
    seed_file: str | Path,
) -> list[dict[str, Any]]:
    entries = []
    for position, row in enumerate(load_jsonl(seed_file)):
        if task == "integration":
            original = str(row.get("original", "")).strip()
            if not original:
                continue
            messages = setter_messages(original)
        elif task == "math":
            problem = str(row.get("seed_problem", "")).strip()
            solution = str(row.get("seed_solution", "")).strip()
            if not problem or not solution:
                continue
            messages = make_math_setter_messages(problem, solution, objective="hard")
        else:
            raise ValueError(f"Unknown task: {task}")
        entries.append(
            {
                "position": position,
                "seed": row,
                "prompt": render_generation_prompt(messages),
            }
        )
    return entries


def sample_setter_checkpoint(
    task: str,
    entries: list[dict[str, Any]],
    *,
    model_path: str,
    visible_devices: str,
    samples_per_seed: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    gpu_memory_utilization: float,
    checkpoint_tag: str | None = None,
    sampling_seed: int | None = None,
) -> list[dict[str, Any]]:
    indexed_prompts = [
        {"idx": str(entry["position"]), "prompt": entry["prompt"]}
        for entry in entries
    ]
    sampled = sample_local_solver_prompts(
        indexed_prompts,
        config=VHGRuntimeConfig(
            solver_ckpt_path=model_path,
            solver_visible_devices=normalize_visible_devices(visible_devices),
            solver_gpu_memory_utilization=gpu_memory_utilization,
        ),
        sample_num=samples_per_seed,
        temperature=temperature,
        top_p=top_p,
        top_k=None,
        max_tokens=max_tokens,
        sampling_seed=sampling_seed,
    )
    rows = []
    checkpoint = str(Path(model_path).resolve())
    checkpoint_tag = checkpoint_tag or Path(model_path).name
    for entry in entries:
        seed = entry["seed"]
        seed_id = seed.get("seed_id", seed.get("id", entry["position"]))
        for sample_idx, response in enumerate(sampled.get(str(entry["position"]), [])):
            base = {
                "id": f"{checkpoint_tag}:{seed_id}:{sample_idx}",
                "checkpoint_tag": checkpoint_tag,
                "checkpoint": checkpoint,
                "seed_id": seed_id,
                "sample_idx": sample_idx,
                "input_prompt": entry["prompt"],
                "response": response,
            }
            if task == "integration":
                base["question_type"] = seed.get("question_type", "integration")
                base["root_seed_id"] = seed.get("root_seed_id", seed_id)
            else:
                base.update(
                    {
                        "seed_problem": seed["seed_problem"],
                        "seed_solution": seed["seed_solution"],
                        "metadata": seed.get("metadata") or {},
                    }
                )
            rows.append(base)
    return rows
