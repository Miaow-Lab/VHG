from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vhg.collect import (
    collect_api_candidates,
    integration_cold_sft_jobs,
    math_cold_sft_jobs,
    process_integration_cold_sft,
    process_math_cold_sft,
)
from vhg.generate import build_verifier_config
from vhg.service import correctness_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and prepare VHG cold-SFT data.")
    parser.add_argument("--task", choices=("integration", "math"), required=True)
    parser.add_argument("--seed_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--api_base_url", required=True)
    parser.add_argument("--api_model", required=True)
    parser.add_argument(
        "--api_key",
        default=os.getenv("COLD_SFT_API_KEY", ""),
    )
    parser.add_argument("--icl_examples", default="data/integration_cold_sft_icl.txt")
    parser.add_argument("--samples_per_seed", type=int, default=0)
    parser.add_argument("--generation_budget", type=int,
                        help="Teacher candidate jobs before verification; defaults to all seed jobs.")
    parser.add_argument("--concurrency", type=int, default=0)
    parser.add_argument("--tokenizer_path", help="Base-model tokenizer for Soft's length filters.")
    parser.add_argument("--soft_verifier_base_url", default=os.getenv("SOFT_VERIFIER_BASE_URL", ""))
    parser.add_argument("--soft_verifier_api_key", default=os.getenv("SOFT_VERIFIER_API_KEY", ""))
    parser.add_argument("--soft_verifier_model", default=os.getenv("SOFT_VERIFIER_MODEL", ""))
    parser.add_argument("--soft_verifier_concurrency", type=int, default=256)
    parser.add_argument("--service_port", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task == "math" and not args.tokenizer_path:
        raise ValueError("Soft cold SFT requires --tokenizer_path for its token-length filters.")
    if not args.api_key:
        raise ValueError("Set COLD_SFT_API_KEY or pass --api_key.")
    if args.generation_budget is not None and args.generation_budget < 1:
        raise ValueError("The generation budget must be positive.")
    output_dir = Path(args.output_dir)
    raw_file = output_dir / "raw_candidates.jsonl"

    if args.task == "integration":
        samples_per_seed = args.samples_per_seed or 10
        concurrency = args.concurrency or 32
        icl_examples = Path(args.icl_examples).read_text(encoding="utf-8")
        jobs = integration_cold_sft_jobs(
            args.seed_file,
            icl_examples=icl_examples,
            samples_per_seed=samples_per_seed,
        )
        temperature = 1.0
    else:
        samples_per_seed = args.samples_per_seed or 1
        concurrency = args.concurrency or 128
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
        jobs = math_cold_sft_jobs(args.seed_file, samples_per_seed=samples_per_seed)
        temperature = 0.0

    # Limit raw generation jobs, not the number surviving verification.
    jobs = jobs[:args.generation_budget]
    asyncio.run(collect_api_candidates(
        jobs, raw_file, base_url=args.api_base_url, api_key=args.api_key,
        model=args.api_model, concurrency=concurrency, temperature=temperature,
        max_tokens=4096,
    ))
    if args.task == "integration":
        with correctness_service(args.service_port) as service_url:
            summary = process_integration_cold_sft(
                raw_file, output_dir, service_url=service_url,
            )
    else:
        verifier_config = build_verifier_config(
            args.soft_verifier_base_url, args.soft_verifier_api_key, args.soft_verifier_model,
        )
        summary = asyncio.run(
            process_math_cold_sft(
                raw_file,
                output_dir,
                verifier_config=verifier_config,
                tokenizer=tokenizer,
                verifier_concurrency=args.soft_verifier_concurrency,
            )
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
