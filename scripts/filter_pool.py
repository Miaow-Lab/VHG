from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vhg.generate import VHGRuntimeConfig, build_verifier_config
from vhg.pool import (
    filter_integration_pool,
    judge_math_pool_file,
    prefilter_math_pool,
    score_math_pool_file,
)
from vhg.service import correctness_service
from vhg.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter and score a sampled VHG problem pool.")
    parser.add_argument("--task", choices=("integration", "math"), required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--solver_model", required=True)
    parser.add_argument("--visible_devices", required=True)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--soft_verifier_base_url", default=os.getenv("SOFT_VERIFIER_BASE_URL", ""))
    parser.add_argument("--soft_verifier_api_key", default=os.getenv("SOFT_VERIFIER_API_KEY", ""))
    parser.add_argument("--soft_verifier_model", default=os.getenv("SOFT_VERIFIER_MODEL", ""))
    parser.add_argument("--soft_verifier_concurrency", type=int, default=256)
    parser.add_argument("--soft_verifier_batch_size", type=int, default=1024)
    parser.add_argument("--service_port", type=int, default=5000)
    parser.add_argument("--previous_archive", help="Previous round's verified_records.jsonl (integration only).")
    parser.add_argument("--round_number", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.previous_archive and args.task != "integration":
        raise ValueError("Cumulative archives are only defined for integration.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "SOLVER_CKPT_PATH": args.solver_model,
            "SOLVER_VISIBLE_DEVICES": args.visible_devices,
            "SOLVER_GPU_MEMORY_UTIL": str(args.gpu_memory_utilization),
            "SOLVER_EVAL_SAMPLES": "10",
            "SOLVER_MAX_TOKENS": "4096",
            "SOLVER_TEMPERATURE": "0.7" if args.task == "math" else "1.0",
            "SOLVER_TOP_P": "0.95",
        }
    )

    if args.task == "integration":
        with correctness_service(args.service_port) as service_url:
            summary = filter_integration_pool(
                args.input_file,
                output_dir,
                service_url=service_url,
                previous_archive=args.previous_archive,
                round_number=args.round_number,
            )
    else:
        template_file = output_dir / "template_deduplicated_records.jsonl"
        prefilter_summary_file = output_dir / "prefilter_summary.json"
        if template_file.exists() and prefilter_summary_file.exists():
            prefilter_summary = json.loads(prefilter_summary_file.read_text(encoding="utf-8"))
        else:
            prefilter_summary = prefilter_math_pool(args.input_file, output_dir)

        pass_band_file = output_dir / "pass_band_records.jsonl"
        difficulty_summary_file = output_dir / "difficulty_summary.json"
        solver_config = VHGRuntimeConfig(
            solver_ckpt_path=args.solver_model,
            solver_visible_devices=args.visible_devices,
            solver_gpu_memory_utilization=args.gpu_memory_utilization,
            solver_max_tokens=4096,
            solver_temperature=0.7,
            solver_top_p=0.95,
        )
        if pass_band_file.exists() and difficulty_summary_file.exists():
            difficulty_summary = json.loads(difficulty_summary_file.read_text(encoding="utf-8"))
        else:
            difficulty_summary = score_math_pool_file(
                template_file,
                pass_band_file,
                config=solver_config,
            )
            save_json(difficulty_summary_file, difficulty_summary)

        judge_summary_file = output_dir / "judge_summary.json"
        accepted_file = output_dir / "accepted_records.jsonl"
        if accepted_file.exists() and judge_summary_file.exists():
            judge_summary = json.loads(judge_summary_file.read_text(encoding="utf-8"))
            if judge_summary.get("target_valid_records") != 20000:
                raise ValueError("Saved judging used a different stopping rule; use a new output directory.")
            if judge_summary.get("batch_size") != args.soft_verifier_batch_size:
                raise ValueError("Judge batch size changed; use a new output directory.")
        else:
            verifier_config = build_verifier_config(
                args.soft_verifier_base_url, args.soft_verifier_api_key, args.soft_verifier_model,
            )
            judge_summary = asyncio.run(
                judge_math_pool_file(
                    pass_band_file,
                    output_dir,
                    verifier_config=verifier_config,
                    concurrency=args.soft_verifier_concurrency,
                    batch_size=args.soft_verifier_batch_size,
                )
            )
            save_json(judge_summary_file, judge_summary)
        summary = {
            **prefilter_summary,
            **difficulty_summary,
            **judge_summary,
        }
        save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
