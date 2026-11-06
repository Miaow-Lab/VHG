from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vhg.evaluate import (
    build_integration_solver_prompts,
    build_math_solver_prompts,
    build_solver_eval_config,
    evaluate_integration_samples,
    evaluate_math_solver_samples,
    sample_solver_prompt_outputs,
)
from vhg.service import correctness_service
from vhg.utils import aggregate_pass_metrics, load_jsonl, save_json, save_jsonl
from vhg.verify import ensure_latex_parser_available


MATH_SCORING_RULE = "last_boxed_answer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VHG solver samples.")
    parser.add_argument("--task", choices=("math", "integration"), default="math")
    parser.add_argument("--records_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--visible_devices", required=True)
    parser.add_argument("--samples_per_problem", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task == "integration":
        ensure_latex_parser_available()
    records = load_jsonl(args.records_file)
    samples_per_problem = args.samples_per_problem
    max_tokens = args.max_tokens
    gpu_memory_utilization = args.gpu_memory_utilization
    if args.task == "integration":
        max_tokens = 4096 if max_tokens is None else max_tokens
        gpu_memory_utilization = (
            0.95 if gpu_memory_utilization is None else gpu_memory_utilization
        )

    if args.task == "integration":
        prompts, questions = build_integration_solver_prompts(records)
        entries = questions
    else:
        math_entries = build_math_solver_prompts(records)
        prompts = [entry["prompt"] for entry in math_entries]
        entries = math_entries

    config = build_solver_eval_config(
        model_path=args.model_path,
        visible_devices=args.visible_devices,
        gpu_memory_utilization=gpu_memory_utilization,
        max_tokens=max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    samples = sample_solver_prompt_outputs(
        prompts,
        config=config,
        samples_per_problem=samples_per_problem,
    )

    if args.task == "integration":
        with correctness_service(shutdown_timeout=5):
            results = evaluate_integration_samples(entries, samples)
    else:
        results = evaluate_math_solver_samples(
            entries,
            samples,
            scoring_rule=MATH_SCORING_RULE,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(output_dir / "evaluation_results.jsonl", results)
    stats = aggregate_pass_metrics(results)
    stats.update(
        {
            "task": args.task,
            "model_path": config.solver_ckpt_path,
            "records_file": args.records_file,
            "samples_per_problem": samples_per_problem,
            "temperature": config.solver_temperature if samples_per_problem > 1 else 0.0,
            "top_p": config.solver_top_p if samples_per_problem > 1 else None,
            "top_k": config.solver_top_k if samples_per_problem > 1 else None,
            "max_new_tokens": config.solver_max_tokens,
            "visible_devices": config.solver_visible_devices,
            "gpu_memory_utilization": config.solver_gpu_memory_utilization,
            "scoring_rule": (
                "hard_verifier" if args.task == "integration" else MATH_SCORING_RULE
            ),
            "prompt_render_mode": "plain",
        }
    )
    save_json(output_dir / "statistics.json", stats)
    print(f"Wrote solver evaluation to {output_dir}")


if __name__ == "__main__":
    main()
