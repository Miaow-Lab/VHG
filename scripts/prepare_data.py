from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vhg.data import write_integration_auxiliary, write_setter_dataset, write_solver_dataset
from vhg.service import correctness_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VHG paper datasets.")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    setter = subparsers.add_parser("setter")
    setter.add_argument("--task", choices=("integration", "math"), required=True)
    setter.add_argument("--seed_file", required=True)
    setter.add_argument("--output_root", required=True)

    auxiliary = subparsers.add_parser("auxiliary", help="Freeze first-round integration auxiliary data.")
    auxiliary.add_argument("--seed_file", required=True)
    auxiliary.add_argument("--cold_sft_data_root", required=True)
    auxiliary.add_argument("--output_root", required=True)
    auxiliary.set_defaults(task="integration")

    solver = subparsers.add_parser("solver")
    solver.add_argument("--task", choices=("integration", "math"), required=True)
    solver.add_argument("--records_file", action="append", required=True)
    solver.add_argument("--output_root", required=True)
    solver.add_argument("--math_seed_data_root", default="")
    solver.add_argument("--math_cold_sft_data_root", default="")
    solver.add_argument("--tokenizer_path", help="Base-model tokenizer for the 4,096-token Soft solver prompt limit.")
    solver.add_argument("--first_round", action="store_true",
                        help="Use the original Hard component order and auxiliary prompts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "setter":
        result = write_setter_dataset(args.task, args.seed_file, args.output_root)
    elif args.task == "integration":
        with correctness_service() as service_url:
            if args.stage == "auxiliary":
                result = write_integration_auxiliary(args.seed_file, args.cold_sft_data_root,
                    args.output_root, service_url=service_url)
            else:
                result = write_solver_dataset(args.task, args.records_file, args.output_root,
                    service_url=service_url, first_round=args.first_round)
    else:
        if not args.tokenizer_path:
            raise ValueError("Soft solver data requires --tokenizer_path for prompt-length filtering.")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
        result = write_solver_dataset(
            args.task,
            args.records_file,
            args.output_root,
            math_seed_data_root=args.math_seed_data_root or None,
            math_cold_sft_data_root=args.math_cold_sft_data_root or None,
            tokenizer=tokenizer,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
