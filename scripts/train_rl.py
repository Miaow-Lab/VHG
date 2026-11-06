from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vhg.service import correctness_service
from vhg.checkpoints import training_finished
from vhg.train import (
    build_setter_rl_config,
    build_solver_rl_config,
    build_verl_ppo_command,
    run_command,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VHG solver or setter RL.")
    parser.add_argument("--role", choices=("solver", "setter"), default="solver")
    parser.add_argument("--domain", choices=("integration", "math"), default="integration")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--project_name", default="vhg_solver_rl")
    parser.add_argument("--experiment_name", default="solver")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument(
        "--verl_override", action="append", default=[],
        help="An explicit Hydra override, repeatable (KEY=VALUE).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = dict(value.lstrip("+").split("=", 1) for value in args.verl_override)
    target = overrides.get("trainer.total_training_steps")
    if target and overrides.get("trainer.resume_mode", "auto") != "disable":
        if training_finished(Path(args.output_dir), int(target), args.gpus):
            print(f"Training already completed {target} updates in {args.output_dir}")
            return
    if args.role == "solver":
        config = build_solver_rl_config(
            data_root=args.data_root,
            model_path=args.model_path,
            project_name=args.project_name,
            experiment_name=args.experiment_name,
            output_dir=args.output_dir,
            n_gpus_per_node=args.gpus,
            domain=args.domain,
        )
    elif args.role == "setter":
        config = build_setter_rl_config(
            data_root=args.data_root,
            model_path=args.model_path,
            project_name=args.project_name,
            experiment_name=args.experiment_name,
            output_dir=args.output_dir,
            n_gpus_per_node=args.gpus,
            domain=args.domain,
        )
    command = build_verl_ppo_command(config) + args.verl_override
    if args.domain == "integration":
        with correctness_service(shutdown_timeout=5):
            raise SystemExit(run_command(command, cwd=REPO_ROOT))
    if args.domain == "math":
        os.environ.pop("STANDALONE_SERVICE_PORT", None)
    raise SystemExit(run_command(command, cwd=REPO_ROOT))


if __name__ == "__main__":
    main()
