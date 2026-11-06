from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vhg.train import VerlSFTConfig, build_verl_sft_command, run_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the VHG cold-SFT setter.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--save_freq", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = VerlSFTConfig(
        data_root=args.data_root,
        model_path=args.model_path,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        n_gpus_per_node=args.gpus,
        train_batch_size=args.train_batch_size,
        save_freq=args.save_freq,
    )
    raise SystemExit(run_command(build_verl_sft_command(config), cwd=REPO_ROOT))


if __name__ == "__main__":
    main()
