from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vhg.benchmarks import prepare_integration_benchmarks, prepare_math_benchmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the public VHG evaluation suites.")
    parser.add_argument("--task", choices=("integration", "math"), required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--antiderivbench_root", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task == "integration":
        counts = prepare_integration_benchmarks(args.antiderivbench_root, args.output_dir)
    else:
        counts = prepare_math_benchmarks(args.output_dir)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
