from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vhg.train import build_merge_checkpoint_command, run_command
from vhg.checkpoints import link_selected_checkpoint
from vhg.pipeline import write_json


STEP_RE = re.compile(r"^global_step_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge VERL FSDP checkpoints for vLLM.")
    parser.add_argument("--checkpoint_root", required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--steps",
        default="",
        help="Comma-separated steps. The default merges every global_step_* checkpoint.",
    )
    selection.add_argument("--preferred_step", type=int,
                          help="Cold SFT: select this step, or the latest saved step if absent.")
    return parser.parse_args()


def merged_checkpoint_is_valid(path: Path) -> bool:
    if not (path / "config.json").is_file() or not any(
        (path / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json")
    ):
        return False
    weights = path / "model.safetensors"
    if weights.is_file():
        return weights.stat().st_size > 0
    index = path / "model.safetensors.index.json"
    try:
        shards = set(json.loads(index.read_text())["weight_map"].values())
    except (OSError, ValueError, KeyError):
        return False
    return bool(shards) and all((path / name).is_file() and (path / name).stat().st_size > 0 for name in shards)


def main() -> None:
    args = parse_args()
    root = Path(args.checkpoint_root)
    selected_steps = {int(step) for step in args.steps.split(",") if step.strip()}
    checkpoints = []
    for path in root.glob("global_step_*"):
        match = STEP_RE.match(path.name)
        if path.is_dir() and match and (not selected_steps or int(match.group(1)) in selected_steps):
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()
    missing = selected_steps - {step for step, _ in checkpoints}
    if missing:
        raise ValueError(f"Missing requested checkpoint steps under {root}: {sorted(missing)}")
    if not checkpoints:
        raise ValueError(f"No requested global_step_* checkpoints found under {root}.")
    if args.preferred_step is not None:
        if args.preferred_step < 1:
            raise ValueError("The preferred checkpoint step must be positive.")
        checkpoints = [next((item for item in checkpoints if item[0] == args.preferred_step), checkpoints[-1])]

    for _, checkpoint in checkpoints:
        target = checkpoint.with_name(f"{checkpoint.name}_merged")
        if merged_checkpoint_is_valid(target):
            print(f"Reusing {target}")
            continue
        if target.exists():
            raise ValueError(f"Incomplete merge target already exists: {target}")
        local_dir = checkpoint / "actor" if (checkpoint / "actor").is_dir() else checkpoint
        return_code = run_command(
            build_merge_checkpoint_command(local_dir, target),
            cwd=REPO_ROOT,
        )
        if return_code:
            raise SystemExit(return_code)
        if not merged_checkpoint_is_valid(target):
            raise RuntimeError(f"Merged checkpoint is incomplete: {target}")
        print(target)

    if args.preferred_step is not None:
        step = checkpoints[0][0]
        link_selected_checkpoint(root, step)
        write_json(root / "selection.json", {
            "preferred_step": args.preferred_step, "selected_step": step,
            "selection_rule": "preferred_step_or_latest",
        })


if __name__ == "__main__":
    main()
