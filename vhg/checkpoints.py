"""Checkpoint selection from complete development evaluations, without GPU imports."""
from collections import defaultdict
import json
import math
from pathlib import Path


def link_selected_checkpoint(root: Path, step: int) -> None:
    selected = root / "selected"
    target = Path(f"global_step_{step}_merged")
    if selected.is_symlink():
        if selected.readlink() != target:
            raise ValueError("A different checkpoint is already selected; use a new run directory.")
    elif selected.exists():
        raise ValueError(f"Selection path is not a symlink: {selected}")
    else:
        selected.symlink_to(target, target_is_directory=True)


def training_finished(root: Path, target: int, gpus: int) -> bool:
    """Avoid an extra VERL update when retrying after the final save."""
    marker = root / "latest_checkpointed_iteration.txt"
    if not marker.exists():
        return False
    step = int(marker.read_text().strip())
    if step > target:
        raise ValueError(f"Checkpoint step {step} exceeds requested training limit {target}.")
    if step < target:
        return False
    checkpoint = root / f"global_step_{step}"
    files = [checkpoint / "data.pt"] + [
        checkpoint / "actor" / f"{kind}_world_size_{gpus}_rank_{rank}.pt"
        for rank in range(gpus) for kind in ("model", "optim", "extra_state")
    ]
    if not all(p.is_file() and p.stat().st_size > 0 for p in files):
        raise ValueError(f"Final checkpoint is incomplete for {gpus} ranks: {checkpoint}")
    return True


def select_by_validation(root: Path, expected_prompts: set[str], steps: list[int], samples: int = 10):
    if not expected_prompts or not steps:
        raise ValueError("Selection needs development prompts and scheduled checkpoint steps.")
    metrics = {}
    for step in steps:
        if step <= 0:
            raise ValueError("Select among trained checkpoints only (positive steps).")
        if not (root / f"global_step_{step}").is_dir():
            raise ValueError(f"Missing checkpoint at step {step}")
        groups = defaultdict(list)
        with (root / "validation_data" / f"{step}.jsonl").open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                score = float(row["score"])
                if not math.isfinite(score):
                    raise ValueError(f"Non-finite validation score at step {step}")
                groups[row["input"].strip()].append(score)
        if set(groups) != expected_prompts or any(len(v) != samples for v in groups.values()):
            raise ValueError(f"Incomplete development evaluation at step {step}; expected {samples} samples per prompt.")
        metrics[step] = sum(sum(v) for v in groups.values()) / (samples * len(groups))
    selected = max(metrics, key=lambda step: (metrics[step], step))
    return {"selected_step": selected, "mean_rewards": metrics,
            "rule": "Highest development mean reward; latest step in an exact tie."}
