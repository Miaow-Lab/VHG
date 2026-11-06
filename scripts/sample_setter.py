from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vhg.generate import load_setter_sampling_entries, sample_setter_checkpoint
from vhg.pipeline import write_json
from vhg.sampling import allocate_budget, portfolio_order
from vhg.utils import iter_jsonl, save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample the VHG setter checkpoint series.")
    parser.add_argument("--task", choices=("integration", "math"), required=True)
    parser.add_argument("--seed_file", required=True)
    parser.add_argument("--seed_order", help="Later-round train-then-development seed export.")
    parser.add_argument("--sampling_seed", type=int, help="Recorded generation RNG seed.")
    parser.add_argument("--init_checkpoint")
    parser.add_argument("--exclude_init", action="store_true", help="Later rounds sample only new setter checkpoints.")
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--visible_devices", required=True)
    parser.add_argument("--steps", default="")
    parser.add_argument("--samples_per_seed", type=int, default=0)
    parser.add_argument("--generation_budget", type=int, help="Exact total; evenly divided across checkpoints.")
    parser.add_argument("--max_tokens", type=int, default=0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = args.steps or "25,50,75,100,125,150,175,200"
    if not args.exclude_init and not args.init_checkpoint:
        raise ValueError("Provide --init_checkpoint or --exclude_init.")
    if args.samples_per_seed < 0 or (args.generation_budget is not None and args.generation_budget < 1):
        raise ValueError("Sampling counts must be positive.")
    checkpoints = [] if args.exclude_init else [("init", Path(args.init_checkpoint))]
    checkpoints.extend(
        (f"step_{step}", Path(args.checkpoint_root) / f"global_step_{step}_merged")
        for step in (int(value) for value in steps.split(",") if value.strip())
    )
    entries = load_setter_sampling_entries(args.task, args.seed_file)
    seed_order = []
    if args.seed_order:
        ordered = load_setter_sampling_entries(args.task, args.seed_order)
        identity = lambda entry: str(entry["seed"]["seed_id"])
        if len(ordered) != len(entries) or {identity(e) for e in ordered} != {identity(e) for e in entries}:
            raise ValueError("Sampling order must contain exactly the input seed bank.")
        by_id = {identity(e): e["seed"] for e in entries}
        if any(by_id[identity(e)] != e["seed"] for e in ordered):
            raise ValueError("Sampling export changed seed content.")
        entries = ordered
        seed_order = [identity(e) for e in entries]
    if not entries:
        raise ValueError("No usable setter seeds.")
    if len({tag for tag, _ in checkpoints}) != len(checkpoints):
        raise ValueError("Duplicate checkpoint steps.")
    groups = (allocate_budget(entries, len(checkpoints), args.generation_budget)
              if args.generation_budget is not None else
              [[(args.samples_per_seed or (64 if args.task == "integration" else 8), entries)]
               for _ in checkpoints])
    output_file = Path(args.output_file)
    shard_dir = output_file.with_suffix("")
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"task": args.task, "seed_sha256": hashlib.sha256(Path(args.seed_file).read_bytes()).hexdigest(),
                "seed_order": seed_order, "sampling_seed": args.sampling_seed,
                "samples_per_seed": args.samples_per_seed, "generation_budget": args.generation_budget,
                "max_tokens": args.max_tokens, "checkpoints": []}
    for tag, checkpoint in checkpoints:
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Missing setter checkpoint: {checkpoint}")
        files = sorted([*checkpoint.glob("*.safetensors"), *checkpoint.glob("*.json")])
        manifest["checkpoints"].append({"tag": tag, "path": str(checkpoint.resolve()),
            "files": {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in files}})
    manifest_path = shard_dir / "sampling.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Sampling inputs changed; use a new output file.")
    elif list(shard_dir.glob("*.jsonl")):
        raise ValueError("Existing shards have no sampling manifest; use a new output file.")
    else:
        write_json(manifest_path, manifest)

    shard_files = []
    orders = portfolio_order(len(entries), len(checkpoints)) if args.generation_budget is not None else None
    for index, ((tag, checkpoint), jobs) in enumerate(zip(checkpoints, groups)):
        positions = {position: request for request, position in orders[index]} if orders else None
        for samples, seed_entries in jobs:
            if positions is not None:
                seed_entries = [{**entry, "position": positions[entry["position"]]} for entry in seed_entries]
            shard_file = shard_dir / f"{tag}_n{samples}.jsonl"
            expected = samples * len(seed_entries)
            if shard_file.exists():
                if sum(1 for _ in iter_jsonl(shard_file)) != expected:
                    raise ValueError(f"Incomplete sampling shard: {shard_file}")
                print(f"Reusing {shard_file}")
            else:
                rows = sample_setter_checkpoint(
                    args.task, seed_entries, model_path=str(checkpoint),
                    visible_devices=args.visible_devices, samples_per_seed=samples,
                    temperature=1.0, top_p=0.95 if args.task == "integration" else 1.0,
                    max_tokens=(args.max_tokens or (4096 if args.task == "integration" else 8192)),
                    gpu_memory_utilization=args.gpu_memory_utilization, checkpoint_tag=tag,
                    sampling_seed=args.sampling_seed,
                )
                if len(rows) != expected:
                    raise RuntimeError(f"Sampler returned {len(rows)} responses; expected {expected}.")
                temporary = shard_file.with_suffix(".tmp")
                save_jsonl(temporary, rows)
                temporary.replace(shard_file)
            shard_files.append(shard_file)

    total = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as output_handle:
        for shard_file in shard_files:
            with shard_file.open("r", encoding="utf-8") as shard_handle:
                for line in shard_handle:
                    if line.strip():
                        output_handle.write(line)
                        total += 1
    temporary.replace(output_file)
    print(f"Wrote {total} sampled setter outputs to {output_file}")


if __name__ == "__main__":
    main()
