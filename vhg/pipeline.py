"""Run the VHG Hard, Soft, and multi-round integration experiments."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VARIABLE = re.compile(r"\$\{([A-Z_]+)\}")


@dataclass
class Stage:
    name: str
    commands: list[list[str]]
    outputs: list[Path]
    env: dict[str, str] = field(default_factory=dict)


def command(script: str, *args, **options) -> list[str]:
    result = [sys.executable, str(ROOT / "scripts" / f"{script}.py"), *map(str, args)]
    for key, value in options.items():
        if value is not None:
            result.extend([f"--{key}", str(value)])
    return result


def shell_argument(value: str) -> str:
    # Leave documented environment placeholders expandable when copying a plan.
    return f'"{value}"' if VARIABLE.fullmatch(value) else shlex.quote(value)


def build_plan(profile: dict, work: Path, devices: str, rounds: int) -> list[Stage]:
    """Cold SFT once, then setter → sample → filter → solver → evaluate per round."""
    task = profile["task"]
    base = "${BASE_MODEL_PATH}"
    gpus = len(devices.split(","))
    cold = work / "cold_sft"
    initializer = cold / "checkpoints/selected"
    soft = ({"soft_verifier_base_url": "${SOFT_VERIFIER_BASE_URL}",
             "soft_verifier_model": "${SOFT_VERIFIER_MODEL}"} if task == "math" else {})
    stages = [
        Stage("cold-data", [command("prepare_cold_sft", task=task,
              seed_file=ROOT / profile["cold_seeds"], output_dir=cold / "data",
              generation_budget=profile.get("cold_generation_budget"),
              api_base_url="${TEACHER_BASE_URL}", api_model="${TEACHER_MODEL}",
              tokenizer_path=base if task == "math" else None, **soft)],
              [cold / "data/train.parquet", cold / "data/test.parquet"]),
        Stage("cold-sft", [
            command("train_sft", data_root=cold / "data", model_path=base,
                    output_dir=cold / "checkpoints", experiment_name=f"{task}_cold_sft",
                    gpus=profile["sft_gpus"], train_batch_size=profile["sft_batch_size"]),
            command("merge_checkpoints", checkpoint_root=cold / "checkpoints",
                    preferred_step=profile["sft_checkpoint"])], [initializer]),
    ]
    for number in range(1, rounds + 1):
        run = work / f"round{number}"
        previous = work / f"round{number - 1}"
        later = profile.get("later_rounds", {}) if number > 1 else {}
        seeds = run / "seeds/seeds.jsonl" if later else ROOT / profile["setter_seeds"]
        sampling_seeds = seeds if later else ROOT / profile["sampling_seeds"]
        setter_model = previous / "setter/selected" if later else initializer
        solver_model = previous / "solver/selected" if later else base
        setter_steps = profile["setter_steps"]
        solver_steps = profile["solver_steps"]
        steps = ",".join(map(str, range(25, setter_steps + 1, 25)))
        env = {
            "CUDA_VISIBLE_DEVICES": devices, "SOLVER_VISIBLE_DEVICES": devices,
            "SOLVER_CKPT_PATH": str(solver_model), "SOLVER_GPU_MEMORY_UTIL": "0.9",
            "SOLVER_EVAL_SAMPLES": "4" if task == "math" else "10",
            "SOLVER_MAX_TOKENS": "2048" if task == "math" else "4096",
            "SOLVER_TEMPERATURE": "0.7" if task == "math" else "1.0",
            "SOLVER_TOP_P": "0.95",
        }
        if soft:
            env.update(SOFT_VERIFIER_BASE_URL=soft["soft_verifier_base_url"],
                       SOFT_VERIFIER_MODEL=soft["soft_verifier_model"])

        def add(name, commands, outputs):
            stages.append(Stage(f"r{number}.{name}", commands, outputs, dict(env)))

        def train(role, model, total):
            args = command("train_rl", role=role, domain=task, data_root=run / "data",
                           model_path=model, output_dir=run / role, gpus=gpus,
                           project_name=f"vhg_{task}", experiment_name=f"round{number}_{role}",
                           verl_override=f"trainer.total_training_steps={total}") + [
                               # Allow the update limit to govern even with one batch per epoch.
                               "--verl_override", f"trainer.total_epochs={total}"]
            if later:
                args += ["--verl_override", f"data.seed={later['data_seeds'][number - 2]}"]
            return args

        def select(role):
            interval = 25 if role == "setter" else 50
            total = setter_steps if role == "setter" else solver_steps
            return command("select_checkpoint", checkpoint_root=run / role,
                           validation_file=run / "data" / role / "test.parquet",
                           steps=",".join(map(str, range(interval, total + 1, interval))))

        if later:
            previous_seeds = ROOT / profile["setter_seeds"] if number == 2 else previous / "seeds/seeds.jsonl"
            # Productivity uses the solver that scored the preceding setter,
            # not the newly trained solver used by the upcoming setter.
            screening_solver = base if number == 2 else work / f"round{number - 2}" / "solver/selected"
            add("seeds", [command("expand_seeds", seed_file=previous_seeds, previous_run=previous,
                output_dir=run / "seeds", round_number=number, solver_model=screening_solver,
                embedding_model="${EMBED_MODEL_PATH}", visible_devices=devices,
                seed=later["expansion_seed"])], [seeds, run / "seeds/summary.json"])
            stages[-1].env["SOLVER_CKPT_PATH"] = str(screening_solver)
        add("data", [command("prepare_data", "setter", task=task, seed_file=seeds,
                             output_root=run / "data")], [run / "data/setter" / name for name in
                             ("train.parquet", "test.parquet", "sampling_seeds.jsonl")])
        add("setter", [train("setter", setter_model, setter_steps),
                       command("merge_checkpoints", checkpoint_root=run / "setter", steps=steps),
                       select("setter")], [run / "setter/selected"])
        sample = command("sample_setter", task=task, seed_file=sampling_seeds,
                         seed_order=run / "data/setter/sampling_seeds.jsonl" if later else None,
                         sampling_seed=later["generation_seeds"][number - 2] if later else None,
                         init_checkpoint=None if later else initializer, checkpoint_root=run / "setter",
                         output_file=run / "pool/generated.jsonl", visible_devices=devices, steps=steps,
                         generation_budget=later.get("generation_budget"),
                         samples_per_seed=None if later else profile["samples_per_seed"])
        if later:
            sample.append("--exclude_init")
        add("sample", [sample], [run / "pool/generated.jsonl"])
        add("filter", [command("filter_pool", task=task, input_file=run / "pool/generated.jsonl",
                               output_dir=run / "pool/filtered", solver_model=solver_model,
                               visible_devices=devices,
                               previous_archive=previous / "pool/filtered/verified_records.jsonl" if later else None,
                               round_number=number, **soft)], [run / "pool/filtered/summary.json"])
        accepted = run / "pool/filtered" / ("solver_pass_band_records.jsonl" if task == "integration" else "accepted_records.jsonl")
        prepare = command("prepare_data", "solver", task=task, records_file=accepted, output_root=run / "data")
        preparation = []
        if task == "integration":
            auxiliary = work / "auxiliary"
            if number == 1:
                preparation.append(command("prepare_data", "auxiliary",
                    seed_file=ROOT / profile["setter_seeds"], cold_sft_data_root=cold / "data",
                    output_root=auxiliary))
            # Freeze the first-round representatives and order once, then reuse.
            prepare += ["--records_file", str(auxiliary / "records.jsonl")]
            if number == 1:
                prepare.append("--first_round")
        else:
            prepare += ["--math_seed_data_root", str(run / "data/setter"),
                        "--math_cold_sft_data_root", str(cold / "data"), "--tokenizer_path", base]
        add("solver-data", [*preparation, prepare], [run / "data/solver/train.parquet", run / "data/solver/test.parquet"])
        if task == "integration" and number == 1:
            stages[-1].outputs.append(auxiliary / "records.jsonl")
        add("solver", [train("solver", solver_model, solver_steps),
                       select("solver")], [run / "solver/selected"])
        evaluations = [command("prepare_benchmarks", task=task, output_dir=work / "benchmarks",
                              antiderivbench_root="${ANTIDERIVBENCH_ROOT}" if task == "integration" else None)]
        for benchmark in profile["benchmarks"]:
            evaluations.append(command("eval_solver", task=task,
                records_file=work / "benchmarks" / f"{benchmark}.jsonl",
                output_dir=run / "evaluation" / benchmark, model_path=run / "solver/selected",
                visible_devices=devices, samples_per_problem=profile["evaluation_samples"],
                temperature=0.7, top_p=0.95,
                max_tokens=8192 if benchmark in {"amc", "aime2024", "aime2025", "aime2026"} else 4096,
                gpu_memory_utilization=0.95))
        add("eval", evaluations, [run / "evaluation" / b / "statistics.json" for b in profile["benchmarks"]])
    return stages


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def source_digest() -> str:
    digest = hashlib.sha256()
    for folder in ("vhg", "scripts", "configs", "data"):
        for path in sorted((ROOT / folder).iterdir()):
            if path.is_file() and path.suffix in {".py", ".json", ".jsonl", ".txt"}:
                digest.update(str(path.relative_to(ROOT)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def execute(stages: list[Stage], work: Path, specification: dict, *, resume: bool) -> None:
    # One owner per output directory; failed commands never receive a completion marker.
    import fcntl

    work.mkdir(parents=True, exist_ok=True)
    with (work / ".pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = work / "run.json"
        if manifest_path.exists():
            if not resume:
                raise ValueError("Run exists; use --resume or a new --work-dir.")
            manifest = json.loads(manifest_path.read_text())
            if manifest["specification"] != specification:
                raise ValueError("Code, data, or configuration changed; use a new --work-dir.")
        else:
            if any(p.name != ".pipeline.lock" for p in work.iterdir()):
                raise ValueError("Output directory is not empty and has no run.json; use a new --work-dir.")
            manifest = {"specification": specification, "bindings": {}, "completed": [],
                        "commands_completed": {}, "running": None}
        pending = [s for s in stages if s.name not in manifest["completed"]]
        required = {name for s in pending for value in [*s.env.values(), *(v for c in s.commands for v in c)]
                    for name in VARIABLE.findall(value)}
        bindings = dict(manifest["bindings"])
        for name in sorted(required):
            value = os.environ.get(name) or bindings.get(name)
            if not value:
                raise ValueError(f"Set {name} before running these stages.")
            if name in bindings and value != bindings[name]:
                raise ValueError(f"{name} changed; use a new --work-dir.")
            bindings[name] = value
        # Credentials are inherited by child processes, never placed in commands or run.json.
        manifest["bindings"] = bindings
        write_json(manifest_path, manifest)
        def resolve(value):
            return VARIABLE.sub(lambda match: bindings[match[1]], value)
        for stage in stages:
            if stage.name in manifest["completed"]:
                if not all(p.exists() for p in stage.outputs):
                    raise ValueError(f"Completed stage {stage.name} has missing outputs; restore them or use a new run.")
                print(f"[{stage.name}] complete", flush=True)
                continue
            print(f"[{stage.name}] running", flush=True)
            completed_commands = manifest["commands_completed"].get(stage.name, 0)
            if stage.name == "cold-sft" and manifest["running"] == ["cold-sft", 0]:
                raise ValueError("Cold SFT was interrupted and has no resume support; use a new --work-dir.")
            env = {**os.environ, **bindings, "CUDA_VISIBLE_DEVICES": specification["devices"],
                   **{key: resolve(value) for key, value in stage.env.items()}}
            for index, args in enumerate(stage.commands):
                if index < completed_commands:
                    continue
                resolved = [resolve(value) for value in args]
                print(shlex.join(resolved), flush=True)
                manifest["running"] = [stage.name, index]
                write_json(manifest_path, manifest)
                subprocess.run(resolved, cwd=ROOT, env=env, check=True)
                manifest["commands_completed"][stage.name] = index + 1
                manifest["running"] = None
                write_json(manifest_path, manifest)
            if not all(p.exists() for p in stage.outputs):
                raise RuntimeError(f"Stage {stage.name} returned without its expected outputs.")
            manifest["completed"].append(stage.name)
            write_json(manifest_path, manifest)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("integration", "general_math"))
    parser.add_argument("--rounds", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--devices", default=os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"))
    parser.add_argument("--from-stage")
    parser.add_argument("--through-stage")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without importing GPU libraries or writing files.")
    parser.add_argument("--resume", action="store_true", help="Skip successful stages with the same code, inputs and settings.")
    args = parser.parse_args(argv)
    if args.experiment == "general_math" and args.rounds != 1:
        parser.error("Only integration has a multi-round paper experiment.")
    devices = args.devices.split(",")
    if not all(d.strip() for d in devices) or len(set(devices)) != len(devices):
        parser.error("--devices must be a nonempty, unique comma-separated list.")
    profile = json.loads((ROOT / "configs" / f"{args.experiment}.json").read_text())
    work = (args.work_dir or ROOT / "outputs" / args.experiment).resolve()
    stages = build_plan(profile, work, args.devices, args.rounds)
    names = [s.name for s in stages]
    for name in (args.from_stage, args.through_stage):
        if name and name not in names:
            parser.error(f"Unknown stage {name}. Stages: {', '.join(names)}")
    start = names.index(args.from_stage) if args.from_stage else 0
    stop = names.index(args.through_stage) + 1 if args.through_stage else len(stages)
    if start >= stop:
        parser.error("--from-stage must precede --through-stage.")
    stages = stages[start:stop]
    if args.dry_run:
        for stage in stages:
            print(f"\n[{stage.name}]")
            if stage.env:
                print("export " + " ".join(f"{key}={shell_argument(value)}" for key, value in stage.env.items()))
            for cmd in stage.commands:
                print(" ".join(map(shell_argument, cmd)))
        return
    if "cold-sft" in [s.name for s in stages] and len(devices) < profile["sft_gpus"]:
        parser.error(f"Cold SFT requires at least {profile['sft_gpus']} visible GPUs for this profile.")
    try:
        execute(stages, work, {"profile": profile, "rounds": args.rounds,
                "devices": args.devices, "source_sha256": source_digest()}, resume=args.resume)
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
