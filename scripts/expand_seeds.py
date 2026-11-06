"""Expand the next integration seed bank from the current run's saved evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vhg.pipeline import write_json
from vhg.sampling import allocate_budget
from vhg.seeds import embed_seed_texts, expand_bank, screen_seeds
from vhg.utils import iter_jsonl, load_jsonl, save_jsonl


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_draws(previous, seeds, seed_file):
    """Use the selected setter's saved samples, checking every seed's draw count."""
    selection = json.loads((previous / "setter/selection.json").read_text())
    tag = f"step_{selection['selected_step']}"
    sampling = json.loads((previous / "pool/generated/sampling.json").read_text())
    if sampling["seed_sha256"] != file_hash(seed_file):
        raise ValueError("Sampling seed bank differs from the bank being expanded.")
    tags = [checkpoint["tag"] for checkpoint in sampling["checkpoints"]]
    index = tags.index(tag)
    if sampling.get("seed_order"):
        by_id = {str(row["seed_id"]): row for row in seeds}
        order = sampling["seed_order"]
        if len(order) != len(seeds) or set(order) != set(by_id):
            raise ValueError("Sampling order differs from the bank being expanded.")
        seeds = [by_id[seed_id] for seed_id in order]
    if sampling["generation_budget"] is not None:
        jobs = allocate_budget(seeds, len(tags), sampling["generation_budget"])[index]
        expected = {str(row["seed_id"]): n for n, rows in jobs for row in rows}
    else:
        expected = {str(row["seed_id"]): sampling["samples_per_seed"] or 64 for row in seeds}
    rows = [row for row in iter_jsonl(previous / "pool/generated.jsonl") if row["checkpoint_tag"] == tag]
    if Counter(str(row["seed_id"]) for row in rows) != expected:
        raise ValueError("Incomplete selected-checkpoint generation coverage; cannot prune seeds.")
    if len({(str(row["seed_id"]), row["sample_idx"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate saved selected-checkpoint draws.")
    return rows


def assess_responses(responses):
    from vhg.generate import sample_local_solver_prompts
    from vhg.score import score_hard_verifier_setter_candidates_batched

    def sample(indexed_prompts, **options):
        sampled = sample_local_solver_prompts(indexed_prompts, **options)
        if set(sampled) != {str(row["idx"]) for row in indexed_prompts} or any(
            len(values) != options["sample_num"] for values in sampled.values()
        ):
            raise RuntimeError("Incomplete ten-attempt productivity assessment.")
        return sampled

    return score_hard_verifier_setter_candidates_batched(
        responses, [{"question_type": "integration", "sample_num": 10, "temperature": 1.0,
                     "top_p": 0.95, "max_tokens": 4096, "objective": "hard"} for _ in responses],
        solver_sampler=sample)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed_file", type=Path, required=True)
    parser.add_argument("--previous_run", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--round_number", type=int, choices=(2, 3), required=True)
    parser.add_argument("--solver_model", required=True, help="Frozen solver used by the PREVIOUS setter (S0 for Q1; S1 for Q2).")
    parser.add_argument("--embedding_model", required=True)
    parser.add_argument("--visible_devices", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--service_port", type=int, default=5000)
    args = parser.parse_args()
    previous, output = args.previous_run, args.output_dir
    archive_file = previous / "pool/filtered/verified_records.jsonl"
    audit_file = previous / "pool/filtered/audited_records.jsonl"
    evidence_files = [*sorted((previous / "setter/validation_data").glob("*.jsonl")),
                      *sorted((previous / "setter/rollout_data").glob("*.jsonl"))]
    inputs = [args.seed_file, archive_file, audit_file, previous / "setter/selection.json",
              previous / "pool/generated/sampling.json", previous / "pool/generated.jsonl", *evidence_files]
    manifest = {"round": args.round_number, "seed": args.seed,
                "source_sha256": {str(p.resolve()): file_hash(p) for p in inputs},
                "code_sha256": {p.name: file_hash(p) for p in (Path(__file__), ROOT / "vhg/seeds.py")},
                "solver_model": str(Path(args.solver_model).resolve()),
                "embedding_model": str(Path(args.embedding_model).resolve())}
    output.mkdir(parents=True, exist_ok=True)
    manifest_file = output / "inputs.json"
    if not manifest_file.exists() and any(output.iterdir()):
        raise ValueError("Existing seed-expansion outputs have no input manifest; use a new directory.")
    if manifest_file.exists() and json.loads(manifest_file.read_text()) != manifest:
        raise ValueError("Seed-expansion inputs changed; use a new output directory.")
    write_json(manifest_file, manifest)
    seeds_file, summary_file = output / "seeds.jsonl", output / "summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
        if not seeds_file.exists() or file_hash(seeds_file) != summary["seeds_sha256"]:
            raise ValueError("Completed expanded seed bank is missing or changed.")
        print(json.dumps(summary, indent=2))
        return

    seeds = load_jsonl(args.seed_file)
    generations = selected_draws(previous, seeds, args.seed_file)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.visible_devices, SOLVER_VISIBLE_DEVICES=args.visible_devices,
                      SOLVER_CKPT_PATH=args.solver_model,
                      CORRECTNESS_SERVICE_URL=f"http://localhost:{args.service_port}")
    from vhg.service import CorrectnessClient, correctness_service
    from transformers import AutoTokenizer

    with correctness_service(args.service_port, pool_size=32) as url:
        try:
            productivity_file = output / "productivity.json"
            if productivity_file.exists():
                productivity = json.loads(productivity_file.read_text())
                retained_ids = set(productivity["retained_ids"])
                retained = [row for row in seeds if str(row["seed_id"]) in retained_ids]
            else:
                def evidence():
                    # An older archive member's parent setter is not the current setter.
                    yield from (row for row in iter_jsonl(audit_file)
                                if row["archive_first_round"] == args.round_number - 1)
                    for path in evidence_files:
                        yield from iter_jsonl(path)
                retained, productivity = screen_seeds(seeds, evidence(), generations, assess_responses)
                write_json(productivity_file, productivity)
            # Release solver workers before loading the embedding model on cuda:0.
            if "ray" in sys.modules:
                sys.modules["ray"].shutdown()
            tokenizer = AutoTokenizer.from_pretrained(previous / "setter/selected", local_files_only=True)
            with CorrectnessClient(url) as client:
                bank, summary = expand_bank(retained, load_jsonl(archive_file), client=client, tokenizer=tokenizer,
                    embed=lambda texts: embed_seed_texts(texts, args.embedding_model),
                    round_number=args.round_number, seed=args.seed)
            temporary = seeds_file.with_suffix(".tmp")
            save_jsonl(temporary, bank)
            temporary.replace(seeds_file)
            summary.update(round=args.round_number, input_seeds=len(seeds),
                           removed_unproductive=len(productivity["removed_ids"]), seeds_sha256=file_hash(seeds_file))
            write_json(summary_file, summary)
            print(json.dumps(summary, indent=2))
        finally:
            if "ray" in sys.modules:
                sys.modules["ray"].shutdown()


if __name__ == "__main__":
    main()
