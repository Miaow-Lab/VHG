"""Round-boundary seed expansion: productivity, identity, then farthest-first selection."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import random

import numpy as np

from .prompts import setter_messages
from .utils import render_generation_prompt
from .verify import normalize_expression_text_for_exact_dedup as normalize


def seed_prompt(row):
    return render_generation_prompt(setter_messages(row["original"]))


def productive(row):
    rate = row.get("solver_hard_verifier_pass_rate")
    return (row.get("reference_symbolic_valid") is True and rate is not None
            and math.isfinite(float(rate)) and 0 <= float(rate) < 1)


def screen_seeds(seeds, evidence, generations, assess):
    """Retain seeds with a verified difficult outcome under the preceding setter's solver.

    Saved rollout/audit positives are sufficient. For unresolved seeds, assess
    the selected setter checkpoint's saved draws, including ones lost to pool
    deduplication. Missing draws are an error, never a reason to prune a seed.
    """
    ids = [str(row["seed_id"]) for row in seeds]
    if len(set(ids)) != len(ids):
        raise ValueError("Seed identifiers must be unique.")
    by_prompt = defaultdict(list)
    for row, identity in zip(seeds, ids):
        by_prompt[seed_prompt(row).strip()].append(identity)
    proof = {}
    for row in evidence:
        if productive(row):
            prompt = str(row.get("input", row.get("input_prompt", ""))).strip()
            for identity in by_prompt.get(prompt, []):
                proof.setdefault(identity, {"source": "saved_evidence", "pass_rate": row["solver_hard_verifier_pass_rate"]})
            if len(proof) == len(ids):
                break
    unresolved = set(ids) - proof.keys()
    draws = defaultdict(list)
    for row in generations:
        identity = str(row["seed_id"])
        if identity in unresolved:
            draws[identity].append(str(row["response"]))
    missing = unresolved - draws.keys()
    if missing:
        raise ValueError(f"No saved selected-checkpoint draws for seeds: {sorted(missing)}")
    # Identical full responses share a check, without transferring validity
    # between different references for the same integrand.
    responses = list(dict.fromkeys(text for values in draws.values() for text in values))
    scores = assess(responses) if responses else []
    if len(scores) != len(responses):
        raise RuntimeError("Incomplete productivity assessment.")
    scored = dict(zip(responses, scores))
    for identity, texts in draws.items():
        for text in texts:
            if productive(scored[text]):
                proof[identity] = {"source": "selected_checkpoint_draw", "pass_rate": scored[text]["solver_hard_verifier_pass_rate"]}
                break
    retained = [row for row in seeds if str(row["seed_id"]) in proof]
    return retained, {"retained_ids": [str(r["seed_id"]) for r in retained],
                      "removed_ids": [i for i in ids if i not in proof],
                      "proof": proof, "newly_assessed_responses": len(responses),
                      "assessed_draws_per_seed": {i: len(v) for i, v in draws.items()}}


def order_archive(rows, seed=42):
    """Seeded round-robin order across parent lineages, saved before selection."""
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get("root_seed_id", row.get("seed_id", "unknown")))].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    ordered = []
    while groups:
        keys = list(groups)
        rng.shuffle(keys)
        for key in keys:
            ordered.append(groups[key].pop())
            if not groups[key]:
                del groups[key]
    return ordered


def deduplicate_seeds(retained, candidates, client):
    """Exact and 20-point identity of BOTH antiderivative and integrand.

    Keep old seeds first, then the first archive representative. Unavailable
    numerical fingerprints fall back to exact text, as in the original runs.
    """
    records = retained + candidates
    expressions = list(dict.fromkeys(row[key] for row in records for key in ("original", "expr")))
    fingerprints = {}
    for start in range(0, len(expressions), 128):
        batch = expressions[start:start + 128]
        results = client.analyze_batch([{"operation": "expression_fingerprint", "expression": text, "num_points": 20}
                                        for text in batch])
        if len(results) != len(batch):
            raise RuntimeError("Incomplete seed fingerprint response.")
        fingerprints.update({text: result.get("fingerprint") for text, result in zip(batch, results)})
    seen = [set() for _ in range(4)]
    old, new = [], []
    for index, row in enumerate(records):
        identity = [normalize(row["original"]), fingerprints[row["original"]],
                    normalize(row["expr"]), fingerprints[row["expr"]]]
        if any(value and value in known for value, known in zip(identity, seen)):
            continue
        for value, known in zip(identity, seen):
            if value:
                known.add(value)
        (old if index < len(retained) else new).append(row)
    return old, new


def farthest_first(vectors, retained_count, count):
    """Greedily maximize cosine distance to retained and already-added seeds."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not 0 < retained_count <= len(vectors) or not np.isfinite(vectors).all():
        raise ValueError("Expected finite embedding rows with a nonempty retained bank.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("Zero-length seed embedding.")
    vectors = vectors / norms
    candidates = vectors[retained_count:]
    maximum = (candidates @ vectors[:retained_count].T).max(axis=1)
    available = np.ones(len(candidates), dtype=bool)
    selected = []
    for _ in range(min(count, len(candidates))):
        index = int(np.argmin(np.where(available, maximum, np.inf)))
        selected.append(index)
        available[index] = False
        maximum = np.maximum(maximum, candidates @ candidates[index])
    return selected


def expand_bank(retained, archive, *, client, tokenizer, embed, round_number, seed=42):
    """Add ceil(N/2) archive seeds to N productive, unique retained seeds."""
    if not retained:
        raise ValueError("No productive seeds remain; cannot expand an empty bank.")
    if any(len(tokenizer.encode(seed_prompt(row), add_special_tokens=False)) > 256 for row in retained):
        raise ValueError("A retained seed exceeds the 256-token setter prompt limit.")
    candidates = [row for row in order_archive(archive, seed)
                  if len(tokenizer.encode(seed_prompt(row), add_special_tokens=False)) <= 256]
    compatible_count = len(candidates)
    old, candidates = deduplicate_seeds(retained, candidates, client)
    target = math.ceil(len(old) / 2)
    chosen = []
    if candidates:
        texts = [f"Reference antiderivative: {row['original']}\n"
                 f"Associated integral: integrate {row['expr']} with respect to {row['var']}."
                 for row in old + candidates]
        vectors = embed(texts)
        if len(vectors) != len(texts):
            raise RuntimeError("Incomplete seed embeddings.")
        chosen = [candidates[i] for i in farthest_first(vectors, len(old), target)]
    additions = []
    for row in chosen:
        identity = "\n".join(normalize(row[k]) for k in ("original", "expr", "var"))
        identifier = hashlib.sha256(identity.encode()).hexdigest()[:24]
        additions.append({"seed_id": f"r{round_number}:{identifier}",
            **{key: row[key] for key in ("original", "expr", "var")},
            "question_type": "integration", "source": "verified_archive",
            "archive_id": row.get("id", row.get("archive_id", identifier)),
            "root_seed_id": row.get("root_seed_id", row.get("seed_id", "unknown"))})
    bank = old + additions
    if len(bank) < 2 or len({str(row["seed_id"]) for row in bank}) != len(bank):
        raise ValueError("Expanded bank needs at least two seeds with unique identifiers.")
    return bank, {"retained_before_dedup": len(retained), "retained_seeds": len(old),
                  "archive_rows": len(archive), "prompt_length_exclusions": len(archive) - compatible_count,
                  "unique_candidates": len(candidates), "requested_additions": target,
                  "added_seeds": len(additions), "addition_shortfall": target - len(additions),
                  "total_seeds": len(bank), "candidate_order_seed": seed,
                  "selected_archive_ids": [r["archive_id"] for r in additions]}


def embed_seed_texts(texts, model_path):
    """Local Qwen3 embeddings, last-token pooling with no input truncation."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
    encoded = tokenizer(texts, padding=False, truncation=False)["input_ids"]
    lengths = [len(tokens) for tokens in encoded]
    if max(lengths) > 32768:
        raise ValueError("Seed embedding input exceeds the model's 32,768-token context.")
    model = AutoModel.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16,
                                      attn_implementation="flash_attention_2").to("cuda:0").eval()
    vectors = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    remaining = sorted(range(len(texts)), key=lambda i: lengths[i])
    try:
        while remaining:
            size = min(64, len(remaining))
            while size > 1 and size * max(lengths[i] for i in remaining[:size]) > 16384:
                size //= 2
            indices = remaining[:size]
            batch = tokenizer.pad([{"input_ids": encoded[i], "attention_mask": [1] * lengths[i]} for i in indices],
                                  padding=True, return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                output = model(**batch, use_cache=False)
                vectors[indices] = torch.nn.functional.normalize(output.last_hidden_state[:, -1].float(), dim=1).cpu().numpy()
            del batch, output
            remaining = remaining[size:]
    finally:
        del model
        torch.cuda.empty_cache()
    return vectors
