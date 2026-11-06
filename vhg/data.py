from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable

from .prompts import (
    make_math_setter_messages,
    math_solver_messages,
    question_format_prompt,
    setter_messages,
    solver_messages,
)
from .service import CorrectnessClient
from .utils import load_jsonl, save_json, save_jsonl
from .verify import (
    extract_last_boxed,
    normalize_expression_text_for_exact_dedup,
    normalize_boxed_text,
    normalize_problem_identity,
    parse_boxed_answer,
    parse_math_setter_response,
    parse_symbolic_generation_question,
)


def write_parquet_splits(
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, int]:
    from datasets import Dataset

    if len(rows) < 2:
        raise ValueError(f"Need at least two rows for a train/test split; got {len(rows)}.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = Dataset.from_list(rows).train_test_split(test_size=test_size, seed=seed)
    split["train"].to_parquet(output_dir / "train.parquet")
    split["test"].to_parquet(output_dir / "test.parquet")
    return {"train_rows": len(split["train"]), "test_rows": len(split["test"])}


def build_setter_rows(
    task: str,
    seed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for idx, row in enumerate(seed_rows):
        if task == "integration":
            original = str(row.get("original", "")).strip()
            var = str(row.get("var") or row.get("variable") or "").strip()
            question_type = str(row.get("question_type") or "integration")
            if not original or not var:
                continue
            prompt = setter_messages(original)
            extra_info = {
                "question": question_format_prompt(question_type, original, var),
                "question_type": question_type,
                "sample_num": 10,
                "temperature": 1.0,
                "max_tokens": 4096,
                "objective": "hard",
            }
            data_source = "hard_verifier_setter"
        elif task == "math":
            problem = str(row.get("seed_problem") or row.get("problem") or "").strip()
            solution = str(row.get("seed_solution") or row.get("solution") or "").strip()
            if not problem or not solution:
                continue
            prompt = make_math_setter_messages(problem, solution, objective="hard")
            extra_info = {
                "seed_problem": problem,
                "seed_solution": solution,
                "metadata": row.get("metadata") or None,
                "sample_num": 4,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": None,
                "max_tokens": 2048,
                "objective": "hard",
            }
            data_source = "soft_verifier_setter"
        else:
            raise ValueError(f"Unknown task: {task}")
        item = {
            "data_source": data_source,
            "prompt": prompt,
            "reward_model": {"style": "rule", "ground_truth": None},
            "extra_info": {"index": idx, **extra_info},
        }
        if task == "math":
            item = {
                "id": row.get("id", idx),
                "seed_problem": problem,
                "seed_solution": solution,
                "metadata": row.get("metadata") or None,
                **item,
            }
        output.append(item)
    return output


def write_setter_dataset(
    task: str,
    seed_file: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    from datasets import Dataset

    seed_rows = load_jsonl(seed_file)
    rows = build_setter_rows(task, seed_rows)
    split = Dataset.from_list(rows).shuffle(seed=42).train_test_split(test_size=0.2, seed=42)
    output_dir = Path(output_root) / "setter"
    output_dir.mkdir(parents=True, exist_ok=True)
    split_stats = {}
    sampling_seeds = []
    for split_name in ("train", "test"):
        split_rows = split[split_name].to_list()
        sampling_seeds.extend(seed_rows[row["extra_info"]["index"]] for row in split_rows)
        for index, row in enumerate(split_rows):
            row["extra_info"] = {
                **row["extra_info"],
                "split": split_name,
                "index": index,
            }
        Dataset.from_list(split_rows).to_parquet(output_dir / f"{split_name}.parquet")
        split_stats[f"{split_name}_rows"] = len(split_rows)
    save_jsonl(output_dir / "sampling_seeds.jsonl", sampling_seeds)
    summary = {
        "stage": "setter_rl_data",
        "task": task,
        "seed_file": str(seed_file),
        "input_rows": len(seed_rows),
        "prepared_rows": len(rows),
        **split_stats,
    }
    save_json(Path(output_root) / "setter_summary.json", summary)
    return summary


def write_sft_dataset(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    usable = [
        {"prompt": str(row.get("prompt", "")), "response": str(row.get("response", ""))}
        for row in records
        if str(row.get("prompt", "")).strip() and str(row.get("response", "")).strip()
    ]
    split_stats = write_parquet_splits(usable, output_dir, seed=seed)
    summary = {
        "stage": "cold_sft_data",
        "input_rows": len(records),
        "prepared_rows": len(usable),
        **split_stats,
    }
    save_json(Path(output_dir) / "summary.json", summary)
    return summary


def _integration_solver_record(row: dict[str, Any]) -> dict[str, Any] | None:
    expr = str(row.get("expr") or row.get("integrand") or "").strip()
    var = str(row.get("var") or row.get("variable") or "").strip()
    question_type = str(row.get("question_type") or "integration")
    if not expr or not var:
        generated = str(row.get("response") or row.get("question")
                        or (row.get("extra_info") or {}).get("question") or "")
        parsed = parse_symbolic_generation_question(
            generated,
            require_original=False,
            require_boxed=False,
            question_type=question_type,
        )
        if parsed.get("parseable"):
            expr = str(parsed["expr"])
            var = str(parsed["var"])
    if not expr or not var:
        return None
    tagged_question = question_format_prompt(question_type, expr, var)
    return {
        "expr": expr,
        "var": var,
        "question": tagged_question,
        "data_source": "integration_solver_rl",
        "prompt": solver_messages(
            question_format_prompt(question_type, expr, var, latex=True)
        ),
        "reward_model": {"style": "rule", "ground_truth": None},
        "extra_info": {
            "question": tagged_question,
            "question_type": question_type,
            "sample_num": 10,
            "temperature": 1.0,
            "max_tokens": 4096,
        },
        "_expression": expr,
    }


def _math_solver_record(
    row: dict[str, Any],
    *,
    source_component: str = "generated",
) -> dict[str, Any] | None:
    extra_info = row.get("extra_info") or {}
    if not isinstance(extra_info, dict):
        extra_info = {}
    parsed = parse_math_setter_response(str(row.get("response") or "")) or {}
    problem = str(
        row.get("derived_problem")
        or row.get("problem")
        or row.get("seed_problem")
        or extra_info.get("problem")
        or extra_info.get("seed_problem")
        or parsed.get("derived_problem")
        or ""
    ).strip()
    solution = str(
        row.get("modified_solution")
        or row.get("reference_solution")
        or row.get("seed_solution")
        or extra_info.get("reference_solution")
        or extra_info.get("seed_solution")
        or parsed.get("modified_solution")
        or ""
    ).strip()
    reference = normalize_boxed_text(str(
        row.get("modified_reference_boxed")
        or row.get("modified_solution_boxed")
        or row.get("reference_answer")
        or extra_info.get("reference_answer")
        or extract_last_boxed(solution)
        or ""
    ))
    if not problem or not solution or not reference:
        return None
    if source_component != "generated" and not parse_boxed_answer(reference):
        return None
    data_source = (
        "math_solver_rl"
        if source_component == "generated"
        else "math_solver_seed_cold_sft"
    )
    return {
        "data_source": data_source,
        "prompt": math_solver_messages(problem),
        "reward_model": {"style": "rule", "ground_truth": reference},
        "extra_info": {
            "problem": problem,
            "reference_answer": reference,
            "reference_solution": solution,
            "source_record_id": row.get("id"),
            "verification_backend": row.get("verification_backend", "soft_verifier"),
            "source_component": source_component,
        },
        "_dedup_key": normalize_problem_identity(problem),
    }


def _load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _split_math_solver_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) < 2:
        raise ValueError(f"Need at least two math solver rows; got {len(rows)}.")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    test_rows = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * 0.2))))
    return shuffled[test_rows:], shuffled[:test_rows]


def _math_auxiliary_rows(
    seed_data_root: str | Path,
    cold_sft_data_root: str | Path,
    split: str,
) -> list[dict[str, Any]]:
    rows = []
    for row in _load_parquet_rows(Path(seed_data_root) / f"{split}.parquet"):
        converted = _math_solver_record(row, source_component="seed")
        if converted is not None:
            rows.append(converted)
    for row in _load_parquet_rows(Path(cold_sft_data_root) / f"{split}.parquet"):
        converted = _math_solver_record(row, source_component="cold_sft")
        if converted is not None:
            rows.append(converted)
    return rows


def _merge_math_solver_splits(
    generated_train: list[dict[str, Any]],
    generated_test: list[dict[str, Any]],
    auxiliary_train: list[dict[str, Any]],
    auxiliary_test: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    output = {"train": [], "test": []}
    seen = set()
    dropped = 0
    for split, rows in (
        ("train", generated_train),
        ("test", generated_test),
        ("train", auxiliary_train),
        ("test", auxiliary_test),
    ):
        for row in rows:
            key = row.pop("_dedup_key")
            if not key or key in seen:
                dropped += 1
                continue
            seen.add(key)
            output[split].append(row)
    return output["train"], output["test"], dropped


def _write_parquet_rows(rows: list[dict[str, Any]], path: Path) -> None:
    from datasets import Dataset

    Dataset.from_list(rows).to_parquet(path)


def _deduplicate_integration(converted, service_url, *, exact_dedup=True):
    analyses = []
    with CorrectnessClient(service_url) as client:
        for start in range(0, len(converted), 512):
            analyses.extend(
                client.analyze_batch(
                    [
                        {
                            "operation": "expression_fingerprint",
                            "expression": row["_expression"],
                            "num_points": 20,
                        }
                        for row in converted[start : start + 512]
                    ]
                )
            )
    if len(analyses) != len(converted):
        raise RuntimeError("Incomplete numerical fingerprint assessment.")
    deduplicated = []
    seen_exact = set()
    seen_fingerprints = set()
    for row, analysis in zip(converted, analyses):
        expression = row.pop("_expression")
        exact = normalize_expression_text_for_exact_dedup(expression)
        if exact_dedup and exact and exact in seen_exact:
            continue
        if exact:
            seen_exact.add(exact)
        fingerprint = analysis.get("fingerprint")
        if fingerprint and fingerprint in seen_fingerprints:
            continue
        if fingerprint:
            seen_fingerprints.add(fingerprint)
        deduplicated.append(row)
    return deduplicated


def write_solver_dataset(
    task: str,
    record_files: Iterable[str | Path],
    output_root: str | Path,
    *,
    service_url: str | None = None,
    math_seed_data_root: str | Path | None = None,
    math_cold_sft_data_root: str | Path | None = None,
    first_round: bool = False,
    tokenizer=None,
) -> dict[str, Any]:
    source_rows = []
    files = [Path(path) for path in record_files]
    for index, path in enumerate(files):
        rows = _load_parquet_rows(path) if path.suffix == ".parquet" else load_jsonl(path)
        if task == "integration" and first_round and index == 0:
            rows = _integration_component_order(rows)
        source_rows.extend(rows)

    convert = _integration_solver_record if task == "integration" else _math_solver_record
    converted = []
    for row in source_rows:
        item = convert(row)
        if item is None:
            continue
        if task == "integration" and first_round and row.get("data_source") == "integration_solver_rl":
            # S1 preserved the auxiliary component's original solver prompts.
            # Later rounds rebuild them with the current common prompt template.
            item["prompt"] = row["prompt"]
        converted.append(item)

    dropped_overlength = 0
    if task == "math" and tokenizer is not None:
        within_limit = [row for row in converted if len(tokenizer.encode(
            row["prompt"][0]["content"], add_special_tokens=False)) <= 4096]
        dropped_overlength = len(converted) - len(within_limit)
        converted = within_limit

    if task == "integration":
        if not service_url:
            raise ValueError("Integration solver-data construction requires the verifier service.")
        converted = _deduplicate_integration(converted, service_url)
        if len(converted) < 2:
            raise ValueError("Need at least two unique integration problems for train/test splits.")

        import pandas as pd

        output_dir = Path(output_root) / "solver"
        output_dir.mkdir(parents=True, exist_ok=True)
        shuffled = pd.DataFrame(converted).sample(frac=1.0, random_state=42).reset_index(drop=True)
        test_rows = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * 0.2))))
        shuffled.iloc[test_rows:].reset_index(drop=True).to_parquet(output_dir / "train.parquet")
        shuffled.iloc[:test_rows].reset_index(drop=True).to_parquet(output_dir / "test.parquet")
        split_stats = {"train_rows": len(shuffled) - test_rows, "test_rows": test_rows}
    else:
        generated_train, generated_test = _split_math_solver_rows(converted)
        if bool(math_seed_data_root) != bool(math_cold_sft_data_root):
            raise ValueError(
                "Math solver data requires both seed and cold-SFT data roots when either is set."
            )
        auxiliary_train = []
        auxiliary_test = []
        if math_seed_data_root and math_cold_sft_data_root:
            auxiliary_train = _math_auxiliary_rows(
                math_seed_data_root,
                math_cold_sft_data_root,
                "train",
            )
            auxiliary_test = _math_auxiliary_rows(
                math_seed_data_root,
                math_cold_sft_data_root,
                "test",
            )
        train_rows, test_rows, dropped = _merge_math_solver_splits(
            generated_train,
            generated_test,
            auxiliary_train,
            auxiliary_test,
        )
        output_dir = Path(output_root) / "solver"
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_parquet_rows(train_rows, output_dir / "train.parquet")
        _write_parquet_rows(test_rows, output_dir / "test.parquet")
        converted = train_rows + test_rows
        split_stats = {
            "generated_rows": len(generated_train) + len(generated_test),
            "auxiliary_rows": len(auxiliary_train) + len(auxiliary_test),
            "deduplicated_rows": dropped,
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
        }
    summary = {
        "stage": "solver_rl_data",
        "task": task,
        "record_files": [str(path) for path in files],
        "input_rows": len(source_rows),
        "prepared_rows": len(converted),
        "deduplicated_rows": len(source_rows) - len(converted),
        "dropped_overlength": dropped_overlength,
        **split_stats,
    }
    save_json(Path(output_root) / "solver_summary.json", summary)
    return summary


def _integration_component_order(rows):
    """Order a solver component as the original shuffle-42, split-42 export."""
    from datasets import Dataset

    if len(rows) < 2:
        return list(rows)
    split = Dataset.from_list(rows).shuffle(seed=42).train_test_split(test_size=0.2, seed=42)
    return split["train"].to_list() + split["test"].to_list()


def write_integration_auxiliary(seed_file, cold_sft_data_root, output_root, *, service_url):
    """Prepare cold-SFT first, then Tongji; freeze the merged train/test order."""
    import pandas as pd

    cold = Path(cold_sft_data_root)
    # The solver export reshuffled the concatenated cold-SFT train/test rows.
    rows = _load_parquet_rows(cold / "train.parquet") + _load_parquet_rows(cold / "test.parquet")
    rows = _integration_component_order(rows)
    cold_count = len(rows)
    rows += _integration_component_order(load_jsonl(seed_file))
    converted = [_integration_solver_record(row) for row in rows]
    if any(row is None for row in converted):
        raise ValueError("An auxiliary problem could not be converted to a solver row.")
    # The original Tongji solver prompts predate the explicit multiplication
    # instruction. S1 kept that wording; later round exports use the common one.
    for row in converted[cold_count:]:
        for message in row["prompt"]:
            message["content"] = message["content"].replace(
                ", and explicitly use \\cdot for multiplication", "")
    # The original auxiliary merge used numerical identity only. The final
    # mixture additionally removes exact duplicates with unavailable fingerprints.
    unique = _deduplicate_integration(converted, service_url, exact_dedup=False)
    if len(unique) < 2:
        raise ValueError("Need at least two unique auxiliary problems.")
    shuffled = pd.DataFrame(unique).sample(frac=1.0, random_state=42).to_dict("records")
    test_count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * 0.2))))
    ordered = shuffled[test_count:] + shuffled[:test_count]
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    save_jsonl(output / "records.jsonl", ordered)
    summary = {"input_rows": len(rows), "prepared_rows": len(ordered),
               "order": "cold-SFT train/test, seed train/test; numerical dedup; shuffle42; train80/test20"}
    save_json(output / "summary.json", summary)
    return summary
