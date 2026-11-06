from __future__ import annotations

import csv
import io
import urllib.request
from pathlib import Path
from typing import Callable

from .utils import iter_jsonl, save_jsonl


MATH_500_URL = "https://openaipublic.blob.core.windows.net/simple-evals/math_500_test.csv"


def prepare_integration_benchmarks(
    antiderivbench_root: str | Path,
    output_dir: str | Path,
) -> dict[str, int]:
    source_root = Path(antiderivbench_root) / "data"
    output_dir = Path(output_dir)
    counts = {}
    for name in ("competition", "qualifier"):
        rows = [
            {
                "id": index,
                "question_type": "integration",
                "expr": row["latex"].strip(),
                "var": "x",
                "metadata": {key: value for key, value in row.items() if key != "latex"},
            }
            for index, row in enumerate(iter_jsonl(source_root / f"{name}.jsonl"))
        ]
        save_jsonl(output_dir / f"{name}.jsonl", rows)
        counts[name] = len(rows)
    return counts


def _load_math500() -> list[dict[str, str]]:
    with urllib.request.urlopen(MATH_500_URL) as response:
        text = response.read().decode("utf-8")
    return [
        {"problem": row["Question"], "answer": row["Answer"]}
        for row in csv.DictReader(io.StringIO(text))
    ]


def _hf_rows(
    dataset_name: str,
    *,
    config: str | None = None,
    split: str = "test",
    convert: Callable,
) -> list[dict[str, str]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, config, split=split)
    return [convert(row) for row in dataset]


def prepare_math_benchmarks(output_dir: str | Path) -> dict[str, int]:
    loaders = {
        "math": _load_math500,
        "gsm8k": lambda: _hf_rows(
            "openai/gsm8k",
            config="main",
            convert=lambda row: {
                "problem": str(row["question"]),
                "answer": str(row["answer"]).split("#### ")[-1],
            },
        ),
        "amc": lambda: _hf_rows(
            "zwhe99/amc23",
            convert=lambda row: {"problem": str(row["question"]), "answer": str(row["answer"])},
        ),
        "minerva": lambda: _hf_rows(
            "zwhe99/simplerl-minerva-math",
            convert=lambda row: {"problem": str(row["problem"]), "answer": str(row["answer"])},
        ),
        "olympiad": lambda: _hf_rows(
            "zwhe99/simplerl-OlympiadBench",
            convert=lambda row: {
                "problem": str(row["question"]),
                "answer": str(
                    (row.get("final_answer") or [""])[0]
                    if isinstance(row.get("final_answer"), list)
                    else row.get("final_answer", "")
                ),
            },
        ),
        "aime2024": lambda: _hf_rows(
            "HuggingFaceH4/aime_2024",
            split="train",
            convert=lambda row: {"problem": str(row["problem"]), "answer": str(row["answer"])},
        ),
        "aime2025": lambda: _hf_rows(
            "yentinglin/aime_2025",
            config="default",
            split="train",
            convert=lambda row: {"problem": str(row["problem"]), "answer": str(row["answer"])},
        ),
        "aime2026": lambda: _hf_rows(
            "MathArena/aime_2026",
            config="default",
            split="train",
            convert=lambda row: {"problem": str(row["problem"]), "answer": str(row["answer"])},
        ),
    }
    output_dir = Path(output_dir)
    counts = {}
    for name, loader in loaders.items():
        rows = loader()
        save_jsonl(output_dir / f"{name}.jsonl", rows)
        counts[name] = len(rows)
    return counts
