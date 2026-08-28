#!/usr/bin/env python3
"""Prepare and validate the official Lightning-OPD DAPO-Math-17k data.

The source parquet is the Dataset Viewer conversion of
``zhuzilin/dapo-math-17k``.  Its SHA-256 is pinned so a mutable Hub update
cannot silently change a reproduction run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


DATASET_ID = "zhuzilin/dapo-math-17k"
SOURCE_URL = (
    "https://huggingface.co/datasets/zhuzilin/dapo-math-17k/resolve/"
    "refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
SOURCE_SHA256 = "768a62a056ff0b2ffa51c3e618d069ebd08ad51ee5546c0c5ad1720fc29b3215"
EXPECTED_SOURCE_ROWS = 17_398
EXPECTED_ROWS = 17_389
EXPECTED_EXCLUDED_INDICES = (2188, 2610, 5042, 6459, 6793, 8664, 8924, 10245, 10526)
EXPECTED_COLUMNS = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
BOXED_INSTRUCTION = r"Answer: \boxed{$Answer}"

OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("data_source", pa.string(), nullable=False),
        pa.field(
            "prompt",
            pa.list_(
                pa.struct(
                    [
                        pa.field("role", pa.string(), nullable=False),
                        pa.field("content", pa.string(), nullable=False),
                    ]
                )
            ),
            nullable=False,
        ),
        pa.field("ability", pa.string(), nullable=False),
        pa.field(
            "reward_model",
            pa.struct(
                [
                    pa.field("style", pa.string(), nullable=False),
                    pa.field("ground_truth", pa.string(), nullable=False),
                ]
            ),
            nullable=False,
        ),
        pa.field(
            "extra_info",
            pa.struct([pa.field("index", pa.int64(), nullable=False)]),
            nullable=False,
        ),
    ]
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prompt(value: Any) -> list[dict[str, str]]:
    """Return a plain Python chat list without changing its text."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"expected one-message prompt, got {value!r}")

    message = value[0]
    if hasattr(message, "as_py"):
        message = message.as_py()
    if not isinstance(message, dict):
        raise TypeError(f"prompt message must be a mapping, got {type(message).__name__}")
    if message.get("role") != "user" or not isinstance(message.get("content"), str):
        raise ValueError(f"invalid user prompt: {message!r}")
    return [{"role": "user", "content": message["content"]}]


def source_rows(source_path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(source_path)
    if table.column_names != ["prompt", "label"]:
        raise ValueError(f"unexpected source columns: {table.column_names}")
    if table.num_rows != EXPECTED_SOURCE_ROWS:
        raise ValueError(f"expected {EXPECTED_SOURCE_ROWS} source rows, got {table.num_rows}")

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(table.to_pylist()):
        prompt = normalize_prompt(row["prompt"])
        label = row["label"]
        if not isinstance(label, str):
            raise TypeError(f"row {index}: label must be str, got {type(label).__name__}")
        rows.append(
            {
                "data_source": "math_dapo",
                "prompt": prompt,
                "ability": "MATH",
                "reward_model": {
                    "style": "rule-lighteval/MATH_v2",
                    "ground_truth": label,
                },
                "extra_info": {"index": index},
            }
        )
    return rows


def _canonical_row(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, got {len(rows)}")
    if len({_canonical_row(row) for row in rows}) != EXPECTED_ROWS:
        raise ValueError("dataset contains duplicate rows")

    expected_indices = [index for index in range(EXPECTED_SOURCE_ROWS) if index not in EXPECTED_EXCLUDED_INDICES]
    actual_indices = [row.get("extra_info", {}).get("index") for row in rows]
    if actual_indices != expected_indices:
        raise ValueError("dataset does not contain the expected deterministic 1,024-token subset")

    for output_index, row in enumerate(rows):
        source_index = row["extra_info"]["index"]
        if list(row) != EXPECTED_COLUMNS:
            raise ValueError(f"row {output_index}: unexpected columns/order: {list(row)}")
        if row["data_source"] != "math_dapo" or row["ability"] != "MATH":
            raise ValueError(f"source row {source_index}: invalid dataset metadata")
        reward_model = row["reward_model"]
        if reward_model.get("style") != "rule-lighteval/MATH_v2":
            raise ValueError(f"source row {source_index}: invalid reward style")
        if not isinstance(reward_model.get("ground_truth"), str):
            raise TypeError(f"source row {source_index}: ground truth is not a string")
        prompt = normalize_prompt(row["prompt"])
        if BOXED_INSTRUCTION not in prompt[0]["content"]:
            raise ValueError(f"source row {source_index}: official boxed-answer instruction is missing")


def read_output_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    if table.column_names != EXPECTED_COLUMNS:
        raise ValueError(f"unexpected output columns: {table.column_names}")
    return table.to_pylist()


def rendered_prompt_length(tokenizer: Any, prompt: Any) -> int:
    token_ids = tokenizer.apply_chat_template(
        normalize_prompt(prompt),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(token_ids, Mapping):
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "shape") and len(token_ids.shape) == 2:
        return int(token_ids.shape[-1])
    if token_ids and isinstance(token_ids[0], (list, tuple)):
        return len(token_ids[0])
    return len(token_ids)


def validate_prompt_lengths(
    rows: Iterable[dict[str, Any]], tokenizer_path: str, max_prompt_length: int
) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    longest = 0
    for output_index, row in enumerate(rows):
        length = rendered_prompt_length(tokenizer, row["prompt"])
        longest = max(longest, length)
        if length > max_prompt_length:
            raise ValueError(
                f"output row {output_index} (source row {row['extra_info']['index']}): "
                f"rendered prompt has {length} tokens, exceeding "
                f"data.max_prompt_length={max_prompt_length}"
            )
    return longest


def filter_overlong_rows(
    rows: list[dict[str, Any]], tokenizer_path: str, max_prompt_length: int
) -> tuple[list[dict[str, Any]], int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    kept: list[dict[str, Any]] = []
    excluded: list[int] = []
    longest_kept = 0
    for row in rows:
        length = rendered_prompt_length(tokenizer, row["prompt"])
        if length > max_prompt_length:
            excluded.append(row["extra_info"]["index"])
        else:
            kept.append(row)
            longest_kept = max(longest_kept, length)
    if tuple(excluded) != EXPECTED_EXCLUDED_INDICES:
        raise ValueError(
            "tokenizer produced an unexpected 1,024-token subset: "
            f"expected exclusions {EXPECTED_EXCLUDED_INDICES}, got {tuple(excluded)}"
        )
    return kept, longest_kept


def validate_output(path: Path, tokenizer_path: str | None, max_prompt_length: int) -> None:
    rows = read_output_rows(path)
    validate_rows(rows)
    print(f"validated {len(rows)} unique filtered {DATASET_ID} rows")
    print(f"excluded official source rows: {list(EXPECTED_EXCLUDED_INDICES)}")
    print(f"boxed-answer prompts: {sum(BOXED_INSTRUCTION in row['prompt'][0]['content'] for row in rows)}")
    if tokenizer_path:
        longest = validate_prompt_lengths(rows, tokenizer_path, max_prompt_length)
        print(f"longest rendered prompt: {longest}/{max_prompt_length} tokens")


def download_source(destination: Path) -> None:
    print(f"downloading {SOURCE_URL}")
    urllib.request.urlretrieve(SOURCE_URL, destination)
    actual_hash = sha256(destination)
    if actual_hash != SOURCE_SHA256:
        raise ValueError(
            "official source parquet hash changed: "
            f"expected {SOURCE_SHA256}, got {actual_hash}"
        )


def prepare(output_path: Path, tokenizer_path: str | None, max_prompt_length: int) -> None:
    if not tokenizer_path:
        raise ValueError("--tokenizer is required to build the deterministic 1,024-token subset")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lightning-dapo-") as temp_dir:
        source_path = Path(temp_dir) / "official.parquet"
        candidate_path = Path(temp_dir) / "converted.parquet"
        download_source(source_path)
        rows, longest = filter_overlong_rows(source_rows(source_path), tokenizer_path, max_prompt_length)
        print(
            f"kept {len(rows)}/{EXPECTED_SOURCE_ROWS} rows; "
            f"longest rendered prompt is {longest}/{max_prompt_length} tokens"
        )
        validate_rows(rows)
        pq.write_table(pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA), candidate_path, compression="snappy")
        validate_output(candidate_path, None, max_prompt_length)
        os.replace(candidate_path, output_path)
    print(f"wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="download, convert, and validate the official split")
    prepare_parser.add_argument("--output", type=Path, default=Path("datasets/dapo-math-17k.parquet"))
    prepare_parser.add_argument("--tokenizer", required=True, help="actor tokenizer path/name used for filtering")
    prepare_parser.add_argument("--max-prompt-length", type=int, default=1024)

    validate_parser = subparsers.add_parser("validate", help="validate an existing converted parquet")
    validate_parser.add_argument("--input", type=Path, default=Path("datasets/dapo-math-17k.parquet"))
    validate_parser.add_argument("--tokenizer", help="actor tokenizer path/name for the prompt-length preflight")
    validate_parser.add_argument("--max-prompt-length", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args.output, args.tokenizer, args.max_prompt_length)
    else:
        validate_output(args.input, args.tokenizer, args.max_prompt_length)


if __name__ == "__main__":
    main()
