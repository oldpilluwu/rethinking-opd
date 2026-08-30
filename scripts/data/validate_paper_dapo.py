#!/usr/bin/env python3
"""Validate the processed DAPO-Math-17K artifact used by the OPD paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


EXPECTED_SHA256 = "500bd8c45eca355b98f9ba6f3213194a72bd42c73c5e9569c6fbbb1b51bd0b39"
EXPECTED_ROWS = 17_917
EXPECTED_COLUMNS = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
EXPECTED_OVERLONG_ROW_POSITIONS = (2650, 5184, 6624, 6958, 8956, 9218, 10577, 10864)
PAPER_INSTRUCTION = r" Please reason step by step, and put your final answer within \boxed{{}}."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_row(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_rows(path: Path) -> list[dict[str, Any]]:
    actual_hash = sha256(path)
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(
            f"paper DAPO parquet hash mismatch: expected {EXPECTED_SHA256}, got {actual_hash}"
        )

    table = pq.read_table(path)
    if table.column_names != EXPECTED_COLUMNS:
        raise ValueError(f"unexpected columns: {table.column_names}")
    if table.num_rows != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, got {table.num_rows}")

    rows = table.to_pylist()
    if len({canonical_row(row) for row in rows}) != EXPECTED_ROWS:
        raise ValueError("paper DAPO dataset contains duplicate rows")

    for position, row in enumerate(rows):
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or len(prompt) != 1:
            raise ValueError(f"row {position}: expected one-message prompt")
        message = prompt[0]
        if message.get("role") != "user" or not isinstance(message.get("content"), str):
            raise ValueError(f"row {position}: invalid user prompt")
        if not message["content"].endswith(PAPER_INSTRUCTION):
            raise ValueError(f"row {position}: paper prompt instruction changed")
        reward_model = row.get("reward_model")
        if not isinstance(reward_model, dict) or not isinstance(reward_model.get("ground_truth"), str):
            raise TypeError(f"row {position}: ground truth must be a string")

    return rows


def rendered_prompt_length(tokenizer: Any, prompt: list[dict[str, str]]) -> int:
    token_ids = tokenizer.apply_chat_template(
        prompt,
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


def validate_prompt_subset(
    rows: list[dict[str, Any]], tokenizer_path: str, max_prompt_length: int
) -> tuple[int, int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    overlong: list[int] = []
    longest_kept = 0
    for position, row in enumerate(rows):
        length = rendered_prompt_length(tokenizer, row["prompt"])
        if length > max_prompt_length:
            overlong.append(position)
        else:
            longest_kept = max(longest_kept, length)

    if tuple(overlong) != EXPECTED_OVERLONG_ROW_POSITIONS:
        raise ValueError(
            "tokenizer produced an unexpected paper DAPO subset: "
            f"expected overlong row positions {EXPECTED_OVERLONG_ROW_POSITIONS}, "
            f"got {tuple(overlong)}"
        )
    return len(rows) - len(overlong), longest_kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = validate_rows(args.input)
    retained, longest = validate_prompt_subset(rows, args.tokenizer, args.max_prompt_length)
    print(f"validated {len(rows)} unique processed DAPO-Math-17K rows")
    print(
        f"runtime prompt filter retains {retained} rows and excludes "
        f"{len(rows) - retained} above {args.max_prompt_length} tokens"
    )
    print(f"longest retained rendered prompt: {longest}/{args.max_prompt_length} tokens")


if __name__ == "__main__":
    main()
