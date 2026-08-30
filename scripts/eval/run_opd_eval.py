#!/usr/bin/env python3
"""Generate and grade paper-compatible OPD checkpoint evaluations."""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import importlib.util
import json
import multiprocessing
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.config.load_opd_config import load_config  # noqa: E402


@dataclass(frozen=True)
class EvalSample:
    example_id: int
    prompt: str
    answer: str


@dataclass(frozen=True)
class EvalSettings:
    responses_per_prompt: int
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    do_sample: bool
    enable_thinking: bool
    prompt_suffix: str
    seed: int
    gpu_memory_utilization: float
    model_dtype: str


def settings_from_config(config: dict[str, Any]) -> EvalSettings:
    validation = config["validation"]
    return EvalSettings(
        responses_per_prompt=validation["responses_per_prompt"],
        temperature=validation["temperature"],
        top_p=validation["top_p"],
        top_k=validation["top_k"],
        max_tokens=validation["max_response_length"],
        do_sample=validation["do_sample"],
        enable_thinking=config["models"]["enable_thinking"],
        prompt_suffix=validation["prompt_suffix"],
        seed=validation["seed"],
        gpu_memory_utilization=validation["gpu_memory_utilization"],
        model_dtype=validation["model_dtype"],
    )


def ensure_prompt_suffix(prompt: str, suffix: str) -> str:
    prompt = prompt.strip()
    suffix_text = suffix.strip()
    if not suffix_text:
        return prompt
    occurrences = prompt.count(suffix_text)
    if occurrences > 1:
        raise ValueError("evaluation prompt already contains the configured suffix more than once")
    if occurrences == 1:
        if not prompt.endswith(suffix_text):
            raise ValueError("evaluation prompt contains the configured suffix before its final position")
        return prompt
    return f"{prompt} {suffix_text}"


def load_samples(path: Path, prompt_suffix: str) -> list[EvalSample]:
    table = pq.read_table(path)
    required = {"prompt", "reward_model"}
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"{path}: missing evaluation columns {sorted(missing)}")

    samples: list[EvalSample] = []
    for position, row in enumerate(table.to_pylist()):
        prompt = row["prompt"]
        if not isinstance(prompt, list) or len(prompt) != 1:
            raise ValueError(f"{path} row {position}: expected one-message prompt")
        message = prompt[0]
        if message.get("role") != "user" or not isinstance(message.get("content"), str):
            raise ValueError(f"{path} row {position}: invalid user prompt")
        reward_model = row["reward_model"]
        answer = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
        if answer is None:
            raise ValueError(f"{path} row {position}: missing ground truth")
        samples.append(
            EvalSample(
                example_id=position,
                prompt=ensure_prompt_suffix(message["content"], prompt_suffix),
                answer=str(answer),
            )
        )
    return samples


def task_name(path: Path) -> str:
    return path.parent.name or path.stem


def output_filename(name: str, settings: EvalSettings) -> str:
    return (
        f"{name.lower()}_t{settings.temperature:g}_p{settings.top_p:g}_"
        f"n{settings.responses_per_prompt}-MNT{settings.max_tokens}.jsonl"
    )


def parse_model_spec(value: str) -> tuple[str, Path | str]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
    else:
        raw_path = value
        label = Path(value.rstrip("/\\")).name
    if not label or not raw_path:
        raise ValueError(f"invalid --model value: {value!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError(f"model label must use only letters, digits, '.', '_' or '-': {label!r}")
    if raw_path.startswith("hf://"):
        model_id = raw_path.removeprefix("hf://")
        if not model_id:
            raise ValueError("hf:// model ID must not be empty")
        return label, model_id
    return label, Path(raw_path)


def prepare_hf_model(model: Path | str, merge_root: Path, label: str) -> str:
    if isinstance(model, str):
        # Remote model IDs are explicit via the hf:// prefix in --model.
        return model
    if not model.exists():
        raise FileNotFoundError(
            f"local model/checkpoint does not exist: {model}. "
            "Use LABEL=hf://ORG/MODEL for a Hub model ID."
        )
    if (model / "config.json").is_file():
        return str(model.resolve())

    actor_dir = model / "actor"
    if not actor_dir.is_dir() or not list(actor_dir.glob("model_world_size_*_rank_*.pt")):
        raise ValueError(
            f"{model} is neither a Hugging Face model nor a raw verl global_step_N checkpoint"
        )

    target = merge_root / label
    if (target / "config.json").is_file():
        return str(target.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    merger = ROOT / "verl" / "scripts" / "legacy_model_merger.py"
    command = [
        sys.executable,
        str(merger),
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor_dir.resolve()),
        "--target_dir",
        str(target.resolve()),
    ]
    print(f"merging raw checkpoint {model} -> {target}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if not (target / "config.json").is_file():
        raise RuntimeError(f"checkpoint merger did not produce {target / 'config.json'}")
    return str(target.resolve())


def _worker_generate(payload: dict[str, Any]) -> list[dict[str, Any]]:
    gpu = payload["gpu"]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # FlashInfer's sampler is JIT-compiled with nvcc, which the torch/vLLM wheels
    # do not ship, so engine startup dies in the profile run without a CUDA
    # toolkit. opd_lightning_a100.sh disables it for training; evaluation has to
    # match, both to start at all and to sample through the same code path.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    import torch
    from vllm import LLM, SamplingParams

    settings = EvalSettings(**payload["settings"])
    llm = None
    try:
        llm = LLM(
            model=payload["model"],
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=settings.gpu_memory_utilization,
            dtype=settings.model_dtype,
            seed=settings.seed + payload["worker_index"],
        )
        tokenizer = llm.get_tokenizer()
        stop_token_ids: list[int] = []
        for token in ("<|im_end|>", "<|endoftext|>"):
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if encoded:
                stop_token_ids.append(encoded[0])

        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": sample["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=settings.enable_thinking,
            )
            for sample in payload["samples"]
        ]
        sampling = SamplingParams(
            n=1,
            temperature=settings.temperature if settings.do_sample else 0.0,
            top_p=settings.top_p,
            top_k=settings.top_k,
            max_tokens=settings.max_tokens,
            stop_token_ids=stop_token_ids or None,
        )

        records: list[dict[str, Any]] = []
        for rollout_id in payload["rollout_ids"]:
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            if len(outputs) != len(payload["samples"]):
                raise RuntimeError("vLLM returned an unexpected number of evaluation outputs")
            for sample, output in zip(payload["samples"], outputs):
                records.append(
                    {
                        "example_id": sample["example_id"],
                        "prompt": sample["prompt"],
                        "answer": sample["answer"],
                        "rollout_id": rollout_id,
                        "response": output.outputs[0].text,
                    }
                )
        return records
    finally:
        if llm is not None:
            del llm
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel

            destroy_model_parallel()
        except Exception:
            pass
        try:
            from vllm.distributed.parallel_state import destroy_distributed_environment

            destroy_distributed_environment()
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()


def generate_records(
    model: str, samples: list[EvalSample], settings: EvalSettings, gpus: list[str]
) -> list[dict[str, Any]]:
    if not gpus:
        raise ValueError("at least one GPU is required")
    sample_dicts = [sample.__dict__ for sample in samples]
    chunks = [[] for _ in gpus]
    for rollout_id in range(settings.responses_per_prompt):
        chunks[rollout_id % len(gpus)].append(rollout_id)
    payloads = [
        {
            "model": model,
            "samples": sample_dicts,
            "rollout_ids": chunk,
            "gpu": gpu,
            "worker_index": index,
            "settings": settings.__dict__,
        }
        for index, (gpu, chunk) in enumerate(zip(gpus, chunks))
        if chunk
    ]

    if len(payloads) == 1:
        records = _worker_generate(payloads[0])
    else:
        context = multiprocessing.get_context("spawn")
        records = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(payloads), mp_context=context
        ) as executor:
            futures = [executor.submit(_worker_generate, payload) for payload in payloads]
            for future in concurrent.futures.as_completed(futures):
                records.extend(future.result())
    return sorted(records, key=lambda item: (item["example_id"], item["rollout_id"]))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_generation_count(
    records: list[dict[str, Any]], sample_count: int, responses_per_prompt: int
) -> None:
    expected = sample_count * responses_per_prompt
    if len(records) != expected:
        raise ValueError(f"expected {expected} generations, found {len(records)}")
    pairs = {(item["example_id"], item["rollout_id"]) for item in records}
    if len(pairs) != expected:
        raise ValueError("generation file contains duplicate or missing example/rollout pairs")


def load_rule_grader() -> Callable[[str, str], bool]:
    path = ROOT / "scripts" / "val" / "eval" / "utils.py"
    spec = importlib.util.spec_from_file_location("opd_author_eval_utils", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load author evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.grade_answer_verl


def grade_records(
    records: list[dict[str, Any]], responses_per_prompt: int, grader: Callable[[str, str], bool]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    graded: list[dict[str, Any]] = []
    for record in records:
        result = dict(record)
        result["correct"] = bool(grader(str(record["response"]), str(record["answer"])))
        result["format_error"] = "\\boxed" not in str(record["response"])
        graded.append(result)
        grouped[int(record["example_id"])].append(result)

    if not grouped:
        raise ValueError("cannot grade an empty generation file")
    for example_id, items in grouped.items():
        if len(items) != responses_per_prompt:
            raise ValueError(
                f"example {example_id} has {len(items)} responses; expected {responses_per_prompt}"
            )

    per_problem = []
    for example_id in sorted(grouped):
        items = grouped[example_id]
        correct = sum(item["correct"] for item in items)
        per_problem.append(
            {
                "example_id": example_id,
                "correct": correct,
                "total": responses_per_prompt,
                "accuracy": correct / responses_per_prompt,
            }
        )
    total_correct = sum(item["correct"] for item in graded)
    summary = {
        "problems": len(per_problem),
        "responses_per_prompt": responses_per_prompt,
        "total_responses": len(graded),
        "correct_responses": total_correct,
        "avg_at_n": total_correct / len(graded),
        "pass_at_n": sum(item["correct"] > 0 for item in per_problem) / len(per_problem),
        "all_at_n": sum(item["correct"] == responses_per_prompt for item in per_problem)
        / len(per_problem),
        "solve_none": sum(item["correct"] == 0 for item in per_problem),
        "format_errors": sum(item["format_error"] for item in graded),
        "per_problem": per_problem,
    }
    return graded, summary


def evaluate_model(
    *,
    label: str,
    model: str,
    config: dict[str, Any],
    output_root: Path,
    gpus: list[str],
    overwrite: bool,
    generate_only: bool,
    grade_only: bool,
) -> dict[str, Any]:
    settings = settings_from_config(config)
    model_dir = output_root / label
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "evaluation_manifest.json"
    manifest = {
        "schema_version": 1,
        "label": label,
        "model": model,
        "experiment_config": {
            key: value for key, value in config.items() if not key.startswith("_")
        },
        "settings": settings.__dict__,
    }
    if manifest_path.exists() and not overwrite:
        with manifest_path.open(encoding="utf-8") as stream:
            previous_manifest = json.load(stream)
        if previous_manifest != manifest:
            raise ValueError(
                f"evaluation settings differ from existing {manifest_path}; "
                "use a new label/output root or pass --overwrite"
            )
    else:
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
    task_summaries: dict[str, Any] = {}
    grader = None if generate_only else load_rule_grader()

    for raw_task_path in config["data"]["validation_files"]:
        task_path = Path(raw_task_path)
        samples = load_samples(task_path, settings.prompt_suffix)
        name = task_name(task_path)
        generations_path = model_dir / output_filename(name, settings)

        if not grade_only:
            if generations_path.exists() and not overwrite:
                records = read_jsonl(generations_path)
                validate_generation_count(records, len(samples), settings.responses_per_prompt)
                print(f"reuse complete generations: {generations_path}")
            else:
                print(f"generate {label} on {name}: {len(samples)} x {settings.responses_per_prompt}")
                records = generate_records(model, samples, settings, gpus)
                validate_generation_count(records, len(samples), settings.responses_per_prompt)
                write_jsonl(generations_path, records)
                print(f"wrote {generations_path}")
        else:
            if not generations_path.is_file():
                raise FileNotFoundError(f"--grade-only requires {generations_path}")
            records = read_jsonl(generations_path)
            validate_generation_count(records, len(samples), settings.responses_per_prompt)

        if not generate_only:
            assert grader is not None
            graded, summary = grade_records(records, settings.responses_per_prompt, grader)
            write_jsonl(generations_path.with_suffix(".graded.jsonl"), graded)
            task_summaries[name] = summary
            print(f"{label} {name} avg@{settings.responses_per_prompt}: {summary['avg_at_n']:.4f}")

    result: dict[str, Any] = {
        "label": label,
        "model": model,
        "config": config["_config_path"],
        "sampling": settings.__dict__,
        "tasks": task_summaries,
    }
    if task_summaries:
        result["macro_avg_at_n"] = sum(item["avg_at_n"] for item in task_summaries.values()) / len(
            task_summaries
        )
        total_correct = sum(item["correct_responses"] for item in task_summaries.values())
        total_responses = sum(item["total_responses"] for item in task_summaries.values())
        result["pooled_avg_at_n"] = total_correct / total_responses
        with (model_dir / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
        print(
            f"{label} macro avg@{settings.responses_per_prompt}: "
            f"{result['macro_avg_at_n']:.4f}"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        help="LABEL=PATH, PATH, or LABEL=hf://ORG/MODEL; repeat as needed",
    )
    parser.add_argument("--gpus", default="0", help="comma-separated physical GPU IDs")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate-only", action="store_true")
    mode.add_argument("--grade-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config["validation"]["grader"] != "verl_rule":
        raise SystemExit(f"unsupported evaluator: {config['validation']['grader']}")
    model_specs = args.model or [f"baseline={config['models']['student']}"]
    parsed_models = [parse_model_spec(value) for value in model_specs]
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus and not args.grade_only:
        raise SystemExit("--gpus must contain at least one GPU ID")
    output_root = args.output_root or (
        Path(config["experiment"]["validation_dir"]) / config["experiment"]["name"]
    )
    merge_root = output_root / "merged_models"

    for label, raw_model in parsed_models:
        model = str(raw_model) if args.grade_only else prepare_hf_model(raw_model, merge_root, label)
        evaluate_model(
            label=label,
            model=model,
            config=config,
            output_root=output_root,
            gpus=gpus,
            overwrite=args.overwrite,
            generate_only=args.generate_only,
            grade_only=args.grade_only,
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
