#!/usr/bin/env python3
"""Create rsync/scp-ready OPD diagnostics or benchmark archives."""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.config.load_opd_config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--kind", choices=("diagnostics", "benchmarks"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("opd_artifacts"))
    return parser.parse_args()


def files_below(root: Path, *, exclude_parts: set[str] | None = None) -> Iterable[Path]:
    if not root.exists():
        return
    excluded = exclude_parts or set()
    for path in sorted(root.rglob("*")):
        if path.is_file() and not any(part in excluded for part in path.relative_to(root).parts):
            yield path


def diagnostics_files(config: dict) -> Iterable[tuple[Path, Path]]:
    experiment = config["experiment"]["name"]
    diagnostics_root = Path(config["experiment"]["diagnostics_dir"]) / experiment

    # Keep Ray's text logs, but exclude object-store/session machinery.
    for path in files_below(diagnostics_root):
        relative = path.relative_to(diagnostics_root)
        if "ray" in relative.parts and "logs" not in relative.parts:
            continue
        yield path, Path("diagnostics") / relative

    log_root = Path(config["experiment"]["log_dir"])
    for path in sorted(log_root.glob(f"{experiment}_*")):
        if path.is_file():
            yield path, Path("training_logs") / path.name

    checkpoint_root = Path(config["experiment"]["output_dir"]) / experiment
    for name in ("experiment.toml", "experiment.json", "resolved_hydra.yaml", "latest_checkpointed_iteration.txt"):
        path = checkpoint_root / name
        if path.is_file():
            yield path, Path("run_metadata") / name


def benchmark_files(config: dict) -> Iterable[tuple[Path, Path]]:
    experiment = config["experiment"]["name"]
    validation_root = Path(config["experiment"]["validation_dir"]) / experiment
    for path in files_below(validation_root, exclude_parts={"merged_models"}):
        yield path, Path("benchmarks") / path.relative_to(validation_root)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    experiment = config["experiment"]["name"]
    entries = list(
        diagnostics_files(config) if args.kind == "diagnostics" else benchmark_files(config)
    )
    if not entries:
        raise SystemExit(f"no {args.kind} artifacts found for {experiment}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / f"{experiment}-{args.kind}.tar.gz"
    manifest = {
        "schema_version": 1,
        "experiment": experiment,
        "kind": args.kind,
        "files": [str(arcname) for _, arcname in entries],
        "checkpoints_included": False,
    }

    with tarfile.open(archive, "w:gz") as tar:
        for source, arcname in entries:
            tar.add(source, arcname=str(arcname), recursive=False)
        payload = (json.dumps(manifest, indent=2) + "\n").encode()
        info = tarfile.TarInfo("artifact-manifest.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    print(f"wrote {archive} ({archive.stat().st_size:,} bytes, {len(entries)} files)")
    print("checkpoints included: no")


if __name__ == "__main__":
    main()
