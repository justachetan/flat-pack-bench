#!/usr/bin/env python3
"""Download Flat-Pack Bench data artifacts from Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

CORE_REPO = "justachetan/flat-pack-bench"
MISC_REPO = "justachetan/flat-pack-bench-misc"
EVAL_REPO = "justachetan/flat-pack-bench-evals"

CORE_METADATA_PATTERNS = (
    "questions/**",
    "segmentation-masks/**",
    "furniture-annotations/part-annotations/**",
)

CORE_FULL_PATTERNS = (
    *CORE_METADATA_PATTERNS,
    "videos/**",
    "rgb-frames/**",
)

MISC_METADATA_PATTERNS = (
    "scrambled-questions/**",
    "scrambled-segmentation-masks/**",
)

MISC_ALL_PATTERNS = (
    "scrambled-questions/**",
    "scrambled-segmentation-masks/**",
    "tva-agent-traces/**",
    "tva-segmentation-masks/**",
)


def _download(
    *,
    repo_id: str,
    local_dir: Path,
    allow_patterns: Sequence[str] | None,
    label: str,
) -> None:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {label}")
    print(f"  repo: {repo_id}")
    print(f"  dest: {local_dir}")
    if allow_patterns is not None:
        print("  paths:")
        for pattern in allow_patterns:
            print(f"    - {pattern}")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=list(allow_patterns) if allow_patterns is not None else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Flat-Pack Bench data artifacts from Hugging Face.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory where benchmark data should be downloaded. Defaults to data/.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help=(
            "Download questions, masks, furniture annotations, scrambled questions, "
            "and scrambled masks."
        ),
    )
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Download metadata plus videos and RGB frames from the core dataset.",
    )
    parser.add_argument(
        "--full-eval-cache",
        action="store_true",
        help="Download the full evaluation cache into <data-dir>/eval-cache.",
    )
    parser.add_argument(
        "--addl-expts",
        action="store_true",
        help="Download all artifacts from the additional experiments dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir

    if not any(
        (
            args.metadata_only,
            args.full_data,
            args.full_eval_cache,
            args.addl_expts,
        )
    ):
        args.metadata_only = True

    if args.full_data:
        _download(
            repo_id=CORE_REPO,
            local_dir=data_dir,
            allow_patterns=CORE_FULL_PATTERNS,
            label="core benchmark metadata, videos, and RGB frames",
        )
    elif args.metadata_only:
        _download(
            repo_id=CORE_REPO,
            local_dir=data_dir,
            allow_patterns=CORE_METADATA_PATTERNS,
            label="core benchmark metadata",
        )

    if args.metadata_only or args.full_data:
        _download(
            repo_id=MISC_REPO,
            local_dir=data_dir,
            allow_patterns=MISC_METADATA_PATTERNS,
            label="scrambled question and mask metadata",
        )

    if args.full_eval_cache:
        _download(
            repo_id=EVAL_REPO,
            local_dir=data_dir / "eval-cache",
            allow_patterns=None,
            label="evaluation cache",
        )

    if args.addl_expts:
        _download(
            repo_id=MISC_REPO,
            local_dir=data_dir,
            allow_patterns=MISC_ALL_PATTERNS,
            label="additional experiment artifacts",
        )


if __name__ == "__main__":
    main()
