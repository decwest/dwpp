#!/usr/bin/env python3
"""Command-line entry point for the DWPP revision simulation studies."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.revision.studies import DEFAULT_OUTPUT_ROOT, STUDY_FUNCTIONS, run_studies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        choices=("all", *STUDY_FUNCTIONS),
        default="all",
        help="study to run (default: all)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"output directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_studies(args.study, args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
