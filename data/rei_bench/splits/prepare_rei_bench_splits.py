#!/usr/bin/env python3
"""
Prepare REI-Bench splits: generate data/rei_bench/splits/rei_bench.json from a source split file.

Usage (from project root):
    python data/rei_bench/splits/prepare_rei_bench_splits.py
    python data/rei_bench/splits/prepare_rei_bench_splits.py --input path/to/source_split.json
"""

import argparse
import json
import logging
import os
import sys

# Default paths (relative to project root)
DEFAULT_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
DEFAULT_INPUT_SPLIT = os.path.join(
    DEFAULT_PROJECT_ROOT, "data", "raw", "alfred", "splits", "pre_experiment_tasks3.json"
)
DEFAULT_OUTPUT_SPLIT = os.path.join(
    DEFAULT_PROJECT_ROOT, "data", "rei_bench", "splits", "rei_bench.json"
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )


def load_split(path: str) -> dict:
    """Load split JSON file."""
    path = os.path.normpath(os.path.abspath(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "valid_seen" not in data:
        raise ValueError("Split file must contain 'valid_seen' key")
    return data


def write_split(data: dict, path: str) -> None:
    """Write split data to JSON file."""
    path = os.path.normpath(os.path.abspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logging.info("Written: %s", path)


def main() -> int:
    """Generate rei_bench.json from source split."""
    parser = argparse.ArgumentParser(
        description="Generate REI-Bench split data/rei_bench/splits/rei_bench.json from a source split file"
    )
    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_INPUT_SPLIT,
        help="Path to source split JSON (with valid_seen and REI-Bench fields)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT_SPLIT,
        help="Path to output REI-Bench split JSON",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    try:
        data = load_split(args.input)
        n = len(data.get("valid_seen", []))
        logging.info("Loaded %d valid_seen entries", n)
        write_split(data, args.output)
        return 0
    except FileNotFoundError as e:
        logging.error("%s", e)
        return 1
    except ValueError as e:
        logging.error("%s", e)
        return 1
    except json.JSONDecodeError as e:
        logging.error("JSON decode error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
