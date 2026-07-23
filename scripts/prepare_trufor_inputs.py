"""Build a checksum-carrying TruFor evaluation index from a dataset manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from forgelens.baselines.trufor import build_trufor_input_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = PROJECT_ROOT.parent
CORD_REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
    )
    args = parser.parse_args()
    destination = (
        PROJECT_ROOT / "artifacts" / "baselines" / "trufor" / f"cord-{args.split}.json"
    )
    count = build_trufor_input_manifest(
        PROJECT_ROOT / "configs" / "data" / "cord_v2_manifest.json",
        STORAGE_ROOT / "data" / "cord-v2-extracted" / CORD_REVISION,
        destination,
        split=args.split,
    )
    print(f"Prepared {count} TruFor inputs at {destination}")


if __name__ == "__main__":
    main()
