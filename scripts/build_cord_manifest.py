"""Build an immutable, portable CORD v2 split manifest."""

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = PROJECT_ROOT.parent
REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
DATA_ROOT = STORAGE_ROOT / "data" / "cord-v2-extracted" / REVISION
OUTPUT = PROJECT_ROOT / "configs" / "data" / "cord_v2_manifest.json"


def main() -> None:
    """Validate official split disjointness and write a portable manifest."""
    items: list[dict[str, Any]] = []
    groups_by_split: dict[str, set[str]] = {}
    expected_counts = {"train": 800, "validation": 100, "test": 100}
    for split, expected_count in expected_counts.items():
        metadata_path = DATA_ROOT / split / "metadata.jsonl"
        rows = [
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != expected_count:
            raise RuntimeError(
                f"{split} has {len(rows)} rows, expected {expected_count}"
            )
        groups = {str(row["source_group"]) for row in rows}
        if len(groups) != len(rows):
            raise RuntimeError(f"duplicate source group within {split}")
        groups_by_split[split] = groups
        for row in rows:
            items.append(
                {
                    "sample_id": row["sample_id"],
                    "source_group": row["source_group"],
                    "split": split,
                    "relative_path": f"{split}/{row['image_path']}",
                    "sha256": row["sha256"],
                    "label": 0,
                }
            )
    split_names = list(expected_counts)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap = groups_by_split[left] & groups_by_split[right]
            if overlap:
                raise RuntimeError(f"source-group leakage between {left} and {right}")
    manifest = {
        "dataset": "naver-clova-ix/cord-v2",
        "revision": REVISION,
        "licence": "CC BY 4.0",
        "split_policy": "preserve official train/validation/test partitions",
        "counts": expected_counts,
        "items": items,
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "counts": expected_counts}, indent=2))


if __name__ == "__main__":
    main()
