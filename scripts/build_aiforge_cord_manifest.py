"""Build a leakage-safe paired CORD/AIForge v2 experiment manifest."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = PROJECT_ROOT.parent
AIFORGE_REVISION = "9fe6f52f073c01b42966d0fd0dda87db7c9725f9"
CORD_REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
AIFORGE_ROOT = STORAGE_ROOT / "data" / "aiforge-doc-v2" / AIFORGE_REVISION
CORD_ROOT = STORAGE_ROOT / "data" / "cord-v2-extracted" / CORD_REVISION
OUTPUT = PROJECT_ROOT / "configs" / "data" / "aiforge_v2_cord_paired.json"
SEED = 20260723


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_splits(
    records: list[dict[str, Any]],
    seed: int = SEED,
    validation_fraction: float = 0.2,
) -> dict[str, str]:
    """Preserve the official test set and split training source groups only."""
    testing = {
        str(record["image_id"])
        for record in records
        if str(record["split"]).lower() == "testing"
    }
    training = sorted(
        {
            str(record["image_id"])
            for record in records
            if str(record["split"]).lower() == "training"
        }
    )
    if testing & set(training):
        raise ValueError("AIForge source group appears in training and testing")
    random.Random(seed).shuffle(training)
    validation_count = max(1, round(len(training) * validation_fraction))
    validation = set(training[:validation_count])
    return {
        image_id: (
            "test"
            if image_id in testing
            else "validation"
            if image_id in validation
            else "train"
        )
        for image_id in testing | set(training)
    }


def cord_source(global_index: int) -> tuple[Path, str]:
    """Map AIForge's global CORD index to the pinned official CORD extraction."""
    if 0 <= global_index < 800:
        split, local_index = "train", global_index
    elif global_index < 900:
        split, local_index = "validation", global_index - 800
    elif global_index < 1000:
        split, local_index = "test", global_index - 900
    else:
        raise ValueError(f"CORD index outside expected range: {global_index}")
    relative = Path(split) / "images" / f"{local_index:06d}.png"
    return CORD_ROOT / relative, str(relative.as_posix())


def storage_relative(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(STORAGE_ROOT.resolve()):
        raise ValueError("manifest path escaped F:\\HYPERVERGE")
    return resolved.relative_to(STORAGE_ROOT.resolve()).as_posix()


def main() -> None:
    metadata_path = AIFORGE_ROOT / "metadata.jsonl"
    records = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [record for record in records if record["source_dataset"] == "cord"]
    if len(records) != 983:
        raise ValueError(f"expected 983 CORD forgeries, found {len(records)}")
    if len({record["image_id"] for record in records}) != len(records):
        raise ValueError("CORD source IDs are not unique")
    split_by_source = assign_splits(records)
    items: list[dict[str, Any]] = []
    group_splits: dict[str, str] = {}
    for record in records:
        source_id = str(record["image_id"])
        global_index = int(source_id.removeprefix("cord_"))
        split = split_by_source[source_id]
        group = f"cord:{global_index:05d}"
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"source leakage detected for {group}")
        group_splits[group] = split
        authentic_path, _ = cord_source(global_index)
        forged_path = AIFORGE_ROOT / str(record["image"])
        mask_path = AIFORGE_ROOT / str(record["mask"])
        for required in (authentic_path, forged_path, mask_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        base_metadata = {
            "source_dataset": "cord",
            "source_image_id": source_id,
            "spec_id": record["spec_id"],
            "doc_type": record["doc_type"],
            "language": record["language"],
            "field_name": record["field_name"],
        }
        items.extend(
            [
                {
                    "sample_id": f"aiforge-v2-{record['spec_id']}-authentic",
                    "source_group": group,
                    "split": split,
                    "image_path": storage_relative(authentic_path),
                    "mask_path": None,
                    "image_sha256": sha256(authentic_path),
                    "label": 0,
                    "tamper_type": "none",
                    **base_metadata,
                },
                {
                    "sample_id": f"aiforge-v2-{record['spec_id']}-forged",
                    "source_group": group,
                    "split": split,
                    "image_path": storage_relative(forged_path),
                    "mask_path": storage_relative(mask_path),
                    "image_sha256": sha256(forged_path),
                    "mask_sha256": sha256(mask_path),
                    "label": 1,
                    "tamper_type": (
                        "numeric_edit"
                        if any(
                            character.isdigit() for character in record["forged_value"]
                        )
                        else "text_edit"
                    ),
                    "affected_fields": [record["field_name"]],
                    "assigned_tool": record["assigned_tool"],
                    **base_metadata,
                },
            ]
        )
    counts = {
        split: {
            "authentic": sum(
                item["split"] == split and item["label"] == 0 for item in items
            ),
            "forged": sum(
                item["split"] == split and item["label"] == 1 for item in items
            ),
        }
        for split in ("train", "validation", "test")
    }
    output = {
        "schema_version": 1,
        "dataset": "Scam-AI/AIForge-Doc-v2 + naver-clova-ix/cord-v2",
        "aiforge_revision": AIFORGE_REVISION,
        "cord_revision": CORD_REVISION,
        "licence_policy": (
            "non-commercial research; CORD-only subset; preserve attribution"
        ),
        "split_policy": (
            "preserve AIForge official test; derive validation only from "
            "official training; keep authentic/forged source pairs together"
        ),
        "seed": SEED,
        "counts": counts,
        "items": items,
    }
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
