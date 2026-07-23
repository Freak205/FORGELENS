"""Build a deterministic traditional-tampering benchmark from CORD v2."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = PROJECT_ROOT.parent
SOURCE_MANIFEST = PROJECT_ROOT / "configs" / "data" / "cord_v2_manifest.json"
CORD_REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
SOURCE_ROOT = STORAGE_ROOT / "data" / "cord-v2-extracted" / CORD_REVISION
OUTPUT_ROOT = STORAGE_ROOT / "data" / "cord-copy-move-v1"
OUTPUT_MANIFEST = PROJECT_ROOT / "configs" / "data" / "cord_copy_move_v1.json"
GLOBAL_SEED = 20260723


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rng_for(sample_id: str) -> random.Random:
    seed_material = f"{GLOBAL_SEED}:{sample_id}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(seed_material).digest()[:8]))


def _different_position(
    rng: random.Random,
    limit_x: int,
    limit_y: int,
    source_x: int,
    source_y: int,
    patch_width: int,
    patch_height: int,
) -> tuple[int, int]:
    for _ in range(64):
        x = rng.randint(0, limit_x)
        y = rng.randint(0, limit_y)
        separated = (
            x + patch_width <= source_x
            or source_x + patch_width <= x
            or y + patch_height <= source_y
            or source_y + patch_height <= y
        )
        if separated:
            return x, y
    return limit_x - source_x, limit_y - source_y


def forge_copy_move(
    source: Path, image_out: Path, mask_out: Path, seed_id: str
) -> None:
    """Copy one non-overlapping local patch and write its exact target mask."""
    rng = _rng_for(seed_id)
    with Image.open(source) as image_file:
        image = image_file.convert("RGB")
    width, height = image.size
    patch_width = max(
        8, min(width // 4, rng.randint(max(8, width // 10), max(8, width // 5)))
    )
    patch_height = max(
        8,
        min(height // 4, rng.randint(max(8, height // 10), max(8, height // 5))),
    )
    limit_x = width - patch_width
    limit_y = height - patch_height
    source_x = rng.randint(0, limit_x)
    source_y = rng.randint(0, limit_y)
    target_x, target_y = _different_position(
        rng,
        limit_x,
        limit_y,
        source_x,
        source_y,
        patch_width,
        patch_height,
    )
    patch = image.crop(
        (
            source_x,
            source_y,
            source_x + patch_width,
            source_y + patch_height,
        )
    )
    image.paste(patch, (target_x, target_y))
    mask = Image.new("L", image.size, 0)
    mask.paste(
        255,
        (
            target_x,
            target_y,
            target_x + patch_width,
            target_y + patch_height,
        ),
    )
    image_out.parent.mkdir(parents=True, exist_ok=True)
    mask_out.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_out, format="PNG")
    mask.save(mask_out, format="PNG")


def _authentic_record(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": f"{item['sample_id']}-authentic",
        "source_group": item["source_group"],
        "split": item["split"],
        "image_path": str(
            (
                Path("data")
                / "cord-v2-extracted"
                / CORD_REVISION
                / item["relative_path"]
            ).as_posix()
        ),
        "mask_path": None,
        "image_sha256": item["sha256"],
        "label": 0,
        "tamper_type": "none",
    }


def main() -> None:
    source_payload = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    counts = {
        split: {"authentic": 0, "forged": 0} for split in source_payload["counts"]
    }
    for item in source_payload["items"]:
        source = SOURCE_ROOT / item["relative_path"]
        forged_relative = Path(item["split"]) / "images" / f"{item['sample_id']}.png"
        mask_relative = Path(item["split"]) / "masks" / f"{item['sample_id']}.png"
        forged_path = OUTPUT_ROOT / forged_relative
        mask_path = OUTPUT_ROOT / mask_relative
        forge_copy_move(source, forged_path, mask_path, item["sample_id"])
        records.append(_authentic_record(item))
        records.append(
            {
                "sample_id": f"{item['sample_id']}-copy-move",
                "source_group": item["source_group"],
                "split": item["split"],
                "image_path": str(
                    (Path("data") / "cord-copy-move-v1" / forged_relative).as_posix()
                ),
                "mask_path": str(
                    (Path("data") / "cord-copy-move-v1" / mask_relative).as_posix()
                ),
                "image_sha256": file_sha256(forged_path),
                "mask_sha256": file_sha256(mask_path),
                "label": 1,
                "tamper_type": "copy_move",
                "generator": "forgelens-copy-move-v1",
                "seed": GLOBAL_SEED,
            }
        )
        counts[item["split"]]["authentic"] += 1
        counts[item["split"]]["forged"] += 1
    output = {
        "schema_version": 1,
        "dataset": "forgelens/cord-copy-move-v1",
        "source_dataset": source_payload["dataset"],
        "source_revision": source_payload["revision"],
        "licence": "CC BY 4.0; synthetic derivatives retain attribution",
        "split_policy": "preserve CORD official split and co-locate derivatives",
        "seed": GLOBAL_SEED,
        "counts": counts,
        "items": records,
    }
    temporary = OUTPUT_MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT_MANIFEST)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
