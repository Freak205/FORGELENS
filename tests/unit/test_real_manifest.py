import json
from pathlib import Path


def test_cord_manifest_is_complete_and_leakage_free() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / "configs" / "data" / "cord_v2_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["revision"] == "7f0115a4b758a71d6473b8d085751692da2fef98"
    assert manifest["counts"] == {"train": 800, "validation": 100, "test": 100}
    items = manifest["items"]
    assert len(items) == 1000
    groups: dict[str, set[str]] = {}
    for item in items:
        groups.setdefault(item["split"], set()).add(item["source_group"])
        assert len(item["sha256"]) == 64
        assert item["label"] == 0
    assert not (groups["train"] & groups["validation"])
    assert not (groups["train"] & groups["test"])
    assert not (groups["validation"] & groups["test"])
