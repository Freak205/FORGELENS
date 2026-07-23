from scripts.build_aiforge_cord_manifest import assign_splits


def test_aiforge_split_preserves_test_and_groups() -> None:
    records = [
        {"image_id": f"cord_{index:05d}", "split": "training"} for index in range(10)
    ]
    records.extend(
        {"image_id": f"cord_{index:05d}", "split": "testing"} for index in range(10, 13)
    )
    splits = assign_splits(records, seed=7, validation_fraction=0.2)
    assert all(splits[f"cord_{index:05d}"] == "test" for index in range(10, 13))
    assert sum(split == "validation" for split in splits.values()) == 2
    assert sum(split == "train" for split in splits.values()) == 8


def test_aiforge_split_is_deterministic() -> None:
    records = [
        {"image_id": f"cord_{index:05d}", "split": "training"} for index in range(20)
    ]
    assert assign_splits(records, seed=42) == assign_splits(records, seed=42)
