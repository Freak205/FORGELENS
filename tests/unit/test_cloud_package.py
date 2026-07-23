import ast
import json
from pathlib import Path


def test_kaggle_script_parses_and_metadata_is_private_gpu() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "cloud" / "kaggle" / "train_vlm.py"
    ast.parse(script.read_text(encoding="utf-8"))
    metadata = json.loads(
        (root / "cloud" / "kaggle" / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert metadata["dataset_sources"] == []
    assert (root / "cloud" / "kaggle" / "requirements-vlm.txt").is_file()
