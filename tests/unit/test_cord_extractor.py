import ast
import json
from pathlib import Path


def test_cord_extractor_and_dependency_manifest_exist() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts" / "extract_cord.mjs").read_text(encoding="utf-8")
    assert "cord-v2-extracted" in script
    assert "sha256" in script
    package = json.loads(
        (root / "tools" / "parquet" / "package.json").read_text(encoding="utf-8")
    )
    assert package["dependencies"]["hyparquet"] == "1.26.2"
    assert package["dependencies"]["hyparquet-compressors"] == "1.1.1"
    ast.parse((root / "scripts" / "download_cord.py").read_text(encoding="utf-8"))
