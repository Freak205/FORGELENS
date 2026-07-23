"""Prepare a private, attribution-preserving Kaggle VLM evidence bundle."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = PROJECT_ROOT.parent
MANIFEST = PROJECT_ROOT / "configs" / "data" / "aiforge_v2_cord_paired.json"
OUTPUT = PROJECT_ROOT / "artifacts" / "kaggle" / "forgelens-vlm-evidence"


def link_or_copy(source: Path, target: Path) -> None:
    """Use a space-efficient hard link when possible, then fall back to copying."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def conversation(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Create a concise label-first, evidence-grounded supervision exchange."""
    prompt = (
        "Inspect this document for image tampering. Return exactly one verdict "
        "(AUTHENTIC or FORGED), then one short visual-evidence sentence. Do not "
        "infer identity or personal attributes."
    )
    if int(item["label"]) == 1:
        field = str(item.get("field_name", "document field"))
        answer = (
            f"VERDICT: FORGED\nEVIDENCE: The {field} region is the annotated "
            "localized alteration; this benchmark label is not an identity claim."
        )
    else:
        answer = (
            "VERDICT: AUTHENTIC\nEVIDENCE: No annotated altered region is present "
            "in this paired benchmark source image."
        )
    return [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]


def main() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for item in payload["items"]:
        source = STORAGE_ROOT / str(item["image_path"])
        suffix = source.suffix.lower()
        relative = Path("images") / f"{item['sample_id']}{suffix}"
        link_or_copy(source, OUTPUT / relative)
        rows.append(
            {
                "sample_id": item["sample_id"],
                "source_group": item["source_group"],
                "split": item["split"],
                "label": int(item["label"]),
                "image": relative.as_posix(),
                "messages": conversation(item),
            }
        )
    with (OUTPUT / "vlm_sft.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "title": "ForgeLens AIForge CORD VLM Evidence",
        "id": "ivsanirudh/forgelens-vlm-evidence",
        "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
    }
    (OUTPUT / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    attribution = (
        "# Attribution and use\n\n"
        "Private non-commercial research bundle derived from "
        "Scam-AI/AIForge-Doc-v2 (CC BY-NC-SA 4.0) and "
        "naver-clova-ix/cord-v2 (CC BY 4.0), pinned by the ForgeLens manifest. "
        "Do not make this gated derivative public.\n"
    )
    (OUTPUT / "README.md").write_text(attribution, encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
