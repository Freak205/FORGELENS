"""Download the open, revision-pinned CORD v2 authentic-source corpus."""

import json
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = PROJECT_ROOT.parent
REPOSITORY = "naver-clova-ix/cord-v2"
REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"


def main() -> None:
    """Download CORD v2 entirely below the ForgeLens storage boundary."""
    destination = STORAGE_ROOT / "data" / "cord-v2" / REVISION
    cache = STORAGE_ROOT / ".cache" / "huggingface"
    downloaded = Path(
        snapshot_download(
            repo_id=REPOSITORY,
            repo_type="dataset",
            revision=REVISION,
            local_dir=destination,
            cache_dir=cache,
        )
    ).resolve()
    if not downloaded.is_relative_to(STORAGE_ROOT.resolve()):
        raise RuntimeError("download escaped the F:\\HYPERVERGE storage boundary")
    parquet_files = sorted(downloaded.glob("data/*.parquet"))
    if len(parquet_files) != 6:
        raise RuntimeError(
            f"expected 6 CORD parquet shards, found {len(parquet_files)}"
        )
    manifest = {
        "dataset": REPOSITORY,
        "revision": REVISION,
        "licence": "CC BY 4.0",
        "official_homepage": "https://github.com/clovaai/cord",
        "download_root": str(downloaded),
        "parquet_shards": [
            {"name": path.name, "bytes": path.stat().st_size} for path in parquet_files
        ],
    }
    (destination.parent / "accession.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
