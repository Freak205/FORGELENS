"""Download an authorized, revision-pinned AIForge-Doc snapshot."""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = PROJECT_ROOT.parent
REPOSITORIES = {
    "v1": "Scam-AI/AIForge-Doc-v1",
    "v2": "Scam-AI/AIForge-Doc-v2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("version", choices=sorted(REPOSITORIES))
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable containing a read-only Hugging Face token",
    )
    return parser.parse_args()


def main() -> None:
    """Resolve the immutable revision, download it, and record provenance."""
    args = parse_args()
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(
            f"{args.token_env} is missing; accept dataset terms and provide a "
            "read-only token through a secure environment secret"
        )
    repository = REPOSITORIES[args.version]
    api = HfApi(token=token)
    info = api.dataset_info(repository, files_metadata=True)
    revision = info.sha
    if revision is None:
        raise RuntimeError("Hugging Face did not return an immutable revision SHA")
    destination = STORAGE_ROOT / "data" / f"aiforge-doc-{args.version}" / revision
    cache = STORAGE_ROOT / ".cache" / "huggingface"
    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        snapshot_download(
            repo_id=repository,
            repo_type="dataset",
            revision=revision,
            local_dir=destination,
            cache_dir=cache,
            token=token,
        )
    ).resolve()
    if not downloaded.is_relative_to(STORAGE_ROOT.resolve()):
        raise RuntimeError("download escaped the F:\\HYPERVERGE storage boundary")
    file_count = sum(1 for path in downloaded.rglob("*") if path.is_file())
    manifest = {
        "dataset": repository,
        "revision": revision,
        "gated": info.gated,
        "storage_bytes_reported": info.used_storage,
        "download_root": str(downloaded),
        "downloaded_file_count": file_count,
        "token_stored": False,
        "licence_policy": (
            "enforce source_dataset filtering; SROIE remains research-use only"
        ),
    }
    manifest_path = destination.parent / "accession.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
