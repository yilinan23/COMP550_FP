"""Download and unpack the official BLiMP benchmark from GitHub."""

from __future__ import annotations

import argparse
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from syntax_rl.utils import ensure_dir, resolve_project_path

DEFAULT_BLIMP_URL = "https://github.com/alexwarstadt/blimp/raw/master/BLiMP.zip"


def download_blimp(destination: str | Path = "data/raw/blimp", url: str = DEFAULT_BLIMP_URL) -> Path:
    """Download official BLiMP data and extract JSONL files under destination.

    The upstream archive contains a ``data/`` directory with 67 BLiMP JSONL
    paradigm files. This helper extracts that directory directly into
    ``data/raw/blimp`` by default, so baseline configs can point at one stable
    local folder without committing the benchmark data to this repository.
    """
    output_dir = ensure_dir(destination)
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "BLiMP.zip"
        urllib.request.urlretrieve(url, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target_name = Path(member.filename).name
                if member.is_dir() or not member.filename.endswith(".jsonl") or target_name.startswith("._"):
                    continue
                target_path = output_dir / target_name
                with archive.open(member) as source, target_path.open("wb") as target:
                    target.write(source.read())
    return output_dir


def main() -> None:
    """CLI entry point for downloading official BLiMP data."""
    parser = argparse.ArgumentParser(description="Download official BLiMP JSONL files from GitHub.")
    parser.add_argument("--destination", default="data/raw/blimp")
    parser.add_argument("--url", default=DEFAULT_BLIMP_URL)
    args = parser.parse_args()
    destination = download_blimp(destination=args.destination, url=args.url)
    print(f"Downloaded BLiMP JSONL files to {destination}")


if __name__ == "__main__":
    main()
