#!/usr/bin/env python
"""Download a HuggingFace model into a local directory, without huggingface_hub.

Why this exists: behind hf-mirror.com, `huggingface_hub` fails with
``FileMetadataError: Distant resource does not seem to be on huggingface.co``.
The mirror answers the metadata HEAD with a 308 pointing at
``https://huggingface.co/api/resolve-cache/...`` and omits ``X-Repo-Commit``,
which the client treats as "not a HF resource". Plain GETs that follow the
redirect work fine, so this script drives the transfer itself.

Usage::

    python scripts/fetch_model.py Qwen/Qwen3-1.7B ./models/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

ENDPOINT = "https://hf-mirror.com"
# Skip weights formats we do not load, and repo furniture.
WANTED_SUFFIXES = (
    ".json",
    ".safetensors",
    ".txt",
    ".model",
)
SKIP_PREFIXES = ("onnx/", "gguf/", ".git")


def list_files(repo: str, revision: str) -> list[str]:
    url = f"{ENDPOINT}/api/models/{repo}"
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(url, params={"revision": revision})
        response.raise_for_status()
        payload = response.json()
    siblings = payload.get("siblings") or []
    if not siblings:
        raise RuntimeError(f"No files listed for {repo}@{revision}")
    return [s["rfilename"] for s in siblings]


def keep(filename: str) -> bool:
    if filename.startswith(SKIP_PREFIXES):
        return False
    name = filename.rsplit("/", 1)[-1]
    if name.startswith("."):
        return False
    return filename.endswith(WANTED_SUFFIXES)


def download(repo: str, dest: Path, revision: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    files = [f for f in list_files(repo, revision) if keep(f)]
    print(f"{len(files)} files to fetch from {ENDPOINT}/{repo}@{revision}")

    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        for index, filename in enumerate(files, 1):
            target = dest / filename
            if target.exists() and target.stat().st_size > 0:
                print(f"  [{index}/{len(files)}] cached  {filename}")
                continue

            url = f"{ENDPOINT}/{repo}/resolve/{revision}/{filename}"
            target.parent.mkdir(parents=True, exist_ok=True)
            # Stream to a temp file so an interrupted run never leaves a
            # truncated file that a later run would treat as cached.
            partial = target.with_suffix(target.suffix + ".part")
            with client.stream("GET", url) as response:
                response.raise_for_status()
                written = 0
                with partial.open("wb") as handle:
                    for chunk in response.iter_bytes(chunk_size=1 << 20):
                        handle.write(chunk)
                        written += len(chunk)
            partial.replace(target)
            print(f"  [{index}/{len(files)}] {written / 1024**2:8.1f} MB  {filename}")

    total = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"done: {dest} ({total / 1024**3:.2f} GB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="e.g. Qwen/Qwen3-1.7B")
    parser.add_argument("dest", type=Path, help="local directory to fill")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    download(args.repo, args.dest, args.revision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
