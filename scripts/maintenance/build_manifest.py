#!/usr/bin/env python3
"""Build deterministic file-size and SHA-256 manifests for the release."""

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXCLUDED = {"MANIFEST.tsv", "CHECKSUMS.sha256"}


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    rows = []
    for path in sorted(ROOT.rglob("*"), key=lambda item: item.as_posix().lower()):
        if (
            not path.is_file()
            or path.name in EXCLUDED
            or ".git" in path.parts
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
        ):
            continue
        relative = path.relative_to(ROOT).as_posix()
        rows.append((relative, path.stat().st_size, digest(path)))

    manifest = ROOT / "MANIFEST.tsv"
    checksums = ROOT / "CHECKSUMS.sha256"
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("path\tbytes\tsha256\n")
        for relative, size, sha256 in rows:
            handle.write(f"{relative}\t{size}\t{sha256}\n")
    with checksums.open("w", encoding="utf-8", newline="\n") as handle:
        for relative, _, sha256 in rows:
            handle.write(f"{sha256}  {relative}\n")
    print(f"Wrote {len(rows)} entries")


if __name__ == "__main__":
    main()
