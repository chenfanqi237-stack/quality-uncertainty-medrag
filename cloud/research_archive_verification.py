"""CPU-only, read-only verification of private research backup archives.

This utility contains no dataset, reviewer mapping, inference or gold reader.
The archive must include INTERNAL_MANIFEST_SHA256.json, covering every other
file with its byte size and SHA256. Verification never extracts or rewrites it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath

MANIFEST = "INTERNAL_MANIFEST_SHA256.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_member(name):
    p = PurePosixPath(name)
    if (not name or p.is_absolute() or ".." in p.parts or "\\" in name
            or ":" in name or p.as_posix() != name):
        raise ValueError("Unsafe archive path")


def verify_archive(path, expected_sha256):
    path = Path(path)
    if digest(path.read_bytes()) != expected_sha256:
        raise ValueError("Archive SHA256 mismatch")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        for info in archive.infolist():
            safe_member(info.filename)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Directories/symlinks are not payload files")
        if archive.testzip() is not None:
            raise ValueError("Archive CRC mismatch")
        manifest = json.loads(archive.read(MANIFEST))
        files = manifest["files"]
        if set(names) != set(files) | {MANIFEST}:
            raise ValueError("Manifest does not cover exact archive inventory")
        for name, expected in files.items():
            safe_member(name)
            data = archive.read(name)
            if len(data) != expected["bytes"] or digest(data) != expected["sha256"]:
                raise ValueError("Member size/hash mismatch: " + name)
    return {"archive_sha256": expected_sha256, "bytes": path.stat().st_size,
            "files_verified": len(files), "crc_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(verify_archive(args.archive, args.sha256), indent=2))


if __name__ == "__main__":
    main()
