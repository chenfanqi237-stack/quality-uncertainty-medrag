"""Small CPU-only synthetic archive tests; no research data or model calls."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "cloud/research_archive_verification.py"
spec = importlib.util.spec_from_file_location("archive_check", SOURCE)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".tmp_archive_test_", dir=SOURCE.parents[1])
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "fixture.zip"

    def make(self, payload=None, manifest=None, duplicate=False):
        payload = payload or {"synthetic.txt": b"SYNTHETIC ONLY"}
        manifest = manifest or {"files": {n: {"bytes": len(b), "sha256": check.digest(b)}
                                         for n, b in payload.items()}}
        with warnings.catch_warnings(), zipfile.ZipFile(self.path, "w") as archive:
            warnings.simplefilter("ignore", UserWarning)
            for name, data in payload.items():
                archive.writestr(name, data)
            if duplicate:
                archive.writestr(next(iter(payload)), b"duplicate")
            archive.writestr(check.MANIFEST, json.dumps(manifest))
        return check.digest(self.path.read_bytes())

    def test_complete_archive_and_immutable_verification(self):
        sha = self.make()
        before = self.path.read_bytes()
        self.assertEqual(check.verify_archive(self.path, sha)["files_verified"], 1)
        self.assertEqual(self.path.read_bytes(), before)

    def test_wrong_outer_hash(self):
        self.make()
        with self.assertRaisesRegex(ValueError, "SHA256"):
            check.verify_archive(self.path, "0" * 64)

    def test_duplicate_members(self):
        sha = self.make(duplicate=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            check.verify_archive(self.path, sha)

    def test_unmanifested_member(self):
        sha = self.make(manifest={"files": {}})
        with self.assertRaisesRegex(ValueError, "inventory"):
            check.verify_archive(self.path, sha)

    def test_member_hash_mismatch(self):
        sha = self.make(manifest={"files": {"synthetic.txt": {"bytes": 14, "sha256": "0" * 64}}})
        with self.assertRaisesRegex(ValueError, "size/hash"):
            check.verify_archive(self.path, sha)

    def test_member_size_mismatch(self):
        sha = self.make(manifest={"files": {"synthetic.txt": {"bytes": 99, "sha256": check.digest(b"SYNTHETIC ONLY")}}})
        with self.assertRaisesRegex(ValueError, "size/hash"):
            check.verify_archive(self.path, sha)

    def test_traversal_and_absolute_paths(self):
        for name in ("../private", "/absolute", "C:/drive", "a\\b", "a//b", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                check.safe_member(name)

    def test_incomplete_write_rejected(self):
        self.make()
        data = self.path.read_bytes()[:40]
        self.path.write_bytes(data)
        with self.assertRaises(zipfile.BadZipFile):
            check.verify_archive(self.path, check.digest(data))


if __name__ == "__main__":
    unittest.main()
