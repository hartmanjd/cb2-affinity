"""Check that models stored outside git are fetched, verified, and required safely."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fetch_models import asset_name, download_models, expected_models, require_models
from provenance_support import sha256

MODEL = 'provenance/models/forest/files/tuning/models.joblib.gz'


class FetchModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / 'repo'
        self.release = Path(self.temporary.name) / 'release'
        self.release.mkdir()
        # A "trained" model and the manifest that records its checksum.
        model = self.root / MODEL
        model.parent.mkdir(parents=True)
        model.write_bytes(b'fitted model bytes')
        self.digest = sha256(model)
        (model.parent / 'manifest.json').write_text(json.dumps(
            {'output_sha256': {MODEL: self.digest, 'provenance/models/forest/files/tuning/metrics.csv': '0' * 64}}))
        # Publish it to the fake release under its asset name, then "clone" without it.
        (self.release / asset_name(MODEL, self.digest)).write_bytes(model.read_bytes())
        model.unlink()
        self.url = self.release.as_uri()

    def tearDown(self):
        self.temporary.cleanup()

    def test_only_model_files_are_expected_and_names_are_unique(self):
        self.assertEqual(expected_models(self.root), {MODEL: self.digest})
        self.assertEqual(asset_name(MODEL, self.digest), f'forest__files__tuning__models.{self.digest[:12]}.joblib.gz')
        self.assertNotEqual(asset_name(MODEL, '1' * 64), asset_name(MODEL, '2' * 64))

    def test_missing_model_gives_fetch_instruction(self):
        with self.assertRaisesRegex(FileNotFoundError, 'fetch_models.py'):
            require_models(self.root)

    def test_download_places_verified_model_and_is_repeatable(self):
        self.assertEqual(download_models(self.root, self.url), 1)
        self.assertEqual(sha256(self.root / MODEL), self.digest)
        require_models(self.root)
        self.assertEqual(download_models(self.root, self.url), 0)

    def test_corrupt_download_is_rejected_without_leaving_files(self):
        (self.release / asset_name(MODEL, self.digest)).write_bytes(b'tampered')
        with self.assertRaisesRegex(RuntimeError, 'Checksum mismatch'):
            download_models(self.root, self.url)
        self.assertEqual(list((self.root / MODEL).parent.glob('*.gz')), [])
        self.assertEqual(list((self.root / MODEL).parent.glob('.download-*')), [])

    def test_changed_local_model_is_never_overwritten(self):
        (self.root / MODEL).write_bytes(b'a newer, unrecorded run')
        with self.assertRaisesRegex(RuntimeError, 'does not match'):
            download_models(self.root, self.url)
        self.assertEqual((self.root / MODEL).read_bytes(), b'a newer, unrecorded run')


if __name__ == '__main__':
    unittest.main()
