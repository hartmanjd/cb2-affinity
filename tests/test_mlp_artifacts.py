"""Check that accepting one MLP result cannot silently destroy the previous one."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from mlp_artifacts import MLP, new_mlp_candidate, publish_mlp
from provenance_support import sha256


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for variant in ['baseline','tuning']:
            folder = self.root/MLP/variant
            folder.mkdir(parents=True)
            (folder/'old.txt').write_text(variant)
        (self.root/MLP/'summary.xlsx').write_bytes(b'previous workbook')
        source = self.root/'source.json'; source.write_text('{}')
        self.candidate = new_mlp_candidate(self.root, 'baseline')
        output = self.candidate/'metrics.csv'; output.write_text('score\n1\n')
        self.manifest = {'completed_at_utc':'2026-01-01','verification':{'science_passed':True},
                         'output_folder':self.candidate.relative_to(self.root).as_posix(),
                         'source_preparation_manifest':'source.json',
                         'source_preparation_manifest_sha256':sha256(source),
                         'output_sha256':{output.relative_to(self.root).as_posix():sha256(output)}}
        self.save()

    def save(self):
        (self.candidate/'manifest.json').write_text(json.dumps(self.manifest))

    def fake_summary(self, root):
        (root/MLP/'summary.xlsx').write_bytes(b'new workbook')

    def test_success_replaces_current_and_clears_dependent_tuning(self):
        with patch('build_mlp_summary.build_summary', side_effect=self.fake_summary):
            result=publish_mlp(self.root,self.candidate,'baseline')
        saved=json.loads((result/'manifest.json').read_text())
        self.assertEqual(saved['output_folder'],(MLP/'baseline').as_posix())
        self.assertTrue(all((self.root/p).is_file() for p in saved['output_sha256']))
        self.assertFalse((result/'old.txt').exists())
        self.assertFalse((self.root/MLP/'tuning').exists())
        self.assertEqual((self.root/MLP/'summary.xlsx').read_bytes(),b'new workbook')
        self.assertFalse(list((self.root/MLP).rglob('.previous-*')))
        self.assertFalse(self.candidate.exists())

    def test_bad_output_hash_preserves_current_results(self):
        (self.candidate/'metrics.csv').write_text('changed')
        with self.assertRaisesRegex(AssertionError,'Output changed'):
            publish_mlp(self.root,self.candidate,'baseline')
        for variant in ['baseline','tuning']:
            self.assertTrue((self.root/MLP/variant/'old.txt').exists())

    def test_report_failure_restores_both_variants_and_candidate(self):
        original=(self.candidate/'manifest.json').read_bytes()
        with patch('build_mlp_summary.build_summary',side_effect=PermissionError('Excel locked')):
            with self.assertRaises(PermissionError):publish_mlp(self.root,self.candidate,'baseline')
        self.assertEqual((self.candidate/'manifest.json').read_bytes(),original)
        for variant in ['baseline','tuning']:
            self.assertTrue((self.root/MLP/variant/'old.txt').exists())
        self.assertEqual((self.root/MLP/'summary.xlsx').read_bytes(),b'previous workbook')

    def test_incomplete_candidate_is_never_selected_as_baseline(self):
        from provenance_support import matching_stage
        self.manifest['source_preparation_manifest_sha256']=sha256(self.root/'source.json')
        self.save()
        with self.assertRaises(FileNotFoundError):
            matching_stage(self.root,MLP/'baseline','source_preparation_manifest',Path('source.json'),
                           'source_preparation_manifest_sha256')

    def test_tuning_replacement_keeps_baseline(self):
        candidate = new_mlp_candidate(self.root,'tuning')
        output=candidate/'metrics.csv'; output.write_text('score\n2\n')
        manifest=dict(self.manifest, output_folder=candidate.relative_to(self.root).as_posix(),
                      output_sha256={output.relative_to(self.root).as_posix():sha256(output)})
        (candidate/'manifest.json').write_text(json.dumps(manifest))
        with patch('build_mlp_summary.build_summary',side_effect=self.fake_summary):
            publish_mlp(self.root,candidate,'tuning')
        self.assertTrue((self.root/MLP/'baseline/old.txt').exists())
        self.assertEqual((self.root/MLP/'summary.xlsx').read_bytes(),b'new workbook')
        self.assertFalse((self.root/MLP/'tuning/old.txt').exists())
