"""Exercise complete XGBoost publication, hash rebasing and rollback."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from xgboost_artifacts import XGB,new_xgboost_candidate,publish_xgboost
from provenance_support import sha256


class XGBoostPublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);parent=self.root/XGB
        (parent/'files').mkdir(parents=True);(parent/'files/old.txt').write_text('old pair of models')
        (parent/'summary.xlsx').write_bytes(b'old workbook')
        (self.root/'prep.json').write_text('{}')
        self.candidate=new_xgboost_candidate(self.root)
        for variant in ['baseline','tuning','diagnostics']:
            folder=self.candidate/variant;folder.mkdir(exist_ok=True)
            (folder/'evidence.csv').write_text('value\n1\n')
            m={'source_preparation_manifest':'prep.json','source_preparation_manifest_sha256':sha256(self.root/'prep.json'),
               'output_folder':folder.relative_to(self.root).as_posix(),'completed_at_utc':'2026-01-01',
               'verification':{'checks_passed':True},'output_sha256':{(folder/'evidence.csv').relative_to(self.root).as_posix():sha256(folder/'evidence.csv')}}
            if variant=='tuning':
                m['baseline_manifest']=(self.candidate/'baseline/manifest.json').relative_to(self.root).as_posix()
                m['baseline_manifest_sha256']=sha256(self.root/m['baseline_manifest'])
            (folder/'manifest.json').write_text(json.dumps(m))

    def test_missing_tuning_does_not_replace_baseline(self):
        (self.candidate/'tuning/manifest.json').unlink()
        with self.assertRaises(FileNotFoundError):publish_xgboost(self.root,self.candidate)
        self.assertTrue((self.root/XGB/'files/old.txt').exists())

    def test_report_failure_restores_both_files_and_workbook(self):
        original=(self.candidate/'tuning/manifest.json').read_bytes()
        with patch('build_xgboost_summary.build_summary',side_effect=PermissionError('locked workbook')):
            with self.assertRaises(PermissionError):publish_xgboost(self.root,self.candidate)
        self.assertTrue((self.root/XGB/'files/old.txt').exists())
        self.assertEqual((self.root/XGB/'summary.xlsx').read_bytes(),b'old workbook')
        self.assertEqual((self.candidate/'tuning/manifest.json').read_bytes(),original)

    def test_success_rebases_hash_links_and_leaves_one_execution(self):
        def report(root):(root/XGB/'summary.xlsx').write_bytes(b'new workbook')
        with patch('build_xgboost_summary.build_summary',side_effect=report):
            destination=publish_xgboost(self.root,self.candidate)
        self.assertFalse(self.candidate.exists())
        self.assertFalse((destination/'old.txt').exists())
        record=json.loads((destination/'tuning/manifest.json').read_text())
        self.assertEqual(record['baseline_manifest_sha256'],sha256(self.root/record['baseline_manifest']))
        for name,expected in record['output_sha256'].items():self.assertEqual(sha256(self.root/name),expected)
        self.assertFalse(list((self.root/XGB).glob('.previous-*')))

    def test_changed_output_cannot_replace_accepted_result(self):
        (self.candidate/'tuning/evidence.csv').write_text('corrupted data')
        with self.assertRaises(AssertionError):publish_xgboost(self.root,self.candidate)
        self.assertEqual((self.root/XGB/'summary.xlsx').read_bytes(),b'old workbook')
        self.assertTrue((self.root/XGB/'files/old.txt').exists())

    def test_failed_scientific_verification_cannot_replace_accepted_result(self):
        path=self.candidate/'tuning/manifest.json'
        record=json.loads(path.read_text());record['verification']['checks_passed']=False
        path.write_text(json.dumps(record))
        with self.assertRaises(AssertionError):publish_xgboost(self.root,self.candidate)
        self.assertTrue((self.root/XGB/'files/old.txt').exists())
