"""Exercise forest diagnostic definitions and complete-experiment publication."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from forest_diagnostics import cliff_pairs,derive
from forest_artifacts import FOREST,new_forest_candidate,publish_forest
from provenance_support import sha256


class DiagnosticTests(unittest.TestCase):
    def test_cliffs_use_only_validation_and_include_threshold_boundary(self):
        features=[[1,1,1,1,0],[1,1,1,1,1],[0,0,0,0,1],[1,1,1,1,0]]
        pairs=cliff_pairs(features,[7.,6.,7.,3.],['a','b','c','test'],
                          {'random':{'validation':[2,1,0],'test':[3]}})
        self.assertEqual(len(pairs),1)
        self.assertEqual((pairs[0]['left_rdkit_smiles'],pairs[0]['right_rdkit_smiles']),('a','b'))
        self.assertEqual(pairs[0]['tanimoto'],.8)
        self.assertEqual(pairs[0]['observed_difference_pki'],1.)

    def test_predicted_ties_are_incorrect_and_empty_groups_are_explicit(self):
        pairs=cliff_pairs([[1],[1]],[7.,6.],['a','b'],{'random':{'validation':[0,1]}})
        predictions=[{'split_strategy':'random','subset':'validation','rdkit_smiles':key,'observed_pki':y,'predicted_pki':6.5}
                     for key,y in [('a',7.),('b',6.)]]
        neighbors=[{'split_strategy':'random','subset':'validation','rdkit_smiles':k,'maximum_tanimoto':.8,
                    'exact_training_fingerprint_match':'False'} for k in ['a','b']]
        result=derive(predictions,pairs,neighbors)
        self.assertEqual(result['Cliff Summary'][0]['predicted_ties'],1)
        self.assertEqual(result['Cliff Summary'][0]['correct_direction_fraction'],0)
        self.assertEqual(result['Cliff Summary'][1]['status'],'undefined_empty')
        self.assertIsNone(result['Cliff Summary'][1]['pair_difference_mae_pki'])


class ForestPublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);parent=self.root/FOREST
        (parent/'files').mkdir(parents=True);(parent/'files/old.txt').write_text('old pair of models')
        (parent/'summary.xlsx').write_bytes(b'old workbook')
        (self.root/'prep.json').write_text('{}')
        self.candidate=new_forest_candidate(self.root)
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
        with self.assertRaises(FileNotFoundError):publish_forest(self.root,self.candidate)
        self.assertTrue((self.root/FOREST/'files/old.txt').exists())

    def test_report_failure_restores_both_files_and_workbook(self):
        original=(self.candidate/'tuning/manifest.json').read_bytes()
        with patch('build_forest_summary.build_summary',side_effect=PermissionError('locked workbook')):
            with self.assertRaises(PermissionError):publish_forest(self.root,self.candidate)
        self.assertTrue((self.root/FOREST/'files/old.txt').exists())
        self.assertEqual((self.root/FOREST/'summary.xlsx').read_bytes(),b'old workbook')
        self.assertEqual((self.candidate/'tuning/manifest.json').read_bytes(),original)

    def test_success_rebases_hash_links_and_leaves_one_execution(self):
        def report(root):(root/FOREST/'summary.xlsx').write_bytes(b'new workbook')
        with patch('build_forest_summary.build_summary',side_effect=report):
            destination=publish_forest(self.root,self.candidate)
        self.assertFalse(self.candidate.exists())
        self.assertFalse((destination/'old.txt').exists())
        record=json.loads((destination/'tuning/manifest.json').read_text())
        self.assertEqual(record['baseline_manifest_sha256'],sha256(self.root/record['baseline_manifest']))
        for name,expected in record['output_sha256'].items():self.assertEqual(sha256(self.root/name),expected)
        self.assertFalse(list((self.root/FOREST).glob('.previous-*')))
