"""Reproducible report-only diagnostics; no model fitting or test evaluation."""
from pathlib import Path
from itertools import combinations
from statistics import mean
import csv
import json
import math
from provenance_support import sha256

PAIR_FIELDS = ['split_strategy','left_rdkit_smiles','right_rdkit_smiles','tanimoto',
               'left_observed_pki','right_observed_pki','observed_difference_pki','exact_fingerprint_pair']


def cliff_pairs(features, targets, keys, split_indices):
    """Freeze all qualifying unordered validation pairs using binary Tanimoto.

    Python integer bit sets avoid a second chemistry dependency. Only validation
    rows are inspected. Thresholds match the existing MLP diagnostic definition.
    """
    result = []
    for split, subsets in split_indices.items():
        indices = sorted(subsets['validation'], key=lambda i: str(keys[i]))
        bits = {}
        for i in indices:
            row = features[i]
            assert all(bit in (0,1) for bit in row)
            bits[i] = sum(1 << j for j, bit in enumerate(row) if bit)
        for left, right in combinations(indices, 2):
            difference = float(targets[left])-float(targets[right])
            if abs(difference) < 1.0:
                continue
            union = (bits[left] | bits[right]).bit_count()
            similarity = (bits[left] & bits[right]).bit_count()/union if union else 1.0
            if similarity >= .8:
                result.append(dict(zip(PAIR_FIELDS, [split,str(keys[left]),str(keys[right]),similarity,
                    float(targets[left]),float(targets[right]),difference,bits[left]==bits[right]], strict=True)))
    return result


def save_cliffs(root, folder, prep_path, features, targets, keys, split_indices,
                stage='random_forest_report_diagnostics'):
    root, folder = Path(root), Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = cliff_pairs(features,targets,keys,split_indices)
    path=folder/'cliff_pairs.csv'
    with path.open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=PAIR_FIELDS);writer.writeheader();writer.writerows(rows)
    record = {'stage':stage,
              'source_preparation_manifest':Path(prep_path).as_posix(),
              'source_preparation_manifest_sha256':sha256(root/prep_path),
              'definition':{'scope':'unordered within outer validation','minimum_tanimoto':.8,
                            'minimum_absolute_observed_pki_difference':1.0},
              'pair_count':len(rows), 'output_sha256':{path.relative_to(root).as_posix():sha256(path)}}
    (folder/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')


def scores(rows):
    if not rows:
        return {'n_structures':0,'mae_pki':None,'rmse_pki':None,'r2':None,'r2_status':'undefined_empty'}
    observed=[float(r['observed_pki']) for r in rows]
    errors=[float(r['predicted_pki'])-y for r,y in zip(rows,observed,strict=True)]
    denominator=sum((y-mean(observed))**2 for y in observed)
    return {'n_structures':len(rows),'mae_pki':mean(abs(e) for e in errors),
            'rmse_pki':math.sqrt(mean(e*e for e in errors)),
            'r2':1-sum(e*e for e in errors)/denominator if denominator and len(rows)>1 else None,
            'r2_status':'defined' if denominator and len(rows)>1 else 'undefined_constant_or_small'}


def derive(predictions, pairs, neighbors):
    """Recompute spreadsheet-only metrics from the saved predictions and pair set."""
    validation={(r['split_strategy'],r['rdkit_smiles']):r for r in predictions if r['subset']=='validation'}
    assert len(validation)==sum(r['subset']=='validation' for r in predictions)
    neighbor_map={(r['split_strategy'],r['rdkit_smiles']):r for r in neighbors if r['subset']=='validation'}
    participants={split:set() for split in ['random','scaffold']}
    pair_rows=[]
    for pair in pairs:
        split=pair['split_strategy'];left=validation[split,pair['left_rdkit_smiles']];right=validation[split,pair['right_rdkit_smiles']]
        observed=float(left['observed_pki'])-float(right['observed_pki'])
        predicted=float(left['predicted_pki'])-float(right['predicted_pki'])
        assert math.isclose(observed,float(pair['observed_difference_pki']),abs_tol=1e-12)
        assert float(pair['tanimoto'])>=.8 and abs(observed)>=1
        participants[split].update([pair['left_rdkit_smiles'],pair['right_rdkit_smiles']])
        pair_rows.append({k:pair[k] for k in ['split_strategy','left_rdkit_smiles','right_rdkit_smiles','tanimoto']} |
                         {'observed_difference_pki':observed,'predicted_difference_pki':predicted,
                          'absolute_difference_error_pki':abs(predicted-observed),'correct_direction':observed*predicted>0})
    cliff_summary=[];molecules=[];similarity=[]
    for split in ['random','scaffold']:
        rows=[r for r in pair_rows if r['split_strategy']==split]
        cliff_summary.append({'split_strategy':split,'n_pairs':len(rows),'n_participating_structures':len(participants[split]),
            'pair_difference_mae_pki':mean(r['absolute_difference_error_pki'] for r in rows) if rows else None,
            'correct_direction_fraction':mean(r['correct_direction'] for r in rows) if rows else None,
            'predicted_ties':sum(r['predicted_difference_pki']==0 for r in rows),'status':'defined' if rows else 'undefined_empty'})
        for group,member in [('cliff_participant',True),('other_validation',False)]:
            chosen=[r for (s,k),r in validation.items() if s==split and ((k in participants[split])==member)]
            molecules.append({'split_strategy':split,'group':group,**scores(chosen)})
        for grouping,group,positive in [('exact_training_fingerprint','exact_match',True),
                                       ('exact_training_fingerprint','no_exact_match',False),
                                       ('nearest_training_tanimoto_0.80','below_0.80',False),
                                       ('nearest_training_tanimoto_0.80','at_least_0.80',True)]:
            chosen=[]
            for (s,k),row in validation.items():
                if s!=split:continue
                n=neighbor_map[s,k]
                flag=n['exact_training_fingerprint_match']=='True' if grouping=='exact_training_fingerprint' else float(n['maximum_tanimoto'])>=.8
                if flag==positive:chosen.append(row)
            similarity.append({'split_strategy':split,'grouping':grouping,'group':group,**scores(chosen)})
    return {'Cliff Summary':cliff_summary,'Cliff Molecule Summary':molecules,
            'Cliff Pair Predictions':pair_rows,'Similarity Summary':similarity}
