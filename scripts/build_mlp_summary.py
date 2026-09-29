"""Build readable MLP summaries from saved evidence; never fit or load binary models."""
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev
import argparse
import csv
import hashlib
import json
import math
import re
from openpyxl import Workbook, load_workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.hyperlink import Hyperlink

ROOT = Path(__file__).resolve().parents[1]
MLP = Path('provenance/models/multilayer_perceptron')
NAVY, TEAL, PALE = '18354A', '007F82', 'EAF4F5'
TABLES = {
    'metrics.csv': ('Scores by Seed', 'Training and validation scores for each initialization. Test sets are reserved.'),
    'seed_summary.csv': ('Score Summary', 'Means and sample standard deviations across initialization seeds.'),
    'baseline_comparison.csv': ('Baseline Comparison', 'Baseline and tuned scores on the same frozen data. Compare matching seeds and splits.'),
    'epoch_selections.csv': ('Selected Epochs', 'Training duration chosen using the internal stopping set, before full-training refits.'),
    'candidate_summary.csv': ('Search Ranking', 'Candidate performance on internal stopping data, summarized across search seeds.'),
    'candidate_scores.csv': ('Search by Seed', 'Every candidate fit and internal stopping score. These are not outer-validation scores.'),
    'subgroup_summary.csv': ('Similarity Summary', 'Validation errors grouped by similarity to the nearest training molecule.'),
    'cliff_summary.csv': ('Cliff Summary', 'Performance on similar molecule pairs with large measured affinity differences.'),
    'cliff_molecule_group_summary.csv': ('Cliff Molecule Summary', 'Validation error on cliff-involved versus other molecules.'),
    'predictions.csv': ('Predictions', 'Every saved training and validation prediction; one row per molecule, split and seed.'),
    'internal_assignments.csv': ('Internal Membership', 'Outer-training molecules assigned to internal fitting or stopping sets.'),
    'internal_history.csv': ('Stopping History', 'Epoch-by-epoch internal fitting and stopping losses.'),
    'selected_internal_history.csv': ('Stopping History', 'Internal loss history for the selected tuned configuration.'),
    'full_refit_history.csv': ('Refit History', 'Epoch history for fresh models fitted to all outer-training molecules.'),
    'candidate_history.csv': ('Search History', 'Epoch histories for every searched candidate and seed.'),
    'validation_subgroups.csv': ('Similarity by Seed', 'Detailed similarity-group errors for each seed.'),
    'cliff_pairs.csv': ('Cliff Pairs', 'Frozen validation pairs defining the activity-cliff diagnostic.'),
    'cliff_molecule_groups.csv': ('Cliff Membership', 'Which validation molecules participate in a qualifying cliff pair.'),
    'cliff_results.csv': ('Cliff Pair Predictions', 'Predicted and observed affinity differences for qualifying pairs.'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def flatten(value, key=''):
    if isinstance(value, dict):
        for name, item in value.items():
            yield from flatten(item, f'{key}.{name}' if key else name)
    elif isinstance(value, list):
        # A short list is easier to read together than across many technical rows.
        if all(not isinstance(item, (dict, list)) for item in value):
            yield key, json.dumps(value)
        else:
            for i, item in enumerate(value):
                yield from flatten(item, f'{key}[{i}]')
    else:
        yield key, value


def numeric(value):
    # Preserve long identifiers, molecular strings and formula-like text literally.
    if value == '':
        return None
    if re.fullmatch(r'-?\d{1,14}', value):
        return int(value)
    if re.fullmatch(r'-?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?', value):
        number = float(value)
        if math.isfinite(number):
            return number
    return value


def build_summary(variant, root=ROOT):
    root = Path(root)
    parent = root / MLP / variant
    folder = parent / 'files'
    manifest = read_json(folder / 'manifest.json')
    assert manifest.get('completed_at_utc'), 'A completed run is required.'
    for path, expected in manifest['output_sha256'].items():
        assert digest(root / path) == expected, f'Changed source file: {path}'
    prep = read_json(root / manifest['source_preparation_manifest'])
    assert digest(root / manifest['source_preparation_manifest']) == manifest['source_preparation_manifest_sha256']
    baseline = manifest
    if variant == 'tuning':
        baseline = read_json(root / manifest['baseline_manifest'])
        assert digest(root / manifest['baseline_manifest']) == manifest['baseline_manifest_sha256']
    config = baseline['configuration']
    metrics = read_csv(folder / 'metrics.csv')
    book = Workbook(write_only=True)
    counts = {}

    def cell(sheet, value, *, header=False, title=False, link=None):
        c = WriteOnlyCell(sheet, value=value)
        if isinstance(value, str):
            c.data_type = 's'
        c.font = Font(name='Calibri', size=19 if title else 11, bold=header or title,
                      color='FFFFFF' if header or title else NAVY)
        c.alignment = Alignment(vertical='top', wrap_text=True)
        if header or title:
            c.fill = PatternFill('solid', fgColor=NAVY if title else TEAL)
        if isinstance(value, float):
            c.number_format = '0.0000'
        if link:
            if link.startswith('#'):
                c.hyperlink = Hyperlink(ref=c.coordinate, location=link[1:])
            else:
                c.hyperlink = link
            c.font = Font(name='Calibri', size=11, color='007F82', underline='single')
        return c

    def sheet(name, title, subtitle, headers, rows, widths=None):
        ws = book.create_sheet(name)
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.sheet_properties.tabColor = TEAL
        ws.sheet_view.showGridLines = False
        ws.freeze_panes = 'A5'
        for i, width in enumerate(widths or [24]*len(headers), 1):
            from openpyxl.utils import get_column_letter
            ws.column_dimensions[get_column_letter(i)].width = width
        from openpyxl.utils import get_column_letter
        ws.merged_cells.add(f'A1:{get_column_letter(len(headers))}1')
        ws.merged_cells.add(f'A2:{get_column_letter(len(headers))}2')
        ws.row_dimensions[1].height = 32
        ws.row_dimensions[2].height = 44
        ws.row_dimensions[4].height = 42
        ws.append([cell(ws, title, title=True)])
        ws.append([cell(ws, subtitle)])
        ws.append([cell(ws, 'Back to overview', link="#'Overview'!A1")])
        ws.append([cell(ws, h, header=True) for h in headers])
        count = 0
        for row in rows:
            if name in {'Training Guide', 'Settings'}:
                ws.row_dimensions[count+5].height = 42
            ws.append([cell(ws, value) for value in row])
            count += 1
        from openpyxl.utils import get_column_letter
        ws.auto_filter.ref = f'A4:{get_column_letter(len(headers))}{4+count}'
        counts[name] = count
        return ws

    cover = book.create_sheet('Overview')
    cover.sheet_view.showGridLines = False
    cover.sheet_properties.tabColor = NAVY
    cover.freeze_panes = 'C6'
    for column, width in zip('ABCDEFGH', [22,22,17,18,18,18,18,25]):
        cover.column_dimensions[column].width = width
    def line(values=(), *, header=False, title=False, height=30, links=None):
        # write-only worksheets do not expose max_row; track the next cover row here.
        line.row += 1
        cover.row_dimensions[line.row].height = height
        if 0 < len(values) < 8:
            first = 'A' if len(values)==1 else 'B' if len(values)==2 else 'C'
            cover.merged_cells.add(f'{first}{line.row}:H{line.row}')
        cover.append([cell(cover,v,header=header,title=title,link=(links or {}).get(i)) for i,v in enumerate(values)])
    line.row = 0
    line(['MLP | ' + ('BASELINE' if variant == 'baseline' else 'TUNED MODEL')],title=True,height=36)
    line(['CURRENT SAVED RESULTS', manifest['completed_at_utc']],height=28)
    line(['Five initialization seeds; mean ± sample SD. SD measures seed variability, not a confidence interval.'],height=30)
    line(['Validation estimates generalization; training scores describe fit. Reserved tests have not been evaluated.'],height=30)
    line(['Split','Subset','Molecules / seed','MAE (pKi)','MAE SD','RMSE (pKi)','RMSE SD','R² ± SD'],header=True)
    groups=defaultdict(list)
    for row in metrics:
        groups[row['split_strategy'], row['subset']].append(row)
    for subset in ['validation','train']:
        for split in ['random','scaffold']:
            rows=groups[split,subset]
            means={k:mean(float(r[k]) for r in rows) for k in ['mae_pki','rmse_pki','r2']}
            sds={k:stdev(float(r[k]) for r in rows) if len(rows)>1 else 0 for k in means}
            line([split.title(),subset.title(),int(rows[0]['n_structures']),means['mae_pki'],sds['mae_pki'],means['rmse_pki'],sds['rmse_pki'],f"{means['r2']:.4f} ± {sds['r2']:.4f}"])
    line(['Lower MAE / RMSE is better. Higher R² is better; zero is the observed-mean benchmark.'],height=28)
    line(['MODEL AT A GLANCE'],header=True)
    params=config['model_parameters']
    layers=params['hidden_layer_sizes']
    line(['Architecture', ('Selected per split' if variant=='tuning' else f"{config['input_bits']} → " + ' → '.join(map(str,layers)) + ' → 1'), 'ReLU hidden layers; linear pKi output'],height=42)
    line(['Inputs','Morgan fingerprints',f"{prep['configuration']['fingerprint']['bits']} bits; radius {prep['configuration']['fingerprint']['radius']}; chirality {prep['configuration']['fingerprint']['include_chirality']}"],height=42)
    line(['Training','Adam optimizer','Internal stopping selects epochs; then a fresh full-training refit.'],height=42)
    if variant == 'tuning':
        for split, selected in manifest['selection'].items():
            line([split.title()+' choice',str(selected['hidden_layers']),f"L2 = {selected['alpha']}; learning rate = {selected['learning_rate_init']}"],height=42)
        line(['Search',f"{len(manifest['configuration']['grid'])} candidates per split",'Chosen by internal stopping MSE; outer validation is not used for selection.'],height=42)
    else:
        line(['Regularization',f"L2 alpha = {params['alpha']}",f"Learning rate = {params['learning_rate_init']}; batch = {params['batch_size']}"],height=42)
    line(['Seeds',', '.join(map(str,config['initialization_seeds'])),'One saved execution includes five initializations on each of two splits.'],height=42)
    line(['Dataset',f"{prep['counts']['structures']:,} structures",f"{prep['counts']['retained_measurements']:,} retained measurements"],height=38)
    line(['Scope','Development validation','Prior development used validation labels; reserved test evaluation is still needed.'],height=44)
    line(['READ NEXT'],header=True)
    navigation=[('Training Guide','Plain-English architecture, fit procedure, data sizes and evaluation definitions.'),
                ('Settings','Complete model, preparation and selection settings, software versions and audit flags.'),
                ('Scores by Seed','Exact training/validation scores, including each initialization.'),
                ('Selected Epochs','Chosen stopping epochs and refit duration.'),
                ('Evidence Files','Links, checksums and table locations for every supporting file.')]
    if variant=='tuning': navigation.insert(3,('Search Ranking','Why these configurations were selected; internal stopping results.'))
    for name, explanation in navigation:
        line([name,explanation],links={0:f"#'{name}'!A1"},height=40)
    line(['Source', 'files/manifest.json', 'This workbook is generated from saved files; it does not retrain models.'],links={1:'files/manifest.json'},height=40)
    # The cover prints as a concise, readable single-page presentation reference.
    cover.sheet_properties.pageSetUpPr.fitToPage=True
    cover.page_setup.orientation='landscape'; cover.page_setup.paperSize='8'
    cover.page_setup.fitToWidth=1; cover.page_setup.fitToHeight=1
    cover.print_options.horizontalCentered=True
    cover.print_area=f'A1:H{line.row}'

    guide=[
        ('Architecture','Fully connected neural network',f"{config['input_bits']} input bits, one continuous pKi output. Per-split parameters are listed below."),
        ('Output','pKi affinity', 'Higher pKi means stronger binding. A change of one pKi unit is a tenfold change in Ki.'),
        ('Target construction','Median within assay, then across assays',prep['target_policy']),
        ('Feature encoding','Binary Morgan fingerprints',json.dumps(prep['configuration']['fingerprint'])),
        ('Optimizer','Adam','Optimizes squared prediction error with L2 regularization (alpha).'),
        ('Activation','ReLU','Hidden units use max(0, x); the regression output is linear.'),
        ('Stopping','Internal fitting and stopping sets', 'Both come only from outer-training molecules; validation is held aside.'),
        ('Epoch selection',config['epoch_selection'],f"At most {config['maximum_epochs']} epochs; patience {config['patience']}."),
        ('Final training',config['full_refit'],'The selected epoch count is reused with a fresh model on all outer-training molecules.'),
        ('Initialization','Five seeds','Independent starts, not five different dataset splits or independent experiments.'),
        ('Random split','Molecular-row assignment','Related structures can appear on both sides of this split.'),
        ('Scaffold split','Bemis–Murcko scaffold groups','Tests generalization across scaffold groups; exact fingerprint overlap is separately audited.'),
        ('MAE','Mean absolute error in pKi','Typical absolute distance between predicted and observed affinity.'),
        ('RMSE','Root mean squared error in pKi','Penalizes larger prediction errors more strongly than MAE.'),
        ('R²','Coefficient of determination','1 is perfect; 0 matches the observed-mean benchmark; negative values are possible.'),
        ('Activity cliffs','Similar molecules with different affinity',json.dumps(config['cliff_definition'])),
        ('Fitted models','models.joblib.gz','Compressed Python model objects are retained as binaries; weights are not expanded into spreadsheet cells.'),
        ('Figures','PNG files','Open the linked figures from Evidence Files for loss curves and prediction diagnostics.'),
    ]
    for split in ['random', 'scaffold']:
        selected_params = dict(params)
        if variant == 'tuning':
            selected = manifest['selection'][split]
            selected_params.update(hidden_layer_sizes=selected['hidden_layers'],
                                   alpha=selected['alpha'], learning_rate_init=selected['learning_rate_init'])
        for key, value in selected_params.items():
            guide.append(('Model parameters', f'{split} / {key.replace("_", " ")}',
                          json.dumps(value) if isinstance(value, list) else value))
    for split, sizes in prep['counts']['split_subsets'].items():
        for subset, count in sizes.items():
            guide.append(('Dataset sizes',f'{split} / {subset}',count['structures']))
    for split, info in config['internal_split'].items():
        for key,value in info.items(): guide.append(('Internal split',f'{split} / {key.replace("_"," ")}',value))
    if variant=='tuning':guide.append(('Search rule','Choose using internal stopping data',manifest['configuration']['selection_rule']))
    sheet('Training Guide','How this model was trained','Definitions and decisions behind the results.', ['Topic','Choice / quantity','Explanation / value'],guide,[25,48,105])
    settings=[]
    for path in sorted(folder.glob('*.json')):
        settings.extend((path.name,key,value) for key,value in flatten(read_json(path)))
    settings.extend(('preparation/manifest.json',key,value) for key,value in flatten(prep))
    if variant=='tuning':
        settings.extend(('baseline configuration',key,value) for key,value in flatten(config))
    sheet('Settings','Complete saved settings','Original JSON keys and values; use Training Guide for the plain-English explanation.', ['Source','Setting','Value'],settings,[32,65,110])
    for filename,(name,explanation) in TABLES.items():
        path=folder/filename
        if not path.exists():continue
        with path.open(newline='',encoding='utf-8') as stream:
            reader=csv.reader(stream); headers=next(reader)
            widths=[min(55,max(18,len(h)+2)) for h in headers]
            sheet(name,name,explanation+' Source: files/'+filename,headers,
                  ([numeric(v) for v in row] for row in reader),widths)
    extra=set(p.name for p in folder.glob('*.csv'))-TABLES.keys()
    assert not extra, f'Add readable names for new reports: {extra}'
    evidence=[]
    for path in sorted(folder.iterdir()):
        if not path.is_file():continue
        tab=TABLES.get(path.name,('Settings' if path.suffix=='.json' else 'Linked file only',))[0]
        purpose=TABLES.get(path.name,('', 'Saved fitted models (binary)' if path.name.endswith('.gz') else 'Settings / provenance' if path.suffix=='.json' else 'Diagnostic figure'))[1]
        evidence.append([path.name,purpose,tab,path.stat().st_size,digest(path)])
    ws=sheet('Evidence Files','Supporting files','Each link opens the original file beside this workbook; SHA-256 identifies its exact bytes.',
             ['File','Purpose','Workbook sheet','Bytes','SHA-256'],[],[43,85,30,15,70])
    for row in evidence:
        ws.append([cell(ws,v,link='files/'+row[0] if i==0 else f"#'{v}'!A1" if i==2 and v!='Linked file only' else None) for i,v in enumerate(row)])
    ws.auto_filter.ref=f'A4:E{len(evidence)+4}'; counts['Evidence Files']=len(evidence)
    destination=parent/'summary.xlsx'; temporary=parent/'.~summary.xlsx'
    book.save(temporary)
    # Reopen the finished file before replacing a reader's existing summary.
    check=load_workbook(temporary,read_only=True,data_only=True)
    assert check.sheetnames[0]=='Overview' and 'Evidence Files' in check.sheetnames
    check.close(); temporary.replace(destination)
    return {'workbook':str(destination.relative_to(root)), 'sheets':len(book.sheetnames),'table_rows':sum(counts.values())}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant',choices=['baseline','tuning','all'],default='all',nargs='?')
    args=parser.parse_args()
    for variant in ['baseline','tuning'] if args.variant=='all' else [args.variant]:
        print(json.dumps(build_summary(variant),indent=2))
