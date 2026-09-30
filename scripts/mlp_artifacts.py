"""Keep one complete MLP execution per variant, replacing it only after audits pass.

Layout: multilayer_perceptron/summary.xlsx (one combined workbook) beside the
baseline/ and tuning/ folders, which each hold that variant's files directly.
"""
from pathlib import Path
import json
import shutil
import tempfile
from provenance_support import sha256

MLP = Path('provenance/models/multilayer_perceptron')


def new_mlp_candidate(root, variant):
    """Write a new execution separately so a failed fit cannot erase current results.

    The candidate sits beside baseline/ and tuning/ (not inside them) because the
    whole variant folder is swapped in when the candidate is accepted.
    """
    if variant not in {'baseline', 'tuning'}:
        raise ValueError(variant)
    parent = Path(root) / MLP
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f'.pending-{variant}-', dir=parent))


def replace_path(value, old, new):
    if isinstance(value, dict):
        return {replace_path(k, old, new): replace_path(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [replace_path(v, old, new) for v in value]
    if isinstance(value, str):
        return value.replace(old, new)
    return value


def publish_mlp(root, candidate, variant):
    """Publish audited files and a matching workbook; roll back on reporting errors.

    A new baseline invalidates tuning derived from the previous baseline. Those
    dependent results are removed only after the replacement baseline is complete.
    Notebook 09 then produces a new tuning execution from this accepted baseline.
    Either publication rebuilds the one combined summary.xlsx beside both folders.
    """
    from build_mlp_summary import build_summary
    root, candidate = Path(root).resolve(), Path(candidate).resolve()
    if variant not in {'baseline', 'tuning'}:
        raise ValueError(variant)
    parent = root / MLP
    assert candidate.parent == parent and candidate.name.startswith(f'.pending-{variant}-')
    destination = parent / variant
    manifest_path = candidate / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest.get('completed_at_utc') and manifest.get('verification')
    assert all(value is True for value in manifest['verification'].values())
    assert manifest['output_folder'] == candidate.relative_to(root).as_posix()
    # Recheck both source and output bytes immediately before accepting the run.
    for name, expected in manifest['output_sha256'].items():
        assert (root/name).resolve().parent == candidate
        assert sha256(root/name) == expected, f'Output changed: {name}'
    for key in ['source_preparation_manifest', 'baseline_manifest']:
        if key in manifest:
            assert sha256(root/manifest[key]) == manifest[key+'_sha256'], f'Source changed: {key}'
    for key in ['source_artifact_sha256', 'reused_baseline_artifact_sha256']:
        for name, expected in manifest.get(key, {}).items():
            assert sha256(root/name) == expected, f'Source changed: {name}'
    assert not manifest.get('test_evaluation_performed')
    # No historical directory survives a successful publication. Temporary backups
    # let a locked workbook or other write failure restore the previous result.
    backups = [(destination, parent/f'.previous-{variant}'),
               (parent/'summary.xlsx', parent/'.previous-summary.xlsx')]
    if variant == 'baseline':
        # Tuning was derived from the old baseline, so it is set aside with it.
        backups.append((parent/'tuning', parent/'.previous-tuning'))
    assert not any(backup.exists() for _, backup in backups), 'An interrupted publication needs recovery before retrying.'
    original_json = {p.name: p.read_bytes() for p in candidate.glob('*.json')}
    old, new = candidate.relative_to(root).as_posix(), destination.relative_to(root).as_posix()
    moved = []
    promoted = False
    try:
        for name, raw in original_json.items():
            value = replace_path(json.loads(raw), old, new)
            (candidate/name).write_text(json.dumps(value, indent=2)+'\n')
        manifest = json.loads(manifest_path.read_text())
        manifest['output_sha256'] = {name: sha256(candidate/Path(name).name) for name in manifest['output_sha256']}
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        for current, backup in backups:
            if current.exists():
                current.rename(backup)
                moved.append((current, backup))
        candidate.rename(destination)
        promoted = True
        build_summary(root)
    except Exception:
        # Restore the previous result and leave the failed candidate for inspection.
        if promoted:
            (parent/'summary.xlsx').unlink(missing_ok=True)
            destination.rename(candidate)
        for current, backup in reversed(moved):
            backup.rename(current)
        for name, raw in original_json.items():
            (candidate/name).write_bytes(raw)
        raise
    for _, backup in moved:
        if backup.is_dir():
            shutil.rmtree(backup)
        else:
            backup.unlink()
    if variant == 'baseline':
        print('Accepted baseline. Previous dependent tuning was cleared; run notebook 09 to refresh it.')
    return destination
