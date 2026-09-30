"""Publish one complete baseline+tuning dummy-baseline execution with its combined workbook."""
from pathlib import Path
import json
import shutil
import tempfile
from provenance_support import sha256
from mlp_artifacts import replace_path

DUMMY=Path('provenance/models/dummy_baselines')


def new_dummy_candidate(root):
    parent=Path(root)/DUMMY;parent.mkdir(parents=True,exist_ok=True)
    candidate=Path(tempfile.mkdtemp(prefix='.pending-',dir=parent))
    (candidate/'baseline').mkdir()
    return candidate


def publish_dummy(root,candidate):
    """Replace both variants together only after both scientific audits succeed.

    A failed tuning stage leaves the accepted baseline and tuning untouched. A
    workbook-write failure rolls back both variants and their previous workbook.
    """
    from build_dummy_summary import build_summary
    root,candidate=Path(root).resolve(),Path(candidate).resolve();parent=root/DUMMY
    assert candidate.parent==parent and candidate.name.startswith('.pending-')
    original={p.relative_to(candidate):p.read_bytes() for p in candidate.rglob('*.json')}
    records={v:json.loads((candidate/v/'manifest.json').read_text()) for v in ['baseline','tuning','diagnostics']}
    for variant,m in records.items():
        if variant!='diagnostics':
            assert m.get('completed_at_utc') and m.get('verification') and all(v is True for v in m['verification'].values())
            assert not m.get('test_evaluation_performed')
        assert m['source_preparation_manifest']==records['baseline']['source_preparation_manifest']
        for key in ['source_preparation_manifest','baseline_manifest']:
            if key in m:assert sha256(root/m[key])==m[key+'_sha256'],f'Source changed: {key}'
        for key in ['output_sha256','source_artifact_sha256']:
            for name,expected in m.get(key,{}).items():
                if key=='output_sha256':assert (root/name).resolve().is_relative_to(candidate)
                assert sha256(root/name)==expected,f'Evidence changed: {name}'
    assert records['tuning']['baseline_manifest']==(candidate/'baseline/manifest.json').relative_to(root).as_posix()
    destination=parent/'files';backup=parent/'.previous-files';old_book=parent/'.previous-summary.xlsx'
    assert not backup.exists() and not old_book.exists(),'Recover an interrupted publication before retrying.'
    moved=[];promoted=False
    try:
        old,new=candidate.relative_to(root).as_posix(),destination.relative_to(root).as_posix()
        for path,raw in original.items():
            value=json.loads(raw)
            relocated=replace_path(value,old,new)
            # Keep path-independent parameter files byte-identical to their source.
            if relocated != value:
                (candidate/path).write_text(json.dumps(relocated,indent=2)+'\n')
        # Update manifest hashes in dependency order after path changes.
        for variant in ['baseline','tuning','diagnostics']:
            path=candidate/variant/'manifest.json';m=json.loads(path.read_text())
            m['output_sha256']={name:sha256(candidate/Path(name).relative_to(destination.relative_to(root))) for name in m['output_sha256']}
            if variant=='tuning':m['baseline_manifest_sha256']=sha256(candidate/'baseline/manifest.json')
            path.write_text(json.dumps(m,indent=2)+'\n')
        for source,target in [(destination,backup),(parent/'summary.xlsx',old_book)]:
            if source.exists():source.rename(target);moved.append((source,target))
        candidate.rename(destination);promoted=True
        build_summary(root)
    except Exception:
        if promoted:
            (parent/'summary.xlsx').unlink(missing_ok=True);destination.rename(candidate)
        for source,target in reversed(moved):target.rename(source)
        for path,raw in original.items():(candidate/path).write_bytes(raw)
        raise
    for _,path in moved:
        if path.is_dir():shutil.rmtree(path)
        else:path.unlink()
    return destination
