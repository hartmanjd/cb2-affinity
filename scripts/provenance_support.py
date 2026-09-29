"""Small shared helpers for locating matching notebook stages and review records.

The notebooks still contain the scientific calculations and their checks. These
helpers keep their storage conventions in one place and prevent mixing datasets.
"""
from pathlib import Path
import hashlib
import json


def read_json(root, path):
    return json.loads((Path(root) / path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def latest_acquisition(root):
    root = Path(root)
    paths = list((root / "provenance/raw").glob("*/acquisition_manifest.json"))
    complete = [p for p in paths if read_json(root, p).get("completed_at_utc")]
    if not complete:
        raise FileNotFoundError("Run notebook 00 first: no completed ChEMBL download exists.")
    return max(complete, key=lambda p: read_json(root, p)["retrieved_at_utc"]).relative_to(root)


def current_selection(root):
    raw = latest_acquisition(root)
    root = Path(root)
    folder = root / "provenance/selection" / raw.parent.name
    # A repeated selection gets a new candidate folder so existing models keep
    # their original source evidence. The direct manifest is the preserved run.
    candidates = []
    for path in folder.glob("**/selection_manifest.json"):
        record = read_json(root, path)
        if path.parent != folder and not record.get("completed_at_utc"):
            continue
        if record.get("source_manifest") == raw.as_posix():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("Run notebook 01 for the newest completed download first.")
    path = max(candidates, key=lambda p: read_json(root, p).get("completed_at_utc", "")).relative_to(root)
    manifest = read_json(root, path)
    assert manifest["source_manifest"] == raw.as_posix(), "Selection belongs to a different download."
    return path


def matching_stage(root, folder, source_key, source, hash_key=None):
    """Choose a completed stage tied to the exact current upstream manifest bytes."""
    root = Path(root)
    source = Path(source)
    expected_hash = sha256(root / source)
    candidates = []
    for path in (root / folder).glob("**/manifest.json"):
        if any(part.startswith(('.pending-', '.previous-')) for part in path.parts):
            continue
        # A model's tuning manifest is a different stage from its baseline.
        if "tuning" in path.relative_to(root / folder).parts[:-1]:
            continue
        record = read_json(root, path)
        if not record.get("completed_at_utc") or record.get(source_key) != source.as_posix():
            continue
        if hash_key:
            actual = record.get(hash_key)
        else:
            actual = record.get("file_sha256", {}).get(source.as_posix())
        if actual == expected_hash:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No completed {folder} stage matches {source}; run its upstream notebook first.")
    return max(candidates, key=lambda p: read_json(root, p)["completed_at_utc"]).relative_to(root)


def current_curation(root):
    return matching_stage(root, "provenance/curation", "selection_manifest", current_selection(root))


def current_preparation(root):
    source = current_curation(root)
    root = Path(root)
    candidates = []
    for path in (root / "provenance/preparation").glob("**/manifest.json"):
        record = read_json(root, path)
        if (record.get("completed_at_utc") and record.get("source_curation_manifest") == source.as_posix()
                and record.get("source_sha256", {}).get(source.as_posix()) == sha256(root / source)):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("Run notebook 03: no completed preparation matches the current curation.")
    return max(candidates, key=lambda p: read_json(root, p)["completed_at_utc"]).relative_to(root)


def current_mlp_baseline(root):
    return matching_stage(root, "provenance/models/multilayer_perceptron/baseline",
                          "source_preparation_manifest", current_preparation(root),
                          "source_preparation_manifest_sha256")


def save_review_record(workbook, path):
    """Save review rows as JSON evidence for audits and reader-facing reports.

    The in-memory workbook retains the notebook's existing independent row/value
    checks. Saving its rows avoids another separate Excel file for each stage.
    """
    sheets = []
    for sheet in workbook:
        sheets.append({"name": sheet.title, "rows": list(sheet.values),
                       "freeze_panes": sheet.freeze_panes,
                       "auto_filter": sheet.auto_filter.ref})
    Path(path).write_text(json.dumps({"sheets": sheets}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_review_record(path, **_options):
    """Recreate the saved review table in memory for the notebook's original audits."""
    from openpyxl import Workbook
    workbook = Workbook()
    workbook.remove(workbook.active)
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    for saved in record["sheets"]:
        sheet = workbook.create_sheet(saved["name"])
        for number, row in enumerate(saved["rows"], 1):
            sheet.append(row)
            for column, value in enumerate(row, 1):
                if isinstance(value, str):
                    sheet.cell(number, column).data_type = "s"
        sheet.freeze_panes = saved.get("freeze_panes")
        sheet.auto_filter.ref = saved.get("auto_filter")
    return workbook
