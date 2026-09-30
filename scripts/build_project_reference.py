"""Verify saved evidence and refresh the model comparison and project file guide.

Run directly or from the final cell of any notebook. This does not download,
train, evaluate test sets, or modify the supplied presentation workbook.
Incomplete and older-dataset results are labeled in the model comparison.
"""
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, median, stdev
import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
from provenance_support import (
    latest_acquisition, current_selection, current_curation, current_preparation,
    load_review_record,
)
from fetch_models import require_models

ROOT = Path(__file__).resolve().parents[1]


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def read_csv(name):
    with (ROOT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def project_files():
    excluded = {".git", ".agents", ".codex", "__pycache__", ".ipynb_checkpoints", ".venv"}
    names = []
    for directory, subdirs, files in os.walk(ROOT):
        subdirs[:] = sorted(d for d in subdirs if d not in excluded and not d.startswith(('.pending-', '.previous-')))
        for name in sorted(files):
            if name.endswith(":Zone.Identifier") or name.startswith((".~", ".pending-", ".previous-")):
                continue
            names.append((Path(directory) / name).relative_to(ROOT).as_posix())
    return sorted(names)


def verify_saved_evidence():
    """Check bytes, notebook syntax, preserved payloads, and prediction arithmetic."""
    # Fitted models live outside git; explain how to fetch them before hashing.
    require_models(ROOT)
    checked = 0
    aliases = {"curation_manifest_sha256": "source_curation_manifest",
               "assignment_sha256": "assignment_file"}

    def check(name, expected):
        nonlocal checked
        assert (ROOT / name).is_file(), f"Missing hashed file: {name}"
        assert sha256(name) == expected, f"Checksum mismatch: {name}"
        checked += 1

    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
                    if "/" in key:
                        check(key, value)
                    elif key.endswith("_sha256"):
                        path_key = aliases.get(key, key[:-7])
                        if isinstance(obj.get(path_key), str):
                            check(obj[path_key], value)
                elif isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    for folder in ["provenance"]:
        for path in (ROOT / folder).rglob("*manifest.json"):
            if any(part.startswith(('.pending-', '.previous-')) for part in path.parts):
                continue
            # This historical log describes earlier locations, not the current contract.
            if path.name != "renaming_manifest.json":
                walk(read_json(path))
    for path in (ROOT / "notebooks").glob("*.ipynb"):
        for cell in read_json(path)["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=str(path))

    # Recompute metrics from stored predictions and check membership against frozen splits.
    metric_rows = 0
    for path in sorted((ROOT / "provenance/models").glob("**/metrics.csv")):
        if any(part.startswith(('.pending-', '.previous-')) for part in path.parts):
            continue
        if not path.with_name("manifest.json").is_file():
            continue
        manifest = read_json(path.with_name("manifest.json"))
        prep = read_json(manifest["source_preparation_manifest"])
        membership = defaultdict(set)
        for row in read_csv(Path(prep["output_folder"]) / "split_assignments.csv"):
            membership[row["split_strategy"], row["subset"]].add(row["rdkit_smiles"])
        curation = read_json(prep["source_curation_manifest"])
        targets = {r["rdkit_smiles"]: float(r["pki_target"]) for r in read_csv(
            Path(curation["output_folder"]) / "curated_structures.csv")}
        predictions = read_csv(path.with_name("predictions.csv"))
        metrics = read_csv(path)
        keys = [key for key in ["split_strategy", "subset", "model", "model_variant", "initialization_seed"]
                if key in metrics[0]]
        groups = defaultdict(list)
        for row in predictions:
            assert row["subset"] in {"train", "validation"}, "Test prediction found"
            assert math.isclose(float(row["observed_pki"]), targets[row["rdkit_smiles"]], abs_tol=1e-10)
            groups[tuple(row[key] for key in keys)].append(row)
        assert len(groups) == len(metrics)
        for metric in metrics:
            rows = groups[tuple(metric[key] for key in keys)]
            assert len(rows) == int(metric["n_structures"])
            assert len({r["rdkit_smiles"] for r in rows}) == len(rows)
            assert {r["rdkit_smiles"] for r in rows} == membership[metric["split_strategy"], metric["subset"]]
            y = [float(r["observed_pki"]) for r in rows]
            errors = [float(r["predicted_pki"]) - a for r, a in zip(rows, y)]
            for row, error in zip(rows, errors):
                for key, value in [("residual", error), ("absolute_error", abs(error)), ("squared_error", error**2)]:
                    assert math.isclose(float(row[key]), value, rel_tol=1e-9, abs_tol=1e-10), (path, key)
            average = mean(y)
            expected = {"mae_pki": mean(abs(e) for e in errors),
                        "rmse_pki": math.sqrt(mean(e*e for e in errors)),
                        "r2": 1 - sum(e*e for e in errors) / sum((a-average)**2 for a in y)}
            for key, value in expected.items():
                assert math.isclose(float(metric[key]), value, rel_tol=1e-9, abs_tol=1e-10), (path, key)
            metric_rows += 1
    return {"checksum_references": checked, "recomputed_metric_rows": metric_rows,
            "active_notebooks": len(list((ROOT / "notebooks").glob("*.ipynb")))}


def build_measurement_tables(accepted):
    from provenance_support import load_review_record
    curation = read_json(accepted["curation"])
    selection = read_json(curation["selection_manifest"])
    raw_folder = Path(curation["source_raw_folder"])
    activities = read_json(raw_folder / "chembl_cb2_activity.json")
    assays = read_json(raw_folder / "chembl_cb2_assay.json")
    documents = read_json(raw_folder / "chembl_cb2_documents.json")
    by_assay = {r["assay_chembl_id"]: r for r in assays}
    by_document = {r["document_chembl_id"]: r for r in documents}
    decisions = read_csv(Path(accepted["curation"]).parent / "measurement_decisions.csv")
    decision_by_id = {int(r["activity_id"]): r for r in decisions}
    assert len(decision_by_id) == len(decisions)
    reasons = {}
    quarantine_file = Path(curation["selection_manifest"]).parent / selection["quarantine_record"]
    book = load_review_record(ROOT / quarantine_file)
    for sheet, expected_count in selection["quarantine_sheets"].items():
        records = book[sheet].iter_rows(values_only=True)
        headers = next(records)
        loaded = [dict(zip(headers, values)) for values in records if any(v is not None for v in values)]
        assert len(loaded) == expected_count, sheet
        for row in loaded:
            key = int(row["activity_id"])
            assert key not in reasons
            reasons[key] = (row["exclusion_reason"], str(quarantine_file) + "#" + sheet)
    book.close()
    kept, removed = [], []
    for activity in activities:
        key = activity["activity_id"]
        decision = decision_by_id.get(key)
        assay = by_assay.get(activity.get("assay_chembl_id"), {})
        document = by_document.get(activity.get("document_chembl_id"), {})
        row = {k: activity.get(k) for k in ["activity_id", "molecule_chembl_id", "canonical_smiles",
               "standard_type", "standard_relation", "standard_value", "standard_units", "assay_chembl_id", "document_chembl_id"]}
        row.update(assay_description=assay.get("description"), assay_type=assay.get("assay_type"),
                   assay_organism=assay.get("assay_organism"), confidence_score=assay.get("confidence_score"),
                   paper_title=document.get("title"), year=document.get("year"), doi=document.get("doi"),
                   pubmed_id=document.get("pubmed_id"), rdkit_smiles=decision.get("rdkit_smiles") if decision else None,
                   pki=float(decision["pki"]) if decision and decision["pki"] else None,
                   source_activity_file=(raw_folder / "chembl_cb2_activity.json").as_posix())
        if decision and decision["status"] == "retained":
            row["decision_source"] = str(Path(accepted["curation"]).parent / "measurement_decisions.csv")
            kept.append(row)
        else:
            if activity.get("standard_type") != "Ki":
                reason, source = "Measurement type is not Ki (outside the modeling endpoint)", "notebooks/01_data_exploration.ipynb"
            elif decision:
                reason, source = decision["review_reason"], str(Path(accepted["curation"]).parent / "measurement_decisions.csv")
            else:
                assert key in reasons, f"Unaccounted raw Ki activity: {key}"
                reason, source = reasons[key]
            row.update(exclusion_reason=reason, decision_source=source)
            removed.append(row)
    kept_ids = {r["activity_id"] for r in kept}
    removed_ids = {r["activity_id"] for r in removed}
    assert kept_ids.isdisjoint(removed_ids) and len(kept_ids | removed_ids) == len(activities)
    assert len(kept) == curation["counts"]["retained_activities"]

    # Reconcile the final pKi aggregation directly from retained measurements.
    grouped = defaultdict(lambda: defaultdict(list))
    for row in kept:
        assert math.isclose(row["pki"], 9-math.log10(float(row["standard_value"])), abs_tol=1e-10)
        grouped[row["rdkit_smiles"]][row["assay_chembl_id"]].append(row["pki"])
    structures = read_csv(Path(accepted["curation"]).parent / "curated_structures.csv")
    assert set(grouped) == {r["rdkit_smiles"] for r in structures}
    for row in structures:
        target = median(median(values) for values in grouped[row["rdkit_smiles"]].values())
        assert math.isclose(target, float(row["pki_target"]), abs_tol=1e-10)
    # Show the bridge from each measurement to the final molecule's modeled target.
    final_by_structure = {row["rdkit_smiles"]: row for row in structures}
    for row in kept:
        final = final_by_structure[row["rdkit_smiles"]]
        row["final_molecule_pki_target"] = float(final["pki_target"])
        row["final_molecule_measurement_count"] = int(final["n_measurements"])

    funnel = []
    def stage(name, count, source, unit="measurements"):
        previous = funnel[-1]["remaining"] if funnel and funnel[-1]["unit"] == unit else None
        funnel.append({"stage": name, "unit": unit, "remaining": count,
                       "removed_at_step": previous-count if previous is not None else 0, "source": source})
    stage("Downloaded target activities", len(activities), str(raw_folder / "acquisition_manifest.json"))
    count = sum(r.get("standard_type") == "Ki" for r in activities)
    stage("Ki endpoint", count, "notebooks/01_data_exploration.ipynb")
    for sheet, label in [("Assay type", "Binding assay type B"), ("Target confidence", "Direct target confidence 9"), ("Assay organism", "Human assay organism")]:
        count -= selection["quarantine_sheets"][sheet]
        stage(label, count, str(quarantine_file) + "#" + sheet)
    assert count == selection["stage_counts"]["Human binding Ki before cleaning"]
    for label, count in selection["stage_counts"].items():
        if label != "Human binding Ki before cleaning":
            stage(label, count, curation["selection_manifest"])
    stage("After structure review", len(decisions)-curation["counts"]["structure_review_activities"], accepted["curation"])
    stage("After conflicting measurements review", len(kept), accepted["curation"])
    stage("Eligible structures before conflict review", curation["counts"]["eligible_structures"], accepted["curation"], "structures")
    stage("Final molecules (canonical structure keys)", len(structures), accepted["curation"], "structures")
    for records, field in [(assays, "assay_chembl_id"), (documents, "document_chembl_id")]:
        used = Counter(r[field] for r in kept)
        downloaded = Counter(r.get(field) for r in activities)
        for row in records:
            row["used_in_final_dataset"] = bool(used[row[field]])
            row["kept_measurements"] = used[row[field]]
            row["downloaded_measurements"] = downloaded[row[field]]
    return activities, assays, documents, kept, removed, funnel


DESCRIPTIONS = {
    "MASTER_REFERENCE.md": "Complete project file and folder guide; regenerated by the reporting script.",
    "RENAME_LOG.md": "Readable old-to-new paths and preservation decisions.",
    "README.md": "Guide to this folder and how to use its contents.",
    "AGENTS.MD": "Project instructions for coding assistants and notebook style.",
    ".gitattributes": "Protects hashed data and results from Git line-ending conversion.",
    ".gitignore": "Keeps caches, environments, and operating-system metadata out of Git.",
    "requirements-reporting.txt": "Spreadsheet dependency for review audits and MLP summaries; not a training environment.",
    "accepted_runs.json": "Explicit accepted curation, preparation, and model runs used by the reports.",
    "model_comparison.csv": "All accepted baseline and tuned models on both validation split protocols, with exact source paths.",
    "model_comparison.md": "Readable side-by-side model comparison, including MLP seed variation.",
    "CB2_data_provenance.xlsx": "Presentation workbook joining raw download, filtering, measurements, papers, settings, and model scores.",
    "build_project_reference.py": "Checks provenance and saved scores; refreshes the comparison and master guide without modifying the workbook.",
    "reorganization_20260929.json": "Every original file's location and checksums before and immediately after reorganization.",
    "renaming_manifest.json": "Historical September 24 rename and retention log; old locations are intentionally preserved.",
    "before_reorganization_20260929.zip": "Original metadata, notebooks, and docs before this reorganization; no duplicated raw data or models.",
    "before_renaming.zip": "Historical metadata and notebooks before the earlier September 24 rename.",
    "manifest.json": "Run completion record with source/output checksums, settings, verification results, and software versions.",
    "acquisition_manifest.json": "ChEMBL download date, version, target, API queries, and record counts.",
    "selection_manifest.json": "Exact-Ki selection rules, counts, and quarantine review-record reference.",
    "chembl_status.json": "ChEMBL service and database release metadata from the original download.",
    "chembl_cb2_target.json": "Identity and organism of the downloaded human CB2 receptor target.",
    "chembl_cb2_activity.json": "All unfiltered downloaded CB2 activity measurements, including non-Ki endpoints.",
    "chembl_cb2_assay.json": "Downloaded experiment descriptions, confidence, type, and organism.",
    "chembl_cb2_documents.json": "Downloaded paper titles, years, identifiers, and DOIs.",
    "cb2_ki_selected.csv": "Exact human binding Ki measurements selected for structure curation.",
    "cb2_ki_quarantine.xlsx": "Measurement-level exclusions from the original Ki selection stage.",
    "curated_structures.csv": "One final canonical structure and aggregated target pKi per molecule.",
    "measurement_decisions.csv": "Each selected measurement's structure, pKi, and retained or review decision.",
    "structure_summary.csv": "Per-structure measurement counts, disagreement, and aggregation diagnostics.",
    "review.xlsx": "Structure and measurement-conflict review from curation.",
    "fingerprints_and_targets.npz": "Frozen molecular fingerprint arrays, targets, and aligned structure keys.",
    "split_assignments.csv": "Frozen train, validation, and test membership for both split protocols.",
    "split_manifest.json": "Preparation checkpoint that freezes assignments before the final audit.",
    "fingerprint_groups.csv": "Summary of structures that share an identical fingerprint.",
    "fingerprint_group_members.csv": "Individual members and source measurements of identical-fingerprint groups.",
    "cross_split_fingerprint_pairs.csv": "Identical-fingerprint pairs that cross train/validation/test boundaries.",
    "heldout_training_neighbors.csv": "Nearest-training similarity for held-out structures; no model test performance.",
    "audit_summary.json": "Preparation audits covering fingerprint equivalence, splits, and chemical similarity.",
    "models.joblib.gz": "Original fitted estimators with lossless gzip compression; kept outside git, fetch with scripts/fetch_models.py.",
    "predictions.csv": "Saved observed and predicted training/validation pKi; reserved tests are absent.",
    "metrics.csv": "Per-model training and validation MAE, RMSE, and R-squared.",
    "training_summary.csv": "Training sample counts, timing, and model-specific information.",
    "training_history.csv": "Historical MLP training and outer-validation loss by epoch.",
    "validation_subgroups.csv": "Validation errors grouped by similarity to training structures.",
    "cv_folds.csv": "Training-only cross-validation fold assignments.",
    "cv_results.csv": "Cross-validation scores for every declared candidate setting.",
    "best_parameters.json": "Selected settings, effective model parameters, and validation scores.",
    "baseline_comparison.csv": "Baseline and selected-model metrics within this model family.",
    "configuration.json": "Internal split, architecture, optimization, seed, and stopping rules.",
    "internal_assignments.csv": "Fit/stopping membership entirely within each outer-training subset.",
    "internal_history.csv": "Baseline MLP fit and internal stopping losses by seed and epoch.",
    "selected_internal_history.csv": "Selected MLP candidate's fit and internal stopping losses.",
    "epoch_selections.csv": "Training duration selected from internal stopping loss for each final network.",
    "full_refit_history.csv": "Fresh full-training refit losses for the selected duration.",
    "seed_summary.csv": "Mean and sample standard deviation of scores across initialization seeds.",
    "subgroup_summary.csv": "Seed summaries of validation errors within similarity groups.",
    "candidate_scores.csv": "Per-seed internal scores for the fixed MLP candidate grid.",
    "candidate_summary.csv": "Aggregate internal scores used to compare MLP candidates.",
    "candidate_history.csv": "Epoch-level loss for each candidate's internal selection runs.",
    "selection.json": "Deterministic MLP configuration selection and its internal scores.",
    "cliff_pairs.csv": "Frozen exploratory validation pairs with high similarity and large observed pKi difference.",
    "cliff_molecule_groups.csv": "Model error for validation molecules participating in provisional activity cliffs.",
    "cliff_molecule_group_summary.csv": "Seed summary of errors for cliff-participating molecules.",
    "cliff_results.csv": "Per-seed predicted-versus-observed differences for provisional cliff pairs.",
    "cliff_summary.csv": "Aggregate provisional activity-cliff pair error across seeds.",
}


def description(name):
    base = Path(name).name
    if base in DESCRIPTIONS:
        return DESCRIPTIONS[base]
    if base.endswith(".ipynb"):
        book = read_json(name)
        title = "".join(book["cells"][0]["source"]).splitlines()[0].lstrip("# ")
        return ("Historical preserved notebook: " if "/archive/" in name else "Notebook: ") + title
    if base.endswith(".png"):
        return "Saved figure: " + base[:-4].replace("_", " ") + "."
    if base == ".gitkeep":
        return "Keeps an otherwise empty project folder in Git."
    raise ValueError(f"Add a plain-English description for {name}")


DESCRIPTIONS.update({
    "build_svr_summary.py": "Builds a combined SVR workbook with exact scores, kernel/settings explanations, diagnostics and evidence links.",
    "svr_artifacts.py": "Publishes baseline and tuned SVR results together with rollback if reporting fails.",
    "test_svr_reporting.py": "Checks SVR publication, hash links, incomplete or corrupt evidence rejection, and rollback.",
    "build_forest_summary.py": "Builds one readable baseline-and-tuned random forest summary with source links and derived diagnostics.",
    "forest_diagnostics.py": "Freezes validation cliff pairs and recomputes report-only cliff and similarity metrics.",
    "forest_artifacts.py": "Publishes a complete forest baseline+tuning execution and workbook, with rollback on failure.",
    "test_forest_reporting.py": "Checks forest diagnostics, incomplete-run rejection, publication and rollback.",
    "fetch_models.py": "Downloads fitted models from the GitHub Release and accepts them only if they match manifest checksums; also uploads new models.",
    "test_fetch_models.py": "Checks model download verification, corrupt-download rejection, and the missing-model instruction.",
    "import_validation.json": "Checksums and verification results for supplied model summary workbooks.",
    "test_mlp_artifacts.py": "Checks successful replacement, failed-candidate preservation and rollback of MLP results.",
    "build_mlp_summary.py": "Builds readable MLP workbooks with scores, training explanations, full tables and file links.",
    "mlp_artifacts.py": "Publishes one audited MLP execution; keeps current results safe until replacement succeeds.",
    "summary.xlsx": "Model overview, architecture, training settings, results and supporting tables for the current execution.",
    "test_provenance_workflow.py": "Tests that reruns use matching datasets and review records retain their values without extra Excel files.",
    "provenance_support.py": "Shared path matching and JSON review storage; prevents mixing incompatible pipeline stages.",
    "cb2_ki_quarantine.json": "Selection review rows and summary used by the pipeline audits.",
    "review.json": "Audited curation review rows and summary retained as supporting evidence.",
    "CB2_data_provenance.xlsx": "User-supplied eight-sheet presentation reference; maintained separately from pipeline reruns.",
})


def pipeline_state():
    """Newest completed acquisition and only matching downstream stages."""
    state, status = {}, []
    for key, function, notebook in [
        ("acquisition", latest_acquisition, "00"),
        ("selection", current_selection, "01"),
        ("curation", current_curation, "02"),
        ("preparation", current_preparation, "03"),
    ]:
        try:
            path = function(ROOT).as_posix()
            state[key] = path
            status.append({"stage": key, "status": "complete for newest download", "manifest": path, "next_action": ""})
        except FileNotFoundError as error:
            state[key] = None
            status.append({"stage": key, "status": "pending", "manifest": "", "next_action": f"Run notebook {notebook}. {error}"})
    return state, status


MODEL_FOLDERS = [
    ("Dummy", "baseline", "dummy_baselines", False),
    ("Dummy", "tuned", "dummy_baselines", True),
    ("Random forest", "baseline", "random_forest", False),
    ("Random forest", "tuned", "random_forest", True),
    ("XGBoost", "baseline", "xgboost", False),
    ("XGBoost", "tuned", "xgboost", True),
    ("Support vector regression", "baseline", "support_vector_regression", False),
    ("Support vector regression", "tuned", "support_vector_regression", True),
    ("Multilayer perceptron", "baseline", "multilayer_perceptron/baseline", False),
    ("Multilayer perceptron", "tuned", "multilayer_perceptron/tuning", False),
]


def latest_models(state, status):
    """Keep completed model results visible until their replacements complete.

    Each row identifies its preparation and dataset, so an old-dataset result is
    never silently presented as the outcome of a fresh acquisition or curation.
    """
    models = []
    for model, variant, folder, tuning_child in MODEL_FOLDERS:
        parent = ROOT / "provenance/models" / folder
        pattern = (("files/tuning/manifest.json" if tuning_child else "files/baseline/manifest.json") if folder in {"random_forest", "support_vector_regression"} else
                   "files/manifest.json" if folder.startswith("multilayer_perceptron/") else
                   "run_*/tuning/manifest.json" if tuning_child else "run_*/manifest.json")
        candidates = [p for p in parent.glob(pattern) if read_json(p).get("completed_at_utc")]
        label = f"{model} / {variant}"
        if not candidates:
            status.append({"stage": label, "status": "pending", "manifest": "", "next_action": "Run the model notebook."})
            continue
        path = max(candidates, key=lambda p: read_json(p)["completed_at_utc"])
        manifest = read_json(path)
        prep_path = manifest["source_preparation_manifest"]
        prep = read_json(prep_path)
        curation = read_json(prep["source_curation_manifest"])
        current = prep_path == state["preparation"] and manifest["source_preparation_manifest_sha256"] == sha256(prep_path)
        run = {"model": model, "variant": variant, "manifest": path.relative_to(ROOT).as_posix(),
               "source_preparation": prep_path, "dataset_snapshot": Path(curation["source_raw_folder"]).name,
               "pipeline_status": "current preparation" if current else "previous preparation; rerun pending"}
        models.append(run)
        status.append({"stage": label, "status": run["pipeline_status"], "manifest": run["manifest"],
                       "next_action": "" if current else "Preserved result; rerun on the new preparation when ready."})
    return models


def comparison_rows(models):
    rows = []
    for run in models:
        path = Path(run["manifest"]).parent / "metrics.csv"
        groups = defaultdict(list)
        for number, metric in enumerate(read_csv(path), 2):
            if metric["subset"] != "validation":
                continue
            model = "Dummy " + metric["model"].split("_")[-1] if "model" in metric else run["model"]
            groups[model, metric["split_strategy"]].append((number, metric))
        for (model, split), values in groups.items():
            row = {"model": model, "variant": run["variant"], "split_strategy": split,
                   "evaluation_subset": "validation", "n_structures": int(values[0][1]["n_structures"]),
                   "n_seeds": len(values)}
            for key in ["mae_pki", "rmse_pki", "r2"]:
                numbers = [float(metric[key]) for _, metric in values]
                row[key] = mean(numbers)
                row[key + "_sample_sd"] = stdev(numbers) if len(numbers) > 1 else None
            row.update(dataset_snapshot=run["dataset_snapshot"], pipeline_status=run["pipeline_status"],
                       source_metrics=path.as_posix(), source_csv_rows=", ".join(str(n) for n, _ in values),
                       source_manifest=run["manifest"], source_preparation=run["source_preparation"],
                       caveat="Development validation. Reserved tests unevaluated. Seed SD is not a confidence interval.")
            rows.append(row)
    return rows


def owner_manifest(name):
    """Identify the producing stage, without attributing incomplete runs to a parent."""
    path = ROOT / name
    for filename in ["manifest.json", "selection_manifest.json", "acquisition_manifest.json"]:
        candidate = path.parent / filename
        if candidate.is_file():
            return candidate.relative_to(ROOT).as_posix()
    if path.parent.name == "identical_fingerprints":
        candidate = path.parent.parent / "manifest.json"
        if candidate.is_file():
            return candidate.relative_to(ROOT).as_posix()
    return None


def write_comparison(rows):
    folder = ROOT / "results"
    folder.mkdir(exist_ok=True)
    lines = ["# Model comparison", "", "Saved baseline and tuned validation results. Each row identifies its dataset and whether it matches the newest preparation. "
             "Results remain visible until new runs complete. Reserved tests are unevaluated. MLP entries are means ± sample SD across seeds.", "",
             "The [provenance workbook](../provenance/CB2_data_provenance.xlsx) is the supplied presentation snapshot. Current exact scores and predictions are under `provenance/models/`; this comparison refreshes independently.", "",
             "| Model | Variant | Split | N | MAE | RMSE | R² | Dataset | Status |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for row in rows:
        values = [f"{row[k]:.4f}" + (f" ± {row[k+'_sample_sd']:.4f}" if row[k+'_sample_sd'] is not None else "") for k in ["mae_pki", "rmse_pki", "r2"]]
        lines.append(f"| {row['model']} | {row['variant']} | {row['split_strategy']} | {row['n_structures']:,} | "
                     + " | ".join(values) + f" | {row['dataset_snapshot']} | {row['pipeline_status']} |")
    if not rows:
        lines += ["", "No completed model results yet. Run the model notebooks after preparation."]
    (folder / "model_comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def build_master_reference():
    names = project_files()
    if "MASTER_REFERENCE.md" not in names:
        names.append("MASTER_REFERENCE.md")
    folders = defaultdict(list)
    for name in sorted(names):
        folders[Path(name).parent.as_posix()].append(name)
    lines = ["# Master reference", "", "Generated by `python scripts/build_project_reference.py`. "
             "All source data, review records, model artifacts, and diagnostics are under `provenance/`. "
             "The data provenance [Excel workbook](provenance/CB2_data_provenance.xlsx) is the supplied presentation snapshot and is not overwritten by reruns. "
             "Git internals, environments, caches, and Windows download tags are excluded.", "", "## Folders", "",
             "| Folder | Purpose |", "| --- | --- |"]
    directories = sorted({str(parent) for name in names for parent in Path(name).parents if str(parent) != "."})
    purpose = {"notebooks": "Ten ordered scientific pipeline notebooks.", "scripts": "Storage helpers, evidence checks, and Markdown reference updates.",
               "tests": "Dataset handoff and review-record checks without downloads or model training.",
               "results": "Current headline comparison; supporting evidence is under provenance/models/.", "provenance": "All source and supporting evidence, plus data and model summary workbooks.",
               "raw": "Original ChEMBL downloads, unchanged.", "selection": "Selected measurements and exclusion records.",
               "curation": "Final molecules, decisions, structure summaries, review records, and curation metadata.",
               "preparation": "Fingerprint arrays, split assignments, similarity diagnostics, and preparation metadata.",
               "identical_fingerprints": "Identical-fingerprint groups, members, and boundary crossings.",
               "models": "Model families with fitted models, predictions, settings, and diagnostics.",
               "files": "Detailed artifacts for the accepted execution; open the model or variant summary.xlsx first.",
               "baseline": "Baseline model executions.", "tuning": "Training-only searches, selected settings, and refits."}
    for directory in directories:
        base = Path(directory).name
        text = purpose.get(base, "Recorded execution; consult its manifest for dataset and settings." if base.startswith("run_") else base.replace("_", " ") + " supporting files.")
        lines.append(f"| `{directory}/` | {text} |")
    for folder, files in sorted(folders.items()):
        lines += ["", "## " + ("Project root" if folder == "." else folder), "", "| File | Description |", "| --- | --- |"]
        lines += [f"| [{Path(name).name}]({name}) | {description(name).replace('|', '/')} |" for name in files]
    (ROOT / "MASTER_REFERENCE.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def build_reference(check_only=False):
    checks = verify_saved_evidence()
    state, status = pipeline_state()
    models = latest_models(state, status)
    comparisons = comparison_rows(models)
    # Check the scientific source records independently of the presentation workbook.
    for path in (ROOT / "provenance/curation").glob("**/manifest.json"):
        if read_json(path).get("completed_at_utc"):
            build_measurement_tables({"curation": path.relative_to(ROOT).as_posix()})
    if not check_only:
        write_comparison(comparisons)
        build_master_reference()
    # The supplied workbook is maintained by the user, so reruns never rewrite it.
    summary = {**checks, "model_comparison_rows": len(comparisons),
               "current_pipeline": state, "reports_written": not check_only,
               "presentation_workbook": "preserved unchanged"}
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify saved scientific evidence without updating the Markdown reports.")
    args = parser.parse_args()
    build_reference(check_only=args.check)


if __name__ == "__main__":
    main()
