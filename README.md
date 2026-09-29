# CB2 affinity

Human CB2 binding-affinity dataset curation and model comparison.

Start with [the data provenance workbook](provenance/CB2_data_provenance.xlsx).
It is the supplied, human-readable presentation reference, with eight sheets:
Overview, Download Record, Filter Funnel, Kept Measurements, Quarantine, Assays,
Documents, and Pipeline Settings. It retains the earlier snapshot's paths and
results exactly as supplied; it does not update automatically when notebooks run.
For current results and file locations, use the links below. All supporting data
and model artifacts remain under `provenance/`.

[MLP summaries](provenance/models/multilayer_perceptron/README.md) · [Model comparison](results/model_comparison.md) · [Every file explained](MASTER_REFERENCE.md)

## Folders

| Folder | Purpose |
| --- | --- |
| `notebooks/` | The ten scientific notebooks, in execution order |
| `provenance/` | Data provenance workbook, model summaries, and all supporting files |
| `results/` | A readable headline model comparison |
| `scripts/` | Shared storage helpers, evidence checks, and Markdown reference updates |

There are no archive folders. The old `curation/run_05` folder is removed; the
supporting evidence for the preserved model results is now directly under
`provenance/curation/`. The original download and all saved model predictions and
scores are preserved. Stage-specific review records remain as JSON files for the pipeline audits.
The supplied workbook is a reference document, not a pipeline input.

## Fresh run

Open this Linux checkout in VS Code: `/home/justin/code/cb2-affinity` (WSL: Ubuntu).
Use the same Python environment throughout the run, with NumPy, pandas, RDKit,
scikit-learn, joblib, matplotlib, XGBoost, and openpyxl available. Notebook checks
still compare recorded software versions; rerun preparation before fitting in a
new environment rather than bypassing the checks.

Run each notebook top to bottom, in order:

1. `00_data_acquisition.ipynb` — fetch the latest available ChEMBL release and human CB2 records.
2. `01_data_exploration.ipynb` — select Ki measurements from that completed download.
3. `02_data_curation.ipynb` — curate structures, decisions, and target pKi.
4. `03_modeling_preparation.ipynb` — freeze fingerprints and random/scaffold splits.
5. `04_random_forest_evaluation.ipynb` — forest baseline and tuning.
6. `05_dummy_baselines.ipynb` — mean/median baselines and selection.
7. `06_xgboost.ipynb` — boosted-tree baseline and tuning.
8. `07_support_vector_regression.ipynb` — SVR baseline and tuning.
9. `08_mlp_baseline.ipynb` — corrected internal stopping and full-training refits.
10. `09_mlp_tuning.ipynb` — bounded MLP search and refits.

Each notebook finds the completed upstream stage for the current acquisition.
No manual edits to an old run number are needed. Missing or incompatible stages
stop with an instruction to run the prerequisite. Candidate outputs receive a
new candidate folder and are published only when verification and the completion
manifest succeed. MLP results use a single `files/` folder per baseline/tuning
variant, beside a generated `summary.xlsx`. A successful baseline replacement
clears dependent tuning results; run notebook 09 to regenerate them. Current model results remain visible until replacements finish;
the model comparison identifies older-dataset/preparation results explicitly.

The final cell of every notebook verifies evidence and refreshes the model
comparison and master reference. It never modifies the presentation workbook.
Update the workbook separately when you want it to reflect a fresh run. Saved notebook outputs remain
from the previous execution until you rerun them; old printed paths are historical.

## Refresh the generated references

With the dependency in `requirements-reporting.txt` installed:

```bash
python scripts/build_project_reference.py --check
python scripts/build_project_reference.py
```

These commands verify or report saved evidence without downloading data or fitting
models. The report builder checks checksums, notebook syntax, prediction metrics,
split membership, measurement accounting, and final pKi aggregation.

All comparisons are development-validation results. Reserved test sets remain
unevaluated; deleting archives does not undo earlier use of validation labels.
