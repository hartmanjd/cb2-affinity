# CB2 affinity

Human CB2 binding-affinity dataset curation and model comparison.

Start with [the data provenance workbook](provenance/CB2_data_provenance.xlsx).
It shows where every number in the dataset came from, with sheets for the
Overview, Filter Funnel, Removal Reasons, Kept Measurements, Removed Measurements,
Assays, Documents, Pipeline Settings, and Evidence Files. It is generated from the
current pipeline files and refreshed by each notebook's final cell, so it always
matches the data. For model results and file locations, use the links below. All
supporting data and model artifacts remain under `provenance/`.

[XGBoost summary](provenance/models/xgboost/summary.xlsx) · [SVR summary](provenance/models/support_vector_regression/summary.xlsx) · [Random forest summary](provenance/models/random_forest/summary.xlsx) · [MLP summary](provenance/models/multilayer_perceptron/summary.xlsx) · [Dummy summary](provenance/models/dummy_baselines/summary.xlsx) · [Model comparison](results/model_comparison.md) · [Every file explained](MASTER_REFERENCE.md)

## Folders

| Folder | Purpose |
| --- | --- |
| `notebooks/` | The eleven scientific notebooks, in execution order |
| `provenance/` | Data provenance workbook, model summaries, and all supporting files |
| `results/` | A readable headline model comparison |
| `scripts/` | Shared storage helpers, evidence checks, and Markdown reference updates |
| `tests/` | Automated checks for dataset handoffs, publication, rollback, and model downloads |

There are no archive folders. The old `curation/run_05` folder is removed; the
supporting evidence for the preserved model results is now directly under
`provenance/curation/`. The original download and all saved model predictions and
scores are preserved. Stage-specific review records remain as JSON files for the pipeline audits.
The data provenance workbook is a generated report, not a pipeline input.

## Fitted models

The fitted estimators (`models.joblib.gz`, about 100 MB in total) are not stored
in git. Git keeps every version of a committed file forever, so each retrain made
the repository, pushes, and clones larger. Instead they are attached to the
[`fitted-models` GitHub Release](https://github.com/hartmanjd/cb2-affinity/releases/tag/fitted-models).
Each run manifest, which *is* committed, still records every model's SHA-256
checksum, so the provenance chain is unchanged.

After cloning, download and verify them once (standard library only):

```bash
python scripts/fetch_models.py
```

A download is kept only if its bytes match the checksum in its manifest, and an
existing local file that differs is never overwritten. Report builders and
notebook 09 stop with this instruction when models are missing. Release file
names include part of the checksum, so older commits can still fetch the exact
models they recorded.

**After retraining** (maintainer step): the notebook writes new models locally
and records their checksums. Publish them *before* pushing that commit, or other
people cannot fetch them. This requires the GitHub CLI (`gh`) and only uploads
files the release does not already have:

```bash
python scripts/fetch_models.py --upload
```

Models committed before this change remain in the git history; untracking them
stops future growth but does not shrink existing clones.

## Fresh run

Open this Linux checkout in VS Code: `/home/justin/code/cb2-affinity` (WSL: Ubuntu).
All saved results were produced on Linux with the exact versions in
`requirements.txt` and Python 3.14.6 (`.python-version`). Create that environment
once with [uv](https://docs.astral.sh/uv/), then select `.venv` as the notebook
kernel in VS Code:

```bash
uv venv --python 3.14.6 .venv
```

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

Notebook checks compare the installed versions **and the operating system** with
the preparation run. Tree models (random forest, XGBoost) can resolve near-tied
splits differently on Windows and Linux, so results are only exactly
reproducible on the same OS. If you change packages or OS, rerun notebook 03
onward rather than bypassing the checks.

In a new clone, first run `python scripts/fetch_models.py` (see [Fitted models](#fitted-models)).
Each notebook's final cell verifies every recorded checksum, including the models.

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
11. `10_noise_ceiling.ipynb` — how close each model gets to the measurement-noise floor (no training).

Each notebook finds the completed upstream stage for the current acquisition.
No manual edits to an old run number are needed. Missing or incompatible stages
stop with an instruction to run the prerequisite. Candidate outputs receive a
new candidate folder and are published only when verification and the completion
manifest succeed. Every model folder has one combined `summary.xlsx` for its baseline
and tuned results. Dummy, random forest, XGBoost, and SVR keep it beside `files/baseline/`,
`files/tuning/`, and `files/diagnostics/`; notebooks 05, 04, 06, and 07 replace their
baseline and tuning together after the complete experiment passes. The MLP keeps it
beside `baseline/` and `tuning/`, because notebooks 08 and 09 publish those variants
separately and each rebuilds the combined workbook. A successful MLP baseline replacement
clears dependent tuning results; run notebook 09 to regenerate them. Current model results remain visible until replacements finish;
the model comparison identifies older-dataset/preparation results explicitly.

The final cell of every notebook verifies evidence and refreshes the model
comparison, the data provenance workbook, and the master reference. Saved notebook outputs remain
from the previous execution until you rerun them; old printed paths are historical.

## How good can a model get?

Repeat lab measurements of the same molecule disagree, so no model can reliably
beat that measurement noise. Notebook 10 uses a published estimate for ChEMBL Ki
data (about 0.54 pKi) and plots every model between "no skill" and that floor:
[results/noise_ceiling.png](results/noise_ceiling.png).

## Refresh the generated references

With the dependency in `requirements-reporting.txt` installed and the
[fitted models](#fitted-models) fetched:

```bash
python scripts/build_project_reference.py --check
python scripts/build_project_reference.py
```

These commands verify or report saved evidence without downloading data or fitting
models. The report builder checks checksums, notebook syntax, prediction metrics,
split membership, measurement accounting, and final pKi aggregation.

All comparisons are development-validation results. Reserved test sets remain
unevaluated; deleting archives does not undo earlier use of validation labels.
