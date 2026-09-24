# Processed data and saved model runs

Start with this guide. Each stage has its own folder. The accepted execution has a numbered
run folder, and `manifest.json` records its dates, inputs, settings, checks, and outputs.
Only one accepted complete run per notebook is retained. Acceptance depends on a
complete, verified implementation, not on whether the scores improved. Run numbers
identify executions; they are not model ranks.

## Current results

These are the retained runs after removing superseded development outputs. This table is an explicit
project checkpoint, not an automatic "latest folder" lookup.

| Stage | Current folder | Contents |
| --- | --- | --- |
| Curation | [curation/run_05](curation/run_05/) | Curated structures, measurement decisions, review workbook |
| Modeling preparation | [modeling_preparation/run_01](modeling_preparation/run_01/) | Fingerprints, targets, frozen splits, similarity audits |
| Random forest baseline | [random_forest/run_03](random_forest/run_03/) | Fitted baseline forests, training and validation predictions, metrics |
| Random forest search | [random_forest/run_03/tuning](random_forest/run_03/tuning/) | CV results, selected settings and forests, predictions, baseline comparison |
| Mean and median baselines | [dummy_baselines/run_03](dummy_baselines/run_03/) | Four fitted constants, predictions, metrics |
| Dummy strategy search | [dummy_baselines/run_03/tuning](dummy_baselines/run_03/tuning/) | CV selection of mean or median for each split protocol |

The dependency chain is `curation/run_05` → `modeling_preparation/run_01` → each
model run. Each model's `tuning` folder also points to its own baseline manifest.
No model run includes predictions or performance evaluation on its protocol's reserved test set.

## What happened when we reran the forest notebook?

The earlier baseline executions were `run_01` and `run_02`. The full execution after
adding hyperparameter search is `run_03`: it trained fresh baseline models and then
searched and refitted selected models in `run_03/tuning`. The superseded runs have
now been removed for both forests and dummies. The accepted baseline and tuning
results remain together. This cleanup did not retrain any model.

## What the filenames mean

Names repeat inside different stage/run folders because their location supplies the context.

| File | Meaning |
| --- | --- |
| `manifest.json` | Completion record: inputs, file hashes, settings, dates, versions, checks, and outputs; fields vary by stage |
| `curated_structures.csv` | One retained molecular structure and its target pKi per row |
| `measurement_decisions.csv` | Individual measurements and their curation decisions |
| `structure_summary.csv` | Counts and disagreement diagnostics for eligible structures |
| `review.xlsx` | Human-readable curation review workbook |
| `fingerprints_and_targets.npz` | NumPy arrays containing molecular fingerprint bits, pKi targets, and structure keys |
| `split_assignments.csv` | Each structure's train/validation/test membership under both protocols |
| `split_manifest.json` | Earlier checkpoint freezing the split before the preparation audit completes |
| `fingerprint_groups.csv`, `fingerprint_group_members.csv` | Groups of structures with identical fingerprints and their members |
| `cross_split_fingerprint_pairs.csv` | Identical-fingerprint pairs crossing subset boundaries |
| `heldout_training_neighbors.csv` | Prepared nearest-training similarity diagnostics for held-out structures |
| `audit_summary.json` | Summary of fingerprint and split-boundary audits |
| `models.joblib.gz` | Fitted models with lossless gzip compression; load directly with `joblib.load` |
| `training_summary.csv` | Training counts and timing, plus model-specific details |
| `predictions.csv` | Observed and predicted training/validation pKi with errors |
| `metrics.csv` | MAE, RMSE, and R² summaries |
| `validation_subgroups.csv` | Forest validation errors grouped by similarity to training molecules |
| `cv_folds.csv` | Training structures assigned to internal CV validation folds |
| `cv_results.csv` | Scores for every searched parameter combination |
| `best_parameters.json` | Selected parameters, effective settings, CV score, and timing |
| `baseline_comparison.csv` | Baseline and CV-selected model metrics together |

A single model file can contain multiple estimators. Forest baseline and tuning
files each contain two forests, one per split protocol. Dummy baseline files contain
four estimators (mean and median under both protocols); dummy tuning files contain two.

## What provenance means here

Provenance is the record of **where a result came from and how it was produced**.
The files save the data and fitted models; manifests connect them to their source
files, settings, software versions, dates, and verification results. A SHA-256 hash
is a fingerprint of a file's bytes: it lets us detect a change even if the filename
stays the same. Tuning also saves the fold assignments and all candidate scores.

The manifest is not a copy of the data, model, or complete source code. Most original
runs recorded a Git commit and whether uncommitted changes existed. Those fields alone
cannot recover the exact uncommitted notebook code from an earlier execution.
The original run manifests retain their actual recorded metadata; missing history
has not been invented. Committing the notebooks with their associated run records
will make future project checkpoints easier to recover.

## Retention and maintenance records

Superseded curation runs, forest/dummy baseline runs, the original combined modeling
experiment, and its duplicate ZIP have been removed. Their removal did not change
any retained dataset, split, prediction, or score. The current model files were
losslessly compressed; their decoded bytes match the original serialized models.
All ten estimators were loaded and checked against their saved predictions.

- `archive/before_renaming.zip` is a small provenance backup containing only the
  original JSON metadata and notebook snapshots from before the naming cleanup.
  It contains no old model or dataset copies.
- [renaming_manifest.json](renaming_manifest.json) records the original rename,
  marks which files were subsequently removed, and records current file locations
  and checksums. Its `retention_cleanup` section records the compression and cleanup.
  Earlier verification fields describe the state at the time of the rename.
- Current run manifests refer to the compressed models and their new checksums.
  Model dates and scientific settings retain their original execution values.
  Historical workbook contents still show the original execution identifier.

The accepted models are small enough after compression to accompany the current
notebooks, reports, and provenance in ordinary Git. These are complete run records;
no required fitted-model file is intentionally left out of the staging set.

## Future reruns

The notebooks create a candidate folder above the highest saved run number and refuse to
overwrite an existing folder. A full forest rerun will create `random_forest/run_04`
and its `tuning` subfolder; the dummy notebook will create `dummy_baselines/run_04`.
Dates remain in the manifests. Candidate creation does not automatically accept
a run or delete earlier results. Running only the tuning export cell a second time
in the same baseline run stops rather than replacing the first tuning export.

A completion `manifest.json` is written only after verification passes. A folder
without that manifest may be an interrupted execution. A baseline manifest confirms
the baseline stage; the separate tuning manifest confirms the search stage.

Downstream notebooks use explicit input manifests. A new curation or preparation
run does not automatically change their inputs. Update the input path and this
current-results table deliberately when adopting a new completed run. Re-running
only a save cell saves the model currently in memory; a fresh full execution is
needed when the intention is to retrain.

After a revised notebook completes all verification, deliberately accept its new
run and remove the superseded development run. For model notebooks, wait for both
the baseline and tuning manifests. If execution fails, keep the previous accepted
run. Retain the required upstream dataset and split files. When an upstream dataset
or split changes, rebuild and verify its downstream model runs before retiring
inputs they still reference. Update the table above, then stage the notebook and
complete run records together. Keep the test sets reserved until model choices
and the final refit procedure are settled.
