# Dummy baselines

Open [summary.xlsx](summary.xlsx) for baseline and tuned results on both frozen
splits. It follows the combined random forest, XGBoost and SVR display: a results
overview, Training Guide, Settings, exact saved tables, search/selection records,
similarity and cliff diagnostics, and portable Evidence Files links.

A dummy ignores the molecule and predicts one constant pKi, learned from the
training labels: their mean or their median. It is the no-skill reference. Any
real model should clearly beat it, and R² near zero is expected. The baseline
saves both a mean and a median dummy for each split. The two-candidate search
(mean vs median) uses five training-only folds, with KFold for random and
GroupKFold for scaffold; it selected mean for random and median for scaffold.
Fitting is deterministic, so there is no seed and no multi-seed SD.

## How this summary was produced

No externally supplied workbook was imported, so there is no `import_validation.json`.
The summary is generated directly from the saved evidence by
`scripts/build_dummy_summary.py`, which checks every manifest checksum, the
tuning-to-baseline link, the preparation source, and the CV arithmetic and
selected candidates before writing.

The previous `run_03/` folder was reorganized into `files/`. CSV data, the
selected-parameter JSON, and model binaries are byte-identical to the originals.
Manifest paths and the tuning-to-baseline hash are updated for relocation.

## Evidence and reruns

- `files/baseline/`: original mean and median dummies, predictions, scores, training constants and manifest.
- `files/tuning/`: selected dummies, predictions, scores, full search records and manifest.
- `files/diagnostics/`: frozen validation cliff pairs and their source manifest.

The diagnostic pair set was rebuilt from this preparation's validation
fingerprints and targets and is byte-identical to the other models' pair sets:
Tanimoto >= 0.8 and absolute pKi difference >= 1.0. A constant predicts zero
difference for every pair, so every pair is a tie and counts as an incorrect
direction. That is the expected no-skill result, not an error. The dummy notebook
saves no per-variant similarity table; the workbook's Similarity Summary is
derived from the saved predictions instead.

Notebook 05 writes a temporary candidate, verifies both variants, then replaces
`files/` and `summary.xlsx` together. Failed tuning or report generation preserves
the previous complete result. Successful replacement removes temporary rollback
copies. Incomplete `.pending-*` folders never appear as accepted results.

Refresh only the workbook, without training:

```bash
python scripts/build_dummy_summary.py
```

Validation scores are development estimates; reserved tests remain unevaluated.
