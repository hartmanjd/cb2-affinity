# XGBoost

Open [summary.xlsx](summary.xlsx) for baseline and tuned results on both frozen
splits. It follows the combined random forest and SVR display: a results overview,
Training Guide, Settings, exact saved tables, search/selection records,
similarity and cliff diagnostics, and portable Evidence Files links.

Both variants use gradient-boosted trees (`reg:squarederror`, `tree_method="hist"`)
on 2,048 binary Morgan fingerprint bits. The baseline uses 250 trees, max_depth=4,
learning_rate=0.05, subsample=0.9 and colsample_bytree=0.8; the selected model
uses learning_rate=0.1, max_depth=4 and 250 trees for each split. The bounded
eight-candidate search uses five training-only folds, with KFold for random and
GroupKFold for scaffold. No early stopping is used, so validation labels never
influence fitting. Each variant has one fitted booster per split with fixed
random_state 42 for row and bit subsampling; there is no multi-seed SD.

## How this summary was produced

Unlike the random forest and SVR folders, no externally supplied workbook was
imported, so there is no `import_validation.json`. The summary is generated
directly from the saved evidence by `scripts/build_xgboost_summary.py`, which
checks every manifest checksum, the tuning-to-baseline link, the preparation
source, and the CV arithmetic and selected candidates before writing.

The previous `run_02/` folder was reorganized into `files/`. CSV data, the
selected-parameter JSON, and model binaries are byte-identical to the originals.
Manifest paths and the tuning-to-baseline hash are updated for relocation.

## Evidence and reruns

- `files/baseline/`: original baseline models, predictions, scores, training details and manifest.
- `files/tuning/`: selected models, predictions, scores, full search records and manifest.
- `files/diagnostics/`: frozen validation cliff pairs and their source manifest.

The diagnostic pair set was rebuilt from this preparation's validation
fingerprints and targets and is byte-identical to the random forest and SVR pair
sets: Tanimoto >= 0.8 and absolute pKi difference >= 1.0. Predicted ties count as
incorrect directions. These descriptive diagnostics never select candidates.

Notebook 06 writes a temporary candidate, verifies both variants, then replaces
`files/` and `summary.xlsx` together. Failed tuning or report generation preserves
the previous complete result. Successful replacement removes temporary rollback
copies. Incomplete `.pending-*` folders never appear as accepted results.

Refresh only the workbook, without training:

```bash
python scripts/build_xgboost_summary.py
```

Validation scores are development estimates; reserved tests remain unevaluated.
