# Support vector regression

Open [summary.xlsx](summary.xlsx) for baseline and tuned results on both frozen
splits. It follows the combined random forest display: a results overview,
Training Guide, Settings, exact saved tables, search/selection records,
similarity and cliff diagnostics, and portable Evidence Files links.

Both variants use an RBF kernel on 2,048 unscaled binary Morgan fingerprint bits.
The baseline uses C=1, epsilon=0.1 and gamma="scale"; the selected model uses
C=10, epsilon=0.1 and gamma="scale" for each split. The bounded eight-candidate
search uses five training-only folds, with KFold for random and GroupKFold for
scaffold. SVR fitting is deterministic and has no random_state parameter;
random-protocol KFold shuffling uses seed 42. There is no multi-seed SD.

## Verified import

The supplied `support_vector_regression_combined_summary.xlsx` passed its numeric
audit: 19 evidence checksum entries, 18,602 saved CSV rows, 735 JSON settings,
eight recomputed metric rows, and 270 derived diagnostic rows. Cover values,
model parameters, selected candidates, and CV arithmetic also matched. The
source workbook has tables rather than embedded chart or image objects.

Two Training Guide rows incorrectly called SVR's determinism a fixed random_state.
The installed summary corrects this wording and explains the separate seeded CV
splitter. It uses the existing forest summary's display and links to the new file
locations. The original download remains unchanged. [The import record](files/import_validation.json)
records its checksum, checks performed, corrections, and original-to-current paths.

## Evidence and reruns

- `files/baseline/`: original baseline models, predictions, scores, training details and manifest.
- `files/tuning/`: selected models, predictions, scores, full search records and manifest.
- `files/diagnostics/`: frozen validation cliff pairs and their source manifest.

CSV data and model binaries are preserved byte for byte. Manifest paths and the
tuning-to-baseline hash are updated for relocation. The diagnostic pair set was
copied from the verified MLP pair set for this same preparation, so the workbook
can be refreshed independently of later MLP changes. Notebook 07 reruns create
the same pairs directly from their own validation fingerprints and targets:
Tanimoto >= 0.8 and absolute pKi difference >= 1.0. Predicted ties count as
incorrect directions. These descriptive diagnostics never select candidates.

Notebook 07 writes a temporary candidate, verifies both variants, then replaces
`files/` and `summary.xlsx` together. Failed tuning or report generation preserves
the previous complete result. Successful replacement removes temporary rollback
copies. Incomplete `.pending-*` folders never appear as accepted results.

Refresh only the workbook, without training:

```bash
python scripts/build_svr_summary.py
```

Validation scores are development estimates; reserved tests remain unevaluated.
