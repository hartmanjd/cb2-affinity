# Random forest

Open [summary.xlsx](summary.xlsx) for baseline and tuned results in one place.
The cover puts training and validation scores side by side on both random and
scaffold splits. Each variant has one fixed-seed fitted forest per split, so
there is no five-seed spread as in the MLP summaries.

- **Training Guide / Settings:** architecture, parameter meanings, exact settings,
  software versions, split sizes, and training decisions.
- **Scores / Baseline Comparison:** exact saved metrics for both variants.
- **Search Ranking / Selected Parameters / CV Folds:** the training-only search
  and selection evidence. Cross-validation variability is across folds, not seeds.
- **Similarity / Cliff sheets:** clearly labeled calculations derived from saved
  predictions and frozen diagnostic sources; no additional fitting occurs.
- **Evidence Files:** portable links, file sizes, and SHA-256 checksums.

Original files sit under `files/baseline/` and `files/tuning/`. Frozen cliff pairs
and their source manifest are under `files/diagnostics/`. Compressed fitted models
remain binaries; the workbook links them rather than expanding their weights.

The two supplied summaries were checked before consolidation. Their recorded
source checksums, saved CSV tables, cover metrics, JSON settings, and derived
cliff/similarity arithmetic matched the repository evidence. `files/import_validation.json`
records their filenames, hashes and check results. Paths and manifest checksums
in the combined workbook reflect the new locations. The original downloads
remain unchanged outside this repository.

The current cliff set was copied from the verified MLP set for the identical
preparation. Its local copy avoids a continuing MLP dependency. Future notebook
04 runs construct the same definition directly from their own frozen validation
fingerprints and targets: Tanimoto ≥ 0.8 and absolute pKi gap ≥ 1.0. No test rows
enter those diagnostics. Report-only diagnostics do not influence tuning.

## Reruns

Notebook 04 writes a temporary candidate and audits both baseline and tuning.
Only after both stages succeed does it replace the current `files/` folder and
combined workbook. Failed tuning or workbook generation preserves the previous
complete result. Successful publication removes temporary rollback copies;
interrupted `.pending-*` candidates are never selected as accepted results.

Refresh only the workbook, without training:

```bash
python scripts/build_forest_summary.py
```

The separate data provenance presentation workbook and MLP summaries are unchanged.
Validation scores remain development estimates; test sets are reserved.
