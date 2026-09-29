# Multilayer perceptron

Open the summary that matches your question:

| Variant | Quick overview | Detailed evidence |
| --- | --- | --- |
| Baseline | [baseline/summary.xlsx](baseline/summary.xlsx) | [baseline/files/](baseline/files/) |
| Tuned | [tuning/summary.xlsx](tuning/summary.xlsx) | [tuning/files/](tuning/files/) |

The first sheet prominently displays random/scaffold validation and training
scores, including variation across five initialization seeds. One accepted
execution includes all five seeds on both splits; these are not obsolete runs.
Reserved test sets remain unevaluated.

**Training Guide** explains architecture, input features, optimization, early
stopping, data sizes, and the meaning of the scores. **Settings** preserves the
complete JSON configuration and software versions. Subsequent sheets contain
exact scores and all CSV tables, including predictions, training histories,
similarity diagnostics and activity cliffs. The tuning summary also includes
search rankings, per-candidate results and comparison with its baseline.

**Evidence Files** links every original file and records its checksum. Compressed
`models.joblib.gz` files hold the fitted Python models and remain binaries;
their weights are not expanded into spreadsheet cells. PNG plots remain linked
files. The summary workbooks are reports, not model inputs.

## One accepted execution

The current baseline and tuned results were retained, with numbered execution
folders removed. Older baseline attempts were deleted. Their deletion does not
change the scientific limitation that validation labels were used during earlier
development; these are development-validation scores, not final test estimates.

Notebooks 08 and 09 save a candidate separately, run their scientific audits, and
then replace `files/` and `summary.xlsx`. A failed fit or workbook write preserves
the previous accepted result. After successful replacement, temporary rollback
copies are deleted. An interrupted candidate may remain hidden as `.pending-*`;
it is never selected as an accepted result.

Tuning uses the baseline's frozen internal assignments and activity-cliff pairs.
Therefore, successfully replacing the baseline clears its now-outdated dependent
tuning result and summary. Run notebook 09 next to populate them again. Tuning
replacement leaves the current baseline intact. Each successful execution keeps
one accepted result per variant, regardless of whether the score improved.

To refresh both summaries from saved files without retraining:

```bash
python scripts/build_mlp_summary.py
```

This requires `requirements-reporting.txt`. It does not modify the separately
maintained `provenance/CB2_data_provenance.xlsx` presentation workbook.
