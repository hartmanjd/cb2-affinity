# History

One-time records of past events. Nothing in the pipeline reads these files, and
notebook reruns never change them.

| File | What happened |
| --- | --- |
| `random_forest_import_validation.json` | Two externally supplied random forest summary workbooks were checked against the saved evidence before the combined forest summary replaced them. |
| `support_vector_regression_import_validation.json` | An externally supplied SVR summary workbook was checked the same way; it also lists where each file moved from `run_01/`. |
| `CB2_data_provenance.xlsx` (git history only) | A hand-made presentation workbook built from an earlier ChEMBL download (3,935 kept measurements). It was replaced by the generated workbook; see commit `92686ec` or earlier. |

The two import records describe the earlier Windows-produced results. The current results
were regenerated on Linux in the pinned environment (see the main README), so the
checksums inside these records refer to earlier file versions kept in git history.
They were moved here from each model's `files/` folder so that republishing a
model can no longer delete them.
