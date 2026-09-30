# One provenance reference

Open [CB2_data_provenance.xlsx](CB2_data_provenance.xlsx). This is the supplied
presentation workbook, copied unchanged from `CB2_data_provenance(claude).xlsx`.
Start with **Overview**, which explains the eight sheets: Download Record,
Filter Funnel, Kept Measurements, Quarantine, Assays, Documents, and Pipeline
Settings, alongside Overview itself.

The workbook is a static reference, not an input to the analysis. It retains
historical paths (including `data/raw/chembl_cb2_20260922T020439072711Z`) and results
from the earlier project snapshot. The current source download is under
`raw/chembl_cb2_20260922T165756862384Z/`. Notebook reruns and reference commands
never overwrite the workbook; update it separately after a fresh run.

Use [the master reference](../MASTER_REFERENCE.md) for current file locations and
[the model comparison](../results/model_comparison.md) for saved validation results.

All original source data and generated evidence are stored below this folder:

| Folder | Evidence |
| --- | --- |
| `raw/` | Original ChEMBL JSON, target/release metadata, queries, download dates |
| `selection/` | Selected Ki CSV, exclusions/reasons, selection records |
| `curation/` | Measurement decisions, final molecules, structure summaries, review records |
| `preparation/` | Fingerprints, targets, frozen splits, nearest neighbors, audits |
| `preparation/identical_fingerprints/` | Current groups, members, cross-subset fingerprint pairs |
| `models/` | Model families, fitted models, predictions, metrics, tuning, seeds, and diagnostic figures |

Future preparation runs place their identical-fingerprint tables inside their own
candidate folder. This keeps each set of diagnostics with the preparation it audits.

## Trace the supporting evidence

Use the workbook's measurement, assay, and document sheets during a presentation.
For current pipeline evidence, follow structure keys and activity/assay/document
IDs through `curation/measurement_decisions.csv`, `curation/structure_summary.csv`,
and the original ChEMBL JSON. Each stage's manifest records its input files,
checksums, settings, and outputs. Model folders contain the exact metrics,
predictions, and model settings.

The former stage-specific review Excel files are JSON review records:
`selection/<snapshot>/cb2_ki_quarantine.json` and `curation/review.json`.
Notebooks reload and audit these records independently of the presentation workbook.
Raw downloads, model artifacts, scores, and prediction values are unchanged.

## Existing evidence and reruns

The `curation/run_05` folder and the archive folders were removed. Supporting
curation files were consolidated directly here so existing model results remain
traceable until the fresh run replaces them. The MLP folder now keeps one baseline and one tuned execution,
each with `summary.xlsx` beside its detailed `files/` folder. The older baseline
attempts were deleted. Start with [the MLP guide](models/multilayer_perceptron/README.md).
No archive restoration is required to use this project.

Run notebooks 00–09 in order. They select only completed stages matching the newest
acquisition and upstream manifest hashes. Model comparisons carry their dataset
identity and current/previous-preparation status. Refreshing the download does not
make an earlier saved model a result on the new dataset. Original sources stay
available while saved models still depend on them.

The final cell in each notebook checks saved evidence and refreshes the Markdown
model comparison and master reference. You can also run
`python scripts/build_project_reference.py` from the project root. Both preserve
the supplied data provenance workbook unchanged. MLP summaries are separate,
automatically generated workbooks refreshed when a new MLP execution is accepted.

## Random forest reference

[The combined random forest summary](models/random_forest/summary.xlsx) presents
baseline and tuned results together. Its supporting files are grouped under
`models/random_forest/files/`, with separate baseline, tuning, and diagnostic
subfolders. See [the forest guide](models/random_forest/README.md) for verification
and rerun behavior.

## Support vector regression reference

[The combined SVR summary](models/support_vector_regression/summary.xlsx) uses
the same display as the forest summary, with baseline and tuned scores, kernel
and parameter explanations, the complete search, and derived cliff diagnostics.
Supporting evidence sits in `models/support_vector_regression/files/` under
baseline, tuning, and diagnostics. See [the SVR guide](models/support_vector_regression/README.md)
for the supplied-workbook audit and automatic replacement on notebook 07 reruns.
