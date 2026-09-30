# Headline model results

[model_comparison.md](model_comparison.md) provides the readable comparison.
The [data provenance workbook](../provenance/CB2_data_provenance.xlsx) shows where the
dataset came from. Exact current scores
and supporting tables are under `provenance/models/`.

All fitted models, predictions, metrics, settings, and diagnostic figures are
under `provenance/models/`, grouped by model family and execution. The headline
comparison selects the latest completed baseline and tuning run for each family.
Every row states which dataset produced it and whether it matches the newest
preparation. Existing results remain visible while a fresh pipeline is incomplete.

Scores use random/scaffold validation, with test sets still reserved. MLP means
and sample SD describe five initialization seeds, not independent experiments.
