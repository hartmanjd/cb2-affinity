# Headline model results

[model_comparison.md](model_comparison.md) provides the readable comparison, and
[noise_ceiling.png](noise_ceiling.png) shows how close each model gets to the measurement-noise floor (notebook 10).
Notebook 11 adds three figures comparing a fine-tuned ChemBERTa transformer with the tuned SVR:
[chemberta_vs_svr.png](chemberta_vs_svr.png) (scores), [chemberta_training.png](chemberta_training.png)
(training curves) and [chemberta_chemical_space.png](chemberta_chemical_space.png) (chemical space before and after fine-tuning).
Notebook 12 writes `assistant_evaluation.csv`, the latest known-answer evaluation of the LLM research assistant
(it appears after the first run with a DeepSeek API key, and varies from run to run).
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
