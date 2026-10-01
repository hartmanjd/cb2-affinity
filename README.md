# CB2 affinity

Human CB2 binding-affinity dataset curation and model comparison.

Start with [the data provenance workbook](provenance/CB2_data_provenance.xlsx).
It shows where every number in the dataset came from, with sheets for the
Overview, Filter Funnel, Removal Reasons, Kept Measurements, Removed Measurements,
Assays, Documents, Pipeline Settings, and Evidence Files. It is generated from the
current pipeline files and refreshed by each notebook's final cell, so it always
matches the data. For model results and file locations, use the links below. All
supporting data and model artifacts remain under `provenance/`.

[XGBoost summary](provenance/models/xgboost/summary.xlsx) · [SVR summary](provenance/models/support_vector_regression/summary.xlsx) · [Random forest summary](provenance/models/random_forest/summary.xlsx) · [MLP summary](provenance/models/multilayer_perceptron/summary.xlsx) · [Dummy summary](provenance/models/dummy_baselines/summary.xlsx) · [Model comparison](results/model_comparison.md) · [Every file explained](MASTER_REFERENCE.md)

## Folders

| Folder | Purpose |
| --- | --- |
| `notebooks/` | The thirteen scientific notebooks, in execution order |
| `provenance/` | Data provenance workbook, model summaries, and all supporting files |
| `results/` | A readable headline model comparison |
| `scripts/` | Shared storage helpers, evidence checks, Markdown reference updates, and the LLM research assistant |
| `tests/` | Automated checks for dataset handoffs, publication, rollback, and model downloads |

There are no archive folders. The old `curation/run_05` folder is removed; the
supporting evidence for the preserved model results is now directly under
`provenance/curation/`. The original download and all saved model predictions and
scores are preserved. Stage-specific review records remain as JSON files for the pipeline audits.
The data provenance workbook is a generated report, not a pipeline input.

## Fitted models

The fitted estimators (`models.joblib.gz`, about 100 MB in total) are not stored
in git. Git keeps every version of a committed file forever, so each retrain made
the repository, pushes, and clones larger. Instead they are attached to the
[`fitted-models` GitHub Release](https://github.com/hartmanjd/cb2-affinity/releases/tag/fitted-models).
Each run manifest, which *is* committed, still records every model's SHA-256
checksum, so the provenance chain is unchanged.

After cloning, download and verify them once (standard library only):

```bash
python scripts/fetch_models.py
```

A download is kept only if its bytes match the checksum in its manifest, and an
existing local file that differs is never overwritten. Report builders and
notebook 09 stop with this instruction when models are missing. Release file
names include part of the checksum, so older commits can still fetch the exact
models they recorded.

**After retraining** (maintainer step): the notebook writes new models locally
and records their checksums. Publish them *before* pushing that commit, or other
people cannot fetch them. This requires the GitHub CLI (`gh`) and only uploads
files the release does not already have:

```bash
python scripts/fetch_models.py --upload
```

Models committed before this change remain in the git history; untracking them
stops future growth but does not shrink existing clones.

## Fresh run

Open this Linux checkout in VS Code: `/home/justin/code/cb2-affinity` (WSL: Ubuntu).
All saved results were produced on Linux with the exact versions in
`requirements.txt` and Python 3.14.6 (`.python-version`). Create that environment
once with [uv](https://docs.astral.sh/uv/), then select `.venv` as the notebook
kernel in VS Code:

```bash
uv venv --python 3.14.6 .venv
```

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

Notebook checks compare the installed versions **and the operating system** with
the preparation run. Tree models (random forest, XGBoost) can resolve near-tied
splits differently on Windows and Linux, so results are only exactly
reproducible on the same OS. If you change packages or OS, rerun notebook 03
onward rather than bypassing the checks.

In a new clone, first run `python scripts/fetch_models.py` (see [Fitted models](#fitted-models)).
Each notebook's final cell verifies every recorded checksum, including the models.

Run each notebook top to bottom, in order:

1. `00_data_acquisition.ipynb` — fetch the latest available ChEMBL release and human CB2 records.
2. `01_data_exploration.ipynb` — select Ki measurements from that completed download.
3. `02_data_curation.ipynb` — curate structures, decisions, and target pKi.
4. `03_modeling_preparation.ipynb` — freeze fingerprints and random/scaffold splits.
5. `04_random_forest_evaluation.ipynb` — forest baseline and tuning.
6. `05_dummy_baselines.ipynb` — mean/median baselines and selection.
7. `06_xgboost.ipynb` — boosted-tree baseline and tuning.
8. `07_support_vector_regression.ipynb` — SVR baseline and tuning.
9. `08_mlp_baseline.ipynb` — corrected internal stopping and full-training refits.
10. `09_mlp_tuning.ipynb` — bounded MLP search and refits.
11. `10_noise_ceiling.ipynb` — how close each model gets to the measurement-noise floor (no training).
12. `11_fine_tuned_chemberta.ipynb` — optional: fine-tunes a pretrained transformer in PyTorch and compares it with the tuned SVR (needs extra packages; see below).
13. `12_llm_research_assistant.ipynb` — optional: an LLM that answers questions about the data by calling pandas, RDKit and SVR tools, with a known-answer evaluation (needs extra packages and a DeepSeek API key; see below).

Each notebook finds the completed upstream stage for the current acquisition.
No manual edits to an old run number are needed. Missing or incompatible stages
stop with an instruction to run the prerequisite. Candidate outputs receive a
new candidate folder and are published only when verification and the completion
manifest succeed. Every model folder has one combined `summary.xlsx` for its baseline
and tuned results. Dummy, random forest, XGBoost, and SVR keep it beside `files/baseline/`,
`files/tuning/`, and `files/diagnostics/`; notebooks 05, 04, 06, and 07 replace their
baseline and tuning together after the complete experiment passes. The MLP keeps it
beside `baseline/` and `tuning/`, because notebooks 08 and 09 publish those variants
separately and each rebuilds the combined workbook. A successful MLP baseline replacement
clears dependent tuning results; run notebook 09 to regenerate them. Current model results remain visible until replacements finish;
the model comparison identifies older-dataset/preparation results explicitly.

The final cell of every notebook verifies evidence and refreshes the model
comparison, the data provenance workbook, and the master reference. Saved notebook outputs remain
from the previous execution until you rerun them; old printed paths are historical.

## How good can a model get?

Repeat lab measurements of the same molecule disagree, so no model can reliably
beat that measurement noise. Notebook 10 uses a published estimate for ChEMBL Ki
data (about 0.54 pKi) and plots every model between "no skill" and that floor:
[results/noise_ceiling.png](results/noise_ceiling.png).

## Does a more advanced model help?

Notebook 11 fine-tunes ChemBERTa, a transformer pretrained on 77 million molecules,
on the same splits and folds as the tuned SVR, and shows its training curves live.
It needs PyTorch and an NVIDIA GPU (about 10 minutes; hours on a CPU). Install the
extra packages once, on top of the pinned environment:

```bash
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements-deep-learning.txt
```

Results: [results/chemberta_vs_svr.png](results/chemberta_vs_svr.png),
[results/chemberta_training.png](results/chemberta_training.png) and
[results/chemberta_chemical_space.png](results/chemberta_chemical_space.png).

## Ask the data questions in plain English

Notebook 12 and a Streamlit chat app let a researcher ask questions such as "which
scaffolds are most potent?" or "predict this molecule and tell me how far to trust it".
A large language model (DeepSeek) answers by calling 21 tools in
`scripts/research_assistant.py`, which cover filtering and grouping, substructure
comparisons, similarity search, activity cliffs, source papers, structure drawings,
plots and SVR predictions. For questions about the project itself, it can compare
every trained model (with bootstrap intervals) and search and read every text file in
the project: READMEs, notebooks, scripts, tests, run manifests and result tables. The
only thing it can never read is the API key: `.env` and other secret-looking files are
excluded, any file containing the key is skipped, and the key is redacted from every
tool result (`scripts/project_knowledge.py`).
Every number comes from a tool, and the app shows each tool call. Linker SAR and the
composition of any group of molecules are computed in RDKit rather than left to the
model, because reviewing real answers showed it guessing both.

A known-answer evaluation measures whether the answers are right, and can be rerun to
check whether a change helped:

```bash
python scripts/assistant_evaluation.py --label "my change"
```

That evaluation also chose the model. Five runs are recorded in
[results/assistant_model_trials.csv](results/assistant_model_trials.csv): DeepSeek
answered 18/18, two locally run models (Gemma 4 26B and Qwen3 14B) reached 9/18 and
6/18 on the same brief and tools, and notebook 12 tells the story of how each run led
to the next. Every number also carries its uncertainty: 95% confidence intervals for group
means, differences and correlations; split-conformal 95% prediction intervals for
predictions; and measurement-noise ranges for measured values. Scaffold names come
from RDKit ring-system matching rather than the LLM, and each series reports how many
papers it comes from. The prediction tool refuses reserved test molecules, so
the test sets stay unevaluated.

Install the extra packages once, on top of the pinned environment:

```bash
uv pip install --python .venv/bin/python -r requirements-llm.txt
```

Put your key in a `.env` file in the project root (git ignores it) as
`DEEPSEEK_API_KEY=your-key`, then start the app and open the address it prints:

```bash
.venv/bin/streamlit run scripts/assistant_app.py
```

### Running a model on your own machine instead

Any server speaking the OpenAI chat-completions format works, so a local model keeps every
question on your machine and costs nothing per token. Point the assistant at it with
`ASSISTANT_BASE_URL`, and name the model it serves:

```bash
ASSISTANT_BASE_URL=http://localhost:11434/v1 python scripts/assistant_evaluation.py --model gemma4:26b --label "local"
```

The same evaluation then says whether the local model matches DeepSeek's 18/18 on this
project's questions, which is the only comparison that matters here. The model must support
tool calling; a chemistry-tuned model that cannot call tools is not usable for this app,
because every number comes from a tool rather than from the model's knowledge.

From WSL, a server running on Windows is not on `localhost`. Start it listening on all
interfaces (for Ollama, set `OLLAMA_HOST=0.0.0.0`) and use the Windows host address:

```bash
ASSISTANT_BASE_URL=http://$(ip route show default | awk '{print $3}'):11434/v1 .venv/bin/streamlit run scripts/assistant_app.py
```

Questions and tool results are sent to DeepSeek's servers unless you run a local model.
That is fine for this public ChEMBL data; don't enter unpublished structures. Notebook 12's tool demonstrations
run without a key. The latest evaluation run is saved to `results/assistant_evaluation.csv`.

## Refresh the generated references

With the dependency in `requirements-reporting.txt` installed and the
[fitted models](#fitted-models) fetched:

```bash
python scripts/build_project_reference.py --check
python scripts/build_project_reference.py
```

These commands verify or report saved evidence without downloading data or fitting
models. The report builder checks checksums, notebook syntax, prediction metrics,
split membership, measurement accounting, and final pKi aggregation.

All comparisons are development-validation results. Reserved test sets remain
unevaluated; deleting archives does not undo earlier use of validation labels.
