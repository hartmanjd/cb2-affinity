"""Read-only project knowledge for the research assistant: documents and model comparisons.

The data tools in research_assistant.py answer questions about molecules. This module
answers questions about the *project*: how the data were curated, which models were
tried, how they compare, and which settings were chosen. It gives the LLM two things:

1. A document library: every text file in the project (READMEs, notebooks with their
   printed results, scripts, tests, requirements, run manifests and result tables), split
   into sections. The one thing it must never see is the DeepSeek API key, so three
   independent safeguards protect it:
     a. `.env*` files and secret-looking names are never added to the library, and only
        names in the library can be opened (so "../.env" or "/etc/passwd" fail too);
     b. any file whose bytes contain the key is left out, wherever it is;
     c. the key, and anything shaped like an API key, is replaced by "[redacted]" in
        every tool result before it is sent to DeepSeek.
2. A model comparison computed from the saved validation predictions of every model,
   with paired bootstrap intervals, the same method notebook 10 uses.
"""
from pathlib import Path
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

NOISE_FLOOR_PKI = 0.54
# Practical, not a privacy limit: bigger CSV/JSON files (raw downloads, per-molecule predictions)
# would bury search results in thousands of data rows. The data tools already cover their contents.
MAX_TABLE_FILE_BYTES = 300_000
MAX_SECTION_CHARS = 6_000        # One notebook cell or README section is cut at this length.
MAX_READ_CHARS = 15_000          # One read returns at most this much text.
CSV_ROWS_PER_SECTION = 40

# Text file types the library reads; images, workbooks and model files are binary and skipped.
TEXT_SUFFIXES = {".md", ".ipynb", ".json", ".csv", ".py", ".txt", ".toml", ".cfg", ".ini", ".yaml", ".yml"}
TEXT_DOTFILES = {".gitignore", ".gitattributes", ".python-version"}
# Folders that are not project content (git internals, the Python environment, caches).
SKIPPED_FOLDERS = {".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints", "tensorboard", ".ruff_cache",
                   ".pytest_cache", ".mypy_cache", ".streamlit"}
API_KEY_SHAPE = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")   # \b: "task-..." is not a key.


def secret_looking(name):
    """File names that might hold credentials are never read: .env, .env.local, secrets.toml, *.key ..."""
    lowered = name.lower()
    return (lowered.startswith(".env") or "secret" in lowered or "credential" in lowered
            or lowered.endswith((".key", ".pem")) or lowered in {".netrc", ".pypirc"})


def api_key():
    from dotenv import dotenv_values
    return (os.environ.get("DEEPSEEK_API_KEY") or dotenv_values(ROOT / ".env").get("DEEPSEEK_API_KEY") or "").strip()


def redact_secrets(text):
    """Replace the API key, and anything shaped like one, before text can reach the LLM."""
    key = api_key()
    if key and len(key) >= 8:
        text = text.replace(key, "[redacted]")
    return API_KEY_SHAPE.sub("[redacted]", text)


# ---------------------------------------------------------------------------
# Turning each kind of file into numbered sections
# ---------------------------------------------------------------------------
def markdown_sections(text):
    """Split Markdown at its headings; each heading starts a section."""
    sections, title, lines = [], "Introduction", []
    for line in text.splitlines():
        if re.match(r"^#{1,4} ", line):
            if any(l.strip() for l in lines):
                sections.append((title, "\n".join(lines).strip()))
            title, lines = line.lstrip("# ").strip(), [line]
        else:
            lines.append(line)
    if any(l.strip() for l in lines):
        sections.append((title, "\n".join(lines).strip()))
    return sections


def notebook_sections(path):
    """One section per cell: explanation text, or code plus its printed output (figures skipped)."""
    sections = []
    for number, cell in enumerate(json.loads(path.read_text(encoding="utf-8"))["cells"]):
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            heading = next((l.lstrip("# ").strip() for l in source.splitlines() if l.startswith("#")), None)
            first = heading or source.strip().splitlines()[0][:80] if source.strip() else "(empty)"
            sections.append((f"cell {number} (text): {first}", source))
            continue
        outputs = []
        for output in cell.get("outputs", []):
            if output.get("name") == "stdout":
                outputs.append("".join(output["text"]))
            elif output["output_type"] in ("execute_result", "display_data"):
                data = output.get("data", {})
                if "image/png" in data:
                    outputs.append("[figure]")
                elif "text/markdown" in data:
                    outputs.append("".join(data["text/markdown"]))
                elif "text/plain" in data:
                    outputs.append("".join(data["text/plain"]))
            elif output["output_type"] == "error":
                outputs.append(f"[error: {output.get('ename')}]")
        first = next((l for l in source.splitlines() if l.strip() and not l.startswith(("import", "from"))), "code")
        text = "CODE:\n" + source + ("\n\nPRINTED OUTPUT:\n" + "\n".join(outputs) if outputs else "")
        sections.append((f"cell {number} (code): {first.strip()[:80]}", text))
    return sections


def python_sections(text):
    """Split a Python file at its top-level functions and classes (the module preamble comes first)."""
    sections, title, lines = [], "module header and imports", []
    for line in text.splitlines():
        match = re.match(r"^(def|class) (\w+)", line)
        if match:
            if any(l.strip() for l in lines):
                sections.append((title, "\n".join(lines)))
            title, lines = f"{match.group(1)} {match.group(2)}", [line]
        else:
            lines.append(line)
    if any(l.strip() for l in lines):
        sections.append((title, "\n".join(lines)))
    return sections


def strip_checksums(value):
    """Drop 64-character checksums and shorten long lists; they cost tokens and explain nothing."""
    if isinstance(value, dict):
        return {k: strip_checksums(v) for k, v in value.items()
                if not (isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v)) and not k.endswith("sha256")}
    if isinstance(value, list):
        shown = [strip_checksums(v) for v in value[:30]]
        return shown + [f"... {len(value) - 30} more items"] if len(value) > 30 else shown
    return value


def json_sections(path):
    value = strip_checksums(json.loads(path.read_text(encoding="utf-8")))
    if isinstance(value, dict):
        return [(str(key), json.dumps(item, indent=1, default=str)) for key, item in value.items()]
    return [("contents", json.dumps(value, indent=1, default=str))]


def csv_sections(path):
    """A CSV as blocks of rows, each with the header, so any block reads on its own."""
    frame = pd.read_csv(path)
    sections = [("summary", f"{len(frame):,} rows; columns: {', '.join(frame.columns)}")]
    for start in range(0, len(frame), CSV_ROWS_PER_SECTION):
        block = frame.iloc[start:start + CSV_ROWS_PER_SECTION]
        sections.append((f"rows {start + 1}-{start + len(block)}", block.to_csv(index=False, float_format="%.4g")))
    return sections


class ProjectLibrary:
    """The allowlisted project documents, split into readable sections."""

    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.paths = {}
        key = api_key().encode()
        for directory, folders, files in os.walk(self.root):
            # Prune folders that are not project content, plus unfinished or rollback model runs.
            folders[:] = sorted(f for f in folders if f not in SKIPPED_FOLDERS and not secret_looking(f)
                                and not f.startswith((".pending-", ".previous-")))
            for file in sorted(files):
                path = Path(directory) / file
                if secret_looking(file) or file.startswith((".~", ".pending-", ".previous-")):
                    continue                                           # Safeguard (a).
                if not (path.suffix.lower() in TEXT_SUFFIXES or file in TEXT_DOTFILES) or file.endswith(":Zone.Identifier"):
                    continue
                if path.suffix in (".csv", ".json") and path.stat().st_size > MAX_TABLE_FILE_BYTES:
                    continue
                if len(key) >= 8 and key in path.read_bytes():
                    continue                                           # Safeguard (b).
                self.paths[path.relative_to(self.root).as_posix()] = path
        self.cache = {}

    def sections(self, name):
        if name not in self.paths:
            raise ValueError(f"{name!r} is not in the project library. Use list_project_documents or "
                             f"search_project to find a document name.")
        if name not in self.cache:
            path = self.paths[name]
            suffix = path.suffix.lower()
            if suffix == ".md":
                found = markdown_sections(path.read_text(encoding="utf-8"))
            elif suffix == ".ipynb":
                found = notebook_sections(path)
            elif suffix == ".json":
                found = json_sections(path)
            elif suffix == ".csv":
                found = csv_sections(path)
            elif suffix == ".py":
                found = python_sections(path.read_text(encoding="utf-8"))
            else:
                found = [(path.name, path.read_text(encoding="utf-8", errors="replace"))]
            self.cache[name] = [(title, text if len(text) <= MAX_SECTION_CHARS else
                                 text[:MAX_SECTION_CHARS] + "\n... [section cut; it continues in the file]")
                                for title, text in found]
        return self.cache[name]

    def describe(self, name):
        """One-line description from the project reference builder, when it has one."""
        try:
            from build_project_reference import description
            return description(name)
        except Exception:
            return ""


def list_project_documents(library, folder=None):
    """Without a folder: every readable file name, grouped by folder (compact). With a folder: descriptions too."""
    folder = (folder or "").strip().strip("/")
    if folder == ".":
        names = [n for n in library.paths if "/" not in n]   # Files in the project root.
    else:
        names = [n for n in library.paths if not folder or n.startswith(folder + "/") or n == folder]
    if folder:
        rows = [{"name": n, "description": library.describe(n)[:160]} for n in names]
        return {"folder": folder, "documents": rows, "total": len(rows)}
    grouped = {}
    for name in names:
        parent, _, file = name.rpartition("/")
        grouped.setdefault(parent or ".", []).append(file)
    return {"total": len(names), "documents_by_folder": grouped,
            "note": ("Pass folder=... to see descriptions (use '.' for the project root). Read one with read_project_document (full path); search them "
                     "all with search_project. Notebook names give the pipeline order (00 acquisition ... 12 this assistant).")}


def read_project_document(library, name, sections=None):
    """Table of contents, or the text of chosen sections (e.g. [3], [3, 4] or '2-5')."""
    found = library.sections(name)
    contents = [{"section": i, "title": title, "characters": len(text)} for i, (title, text) in enumerate(found)]
    if sections is None:
        total = sum(len(text) for _, text in found)
        if total <= MAX_READ_CHARS:   # Small documents are returned whole.
            return {"name": name, "description": library.describe(name),
                    "text": redact_secrets("\n\n".join(text for _, text in found))}
        return {"name": name, "description": library.describe(name), "table_of_contents": contents,
                "note": "Too long to return whole; call again with the section numbers you need."}
    if isinstance(sections, str):
        start, _, end = sections.partition("-")
        sections = list(range(int(start), int(end or start) + 1))
    if isinstance(sections, int):
        sections = [sections]
    parts, used = [], 0
    for index in sections:
        if not 0 <= int(index) < len(found):
            raise ValueError(f"Section {index} does not exist; {name} has sections 0-{len(found) - 1}.")
        title, text = found[int(index)]
        if used + len(text) > MAX_READ_CHARS:
            parts.append({"section": int(index), "title": title, "text": "[not returned: read limit reached; ask again]"})
            continue
        parts.append({"section": int(index), "title": title, "text": redact_secrets(text)})
        used += len(text)
    return {"name": name, "sections": parts}


STOPWORDS = set("""a an and are as at be by can do does did for from has have how i in is it its of on or so that
the this to was were what when where which who why will with you your we our there their them they than then
used use using about into more most much many any all each also only not no yes""".split())


SYNONYMS = {"hyperparameter": "parameter", "hyper-parameter": "parameter", "settings": "setting"}


def stem(word):
    """Crude stemming so 'searched', 'searches' and 'search' all match: strip a common ending."""
    word = SYNONYMS.get(word, word)
    for ending in ("ing", "ed", "es", "s"):
        if word.endswith(ending) and len(word) - len(ending) >= 4:
            return SYNONYMS.get(word[:-len(ending)], word[:-len(ending)])
    return word


def search_project(library, text, limit=10):
    """Rank sections by the search words they contain, weighting rare words more (TF-IDF style).

    A word found in few sections (e.g. 'Kramer') says more about relevance than one found
    almost everywhere (e.g. 'model'), so each word counts log(sections / sections containing it).
    """
    words = [stem(w.lower()) for w in re.findall(r"[\w.+-]+", str(text)) if len(w) > 1 and w.lower() not in STOPWORDS]
    if not words:
        raise ValueError("Give at least one meaningful search word.")
    every = [(name, index, title, (title + "\n" + body).lower(), body)
             for name in library.paths for index, (title, body) in enumerate(library.sections(name))]
    weight = {w: math.log(len(every) / (1 + sum(w in lowered for *_, lowered, _ in every))) + 1 for w in set(words)}
    hits = []
    for name, index, title, lowered, body in every:
        present = [w for w in dict.fromkeys(words) if w in lowered]
        if not present:
            continue
        score = sum(weight[w] * (1 + 0.2 * min(lowered.count(w), 5)) for w in present)
        position = lowered.find(max(present, key=weight.get))
        snippet = (title + "\n" + body)[max(0, position - 150):position + 250].replace("\n", " ")
        hits.append({"name": name, "section": index, "title": title, "score": round(score, 1),
                     "matched_words": present, "snippet": redact_secrets(snippet)})
    hits.sort(key=lambda h: (-h["score"], h["name"], h["section"]))
    return {"query": text, "total_hits": len(hits), "hits": hits[:max(1, min(int(limit), 25))],
            "note": "Read a hit in full with read_project_document(name, sections=[section])."}


# ---------------------------------------------------------------------------
# Model comparison from the saved validation predictions
# ---------------------------------------------------------------------------
def validation_error_table(root, row):
    """Squared validation errors for one model row: one row per molecule, one column per seed."""
    frame = pd.read_csv(root / Path(row["source_metrics"]).with_name("predictions.csv"))
    frame = frame[(frame["split_strategy"] == row["split_strategy"]) & (frame["subset"] == "validation")]
    if row["model"].startswith("Dummy ") and "model" in frame:   # Baseline dummies: mean or median rows.
        frame = frame[frame["model"].str.endswith(row["model"].split()[-1])]
    frame = frame.assign(squared_error=(frame["predicted_pki"] - frame["observed_pki"]) ** 2)
    if "initialization_seed" in frame:   # MLP: one column per seed.
        return frame.pivot(index="rdkit_smiles", columns="initialization_seed", values="squared_error")
    return frame.set_index("rdkit_smiles")[["squared_error"]]


def build_model_comparison(root=ROOT, resamples=2000, seed=42):
    """Every model's validation scores with 95% bootstrap RMSE intervals and paired comparisons.

    The same validation molecules are redrawn for every model in each resample, so differences
    between models are paired: luck in which molecules are drawn affects both models equally.
    """
    from build_project_reference import pipeline_state, latest_models, comparison_rows
    state, status = pipeline_state()
    rows = pd.DataFrame(comparison_rows(latest_models(state, status)))
    tables = {index: validation_error_table(root, row) for index, row in rows.iterrows()}

    # Fine-tuned ChemBERTa (notebook 11): five fold models; their validation predictions are averaged.
    chemberta_folder = root / "provenance/models/fine_tuned_chemberta"
    if (chemberta_folder / "predictions.csv").is_file():
        predictions = pd.read_csv(chemberta_folder / "predictions.csv")
        predictions = predictions[predictions["role"] == "validation"]
        for split, part in predictions.groupby("split_strategy"):
            averaged = part.groupby("rdkit_smiles").agg(observed=("observed_pki", "first"), predicted=("predicted_pki", "mean"))
            errors = averaged["predicted"] - averaged["observed"]
            index = len(rows)
            rows.loc[index] = {"model": "Fine-tuned ChemBERTa", "variant": "fixed settings", "split_strategy": split,
                               "n_structures": len(averaged), "n_seeds": 1, "mae_pki": errors.abs().mean(),
                               "rmse_pki": math.sqrt((errors ** 2).mean()),
                               "r2": 1 - (errors ** 2).sum() / ((averaged["observed"] - averaged["observed"].mean()) ** 2).sum(),
                               "source_metrics": "provenance/models/fine_tuned_chemberta/configuration.json"}
            tables[index] = (errors ** 2).to_frame("squared_error")
            # Cross-check against the score notebook 11 saved for the same averaged predictions.
            saved = json.loads((chemberta_folder / "configuration.json").read_text(encoding="utf-8"))["validation_scores"]
            saved_rmse = next(s["rmse"] for s in saved if s["split_strategy"] == split and "ChemBERTa" in s["model"])
            assert np.isclose(rows.loc[index, "rmse_pki"], saved_rmse), "ChemBERTa RMSE differs from notebook 11"

    rng = np.random.default_rng(seed)
    samples, records = {}, []
    for split, group in rows.groupby("split_strategy"):
        molecules = tables[group.index[0]].index
        draws = rng.integers(0, len(molecules), (resamples, len(molecules)))
        for index, row in group.iterrows():
            errors = tables[index].loc[molecules].to_numpy()
            # The recomputed RMSE must match the saved score before it is used (seeds averaged for the MLP).
            assert np.isclose(np.sqrt(errors.mean(axis=0)).mean(), row["rmse_pki"]), row["model"]
            samples[index] = np.sqrt(errors[draws].mean(axis=1)).mean(axis=1)
        dummy = group[group["model"].str.startswith("Dummy") & (group["variant"] == "tuned")]["rmse_pki"].iloc[0]
        real = group[~group["model"].str.startswith("Dummy")]
        best = real["rmse_pki"].idxmin()
        for index, row in group.iterrows():
            difference = samples[index] - samples[best]
            low, high = np.percentile(difference, [2.5, 97.5])
            records.append({
                "split_strategy": split, "model": row["model"], "variant": row["variant"],
                "n_validation_molecules": int(row["n_structures"]), "n_seeds": int(row["n_seeds"]),
                "rmse_pki": row["rmse_pki"], "rmse_ci95_low": np.percentile(samples[index], 2.5),
                "rmse_ci95_high": np.percentile(samples[index], 97.5), "mae_pki": row["mae_pki"], "r2": row["r2"],
                "share_of_gap_to_noise_floor_closed": (dummy - row["rmse_pki"]) / (dummy - NOISE_FLOOR_PKI),
                "rmse_minus_best_ci95_low": None if index == best else low,
                "rmse_minus_best_ci95_high": None if index == best else high,
                "versus_best": ("best model on this split" if index == best else
                                "worse than the best (interval above 0)" if low > 0 else
                                "not distinguishable from the best (interval includes 0)"),
                "source": row["source_metrics"]})
    return pd.DataFrame(records).sort_values(["split_strategy", "rmse_pki"]).reset_index(drop=True)
