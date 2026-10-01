"""Known-answer evaluation of the LLM research assistant (notebook 12 and the command line).

A chat that sounds convincing is not evidence that it is correct. Every question here has
an answer computed independently from the curated files and saved results, with pandas and
RDKit, not through the assistant's own tools. The assistant answers each question in a fresh
conversation and must end with a line "FINAL ANSWER: ...", which is compared with the truth.

Run it directly to measure a change (each run calls DeepSeek and costs roughly ten cents):

    python scripts/assistant_evaluation.py --out results/assistant_evaluation.csv
    python scripts/assistant_evaluation.py --no-thinking --label "thinking off"
"""
from pathlib import Path
import argparse
import json
import re
import sys
import time

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_assistant as assistant  # noqa: E402

ANSWER_FORMAT = ("\n\nEnd your reply with one line 'FINAL ANSWER: <answer>'. For a number give only the number. If a "
                 "prediction was withheld write 'FINAL ANSWER: WITHHELD'. If this dataset cannot answer, write "
                 "'FINAL ANSWER: NOT IN DATA'.")
# How close an answer must be: counts exact, pKi within 0.01, other decimals within 0.1.
TOLERANCE = {"count": 0.5, "pki": 0.01, "decimal": 0.1}


def build_questions(root=ROOT):
    """The question set with its independently computed answers: (question, expected, kind)."""
    curated = pd.read_csv(root / "provenance/curation/curated_structures.csv")
    decisions = pd.read_csv(root / "provenance/curation/measurement_decisions.csv")
    splits = pd.read_csv(root / "provenance/preparation/split_assignments.csv")
    retained = decisions[decisions["status"] == "retained"]
    scaffolds = splits[splits["split_strategy"] == "random"]["scaffold"]
    weights = curated["rdkit_smiles"].map(lambda s: Descriptors.MolWt(Chem.MolFromSmiles(s)))
    sulfonamide = Chem.MolFromSmarts("[SX4](=[OX1])(=[OX1])[NX3]")
    has_sulfonamide = curated["rdkit_smiles"].map(lambda s: Chem.MolFromSmiles(s).HasSubstructMatch(sulfonamide))
    strongest = curated.loc[curated["pki_target"].idxmax(), "rdkit_smiles"]
    nabilone = retained.loc[retained["molecule_chembl_id"] == "CHEMBL2218896", "rdkit_smiles"].iloc[0]
    nabilone_pki = curated.loc[curated["rdkit_smiles"] == nabilone, "pki_target"].iloc[0]
    random_rows = splits[splits["split_strategy"] == "random"]
    largest_series = random_rows.loc[random_rows["scaffold"] == scaffolds.value_counts().index[0], "rdkit_smiles"]
    largest_series_papers = retained.loc[retained["rdkit_smiles"].isin(largest_series), "document_chembl_id"].nunique()

    # The best scaffold-split model, read from the table rows of results/model_comparison.md:
    # | Model | Variant | Split | N | MAE | RMSE | ... ; MLP RMSE cells read "0.908 ± 0.029".
    table_rows = [[cell.strip() for cell in line.strip().strip("|").split("|")]
                  for line in (root / "results/model_comparison.md").read_text().splitlines()
                  if line.startswith("| ") and not line.startswith(("| Model", "| ---"))]
    scaffold_rmse = {(row[0], row[1]): float(row[5].split("±")[0]) for row in table_rows if row[2] == "scaffold"}
    best_scaffold_model = min(scaffold_rmse, key=scaffold_rmse.get)[0]
    svr_manifest = json.loads((root / "provenance/models/support_vector_regression/files/tuning/manifest.json").read_text())
    selected_c = svr_manifest["selection"]["scaffold"]["best_parameters"]["C"]
    notebook_10 = "".join("".join(c["source"]) for c in
                          json.loads((root / "notebooks/10_noise_ceiling.ipynb").read_text())["cells"])
    noise_floor = float(re.search(r"NOISE_FLOOR = ([0-9.]+)", notebook_10).group(1))
    test_molecule = splits[(splits["split_strategy"] == assistant.PREDICTION_SPLIT)
                           & (splits["subset"] == "test")]["rdkit_smiles"].iloc[0]

    # A linker question with an answer RDKit can check: sp3 chains between an aromatic ring and a
    # phenyl, counted only when the chain itself is acyclic (see find_linkers).
    molecules = [Chem.MolFromSmiles(s) for s in curated["rdkit_smiles"]]
    pattern = Chem.MolFromSmarts("[a]-[CX4]-[CX4]-[CX4]-c1ccccc1")
    genuine = sum(any(all(not molecule.GetAtomWithIdx(match[i]).IsInRing() for i in (1, 2, 3))
                      for match in molecule.GetSubstructMatches(pattern)) for molecule in molecules)

    return [
        ("How many unique molecules are in the curated modeling dataset?", len(curated), "count"),
        ("How many molecules have a curated pKi of 9 or higher?", int((curated["pki_target"] >= 9).sum()), "count"),
        ("What is the curated pKi of nabilone?", nabilone_pki, "pki"),
        ("What is the highest curated pKi in the dataset?", curated["pki_target"].max(), "pki"),
        ("Give the ChEMBL molecule ID of the molecule with the highest curated pKi.",
         retained.loc[retained["rdkit_smiles"] == strongest, "molecule_chembl_id"].iloc[0], "text"),
        ("How many measurements were removed because repeat measurements disagreed by 1 pKi or more?",
         int((decisions["status"] == "measurement_conflict").sum()), "count"),
        ("How many distinct Bemis-Murcko scaffolds are there among the curated molecules?", scaffolds.nunique(), "count"),
        ("How many molecules share the most common Bemis-Murcko scaffold (count every scaffold, including plain benzene)?",
         int(scaffolds.value_counts().iloc[0]), "count"),
        ("How many different source papers (documents) do the molecules with the most common Bemis-Murcko scaffold "
         "come from (count every scaffold, including plain benzene)?", largest_series_papers, "count"),
        ("Which source document (ChEMBL document ID) contributed the most retained measurements?",
         retained["document_chembl_id"].value_counts().index[0], "text"),
        ("What is the median curated pKi of molecules containing a sulfonamide group?",
         curated.loc[has_sulfonamide, "pki_target"].median(), "pki"),
        ("What is the mean molecular weight of molecules with a curated pKi of at least 8? Round to one decimal.",
         weights[curated["pki_target"] >= 8].mean(), "decimal"),
        (f"Predict the CB2 pKi of {test_molecule} with the project's model.", "WITHHELD", "text"),
        ("According to this dataset, how selective is nabilone for CB2 over CB1?", "NOT IN DATA", "text"),
        ("Which model family has the lowest validation RMSE on the scaffold split?",
         ("SUPPORT VECTOR", "SVR") if best_scaffold_model == "Support vector regression" else best_scaffold_model.upper(),
         "text"),
        ("What value of the SVR regularisation parameter C was selected for the scaffold split?", float(selected_c), "decimal"),
        ("What measurement-noise floor, in pKi, does the project use when judging model error?", noise_floor, "pki"),
        ("How many molecules have a genuine three-carbon chain, not part of any ring, joining an aromatic ring to a "
         "phenyl group?", genuine, "count"),
    ]


def final_answer(text):
    """The text after the last 'FINAL ANSWER:' line, or '' if the assistant forgot the format."""
    lines = [line for line in (text or "").splitlines() if "FINAL ANSWER:" in line.upper()]
    return lines[-1].upper().split("FINAL ANSWER:")[-1].strip(" *`.") if lines else ""


def is_correct(answer, expected, kind):
    if kind == "text":
        # Some answers have more than one correct spelling (e.g. "SVR" or "support vector regression").
        options = expected if isinstance(expected, tuple) else (expected,)
        return any(str(option).upper() in answer for option in options)
    try:
        value = float(answer.replace(",", "").split()[0])
    except (ValueError, IndexError):
        return False
    return abs(value - float(expected)) <= TOLERANCE[kind]


def run_evaluation(client, data, model=None, thinking=None, questions=None, progress=print, compact=None,
                   prefetch_tools=True):
    """Ask every question in a fresh conversation and score the answers. Returns a DataFrame."""
    model = model or assistant.DEFAULT_MODEL
    thinking = assistant.DEFAULT_THINKING if thinking is None else thinking
    questions = questions or build_questions(data.root)
    profile_compact = assistant.conversation_profile(str(client.base_url), compact)[0] is assistant.COMPACT_SYSTEM_PROMPT
    rows = []
    for number, (question, expected, kind) in enumerate(questions, 1):
        # A fresh conversation per question, so earlier answers cannot help later ones.
        messages = (assistant.new_conversation(str(client.base_url), compact)
                    + [{"role": "user", "content": question + ANSWER_FORMAT}])
        usage, tools_used, failure = {}, [], None
        started = time.time()
        try:
            reply = assistant.chat_turn(client, data, messages, model=model, thinking=thinking, usage=usage,
                                        compact=compact, prefetch_tools=prefetch_tools,
                                        on_tool=lambda name, *rest: tools_used.append(name))
        except Exception as error:   # A failed call is scored as wrong rather than stopping the run.
            reply = f"ERROR {type(error).__name__}: {error}"
            failure = f"{type(error).__name__}: {str(error)[:200]}"
        answer = final_answer(reply)
        rows.append({"number": number, "question": question, "kind": kind, "expected": expected, "answer": answer,
                     "error": failure,
                     "correct": is_correct(answer, expected, kind), "tools": ", ".join(tools_used),
                     # Tool calls the LLM asked for, and look-ups the code ran before it saw the question.
                     "n_tool_calls": sum("(pre-fetched)" not in name for name in tools_used),
                     "n_prefetched": sum("(pre-fetched)" in name for name in tools_used),
                     "prefetch": prefetch_tools, "seconds": round(time.time() - started, 1),
                     "cost_usd": usage.get("cost_usd", 0.0), "model": model, "thinking": thinking,
                     "profile": "compact" if profile_compact else "full"})
        progress(f"{number:>2}/{len(questions)} {'correct' if rows[-1]['correct'] else 'WRONG  '} "
                 f"{rows[-1]['seconds']:>5.1f}s {len(tools_used)} tools | expected {str(expected)[:14]:<14} "
                 f"got {failure or answer[:60]}")
        if failure and number == 1:
            # A first-question failure is almost always the endpoint, not the model: stop rather
            # than spend the whole run producing identical errors.
            raise SystemExit(f"\nThe first question failed, so the rest were not attempted:\n  {failure}\n"
                             "Check that the server is running and reachable, and that --model names a model it serves.")
    return pd.DataFrame(rows)


def summarise(frame, label=""):
    failed = int(frame["error"].notna().sum()) if "error" in frame else 0
    return ((f"{failed} of {len(frame)} questions failed with an error. " if failed else "") + (f"{label + ': ' if label else ''}{frame['correct'].sum()}/{len(frame)} correct "
            f"({frame['correct'].mean():.0%}); {frame['seconds'].mean():.1f} s per question "
            f"(total {frame['seconds'].sum():.0f} s); {frame['n_tool_calls'].mean():.1f} tool calls; "
            f"{frame['n_prefetched'].sum() if 'n_prefetched' in frame else 0} pre-fetched look-ups; "
            f"${frame['cost_usd'].sum():.3f} total"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/assistant_evaluation.csv", help="Where to write the scored answers.")
    parser.add_argument("--model", default=None, help="Model name, e.g. deepseek-v4-pro or gemma4:26b.")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-format endpoint. Default DeepSeek; use a local runner's URL to "
                             "evaluate a model on this machine, e.g. http://localhost:11434/v1")
    parser.add_argument("--no-thinking", action="store_true", help="Answer without the model's private reasoning step.")
    parser.add_argument("--thinking", action="store_true", help="Force thinking on.")
    parser.add_argument("--no-prefetch", action="store_true",
                        help="Do not look up molecules named in the question before the LLM sees it.")
    parser.add_argument("--label", default="", help="Name for this run, printed with the summary.")
    parser.add_argument("--full-prompt", action="store_true",
                        help="Give a local model the full prompt and all 21 tools instead of the compact profile.")
    parser.add_argument("--compact-prompt", action="store_true",
                        help="Use the short prompt and core tools even against DeepSeek, for a like-for-like test.")
    arguments = parser.parse_args()
    thinking = True if arguments.thinking else False if arguments.no_thinking else None
    data = assistant.ResearchData(ROOT)
    client = assistant.make_client(base_url=arguments.base_url)
    if arguments.model is None and not assistant.is_hosted(arguments.base_url):
        raise SystemExit("Pass --model with the local model's name, e.g. --model gemma4:26b")
    compact = True if arguments.compact_prompt else False if arguments.full_prompt else None
    prompt, tools = assistant.conversation_profile(str(client.base_url), compact)
    print(f"Endpoint {client.base_url} | model {arguments.model or assistant.DEFAULT_MODEL} | "
          f"{'compact' if prompt is assistant.COMPACT_SYSTEM_PROMPT else 'full'} profile "
          f"({len(prompt) // 4} prompt tokens, {len(tools)} tools)")
    frame = run_evaluation(client, data, model=arguments.model, thinking=thinking, compact=compact,
                           prefetch_tools=not arguments.no_prefetch)
    output = Path(arguments.out)
    output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print("\n" + summarise(frame, arguments.label))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
