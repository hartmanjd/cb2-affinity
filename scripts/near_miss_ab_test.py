"""A/B test: do separator-free name matching and a near-miss structure search stop wrong "not in dataset" answers?

Why this test exists. Asked for the CB2 Ki of SR144528, the assistant said it was not in the dataset and quoted
a model prediction. Two things went wrong. ChEMBL stores the name as "Sr-144528", so "SR144528" matched nothing;
and the SMILES the language model then wrote from memory had its chlorine and methyl swapped, so the structure
search found nothing either. SR144528 has 18 recorded measurements, all removed by curation because they
disagree by more than 1 pKi.

    A  as before: names compared with their hyphens and spaces; an unmatched SMILES is reported as absent
    B  names compared on letters and digits only ("SR144528" finds "Sr-144528", also in pre-fetch), and an
       unmatched SMILES returns recorded molecules with the same formula or Tanimoto >= 0.9 (near_misses)

The answer key does not rely on anyone's memory of these structures: each ligand's SMILES is PubChem's for
the name (fetched 2026-10-01), matched to the recorded measurements by InChIKey skeleton (stereo ignored).
Three ligands are absent from the data; JWH-073 is JWH-018 with one carbon fewer, to test that a close
analogue is not mistaken for the named compound. Both variants run at the same time.

    python -u scripts/near_miss_ab_test.py --variant A --out ab_near_miss_A.csv
"""
from pathlib import Path
import argparse
import re
import sys
import time

import pandas as pd
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_assistant as assistant  # noqa: E402
from assistant_evaluation import final_answer  # noqa: E402

ANSWER_FORMAT = ("\n\nEnd your reply with one line 'FINAL ANSWER: <answer>': the curated pKi as a number; REMOVED if "
                 "it was measured but every measurement was removed during curation; or NOT IN DATA if it was never "
                 "measured in this dataset.")

# PubChem's structure for each name (PUG REST, IsomericSMILES).
LIGANDS = [
    ("SR144528", "CC1=CC=C(C=C1)CN2C(=CC(=N2)C(=O)N[C@H]3[C@]4(CC[C@H](C4)C3(C)C)C)C5=CC(=C(C=C5)Cl)C"),
    ("WIN 55,212-2", "CC1=C(C2=C3N1[C@@H](COC3=CC=C2)CN4CCOCC4)C(=O)C5=CC=CC6=CC=CC=C65"),
    ("CP 55,940", "CCCCCCC(C)(C)C1=CC(=C(C=C1)[C@@H]2C[C@@H](CC[C@H]2CCCO)O)O"),
    ("JTE-907", "CCCCCOC1=C(C=CC2=C1NC(=O)C(=C2)C(=O)NCC3=CC4=C(C=C3)OCO4)OC"),
    ("AM630", "CC1=C(C2=C(N1CCN3CCOCC3)C=C(C=C2)I)C(=O)C4=CC=C(C=C4)OC"),
    ("AM251", "CC1=C(N(N=C1C(=O)NN2CCCCC2)C3=C(C=C(C=C3)Cl)Cl)C4=CC=C(C=C4)I"),
    ("JWH-015", "CCCN1C(=C(C2=CC=CC=C21)C(=O)C3=CC=CC4=CC=CC=C43)C"),
    ("GW405833", "CC1=C(C2=C(N1C(=O)C3=C(C(=CC=C3)Cl)Cl)C=CC(=C2)OC)CCN4CCOCC4"),
    ("AM1241", "CN1CCCCC1CN2C=C(C3=CC=CC=C32)C(=O)C4=C(C=CC(=C4)[N+](=O)[O-])I"),
    ("JWH-018", "CCCCCN1C=C(C2=CC=CC=C21)C(=O)C3=CC=CC4=CC=CC=C43"),
    ("A-836339", "CC1=C(SC(=NC(=O)C2C(C2(C)C)(C)C)N1CCOC)C"),
    ("L-759,633", "CCCCCCC(C)(C)C1=CC2=C([C@@H]3CC(=CC[C@H]3C(O2)(C)C)C)C(=C1)OC"),
    ("L-759,656", "CCCCCCC(C)(C)C1=CC2=C([C@@H]3CC(=C)CC[C@H]3C(O2)(C)C)C(=C1)OC"),
    ("JWH-210", "CCCCCN1C=C(C2=CC=CC=C21)C(=O)C3=CC=C(C4=CC=CC=C43)CC"),
    ("BAY 59-3074", "C1=CC(=CC(=C1)OS(=O)(=O)CCCC(F)(F)F)OC2=CC=CC(=C2C#N)C(F)(F)F"),
    ("UR-144", "CCCCCN1C=C(C2=CC=CC=C21)C(=O)C3C(C3(C)C)(C)C"),
    ("CB-13", "CCCCCOC1=CC=C(C2=CC=CC=C21)C(=O)C3=CC=CC4=CC=CC=C43"),
    ("AM-2201", "C1=CC=C2C(=C1)C=CC=C2C(=O)C3=CN(C4=CC=CC=C43)CCCCCF"),
    ("JWH-073", "CCCCN1C=C(C2=CC=CC=C21)C(=O)C3=CC=CC4=CC=CC=C43"),
    ("XLR-11", "CC1(C(C1(C)C)C(=O)C2=CN(C3=CC=CC=C32)CCCCCF)C"),
]


def skeleton(smiles):
    return MolToInchiKey(Chem.MolFromSmiles(smiles)).split("-")[0]


def build_questions(data):
    """(question, expected, kind): kind "pki" accepts any curated pKi of the ligand (stereo variants) within 0.01."""
    keys = {s: skeleton(s) for s in data.measurements["smiles"].dropna().unique()}
    measured = data.measurements.assign(skeleton=data.measurements["smiles"].map(keys))
    questions = []
    for name, smiles in LIGANDS:
        rows = measured[measured["skeleton"] == skeleton(smiles)]
        curated = sorted({float(data.compounds.at[data.row_of_smiles[s], "pki"]) for s in rows["smiles"] if s in data.row_of_smiles})
        expected, kind = (curated, "pki") if curated else (("REMOVED", "text") if len(rows) else ("NOT IN DATA", "text"))
        questions.append((f"What is the CB2 Ki of {name} in this dataset?", expected, kind))
    return questions


def is_correct(answer, expected, kind):
    if kind == "text":
        return answer.strip().upper().startswith(expected)
    number = re.search(r"\d+(?:\.\d+)?", answer)
    return bool(number) and any(abs(float(number.group(0)) - value) <= 0.01 for value in expected)


def use_original_lookup():
    """Variant A: names compared with their punctuation, and no near-miss search."""
    assistant.name_key = lambda name: str(name).lower()
    assistant.name_pattern = lambda name: r"(?<![\w-])" + re.escape(name.lower()) + r"(?![\w-])"
    assistant.NEAR_MISS_SEARCH = False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=["A", "B"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--questions", type=int, nargs="*", help="Question numbers to run (default: all)")
    parser.add_argument("--repeats", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.variant == "A":
        use_original_lookup()
    data, client = assistant.ResearchData(ROOT), assistant.make_client()
    rows = []
    questions = [(number, *question) for number, question in enumerate(build_questions(data), 1)
                 if not arguments.questions or number in arguments.questions]
    for number, question, expected, kind in [q for _ in range(arguments.repeats) for q in questions]:
        usage, calls = {}, []

        def on_tool(name, call_arguments, result, images):
            calls.append((name, call_arguments, "possible_intended_molecules" in result))

        messages = assistant.new_conversation() + [{"role": "user", "content": question + ANSWER_FORMAT}]
        started, error, reply = time.time(), "", ""
        try:
            reply = assistant.chat_turn(client, data, messages, usage=usage, on_tool=on_tool)
        except Exception as failure:  # Record and continue: one failed call should not lose the run.
            error = f"{type(failure).__name__}: {failure}"
        answer = final_answer(reply)
        rows.append({"variant": arguments.variant, "number": number, "question": question, "kind": kind,
                     "expected": expected, "answer": answer, "correct": is_correct(answer, expected, kind),
                     "near_miss_shown": any(shown for _, _, shown in calls),
                     "answer_says_near_miss": bool(re.search(r"near[ -]miss", reply, re.I)),
                     "tools": ", ".join(name for name, _, _ in calls),
                     "tool_arguments": " | ".join(arguments_text for _, arguments_text, _ in calls),
                     "n_tool_calls": len(calls), "seconds": round(time.time() - started, 1),
                     "cost_usd": usage.get("cost_usd", 0.0), "error": error, "reply": reply})
        row = rows[-1]
        print(f"{arguments.variant} {number:2d} {'OK ' if row['correct'] else 'BAD'} {row['seconds']:5.1f}s "
              f"expected {expected} got {answer!r} [{row['tools']}]{' NEAR-MISS' if row['near_miss_shown'] else ''}",
              flush=True)
        pd.DataFrame(rows).to_csv(arguments.out, index=False)   # Saved after every question.


if __name__ == "__main__":
    main()
