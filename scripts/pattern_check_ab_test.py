"""A/B test: does checking the LLM's own substructure patterns make functional-group answers more accurate, and slower?

Why this test exists. When the LLM writes its own SMARTS instead of a named substructure, a loose pattern
gives a confident, correct-looking answer about the wrong molecules: "C(=O)O" written for a carboxylic acid
also counts esters and carbamates. The earlier evaluations had one substructure question, so they could not
show how often this happens.

    A  no check: tool results as before
    B  pattern_check: each pattern the LLM writes is matched against small reference molecules, and the tool
       result lists the functional groups it matches, with a warning when there is more than one

Every answer is compared with a count or median computed here with RDKit from the curated structures, using
the named substructure patterns. Both variants run at the same time so DeepSeek's load affects them equally.

    python -u scripts/pattern_check_ab_test.py --variant A --out ab_pattern_A.csv
"""
from pathlib import Path
import argparse
import json
import sys
import time

import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_assistant as assistant  # noqa: E402
from assistant_evaluation import ANSWER_FORMAT, final_answer, is_correct  # noqa: E402

GROUPS = dict(assistant.FUNCTIONAL_GROUPS, aniline_nh2="[NX3;H2]c", aliphatic_primary_amine="[NX3;H2;!$(NC=[O,S,N])][CX4]",
              secondary_amide="[NX3;H1][CX3](=[OX1])[#6]", alkyl_aryl_ether="[OD2](c)[CX4]")


def build_questions(root=ROOT):
    """(question, expected, kind): counts must be exact, medians within 0.01 pKi."""
    curated = pd.read_csv(root / "provenance/curation/curated_structures.csv")
    molecules = [Chem.MolFromSmiles(s) for s in curated["rdkit_smiles"]]
    has = {name: pd.Series([m.HasSubstructMatch(Chem.MolFromSmarts(smarts)) for m in molecules])
           for name, smarts in GROUPS.items()}

    def count(mask):
        return int(mask.sum())

    def median(mask):
        return float(curated.loc[mask.to_numpy(), "pki_target"].median())

    return [
        ("How many molecules contain a carboxylic acid group?", count(has["carboxylic_acid"]), "count"),
        ("How many molecules contain an ester group?", count(has["ester"]), "count"),
        ("How many molecules contain a urea group?", count(has["urea"]), "count"),
        ("How many molecules contain a carbamate group?", count(has["carbamate"]), "count"),
        ("How many molecules contain a sulfone group (an SO2 bonded to two carbons)?", count(has["sulfone"]), "count"),
        ("How many molecules contain a sulfonamide?", count(has["sulfonamide"]), "count"),
        ("How many molecules contain a nitrile?", count(has["nitrile"]), "count"),
        ("How many molecules contain a ketone?", count(has["ketone"]), "count"),
        ("How many molecules contain a phenol (an OH directly on an aromatic ring)?", count(has["phenol"]), "count"),
        ("How many molecules contain an aliphatic alcohol (an OH on an sp3 carbon)?", count(has["hydroxyl"]), "count"),
        ("How many molecules have an NH2 attached directly to an aromatic ring, as in aniline?",
         count(has["aniline_nh2"]), "count"),
        ("How many molecules have an aliphatic primary amine (an NH2 on an sp3 carbon, not an amide)?",
         count(has["aliphatic_primary_amine"]), "count"),
        ("How many molecules contain a secondary amide, i.e. an amide with an N-H?", count(has["secondary_amide"]), "count"),
        ("How many molecules have an alkyl aryl ether: an oxygen joining an aromatic ring to an sp3 carbon?",
         count(has["alkyl_aryl_ether"]), "count"),
        ("How many molecules contain both an amide and a sulfonamide?", count(has["amide"] & has["sulfonamide"]), "count"),
        ("How many molecules contain a carboxylic acid or an ester (either one)?",
         count(has["carboxylic_acid"] | has["ester"]), "count"),
        ("What is the median curated pKi of molecules with a carboxylic acid group?", median(has["carboxylic_acid"]), "pki"),
        ("What is the median curated pKi of molecules with an ester but no carboxylic acid?",
         median(has["ester"] & ~has["carboxylic_acid"]), "pki"),
        ("What is the median curated pKi of molecules containing a ketone?", median(has["ketone"]), "pki"),
        ("What is the median curated pKi of molecules containing a urea group?", median(has["urea"]), "pki"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=["A", "B"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--questions", type=int, nargs="*", help="Question numbers to run (default: all)")
    parser.add_argument("--repeats", type=int, default=1)
    arguments = parser.parse_args()
    assistant.CHECK_PATTERNS = arguments.variant == "B"
    data, client = assistant.ResearchData(ROOT), assistant.make_client()
    questions = [(number, *question) for number, question in enumerate(build_questions(), 1)
                 if not arguments.questions or number in arguments.questions]
    rows = []
    for number, question, expected, kind in [q for _ in range(arguments.repeats) for q in questions]:
        usage, calls = {}, []

        def on_tool(name, call_arguments, result, images):
            parsed = json.loads(call_arguments or "{}")
            calls.append({"tool": name, "patterns": [parsed[k] for k in assistant.PATTERN_ARGUMENTS if parsed.get(k)],
                          "warned": '"warning"' in result and "pattern_check" in result})

        messages = assistant.new_conversation() + [{"role": "user", "content": question + ANSWER_FORMAT}]
        started, error, reply = time.time(), "", ""
        try:
            reply = assistant.chat_turn(client, data, messages, usage=usage, on_tool=on_tool)
        except Exception as failure:  # Record and continue: one failed call should not lose the run.
            error = f"{type(failure).__name__}: {failure}"
        answer = final_answer(reply)
        rows.append({"variant": arguments.variant, "number": number, "question": question, "kind": kind,
                     "expected": expected, "answer": answer, "correct": is_correct(answer, expected, kind),
                     "patterns": " | ".join(p for call in calls for p in call["patterns"]),
                     "warned": any(call["warned"] for call in calls), "tools": ", ".join(c["tool"] for c in calls),
                     "n_tool_calls": len(calls), "seconds": round(time.time() - started, 1),
                     "cost_usd": usage.get("cost_usd", 0.0), "error": error, "reply": reply})
        row = rows[-1]
        print(f"{arguments.variant} {number:2d} {'OK ' if row['correct'] else 'BAD'} {row['seconds']:5.1f}s "
              f"expected {expected} got {answer!r} patterns [{row['patterns']}]{' WARNED' if row['warned'] else ''}",
              flush=True)
        pd.DataFrame(rows).to_csv(arguments.out, index=False)   # Saved after every question.


if __name__ == "__main__":
    main()
