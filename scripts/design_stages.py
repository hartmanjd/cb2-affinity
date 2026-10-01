"""Re-run the assistant as it stood at each stage of its design, to measure what each change bought.

The assistant gained two features after its first evaluation: pre-fetch (molecules named in a
question are looked up before DeepSeek sees it) and R/S labels in describe_molecule. Each stage
is recreated here without touching the app's code, so the stages can run side by side at the
same time. That matters: DeepSeek's speed changes with its load, and the same questions ran
several times slower in the afternoon than in the morning, so stages measured hours apart
would compare the hour as much as the design.

    1-original   pre-fetch off, describe_molecule without R/S labels (the design before this work)
    2-prefetch   pre-fetch on,  describe_molecule without R/S labels
    3-final      pre-fetch on,  R/S labels, as first released
    4-note       pre-fetch on,  R/S labels plus the note explaining "unspecified" (the current app)

Stage 4 came from watching stage 3: it slowed down, and erred, on molecules whose stereocentres are
unspecified. It was measured with a targeted A/B test (scripts/stereo_note_ab_test.py) rather
than by re-running every stage, and stage 3 here removes the note so it matches what was recorded.

Chemistry: a fixed 20-question subset covering every category (CHEMISTRY_SUBSET).
Dataset: all 18 known-answer questions. They never involve R/S, so stages 2 and 3 are the same
code for them and only 1-original and 2-prefetch are run.

    python scripts/design_stages.py --stage 1-original --set chemistry --out stage1_chemistry.csv
"""
from pathlib import Path
import argparse
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_assistant as assistant  # noqa: E402
import assistant_evaluation  # noqa: E402
import chemistry_evaluation  # noqa: E402

# Question numbers in chemistry_evaluation.build_questions, chosen to cover every category:
# formulas and counts, all five R/S questions, name <-> structure, calculations, pharmacology
# and a concept. Fixed here so every stage answers exactly the same 20.
CHEMISTRY_SUBSET = [1, 2, 5, 6,              # structure reading
                    10, 11, 12, 13, 14,      # R/S labels
                    15, 16, 17,              # name -> structure
                    23, 24, 26,              # structure -> name
                    31, 35,                  # calculations
                    38, 43,                  # CB2 pharmacology
                    50]                      # medicinal-chemistry concept

# describe_molecule's description before R/S labels were added (from the commit before them).
ORIGINAL_DESCRIBE_DESCRIPTION = (
    "Compute properties of ANY molecule from its SMILES (need not be in the dataset): formula, "
    "descriptors, named substructures, ring_systems (RDKit-matched names), scaffold and scaffold_name, and whether it is "
    "in the dataset. Use this instead of reading a SMILES yourself, and use its names for rings and scaffolds.")

STAGES = {"1-original": {"prefetch": False, "stereo": False, "note": False},
          "2-prefetch": {"prefetch": True, "stereo": False, "note": False},
          "3-final": {"prefetch": True, "stereo": True, "note": False},
          "4-note": {"prefetch": True, "stereo": True, "note": True}}


def remove_unspecified_note():
    """Recreate stage 3: R/S labels without the later note about unspecified stereocentres."""
    current = assistant.TOOL_FUNCTIONS["describe_molecule"]

    def describe_without_note(data, smiles):
        result = current(data, smiles)
        result.get("stereochemistry", {}).pop("note", None)
        return result

    assistant.TOOL_FUNCTIONS["describe_molecule"] = describe_without_note


def remove_stereo_labels():
    """Put describe_molecule back as it was before R/S labels: same output minus 'stereochemistry'.

    Both the function the tools call and the description DeepSeek reads are restored, in this
    process only; the app and the saved code are unchanged.
    """
    current = assistant.TOOL_FUNCTIONS["describe_molecule"]

    def describe_without_stereo(data, smiles):
        result = current(data, smiles)
        result.pop("stereochemistry", None)
        return result

    assistant.TOOL_FUNCTIONS["describe_molecule"] = describe_without_stereo
    for schema in assistant.TOOL_SCHEMAS:
        if schema["function"]["name"] == "describe_molecule":
            schema["function"]["description"] = ORIGINAL_DESCRIBE_DESCRIPTION


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=list(STAGES), required=True)
    parser.add_argument("--set", choices=["chemistry", "dataset"], required=True)
    parser.add_argument("--out", required=True, help="CSV to write the scored answers to.")
    arguments = parser.parse_args()
    stage = STAGES[arguments.stage]
    if not stage["stereo"]:
        remove_stereo_labels()
    elif not stage["note"]:
        remove_unspecified_note()
    data = assistant.ResearchData(ROOT)
    client = assistant.make_client()
    output = Path(arguments.out)
    output.unlink(missing_ok=True)
    progress = lambda line: print(f"{arguments.stage} {arguments.set} | {line}", flush=True)  # noqa: E731

    if arguments.set == "chemistry":
        questions = chemistry_evaluation.build_questions(data)
        chosen = [questions[number - 1] for number in CHEMISTRY_SUBSET]
        frame = chemistry_evaluation.run_evaluation(client, data, modes=("assistant",), questions=chosen,
                                                    prefetch_tools=stage["prefetch"], progress=progress)
        # run_evaluation numbers the subset 1-20; keep the full-set numbers so stages line up with other runs.
        frame["number"] = [CHEMISTRY_SUBSET[n - 1] for n in frame["number"]]
    else:
        frame = assistant_evaluation.run_evaluation(client, data, prefetch_tools=stage["prefetch"], progress=progress)
    frame.insert(0, "stage", arguments.stage)
    frame.insert(1, "question_set", arguments.set)
    frame.to_csv(output, index=False)
    print(f"{arguments.stage} {arguments.set}: {frame['correct'].sum()}/{len(frame)} correct, "
          f"{frame['seconds'].mean():.1f} s per question. Saved {output}", flush=True)


if __name__ == "__main__":
    main()
