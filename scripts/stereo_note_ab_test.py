"""A/B test: does explaining "unspecified" stereocentres stop DeepSeek's slow stereo detours?

Why this test exists. On the final design (R/S labels in describe_molecule), asking for
nabilone's SMILES took 66 s in the design-stage run and 201 s in the full run, against 20-34 s
before R/S labels existed. Nabilone is recorded as a racemate, so the tool reported its two
stereocentres as "unspecified", and DeepSeek set about assigning them itself, testing candidate
stereoisomers one describe_molecule call at a time. 23% of the curated molecules have at least
one unspecified stereocentre, so this was not a one-off.

The candidate fix is one sentence. When a molecule has an unspecified stereocentre,
describe_molecule's stereochemistry adds a note saying what "unspecified" means and that it
should be reported, not resolved:

    A  the design as tested (no note)
    B  the same design plus the note (UNSPECIFIED_NOTE)

Both variants are created here inside the test process. B won and was adopted: the app now
always behaves as B, and this script recreates A by removing the note.

Decided before running: B is adopted only if, on the same questions paired one by one, it is at
least 30% faster on average AND answers at least as many correctly as A.

    python scripts/stereo_note_ab_test.py --variant A --repeats 2 --out ab_A.csv
    python scripts/stereo_note_ab_test.py --variant B --repeats 2 --out ab_B.csv
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_assistant as assistant  # noqa: E402
import chemistry_evaluation  # noqa: E402

# The note as tested. B won (20/20 correct against 18/20, 16.0 s against 23.6 s per answer), so the
# same sentence is now part of describe_molecule; this script still recreates A by removing it.
UNSPECIFIED_NOTE = assistant.UNSPECIFIED_STEREO_NOTE

# Accepted spellings of "the record leaves it undefined".
UNSPECIFIED_ANSWERS = ("UNSPECIFIED", "UNDEFINED", "NOTSPECIFIED", "NOTDEFINED", "RACEMIC", "RACEMATE")


def questions(data):
    """Ten questions about molecules whose stereocentres are recorded as unspecified.

    Six ask for a SMILES (the pattern that caused the slow detour), scored by InChIKey against
    ChEMBL's structure as in the chemistry evaluation; four ask for the configuration, where the
    right answer is that the record does not define it.
    """
    def smiles(chembl_id):
        return chemistry_evaluation.smiles_of(data.compounds, chembl_id)

    rows = []
    for subject, chembl_id in [("nabilone", "CHEMBL2218896"), ("cannabichromene", "CHEMBL422704"),
                               ("rac-ibipinabant", "CHEMBL158784"), ("the dataset molecule CHEMBL3353429", "CHEMBL3353429"),
                               ("the dataset molecule CHEMBL568468", "CHEMBL568468"),
                               ("the dataset molecule CHEMBL571330", "CHEMBL571330")]:
        rows.append({"category": "smiles_request", "kind": "smiles", "reject": (), "source": f"ChEMBL structure of {chembl_id}",
                     "question": f"Give a SMILES for {subject}, with stereochemistry if it has any.",
                     "expected": smiles(chembl_id)})
    for question in [
            "What are the R/S configurations of nabilone's stereocentres as recorded in this dataset?",
            "Is the stereocentre of cannabichromene defined in this dataset's structure? If so, give R or S.",
            f"What is the configuration (R or S) of each stereocentre in {smiles('CHEMBL559035')}?",
            f"What is the configuration (R or S) of the stereocentre in {smiles('CHEMBL2063241')}?"]:
        rows.append({"category": "configuration", "kind": "name", "reject": (), "source": "ChEMBL record: undefined",
                     "question": question + " If the recorded structure leaves it undefined, answer UNSPECIFIED.",
                     "expected": UNSPECIFIED_ANSWERS})
    return rows


def set_variant(variant):
    """Make describe_molecule behave as variant A (no note) or B (note), in this process only."""
    current = assistant.TOOL_FUNCTIONS["describe_molecule"]

    def describe(data, smiles):
        result = current(data, smiles)
        stereo = result.get("stereochemistry", {})
        stereo.pop("note", None)   # Start from A whether or not the app already has the fix.
        if variant == "B" and any(c["label"] == "unspecified" for c in stereo.get("stereocentres", [])):
            stereo["note"] = UNSPECIFIED_NOTE
        return result

    assistant.TOOL_FUNCTIONS["describe_molecule"] = describe


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=["A", "B"], required=True)
    parser.add_argument("--repeats", type=int, default=2, help="Ask every question this many times (times vary a lot).")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    set_variant(arguments.variant)
    data = assistant.ResearchData(ROOT)
    output = Path(arguments.out)
    output.unlink(missing_ok=True)
    frame = chemistry_evaluation.run_evaluation(
        assistant.make_client(), data, modes=("assistant",), questions=questions(data), repeats=arguments.repeats,
        output=output, progress=lambda line: print(f"{arguments.variant} | {line}", flush=True))
    frame.insert(0, "variant", arguments.variant)
    frame.to_csv(output, index=False)
    print(f"{arguments.variant}: {frame['correct'].sum()}/{len(frame)} correct, {frame['seconds'].mean():.1f} s per answer",
          flush=True)


if __name__ == "__main__":
    main()
