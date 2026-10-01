"""Build the research assistant's audit workbook (results/assistant_audit.xlsx) from the saved test results.

The workbook is the evidence behind the app's claim that the assistant answered 55/55 chemistry and
18/18 dataset questions correctly: every question, what it tests, where its true answer came from,
what the assistant answered, and the conditions held fixed. Every row is read from the result files
the evaluation scripts wrote (results/*.csv); nothing is typed in except the descriptions of what
each dataset question tests, which follow scripts/assistant_evaluation.py.

Like the data provenance workbook, it is written deterministically (fixed timestamps), so rebuilding
it from unchanged results gives identical bytes.

    python scripts/build_assistant_audit.py
"""
from pathlib import Path
import io
import json
import sys

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_data_summary import normalize_xlsx  # noqa: E402

WORKBOOK = Path("results/assistant_audit.xlsx")
STAMP = "2026-10-01T00:00:00Z"   # Fixed save time, so unchanged results give identical bytes.
RUN_DATE = "2026-10-01"          # When the accepted runs were made.

# What each of the 18 dataset questions tests, and how its true answer is computed
# (scripts/assistant_evaluation.py, build_questions), by question number.
DATASET_QUESTIONS = {
    1: ("Counting", "Rows of the curated dataset (provenance/curation/curated_structures.csv)."),
    2: ("Filtering", "pandas filter pKi >= 9 on the curated dataset."),
    3: ("Look-up by name", "Curated pKi of nabilone, found by its ChEMBL ID CHEMBL2218896."),
    4: ("Extreme value", "Maximum curated pKi."),
    5: ("Look-up by value", "ChEMBL ID of the molecule with the maximum pKi, from the retained measurements."),
    6: ("Curation provenance", "Measurements with status 'measurement_conflict' in measurement_decisions.csv."),
    7: ("Chemistry (RDKit scaffolds)", "Distinct Bemis-Murcko scaffolds in the split assignments."),
    8: ("Chemistry (RDKit scaffolds)", "Size of the most common scaffold, counting every scaffold including benzene."),
    9: ("Provenance across tables", "Distinct source documents behind the most common scaffold's molecules."),
    10: ("Provenance", "Document contributing the most retained measurements."),
    11: ("Substructure statistics", "Median pKi of molecules matching an RDKit sulfonamide SMARTS."),
    12: ("Descriptors and filtering", "Mean RDKit molecular weight of molecules with pKi >= 8."),
    13: ("Safeguard: test set", "A reserved test molecule; the correct behaviour is to refuse (WITHHELD)."),
    14: ("Safeguard: scope", "CB1 selectivity is not in a CB2-only dataset; the correct answer is NOT IN DATA."),
    15: ("Project results", "Best scaffold-split model, read from results/model_comparison.md."),
    16: ("Project settings", "SVR regularisation C selected for the scaffold split, from its run manifest."),
    17: ("Project settings", "Measurement-noise floor, read from notebook 10."),
    18: ("Chemistry trap (linkers)", "Three-carbon chains outside every ring between an aromatic ring and a phenyl, "
                                     "counted with RDKit; a careless SMARTS also counts ring atoms."),
}

CHEMISTRY_CATEGORIES = {
    "structure_reading": "Reading a structure: formula, atom and ring counts, stereocentres, molecular weight",
    "cip_label": "Stereochemistry: R/S labels by the Cahn-Ingold-Prelog rules",
    "name_to_structure": "Writing the structure (SMILES) of a named compound",
    "structure_to_name": "Naming a compound from its structure, including near-miss isomers",
    "calculation": "Medicinal-chemistry arithmetic: Ki, Cheng-Prusoff, ligand efficiency, free energy, occupancy",
    "pharmacology": "CB2 receptor pharmacology, from the primary literature",
    "medchem_concepts": "General medicinal-chemistry concepts",
}

CONTROLS = [
    ("Answers no language model wrote",
     "Every expected answer is computed from the project's files with pandas and RDKit, taken from ChEMBL's records, or "
     "cited to a primary paper, before and independently of the assistant.",
     "A test graded by another model would only measure agreement between models."),
    ("One question per fresh conversation",
     "Each question starts a new conversation with only the system prompt.",
     "Earlier answers cannot help later ones."),
    ("Same model and settings throughout",
     "deepseek-v4-pro, thinking on, the full system prompt with all 21 tools, pre-fetch on (the deployed design).",
     "Differences between answers come from the questions, not the configuration."),
    ("A fixed answer format, read by code",
     "The assistant must end with 'FINAL ANSWER: ...'; that line is extracted and scored automatically.",
     "No human judgement in scoring; a missing answer line counts as wrong."),
    ("Strict scoring rules",
     "Counts exact; pKi within 0.01; other decimals within 0.1 (chemistry calculations within 3% or a stated decimal); "
     "structures compared by InChIKey; names must match and known near-misses (e.g. honokiol vs magnolol, delta-8 vs "
     "delta-9 THC) are rejected; multiple choice must be the exact letter.",
     "A loose scorer would flatter the result."),
    ("'Unknown' counted separately",
     "An answer of UNKNOWN is neither right nor wrong.",
     "Admitting uncertainty is safer than a confident wrong answer, and the two should be told apart."),
    ("Multiple-choice positions shuffled",
     "Options are shuffled with a fixed seed; a test fails if any letter is right for more than 40% of questions.",
     "Written by hand, most answers had landed on 'B'; a model favouring one letter would have scored well."),
    ("Time limit with a forced answer",
     "Chemistry questions have 120 s; if exceeded, the assistant must answer from the work done so far, and the row "
     "is marked forced.",
     "Slow answers are visible, not hidden; one assistant-mode question in the accepted run was forced (still correct)."),
    ("Decision rule fixed before running",
     "A category needs at least 90%; a gap a tool could close calls for a tool, only a knowledge gap for a specialist "
     "model (scripts/chemistry_evaluation.py, decide).",
     "The conclusion cannot be fitted to the results after seeing them."),
    ("Refinements compared side by side",
     "Each design stage was re-run at the same time on the same questions (scripts/design_stages.py).",
     "DeepSeek's speed varies with its load; the same questions ran several times slower in the afternoon."),
    ("Behavioural safeguards tested",
     "Reserved test molecules must be refused; questions the data cannot answer must say so.",
     "Correct numbers are not enough; the assistant must also know what it must not do."),
    ("Costs and times recorded",
     "Every answer's time, tool calls and approximate cost (peak-hour prices) are saved.",
     "So speed and cost claims can be checked as well as accuracy."),
]

HEADER_FILL = PatternFill("solid", fgColor="1C5CAB")
HEADER_FONT = Font(bold=True, color="FFFFFF")
OUTCOME_FILLS = {"correct": PatternFill("solid", fgColor="DFF2DF"), "wrong": PatternFill("solid", fgColor="F9DEDC"),
                 "unknown": PatternFill("solid", fgColor="FDF0CC")}
WRAP = Alignment(wrap_text=True, vertical="top")


def read(name):
    return pd.read_csv(ROOT / "results" / name)


def outcome_of(row):
    if "outcome" in row and isinstance(row["outcome"], str):
        return row["outcome"]
    return "correct" if bool(row["correct"]) else "wrong"


def add_table(sheet, columns, rows, widths, outcome_column=None, start_row=1):
    """A header row and data rows, wrapped and sized; rows are coloured by outcome when given."""
    for index, (title, width) in enumerate(zip(columns, widths), 1):
        cell = sheet.cell(row=start_row, column=index, value=title)
        cell.fill, cell.font, cell.alignment = HEADER_FILL, HEADER_FONT, WRAP
        sheet.column_dimensions[get_column_letter(index)].width = width
    for offset, values in enumerate(rows, 1):
        for index, value in enumerate(values, 1):
            cell = sheet.cell(row=start_row + offset, column=index, value=value)
            cell.alignment = WRAP
        if outcome_column is not None:
            fill = OUTCOME_FILLS.get(str(values[outcome_column]).lower())
            if fill:
                for index in range(1, len(values) + 1):
                    sheet.cell(row=start_row + offset, column=index).fill = fill
    sheet.freeze_panes = sheet.cell(row=start_row + 1, column=1)
    return start_row + len(rows) + 2


def tools_used(text):
    return "" if pd.isna(text) else str(text)


def summary(frame):
    return (int(frame["correct"].sum()), len(frame), round(float(frame["seconds"].mean()), 1),
            round(float(frame["seconds"].median()), 1), round(float(frame["seconds"].max()), 1),
            round(float(frame["cost_usd"].sum()), 3))


def overview(root=ROOT):
    """The headline: run conditions, scores per question set, and the verdict."""
    dataset = read("assistant_evaluation.csv")
    chemistry = read("chemistry_evaluation.csv")
    chemistry = chemistry[chemistry["mode"] == "assistant"]
    scores = pd.DataFrame([(name,) + summary(frame) for name, frame in
                           [("Dataset questions", dataset), ("Chemistry questions", chemistry)]],
                          columns=["Question set", "Correct", "Asked", "Mean s", "Median s", "Slowest s", "Cost $"])
    scores["Correct"] = scores["Correct"].astype(str) + "/" + scores["Asked"].astype(str)
    return {
        "title": "Affinity, Audited: the research assistant's audit",
        "conditions": f"Accepted runs recorded {RUN_DATE}. Model {dataset['model'].iloc[0]}, thinking "
                      f"{'on' if dataset['thinking'].iloc[0] else 'off'}, deployed design (full prompt, all tools, pre-fetch).",
        "method": "Every question has a true answer computed independently of the assistant: from the project's files "
                  "with pandas and RDKit, from ChEMBL's records, or from a cited paper. The assistant answered each in a "
                  "fresh conversation and its final answer was scored automatically. See Controls for every condition "
                  "held fixed.",
        "scores": scores,
        "verdict": "Every chemistry category cleared the 90% bar set before the runs, so no chemistry-specialist model "
                   "is needed alongside DeepSeek; where it was slow or wrong, the fix was a tool, not a model.",
    }


def audit_sheets(root=ROOT):
    """Every sheet after the overview: name, what it holds, and its tables.

    Each table is (title or None, DataFrame, column widths for Excel, name of the outcome column or None).
    The workbook and the app's audit section are both drawn from this, so they always agree.
    """
    dataset = read("assistant_evaluation.csv")
    chemistry = read("chemistry_evaluation.csv")
    chemistry = chemistry[chemistry["mode"] == "assistant"]
    stages = read("assistant_design_stages.csv")
    ab = read("stereo_note_ab_test.csv")
    development = read("assistant_development_runs.csv")
    history = read("assistant_model_trials.csv")
    assert set(dataset["number"]) == set(DATASET_QUESTIONS), "a dataset question has no description"
    sheets = []

    controls = pd.DataFrame(CONTROLS, columns=["Condition", "How it was controlled", "Why it matters"])
    sheets.append({"name": "Controls", "about": "Every condition held fixed during testing, how, and why.",
                   "tables": [(None, controls, [30, 70, 55], None)]})

    rows = []
    for row in dataset.sort_values("number").to_dict("records"):
        tests, source = DATASET_QUESTIONS[row["number"]]
        rows.append((row["number"], tests, row["question"], row["expected"], source, row["answer"],
                     "correct" if row["correct"] else "wrong", row["seconds"], row["n_tool_calls"],
                     row["n_prefetched"], tools_used(row["tools"]), round(row["cost_usd"], 4)))
    table = pd.DataFrame(rows, columns=["#", "What it tests", "Question", "True answer", "How the true answer was computed",
                                        "Assistant's answer", "Outcome", "Seconds", "Tool calls", "Pre-fetched",
                                        "Tools used", "Cost $"])
    sheets.append({"name": "Dataset questions",
                   "about": "The 18 questions about the data: what each tests, how its answer was computed, the answer given.",
                   "tables": [(None, table, [5, 22, 50, 16, 50, 18, 10, 9, 9, 10, 40, 9], "Outcome")]})

    rows = [(row["number"], CHEMISTRY_CATEGORIES.get(row["category"], row["category"]), row["question"], row["expected"],
             row["source"], row["answer"], outcome_of(row), "yes" if row["forced_by_time_limit"] else "no",
             row["seconds"], row["n_tool_calls"], row["n_prefetched"], tools_used(row["tools"]), round(row["cost_usd"], 4))
            for row in chemistry.sort_values("number").to_dict("records")]
    table = pd.DataFrame(rows, columns=["#", "Category", "Question", "True answer", "Source of the true answer",
                                        "Assistant's answer", "Outcome", "Forced by time limit", "Seconds", "Tool calls",
                                        "Pre-fetched", "Tools used", "Cost $"])
    sheets.append({"name": "Chemistry questions",
                   "about": "The 55 chemistry questions: category, source of the true answer, the answer given.",
                   "tables": [(None, table, [5, 34, 55, 22, 34, 22, 10, 10, 9, 9, 10, 36, 9], "Outcome")]})

    labels = {"1-original": "1. Original design", "2-prefetch": "2. + pre-fetch", "3-final": "3. + R/S labels"}
    rows = []
    for (question_set, stage), group in stages.groupby(["question_set", "stage"]):
        c, n, mean, median, slowest, cost = summary(group)
        unknown = int((group["outcome"] == "unknown").sum())
        rows.append((question_set, labels.get(stage, stage), f"{c}/{n}", unknown, mean, median, slowest, cost))
    totals = pd.DataFrame(rows, columns=["Question set", "Stage", "Correct", "Unknown", "Mean s", "Median s", "Slowest s", "Cost $"])
    detail = pd.DataFrame([(labels.get(r["stage"], r["stage"]), r["question_set"], r["number"], r["question"], r["expected"],
                            r["answer"], outcome_of(r), r["seconds"], tools_used(r["tools"]))
                           for r in stages.sort_values(["question_set", "number", "stage"]).to_dict("records")],
                          columns=["Stage", "Set", "#", "Question", "True answer", "Answer", "Outcome", "Seconds", "Tools used"])
    sheets.append({"name": "Refinement stages",
                   "about": "The assistant re-run as it stood at each design stage, side by side (scripts/design_stages.py), "
                            "on a fixed 20-question chemistry subset and all 18 dataset questions. Stage 4, the note on "
                            "unspecified stereocentres, was measured by the A-B test.",
                   "tables": [("By stage", totals, [14, 22, 10, 10, 10, 10, 10, 10], None),
                              ("Every answer", detail, [22, 12, 5, 55, 18, 18, 10, 9, 40], "Outcome")]})

    totals = pd.DataFrame([(("A: without the note" if variant == "A" else "B: with the note"),) + summary(group)[:2] + summary(group)[2:5]
                           for variant, group in ab.groupby("variant")],
                          columns=["Variant", "Correct", "Asked", "Mean s", "Median s", "Slowest s"])
    totals["Correct"] = totals["Correct"].astype(str) + "/" + totals["Asked"].astype(str)
    detail = pd.DataFrame([(r["variant"], r["repeat"], r["number"], r["question"], r["expected"], r["answer"], outcome_of(r),
                            "yes" if r["forced_by_time_limit"] else "no", r["seconds"], r["n_tool_calls"])
                           for r in ab.sort_values(["number", "variant", "repeat"]).to_dict("records")],
                          columns=["Variant", "Pass", "#", "Question", "True answer", "Answer", "Outcome", "Forced", "Seconds",
                                   "Tool calls"])
    sheets.append({"name": "Stereo note A-B test",
                   "about": "Ten questions about molecules whose stereocentres are recorded as unspecified, each asked twice "
                            "per variant, side by side. Bar set in advance: at least 30% faster and no fewer correct. B was "
                            "adopted: without the note, nabilone's structure came back wrong in both passes.",
                   "tables": [("By variant", totals, [24, 10, 10, 10, 10, 10], None),
                              ("Every answer", detail, [9, 6, 5, 55, 22, 22, 10, 8, 9, 9], "Outcome")]})

    table = pd.DataFrame([(run, group["model"].iloc[0], group["runs_on"].iloc[0], group["profile"].iloc[0]) + summary(group)
                          for run, group in history.groupby("run", sort=False)],
                         columns=["Run", "Model", "Runs on", "Prompt profile", "Correct", "Asked", "Mean s", "Median s",
                                  "Slowest s", "Cost $"])
    sheets.append({"name": "Model history",
                   "about": "Earlier runs that chose DeepSeek: before and after the speed work, and two local models.",
                   "tables": [(None, table, [48, 18, 10, 12, 9, 8, 9, 9, 10, 9], None)]})

    table = pd.DataFrame([(run, group["question_set"].iloc[0], group["time_of_day"].iloc[0]) + summary(group)
                          for run, group in development.groupby("run", sort=False)],
                         columns=["Run", "Question set", "Time of day", "Correct", "Asked", "Mean s", "Median s",
                                  "Slowest s", "Cost $"])
    sheets.append({"name": "Development runs", "about": "Every exploratory run made while refining the assistant, labelled.",
                   "tables": [(None, table, [62, 12, 11, 9, 8, 9, 9, 10, 9], None)]})
    return sheets


def render(root=ROOT):
    """The workbook's bytes: an Overview sheet, then one sheet per audit_sheets() entry."""
    head, sheets = overview(root), audit_sheets(root)
    book = Workbook()
    sheet = book.active
    sheet.title = "Overview"
    sheet.column_dimensions["A"].width = 34
    for row, (text, font) in enumerate([(head["title"], Font(bold=True, size=16)), (head["conditions"], None),
                                        (head["method"], None)], 1):
        cell = sheet.cell(row=row, column=1, value=text)
        if font:
            cell.font = font
    scores = head["scores"]
    next_row = add_table(sheet, list(scores.columns), list(scores.itertuples(index=False)), [34, 12, 10, 10, 10, 12, 10], start_row=5)
    sheet.freeze_panes = None
    guide = [("Sheet", "What it holds")] + [(s["name"], s["about"]) for s in sheets]
    for offset, (name, text) in enumerate(guide):
        a, b = sheet.cell(row=next_row + offset, column=1, value=name), sheet.cell(row=next_row + offset, column=2, value=text)
        if offset == 0:
            for cell in (a, b):
                cell.fill, cell.font = HEADER_FILL, HEADER_FONT
    verdict_row = next_row + len(guide) + 1
    sheet.cell(row=verdict_row, column=1, value="Verdict").font = Font(bold=True)
    sheet.cell(row=verdict_row + 1, column=1, value=head["verdict"])

    for spec in sheets:
        sheet = book.create_sheet(spec["name"])
        row = 1
        if len(spec["tables"]) > 1 or spec["name"] in ("Refinement stages", "Stereo note A-B test"):
            sheet.cell(row=row, column=1, value=spec["about"])
            row += 2
        for title, table, widths, outcome in spec["tables"]:
            if title:
                sheet.cell(row=row, column=1, value=title).font = Font(bold=True)
                row += 1
            outcome_column = list(table.columns).index(outcome) if outcome else None
            row = add_table(sheet, list(table.columns), list(table.itertuples(index=False)), widths, outcome_column, start_row=row)

    output = io.BytesIO()
    book.save(output)
    return normalize_xlsx(output.getvalue(), STAMP)


def build(root=ROOT):
    destination = Path(root) / WORKBOOK
    data = render(root)
    destination.write_bytes(data)
    return {"workbook": WORKBOOK.as_posix(), "bytes": len(data)}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
