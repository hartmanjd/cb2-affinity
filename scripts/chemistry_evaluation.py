"""Chemistry-knowledge evaluation: does DeepSeek need a chemistry-specialist model alongside it?

The known-answer evaluation (assistant_evaluation.py) checks that the assistant reports the
*dataset* correctly. This one checks the *chemistry* the language model brings itself, which is
what a specialist model would replace. Every question has an answer that does not come from an LLM:

- structure_reading   molecular formula, ring and stereocentre counts, CIP labels: computed by RDKit
- name_to_structure   SMILES for a named compound: ChEMBL's structure, compared by InChIKey
- structure_to_name   naming a compound from its SMILES: ChEMBL's preferred name
- calculation         Ki from pKi, Cheng-Prusoff, ligand efficiency, occupancy: computed in Python
- pharmacology        CB2 receptor facts: multiple choice, each answer cited to a primary source
- medchem_concepts    general medicinal chemistry: multiple choice, textbook facts

Each question is asked in two modes:

- bare        DeepSeek alone, no tools: the model's own chemistry, which is what a specialist competes with
- assistant   the deployed assistant (full prompt and all tools), which is what a user actually gets

The decision rule (fixed before running, see decide()): a gap in a category the assistant gets
wrong is a reason for a specialist only if no deterministic tool could close it. Counting rings
or converting units is a missing tool, not a missing model.

Run it (DeepSeek, both modes, roughly 90 calls and well under a dollar):

    python scripts/chemistry_evaluation.py --out results/chemistry_evaluation.csv
    python scripts/chemistry_evaluation.py --mode bare --base-url http://localhost:11434/v1 --model <specialist>
"""
from pathlib import Path
import argparse
import math
import random
import re
import sys
import time

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdCIPLabeler, rdMolDescriptors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_assistant as assistant  # noqa: E402
from assistant_evaluation import final_answer  # noqa: E402

ANSWER_FORMAT = ("\n\nEnd your reply with one line 'FINAL ANSWER: <answer>'. For multiple choice give only the "
                 "letter. For a number give only the number. For a structure give only one SMILES. For a formula "
                 "write it in Hill order, e.g. C6H12O6. If you cannot determine the answer, write "
                 "'FINAL ANSWER: UNKNOWN' rather than guessing.")

# The bare mode's brief: the same role, no tools, no dataset. It is deliberately short so the
# score reflects the model's chemistry rather than the instructions.
BARE_SYSTEM_PROMPT = ("You are an expert medicinal chemist and pharmacologist. Answer from your own knowledge. "
                      "Work carefully; accuracy matters more than speed.")

# Whether a deterministic tool could answer the category, which decides what a gap means.
# CIP labels were the one structure question the tools did not compute at first. The first runs
# showed the assistant taking 30 s to 3 minutes to assign them by reasoning, and once answering
# UNKNOWN at the time limit, so describe_molecule now reports them. They stay a separate category
# so that before-and-after runs can be compared.
TOOL_COULD_ANSWER = {
    "structure_reading": "yes: describe_molecule computes it",
    "cip_label": "yes: describe_molecule computes it (added after the first runs)",
    "calculation": "yes: arithmetic a tool could do",
    "name_to_structure": "partly: lookup_compound knows the 30 named dataset molecules only",
    "structure_to_name": "partly: lookup_compound knows the 30 named dataset molecules only",
    "pharmacology": "no: literature knowledge",
    "medchem_concepts": "no: textbook knowledge",
}

# Time allowed per question, and for the forced answer that follows if it runs out. The language
# model cannot see a clock, so the limit is enforced from outside: at expiry the work so far
# (reasoning in bare mode, tool results in assistant mode) is handed back and an answer demanded.
# 120 s is generous: the dataset evaluation averages 6 s a question with tools, and the first run
# of this one stalled for over four minutes on a single bare-mode question.
DEFAULT_TIME_LIMIT = 120
FORCED_ANSWER_SECONDS = 60

# How close a numeric answer must be: absolute tolerance, or a fraction of the true value.
ABSOLUTE_TOLERANCE = {"count": 0.5, "decimal_2": 0.01, "decimal_1": 0.1, "weight": 0.5}
RELATIVE_TOLERANCE = {"relative": 0.03}


def smiles_of(curated, chembl_id):
    """The curated (ChEMBL-derived, RDKit-standardised) SMILES for one ChEMBL molecule ID."""
    rows = curated[curated["chembl_ids"].str.split(";").map(lambda ids: chembl_id in [i.strip() for i in ids])]
    if len(rows) != 1:
        raise ValueError(f"{chembl_id} matches {len(rows)} curated rows, expected exactly 1")
    return rows["smiles"].iloc[0]


def cip_labels(molecule):
    """CIP labels (R/S) of every stereocentre, from RDKit's new CIP labeller (the accurate one)."""
    rdCIPLabeler.AssignCIPLabels(molecule)
    return [atom.GetProp("_CIPCode") for atom in molecule.GetAtoms() if atom.HasProp("_CIPCode")]


def build_questions(data):
    """The question set with independently known answers, as a list of dicts.

    Molecules are chosen by ChEMBL ID so each question is readable; their structures and names
    come from the curated ChEMBL data, and every structural answer is computed here by RDKit.
    """
    compounds = data.compounds
    questions = []

    def add(category, question, expected, kind, source, reject=()):
        questions.append({"category": category, "question": question, "expected": expected, "kind": kind,
                          "source": source, "reject": tuple(reject)})

    # --- Structure reading: what an LLM must do if it reads a SMILES string itself. ---
    for chembl_id in ["CHEMBL2218896", "CHEMBL297453", "CHEMBL220360", "CHEMBL3139186"]:
        smiles = smiles_of(compounds, chembl_id)
        add("structure_reading", f"What is the molecular formula of {smiles}?",
            rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smiles)), "formula", "RDKit CalcMolFormula")
    smiles = smiles_of(compounds, "CHEMBL562668")
    add("structure_reading", f"How many heavy (non-hydrogen) atoms does {smiles} have?",
        Chem.MolFromSmiles(smiles).GetNumHeavyAtoms(), "count", "RDKit GetNumHeavyAtoms")
    smiles = smiles_of(compounds, "CHEMBL1257246")
    add("structure_reading", f"How many rings (smallest set of smallest rings) does {smiles} contain?",
        rdMolDescriptors.CalcNumRings(Chem.MolFromSmiles(smiles)), "count", "RDKit CalcNumRings")
    add("structure_reading", f"How many stereocentres does {smiles} have?",
        len(Chem.FindMolChiralCenters(Chem.MolFromSmiles(smiles), includeUnassigned=True,
                                      useLegacyImplementation=False)), "count", "RDKit FindMolChiralCenters")
    smiles = smiles_of(compounds, "CHEMBL3139186")
    add("structure_reading", f"How many aromatic rings does {smiles} contain?",
        rdMolDescriptors.CalcNumAromaticRings(Chem.MolFromSmiles(smiles)), "count", "RDKit CalcNumAromaticRings")
    smiles = smiles_of(compounds, "CHEMBL189676")
    add("structure_reading", f"What is the average molecular weight of {smiles} in g/mol? Round to one decimal.",
        Descriptors.MolWt(Chem.MolFromSmiles(smiles)), "weight", "RDKit MolWt")

    # CIP labels: molecules with exactly one stereocentre, so the answer is a single R or S.
    for chembl_id in ["CHEMBL120526", "CHEMBL5803028", "CHEMBL5992157"]:
        smiles = smiles_of(compounds, chembl_id)
        labels = cip_labels(Chem.MolFromSmiles(smiles))
        if len(labels) == 1:
            add("cip_label", f"What is the CIP configuration (R or S) of the stereocentre in {smiles}?",
                labels[0], "exact", "RDKit new CIP labeller")
    for chembl_id in ["CHEMBL220360", "CHEMBL297453"]:
        smiles = smiles_of(compounds, chembl_id)
        labels = cip_labels(Chem.MolFromSmiles(smiles))
        add("cip_label", f"{smiles} has two stereocentres. Give their CIP labels in the order the stereocentres "
            f"appear in the SMILES, as two letters separated by a comma (e.g. R,S).", ",".join(labels), "exact",
            "RDKit new CIP labeller")

    # --- Name to structure, and structure to name, for named dataset molecules (ChEMBL names). ---
    # Answers are compared by the first block of the InChIKey (connectivity), so a correct
    # skeleton with missing or wrong stereochemistry still scores; stereo is recorded separately.
    for chembl_id, name in [("CHEMBL15848", "anandamide"), ("CHEMBL122972", "2-arachidonoylglycerol"),
                            ("CHEMBL2218896", "nabilone"), ("CHEMBL267227", "delta-8-tetrahydrocannabinol"),
                            ("CHEMBL445740", "beta-caryophyllene"), ("CHEMBL422704", "cannabichromene"),
                            ("CHEMBL180920", "magnolol"), ("CHEMBL220360", "taranabant")]:
        add("name_to_structure", f"Give a SMILES for {name}, with stereochemistry if it has any.",
            smiles_of(compounds, chembl_id), "smiles", f"ChEMBL structure of {chembl_id}")
    for chembl_id, names, reject in [
            ("CHEMBL15848", ("ANANDAMIDE", "ARACHIDONOYLETHANOLAMIDE", "ARACHIDONYLETHANOLAMIDE", "AEA"), ()),
            ("CHEMBL16901", ("HONOKIOL",), ("METHYL", "MAGNOLOL")),
            ("CHEMBL180920", ("MAGNOLOL",), ("HONOKIOL",)),
            ("CHEMBL267227", ("DELTA8", "D8THC"), ("DELTA9",)),
            ("CHEMBL2218896", ("NABILONE",), ()),
            ("CHEMBL297453", ("EPIGALLOCATECHINGALLATE", "EPIGALLOCATECHIN3GALLATE", "EGCG"), ()),
            ("CHEMBL220360", ("TARANABANT",), ()),
            ("CHEMBL189676", ("SURINABANT",), ())]:
        smiles = smiles_of(compounds, chembl_id)
        add("structure_to_name", f"Which named compound (drug, natural product or endocannabinoid) has the "
            f"structure {smiles}? Give its common name.", names, "name", f"ChEMBL preferred name of {chembl_id}",
            reject)

    # --- Calculations a medicinal chemist does by hand. Truth computed here. ---
    gas_constant, temperature = 1.987204e-3, 298.15   # kcal/(mol K), K
    add("calculation", "A compound has pKi 7.98. What is its Ki in nM?", 10 ** (9 - 7.98), "relative", "Ki = 10^-pKi")
    add("calculation", "In a radioligand displacement assay the IC50 is 50 nM, the radioligand is used at 0.5 nM and "
        "its Kd is 1.0 nM. What is the Ki in nM by the Cheng-Prusoff equation?", 50 / (1 + 0.5 / 1.0), "relative",
        "Cheng & Prusoff 1973: Ki = IC50 / (1 + [L]/Kd)")
    add("calculation", "Two compounds have pKi 8.3 and 6.1. By what factor do their Ki values differ?",
        10 ** (8.3 - 6.1), "relative", "ratio = 10^(delta pKi)")
    add("calculation", "Two independent Ki measurements of one compound are 1 nM and 100 nM. What is their mean "
        "expressed as pKi (average on the log scale)?", (9 + 7) / 2, "decimal_2", "mean of pKi values")
    add("calculation", "What is the standard binding free energy, in kcal/mol, of a ligand with Ki = 10 nM at 298.15 K? "
        "Give a signed value to one decimal.", gas_constant * temperature * math.log(10e-9), "decimal_1",
        "dG = RT ln(Ki)")
    add("calculation", "A ligand has pKi 8.5 and 30 heavy atoms. What is its ligand efficiency in kcal/mol per heavy "
        "atom at 298.15 K? Give two decimals.",
        gas_constant * temperature * math.log(10) * 8.5 / 30, "decimal_2", "LE = -dG / heavy atoms (Hopkins 2004)")
    add("calculation", "At a free ligand concentration of 9 times its Kd, what fraction of receptors is occupied at "
        "equilibrium (simple one-site binding)? Give two decimals.", 9 / (9 + 1), "decimal_2",
        "occupancy = [L] / ([L] + Kd)")

    # --- Multiple choice: CB2 pharmacology (primary literature) and general medicinal chemistry. ---
    # Each entry is (stem, correct option, wrong options, source). The options are shuffled with a
    # fixed seed so the correct letter varies: written by hand, most answers had landed on "B",
    # and a model that favours one letter would then score well for the wrong reason.
    pharmacology = [
        ("Which G protein family does the CB2 receptor couple to primarily?", "Gi/o", ["Gs", "Gq/11", "G12/13"],
         "Howlett et al. 2002, Pharmacol Rev 54:161 (IUPHAR cannabinoid receptor review)"),
        ("Which human gene encodes the CB2 receptor?", "CNR2", ["CNR1", "GPR55", "FAAH"],
         "Munro et al. 1993, Nature 365:61; HGNC symbol CNR2"),
        ("On which human chromosome is the CB2 receptor gene located?", "1", ["6", "12", "X"],
         "HGNC/NCBI Gene: CNR2 at 1p36.11"),
        ("How many amino acids long is the human CB2 receptor?", "360", ["472", "248", "590"],
         "UniProt P34972 (CNR2_HUMAN), 360 aa; CB1 (P21554) is 472"),
        ("Where is the CB2 receptor most highly expressed?", "Immune cells and spleen",
         ["Cerebellum and basal ganglia", "Liver hepatocytes", "Skeletal muscle"],
         "Munro et al. 1993; Galiegue et al. 1995, Eur J Biochem 232:54"),
        ("Which of these is a CB2-selective agonist?", "HU-308", ["Rimonabant", "CP-55,940", "AM251"],
         "Hanus et al. 1999, PNAS 96:14228"),
        ("SR144528 is best described as:", "a CB2-selective antagonist/inverse agonist",
         ["a CB1-selective antagonist", "a non-selective agonist", "a FAAH inhibitor"],
         "Rinaldi-Carmona et al. 1998, JPET 284:644"),
        ("[3H]CP-55,940 is the usual radioligand in CB2 binding assays. Which best describes CP-55,940?",
         "a high-affinity agonist at both CB1 and CB2",
         ["a CB1-selective antagonist", "a CB2-selective inverse agonist", "an endocannabinoid"],
         "Pertwee 1997, Pharmacol Ther 74:129"),
        ("Which endocannabinoid acts as a full agonist at CB2?", "2-Arachidonoylglycerol",
         ["Anandamide", "Oleamide", "Palmitoylethanolamide"],
         "Sugiura et al. 2000, J Biol Chem 275:605; Gonsiorek et al. 2000, Mol Pharmacol 57:1045"),
        ("At the CB2 receptor, delta-9-THC is:", "a partial agonist", ["a full agonist", "a neutral antagonist", "inactive"],
         "Pertwee 2008, Br J Pharmacol 153:199"),
        ("Which dietary terpene was reported as a selective CB2 agonist?", "Beta-caryophyllene",
         ["Limonene", "Myrcene", "Linalool"], "Gertsch et al. 2008, PNAS 105:9099"),
        ("Rimonabant (SR141716A) is:", "CB1-selective", ["CB2-selective", "equipotent at CB1 and CB2", "a CB2 agonist"],
         "Rinaldi-Carmona et al. 1994, FEBS Lett 350:240"),
    ]
    concepts = [
        ("Which group is the classic carboxylic acid bioisostere?", "1H-Tetrazole", ["Methyl ester", "Nitrile", "Phenyl"],
         "Meanwell 2011, J Med Chem 54:2529"),
        ("Which amine is the most basic (highest conjugate-acid pKa)?", "Piperidine", ["Aniline", "Pyridine", "Morpholine"],
         "pKaH: piperidine 11.1, morpholine 8.4, pyridine 5.2, aniline 4.6"),
        ("In Lipinski's rule of five, the hydrogen-bond donor limit is:", "5", ["3", "10", "15"],
         "Lipinski et al. 1997, Adv Drug Deliv Rev 23:3"),
        ("A Bemis-Murcko scaffold keeps:", "ring systems and the linkers between them",
         ["only the largest ring", "all atoms except hydrogens", "the side chains only"],
         "Bemis & Murcko 1996, J Med Chem 39:2887"),
        ("Replacing an aromatic C-H with C-F at a metabolically labile position most often:",
         "blocks CYP oxidation at that position",
         ["makes the ring basic", "adds a hydrogen-bond donor", "removes aromaticity"],
         "Purser et al. 2008, Chem Soc Rev 37:320"),
        ("Two molecules with Tanimoto similarity 0.9 and a 100-fold Ki difference are called:", "an activity cliff",
         ["a matched series", "a scaffold hop", "a privileged structure"], "Maggiora 2006, J Chem Inf Model 46:1535"),
    ]
    shuffler = random.Random(12)
    for category, entries in [("pharmacology", pharmacology), ("medchem_concepts", concepts)]:
        for stem, correct, wrong, source in entries:
            options = [correct] + wrong
            shuffler.shuffle(options)
            letters = "ABCD"
            choices = " ".join(f"{letter}) {option}" for letter, option in zip(letters, options))
            add(category, f"{stem} {choices}", letters[options.index(correct)], "exact", source)
    return questions


def normalise_name(text):
    """Upper-case, Greek deltas spelled out, everything but letters and digits removed."""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper().replace("Δ", "DELTA"))


def normalise_formula(text):
    subscripts = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    return re.sub(r"[^A-Za-z0-9]", "", str(text).translate(subscripts))


def same_structure(answer, expected_smiles):
    """(skeleton matches, stereo matches): InChIKey first block, then the full InChIKey."""
    molecule = Chem.MolFromSmiles(answer.strip().split()[0]) if answer.strip() else None
    if molecule is None:
        return False, False
    got, want = Chem.MolToInchiKey(molecule), Chem.MolToInchiKey(Chem.MolFromSmiles(expected_smiles))
    return got[:14] == want[:14], got == want


def score(answer, row):
    """'correct', 'wrong' or 'unknown' for one final answer."""
    raw = answer or ""
    if not raw.strip():
        return "wrong"                        # No answer line at all is a failure, not humility.
    if raw.strip().upper().startswith("UNKNOWN"):
        return "unknown"
    kind, expected = row["kind"], row["expected"]
    if kind == "smiles":
        # final_answer upper-cases its output, which would turn aromatic "c" into aliphatic "C",
        # so SMILES answers are read from the original reply text instead (see run_evaluation).
        return "correct" if same_structure(raw, expected)[0] else "wrong"
    if kind == "formula":
        return "correct" if normalise_formula(raw).upper() == normalise_formula(expected).upper() else "wrong"
    if kind == "name":
        got = normalise_name(raw)
        hit = any(normalise_name(option) in got for option in expected)
        return "correct" if hit and not any(normalise_name(bad) in got for bad in row["reject"]) else "wrong"
    if kind == "exact":
        return "correct" if re.sub(r"[\s.)*`]", "", raw.upper()) == expected.upper() else "wrong"
    try:
        value = float(re.search(r"-?[0-9][0-9,]*\.?[0-9]*(e-?[0-9]+)?", raw.replace("−", "-")).group(0).replace(",", ""))
    except (AttributeError, ValueError):
        return "wrong"
    tolerance = ABSOLUTE_TOLERANCE.get(kind, RELATIVE_TOLERANCE.get(kind, 0) * abs(float(expected)))
    return "correct" if abs(value - float(expected)) <= tolerance else "wrong"


def raw_final_answer(text):
    """Like final_answer, but keeps the original letter case (needed for SMILES)."""
    lines = [line for line in (text or "").splitlines() if "FINAL ANSWER:" in line.upper()]
    if not lines:
        return ""
    line = lines[-1]
    return line[line.upper().rindex("FINAL ANSWER:") + len("FINAL ANSWER:"):].strip(" *`.")


def request_options(client, thinking):
    """DeepSeek's thinking switch, or a raised context window for a local runner (see chat_turn)."""
    if assistant.is_hosted(str(client.base_url)):
        return {"extra_body": {"thinking": {"type": "enabled" if thinking else "disabled"}}}
    return {"extra_body": {"options": {"num_ctx": assistant.LOCAL_CONTEXT_TOKENS}}}


def add_estimated_usage(usage, model, input_chars, output_chars):
    """Cost of a request stopped mid-stream, which never receives its usage report.

    Roughly four characters per token. Only used for requests cut off at the time limit, so
    the run's total cost is approximate when some questions timed out.
    """
    price = assistant.PRICE_PER_MILLION.get(model, {"input": 0.0, "output": 0.0})
    usage["cost_usd"] = usage.get("cost_usd", 0.0) + (
        input_chars / 4 * price["input"] + output_chars / 4 * price["output"]) / 1e6


def ask_bare(client, question, model, thinking, usage, time_limit=None):
    """One question to the model alone: no tools, no dataset, a short expert brief.

    The reply is streamed, so the model's reasoning arrives piece by piece. If the time limit
    passes first, the stream is closed and the model is asked, with thinking off, to answer
    from the reasoning it had written so far: what it had worked out, not a fresh guess.
    Returns (reply text, whether the answer was forced by the time limit).
    """
    messages = [{"role": "system", "content": BARE_SYSTEM_PROMPT}, {"role": "user", "content": question}]
    deadline = None if time_limit is None else time.time() + time_limit
    reasoning, content, finished = [], [], False
    # timeout guards against a server that goes silent; max_retries=0 so a slow call is not silently retried.
    stream = client.with_options(timeout=time_limit, max_retries=0).chat.completions.create(
        model=model, messages=messages, stream=True, stream_options={"include_usage": True},
        **request_options(client, thinking))
    try:
        for chunk in stream:
            if chunk.usage:
                assistant.add_usage(usage, chunk.usage, model)
                finished = True
            if chunk.choices:
                delta = chunk.choices[0].delta
                # DeepSeek sends its private reasoning as reasoning_content, outside the OpenAI format.
                reasoning.append(getattr(delta, "reasoning_content", None)
                                 or (delta.model_extra or {}).get("reasoning_content") or "")
                content.append(delta.content or "")
            if deadline is not None and time.time() > deadline:
                stream.close()
                break
        else:
            finished = True
    except Exception as error:
        # A read timeout means the server went quiet past the limit: treat it like running out of time.
        if "Timeout" not in type(error).__name__:
            raise
    if finished:
        return "".join(content), False

    partial = ("".join(reasoning) + "".join(content)).strip()
    add_estimated_usage(usage, model, sum(len(m["content"]) for m in messages), len(partial))
    forced = messages + [{"role": "user", "content": (
        "Time is up. " + (f"This is your reasoning so far:\n\n{partial}\n\n" if partial else
                          "You had not written any reasoning yet. ")
        + "Give your best answer now from that reasoning, without starting over. If it is not enough to "
          "decide, answer UNKNOWN." + ANSWER_FORMAT)}]
    response = client.with_options(timeout=FORCED_ANSWER_SECONDS, max_retries=0).chat.completions.create(
        model=model, messages=forced, **request_options(client, thinking=False))
    assistant.add_usage(usage, response.usage, model)
    return response.choices[0].message.content or "", True


def ask_assistant(client, data, question, model, thinking, usage, tools_used, time_limit=None, prefetch_tools=True):
    """One question to the deployed assistant (full prompt and tools), under the same time limit.

    Between tool rounds, chat_turn's deadline stops further rounds once time is up and forces an
    answer from the tool results gathered so far. If a single request runs past the limit, it is
    abandoned and the same forced answer is requested here. Returns (reply, forced).
    """
    messages = assistant.new_conversation(str(client.base_url)) + [{"role": "user", "content": question}]
    deadline = None if time_limit is None else time.time() + time_limit
    try:
        reply = assistant.chat_turn(client.with_options(timeout=time_limit, max_retries=0), data, messages,
                                    model=model, thinking=thinking, usage=usage, deadline=deadline,
                                    prefetch_tools=prefetch_tools,
                                    on_tool=lambda name, *rest: tools_used.append(name))
        return reply, messages[-2]["content"].startswith("Time limit reached")
    except Exception as error:
        if "Timeout" not in type(error).__name__:
            raise
    # The conversation still holds every tool result received before the stalled request.
    messages.append({"role": "user", "content": "Time limit reached. Answer now from the results you already "
                                                "have, and say what is still unchecked." + ANSWER_FORMAT})
    _, tools = assistant.conversation_profile(str(client.base_url))
    response = client.with_options(timeout=FORCED_ANSWER_SECONDS, max_retries=0).chat.completions.create(
        model=model, messages=messages, tools=tools, tool_choice="none", **request_options(client, thinking=False))
    assistant.add_usage(usage, response.usage, model)
    return response.choices[0].message.content or "", True


def run_evaluation(client, data, modes=("bare", "assistant"), model=None, thinking=None, questions=None,
                   repeats=1, progress=print, time_limit=DEFAULT_TIME_LIMIT, output=None, prefetch_tools=True):
    """Ask every question in every mode, each in a fresh conversation. Returns a DataFrame.

    With `output`, every scored answer is appended to that CSV as soon as it exists, so a run can
    be watched while it goes and resumed after an interruption: questions already in the file
    (same repeat, mode and number) are skipped.
    """
    model = model or assistant.DEFAULT_MODEL
    thinking = assistant.DEFAULT_THINKING if thinking is None else thinking
    questions = questions or build_questions(data)
    output = Path(output) if output else None
    done = set()
    if output and output.exists():
        previous = pd.read_csv(output)
        done = set(zip(previous["repeat"], previous["mode"], previous["number"]))
    rows = []
    for repeat in range(1, repeats + 1):
        for mode in modes:
            for number, row in enumerate(questions, 1):
                if (repeat, mode, number) in done:
                    continue
                prompt = row["question"] + ANSWER_FORMAT
                usage, tools_used, failure, forced = {}, [], None, False
                started = time.time()
                try:
                    if mode == "bare":
                        reply, forced = ask_bare(client, prompt, model, thinking, usage, time_limit)
                    else:
                        reply, forced = ask_assistant(client, data, prompt, model, thinking, usage, tools_used,
                                                      time_limit, prefetch_tools)
                except Exception as error:   # A failed call is scored as wrong rather than stopping the run.
                    reply, failure = "", f"{type(error).__name__}: {str(error)[:200]}"
                answer = raw_final_answer(reply) if row["kind"] == "smiles" else final_answer(reply)
                outcome = score(answer, row)
                stereo = same_structure(answer, row["expected"])[1] if row["kind"] == "smiles" else None
                rows.append({"repeat": repeat, "mode": mode, "number": number, "category": row["category"],
                             "question": row["question"], "kind": row["kind"],
                             "expected": "|".join(row["expected"]) if isinstance(row["expected"], tuple) else row["expected"],
                             "answer": answer, "outcome": outcome, "correct": outcome == "correct",
                             "stereo_correct": stereo,
                             # Answered only because the time limit forced it, from the work done so far.
                             "forced_by_time_limit": forced,
                             # The system prompt asks for general-knowledge claims to be labelled as such.
                             "says_general_knowledge": "general knowledge" in reply.lower(),
                             "tools": ", ".join(tools_used),
                             "n_tool_calls": sum("(pre-fetched)" not in name for name in tools_used),
                             "n_prefetched": sum("(pre-fetched)" in name for name in tools_used),
                             "prefetch": prefetch_tools and mode == "assistant",
                             "seconds": round(time.time() - started, 1), "time_limit": time_limit,
                             "cost_usd": usage.get("cost_usd", 0.0),
                             "model": model, "thinking": thinking, "error": failure, "source": row["source"]})
                if output:
                    # Rewrite the whole file (it is small) rather than appending, so rows saved by an
                    # earlier version with different columns still line up when a run is resumed.
                    saved = pd.read_csv(output) if output.exists() else pd.DataFrame()
                    pd.concat([saved, pd.DataFrame(rows[-1:])], ignore_index=True).to_csv(output, index=False)
                progress(f"{mode:<9} {number:>2}/{len(questions)} {outcome:<7} {rows[-1]['seconds']:>5.1f}s "
                         f"{'forced ' if forced else '       '}{len(tools_used)} tools | {row['category']:<17} | "
                         f"expected {str(rows[-1]['expected'])[:18]:<18} got {failure or answer[:50]}")
                if failure and number == 1:
                    raise SystemExit(f"\nThe first question failed, so the rest were not attempted:\n  {failure}")
    return pd.DataFrame(rows)


def category_table(frame):
    """Accuracy per category and mode, with how many answers were wrong rather than 'unknown'."""
    table = (frame.groupby(["category", "mode"])
             .agg(asked=("outcome", "size"), correct=("correct", "sum"),
                  wrong=("outcome", lambda s: (s == "wrong").sum()), unknown=("outcome", lambda s: (s == "unknown").sum()))
             .reset_index())
    table["accuracy"] = table["correct"] / table["asked"]
    table["tool_could_answer"] = table["category"].map(TOOL_COULD_ANSWER)
    return table


def decide(frame, threshold=0.9):
    """Apply the decision rule to the deployed assistant's results, one line per category.

    - accuracy at or above the threshold: no gap.
    - a gap a deterministic tool could close: add or fix the tool; a specialist model is not needed.
    - a gap only knowledge could close: a candidate for a specialist; score one with --mode bare first.
    """
    table = category_table(frame)
    deployed = table[table["mode"] == "assistant"] if "assistant" in set(table["mode"]) else table
    lines = []
    for row in deployed.itertuples():
        if row.accuracy >= threshold:
            verdict = "no gap"
        elif not row.tool_could_answer.startswith("no"):
            verdict = "gap: fix with a tool, not a model"
        else:
            verdict = "gap: knowledge, a specialist candidate"
        lines.append((row.category, row.accuracy, row.wrong, verdict))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="results/chemistry_evaluation.csv", help="Where to write the scored answers.")
    parser.add_argument("--mode", choices=["both", "bare", "assistant"], default="both")
    parser.add_argument("--model", default=None, help="Model name, e.g. deepseek-v4-pro or a local specialist.")
    parser.add_argument("--base-url", default=None, help="OpenAI-format endpoint; default DeepSeek.")
    parser.add_argument("--no-thinking", action="store_true", help="Answer without the model's private reasoning step.")
    parser.add_argument("--repeats", type=int, default=1, help="Ask every question this many times (LLMs vary).")
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT,
                        help="Seconds per question before an answer is forced from the work so far.")
    parser.add_argument("--no-prefetch", action="store_true",
                        help="Assistant mode: do not look up molecules named in the question before the LLM sees it.")
    parser.add_argument("--resume", action="store_true",
                        help="Keep answers already in --out and ask only the missing questions.")
    arguments = parser.parse_args()
    output = Path(arguments.out)
    output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not arguments.resume:
        output.unlink()   # A fresh run replaces the previous one, which is the one-accepted-run rule.
    data = assistant.ResearchData(ROOT)
    client = assistant.make_client(base_url=arguments.base_url)
    if arguments.model is None and not assistant.is_hosted(arguments.base_url):
        raise SystemExit("Pass --model with the local model's name.")
    modes = ("bare", "assistant") if arguments.mode == "both" else (arguments.mode,)
    run_evaluation(client, data, modes=modes, model=arguments.model,
                   thinking=False if arguments.no_thinking else None, repeats=arguments.repeats,
                   time_limit=arguments.time_limit, output=output, prefetch_tools=not arguments.no_prefetch,
                   progress=lambda line: print(line, flush=True))
    frame = pd.read_csv(output)   # Includes any answers kept by --resume.

    print(f"\n{'Category':<18} | {'Mode':<9} | {'Correct':>9} | {'Wrong':>5} | {'Unknown':>7} | Tool could answer?")
    for row in category_table(frame).itertuples():
        print(f"{row.category:<18} | {row.mode:<9} | {f'{row.correct}/{row.asked}':>9} | {row.wrong:>5} | "
              f"{row.unknown:>7} | {row.tool_could_answer}")
    print(f"\n{'Category':<18} | {'Assistant':>9} | {'Wrong':>5} | Verdict")
    for category, accuracy, wrong, verdict in decide(frame):
        print(f"{category:<18} | {accuracy:>9.0%} | {wrong:>5} | {verdict}")
    forced = frame[frame["forced_by_time_limit"]]
    print(f"\n{len(forced)} of {len(frame)} answers were forced by the {arguments.time_limit:.0f} s time limit; "
          f"{forced['correct'].sum()} of those were correct.")
    print(f"{len(frame)} answers, about ${frame['cost_usd'].sum():.3f} total. Saved {output}")


if __name__ == "__main__":
    main()
