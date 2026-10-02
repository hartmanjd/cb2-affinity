"""Check the research assistant's tools and chat loop without calling DeepSeek.

A fake client replays scripted model replies, so these tests need no API key or
network. They use the real dataset and saved SVR (run scripts/fetch_models.py first).
"""
from pathlib import Path
import json
import sys
import unittest

import pandas as pd
from openai.types.chat import ChatCompletion

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import research_assistant as assistant  # noqa: E402
import project_knowledge  # noqa: E402

DATA = assistant.ResearchData()


def reply(content="", tool_calls=None, reasoning="thinking..."):
    """A Chat Completions response shaped like DeepSeek's, including reasoning_content."""
    message = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if tool_calls:
        message["tool_calls"] = [{"id": f"call_{i}", "type": "function",
                                  "function": {"name": name, "arguments": json.dumps(arguments)}}
                                 for i, (name, arguments) in enumerate(tool_calls)]
    return ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "deepseek-v4-pro",
        "choices": [{"index": 0, "finish_reason": "tool_calls" if tool_calls else "stop", "message": message}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100, "prompt_cache_hit_tokens": 600},
    })


class FakeClient:
    def __init__(self, replies, base_url=assistant.DEEPSEEK_BASE_URL):
        self.replies, self.requests, self.base_url = list(replies), [], base_url
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, **request):
        self.requests.append(json.loads(json.dumps(request["messages"])))   # Snapshot what was sent.
        self.calls.append(request)
        return self.replies.pop(0)


class ToolTests(unittest.TestCase):
    def test_every_tool_has_a_schema_and_errors_are_returned_as_text(self):
        self.assertEqual(len(assistant.TOOL_SCHEMAS), len(assistant.TOOL_FUNCTIONS))
        text, images = assistant.run_tool(DATA, "query_table", {"filters": [{"column": "missing", "op": "==", "value": 1}]})
        self.assertIn("Unknown column", json.loads(text)["error"])
        self.assertIn("error", json.loads(assistant.run_tool(DATA, "no_such_tool", {})[0]))

    def test_query_counts_match_pandas(self):
        result = json.loads(assistant.run_tool(DATA, "query_table", {"filters": [{"column": "pki", "op": ">=", "value": 9}]})[0])
        curated = pd.read_csv(assistant.ROOT / "provenance/curation/curated_structures.csv")
        self.assertEqual(result["total_matching"], int((curated["pki_target"] >= 9).sum()))

    def test_reserved_test_molecules_are_never_predicted(self):
        test_smiles = DATA.compounds.loc[DATA.compounds[f"{assistant.PREDICTION_SPLIT}_subset"] == "test", "smiles"]
        for smiles in test_smiles.head(20):
            result = json.loads(assistant.run_tool(DATA, "predict_pki", {"smiles": smiles})[0])
            self.assertTrue(result["prediction_withheld"])
            self.assertNotIn("predicted_pki", result)

    def test_predictions_match_the_saved_validation_predictions(self):
        saved = pd.read_csv(assistant.SVR_FOLDER / "predictions.csv")
        saved = saved[(saved["split_strategy"] == assistant.PREDICTION_SPLIT) & (saved["subset"] == "validation")].head(10)
        for row in saved.itertuples():
            result = json.loads(assistant.run_tool(DATA, "predict_pki", {"smiles": row.rdkit_smiles})[0])
            self.assertAlmostEqual(result["predicted_pki"], row.predicted_pki, delta=0.006)

    def test_scaffold_names_come_from_rdkit(self):
        # The three scaffolds an early DeepSeek answer misnamed from their SMILES.
        self.assertIn("4-quinolone", assistant.name_scaffold("O=C(NC12CC3CC(CC(C3)C1)C2)c1c[nH]c2ccccc2c1=O"))
        self.assertIn("tetrahydropyran", assistant.name_scaffold("O=C(c1cn(CC2CCOCC2)c2ccccc12)C1CC1"))
        self.assertNotIn("morpholine", assistant.name_scaffold("O=C(c1cn(CC2CCOCC2)c2ccccc12)C1CC1"))
        self.assertEqual(assistant.name_scaffold("O=C(c1c[nH]c(-c2ccccc2)c1)c1cccc2ccccc12"), "pyrrole + benzene + naphthalene")
        self.assertTrue(assistant.name_scaffold("ACYCLIC").startswith("acyclic"))

    def test_scaffold_groups_report_papers_and_intervals(self):
        groups = json.loads(assistant.run_tool(DATA, "aggregate_table", {"group_by": "scaffold", "limit": 15})[0])["groups"]
        thiadiazole = next(g for g in groups if g["scaffold"] == "O=C(N=c1[nH]ncs1)c1ccccc1")
        self.assertEqual(thiadiazole["n_papers"], 1)
        for group in groups:
            self.assertLess(group["mean_ci95_low"], group["mean"])
            self.assertGreater(group["mean_ci95_high"], group["mean"])
            self.assertIn("scaffold_name", group)

    def test_prediction_intervals_cover_95_percent_of_validation_molecules(self):
        # Split conformal guarantees at least 95% coverage on the molecules the width was taken from.
        errors = DATA.validation_errors
        for low, high in assistant.PREDICTION_BANDS:
            band = errors[(errors["maximum_tanimoto"] >= low) & (errors["maximum_tanimoto"] < high)]["absolute_error"]
            self.assertGreaterEqual((band <= assistant.conformal_half_width(band, 0.95)).mean(), 0.95)
        result = json.loads(assistant.run_tool(DATA, "predict_pki", {"smiles": "CCCCCc1cc(O)c2c(c1)OC(C)(C)C1CCC(=O)CC21"})[0])
        low, high = result["prediction_interval_95"]
        inner_low, inner_high = result["prediction_interval_80"]
        self.assertTrue(low < inner_low < result["predicted_pki"] < inner_high < high)

    def test_interval_helpers(self):
        self.assertIsNone(assistant.mean_interval([7.0]))
        self.assertEqual(assistant.measured_interval(8.0, 1), [round(8 - 1.959964 * 0.54, 3), round(8 + 1.959964 * 0.54, 3)])
        low, high = assistant.correlation_interval(0.5, 100)
        self.assertTrue(low < 0.5 < high)
        cliffs = json.loads(assistant.run_tool(DATA, "find_activity_cliffs", {"limit": 50})[0])
        for pair in cliffs["pairs"]:
            self.assertEqual(pair["beyond_measurement_noise"], pair["difference_approx_95_range"][0] > 0)

    def test_private_and_raw_files_cannot_be_read(self):
        names = DATA.library.paths
        self.assertFalse(any(Path(n).name.startswith(".env") or n.startswith((".git/", ".venv/")) for n in names))
        for readable in ["README.md", "AGENTS.MD", "requirements.txt", "scripts/research_assistant.py",
                         "tests/test_research_assistant.py", "notebooks/07_support_vector_regression.ipynb"]:
            self.assertIn(readable, names)
        for name in [".env", "../.env", "/etc/passwd", ".git/config",
                     "provenance/raw/chembl_cb2_20260922T165756862384Z/chembl_cb2_activity.json"]:
            self.assertIn("not in the project library", json.loads(assistant.run_tool(DATA, "read_project_document", {"name": name})[0])["error"])

    def test_files_holding_the_key_or_secret_names_are_never_listed(self):
        import os
        import tempfile
        from unittest.mock import patch
        fake_key = "sk-" + "a1B2" * 8
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"DEEPSEEK_API_KEY": fake_key}):
            root = Path(folder)
            (root / "notes.md").write_text("# Notes\nnothing secret")
            (root / "leak.md").write_text(f"# Oops\nkey = {fake_key}")       # Contains the key: skipped.
            (root / ".env").write_text(f"DEEPSEEK_API_KEY={fake_key}")
            (root / ".env.local").write_text("X=1")
            (root / "secrets.toml").write_text("x = 1")
            (root / ".git").mkdir()
            (root / ".git" / "HEAD").write_text("ref")
            library = project_knowledge.ProjectLibrary(root)
            self.assertEqual(sorted(library.paths), ["notes.md"])
            self.assertEqual(project_knowledge.redact_secrets(f"a {fake_key} b sk-{'z' * 20} task-budgets-2026-03-13"),
                             "a [redacted] b [redacted] task-budgets-2026-03-13")

    def test_api_key_is_redacted_from_tool_results(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "Brc1ccc2c(c1)"}):   # A string that does occur in results.
            text, _ = assistant.run_tool(DATA, "query_table", {"limit": 1, "columns": ["smiles"]})
        self.assertNotIn("Brc1ccc2c(c1)", text)
        self.assertIn("[redacted]", text)

    def test_project_search_and_reading(self):
        hits = json.loads(assistant.run_tool(DATA, "search_project", {"text": "Kramer noise floor"})[0])["hits"]
        self.assertTrue(any(h["name"] == "notebooks/10_noise_ceiling.ipynb" for h in hits[:3]))
        document = json.loads(assistant.run_tool(DATA, "read_project_document", {
            "name": "provenance/models/support_vector_regression/files/tuning/best_parameters.json"})[0])
        self.assertIn('"C": 10.0', document["text"])
        self.assertNotRegex(document["text"], r"[0-9a-f]{64}")   # Checksums are stripped.
        sections = json.loads(assistant.run_tool(DATA, "read_project_document", {
            "name": "notebooks/07_support_vector_regression.ipynb", "sections": "0-1"})[0])["sections"]
        self.assertEqual([s["section"] for s in sections], [0, 1])

    def test_model_comparison_matches_saved_scores(self):
        result = json.loads(assistant.run_tool(DATA, "compare_models", {"split_strategy": "scaffold"})[0])
        models = {(m["model"], m["variant"]): m for m in result["models"]}
        best = next(m for m in result["models"] if m["versus_best"] == "best model on this split")
        self.assertEqual((best["model"], best["variant"]), ("Support vector regression", "tuned"))
        metrics = pd.read_csv(assistant.SVR_FOLDER / "metrics.csv").set_index(["split_strategy", "subset"])
        self.assertAlmostEqual(best["rmse_pki"], metrics.loc[("scaffold", "validation"), "rmse_pki"], places=3)
        self.assertIn(("Fine-tuned ChemBERTa", "fixed settings"), models)
        self.assertIn(("Frozen ChemBERTa + SVR", "tuned SVR on pretrained vectors"), models)
        for model in result["models"]:
            self.assertLessEqual(model["rmse_ci95_low"], model["rmse_pki"])
            self.assertGreaterEqual(model["rmse_ci95_high"], model["rmse_pki"])

    def test_linkers_report_the_difference_between_lengths(self):
        # The step from one chain length to the next comes with its own interval, computed rather than judged
        # from two overlapping mean intervals: for a phenyl, one CH2 vs none is a small but real difference.
        rows = json.loads(assistant.run_tool(DATA, "find_linkers", {"terminal_group": "benzene"})[0])["lengths"]
        one = rows[1]
        self.assertAlmostEqual(one["mean_difference_vs_previous_length"], one["mean_pki"] - rows[0]["mean_pki"], places=2)
        low, high = one["mean_difference_vs_previous_length_ci95"]
        self.assertGreater(low, 0)
        self.assertLess(low, one["mean_difference_vs_previous_length"])
        self.assertGreater(high, one["mean_difference_vs_previous_length"])
        self.assertNotIn("mean_difference_vs_previous_length", rows[0])   # Nothing before length 0.

    def test_linkers_separate_real_chains_from_ring_paths(self):
        result = json.loads(assistant.run_tool(DATA, "find_linkers", {"terminal_group": "benzene", "max_linker_atoms": 4})[0])
        rows = {row["linker_atoms"]: row for row in result["lengths"]}
        # A plain SMARTS reports 10 molecules with a "4-atom linker" to phenyl; none is a real chain.
        self.assertEqual(rows[4]["n_genuine_linker"], 0)
        self.assertEqual(rows[4]["n_ring_path_only"], 10)
        self.assertIsNone(rows[4]["mean_pki"])
        self.assertEqual(rows[3]["n_genuine_linker"], 1)
        self.assertEqual(rows[0]["n_ring_path_only"], 0)        # Directly attached: no chain to be in a ring.
        self.assertGreater(rows[1]["n_genuine_linker"], 400)    # Benzyl is the common, genuine case.
        for row in result["lengths"]:
            matched = row["n_genuine_linker"] + row["n_ring_path_only"]
            mask = assistant.substructure_mask(DATA, DATA.compounds,
                                               "[a]-" + "-".join(["[CX4]"] * row["linker_atoms"]) + "-c1ccccc1"
                                               if row["linker_atoms"] else "[a]-c1ccccc1")
            self.assertEqual(matched, int(mask.sum()))          # Every SMARTS match is in exactly one column.

    def test_loose_patterns_are_flagged_in_the_tool_result(self):
        # "C(=O)O" written for an acid also matches esters and carbamates; the named group is not checked.
        result = json.loads(assistant.run_tool(DATA, "compare_substructure", {"pattern": "C(=O)O"})[0])
        self.assertEqual(result["pattern_check"][0]["matches_reference_groups"], ["carboxylic_acid", "ester", "carbamate"])
        self.assertIn("warning", result["pattern_check"][0])
        self.assertNotIn("pattern_check", json.loads(assistant.run_tool(DATA, "compare_substructure",
                                                                        {"pattern": "carboxylic_acid"})[0]))
        # A strict pattern is reported clean, and each part of "A.B" is checked on its own.
        self.assertIn("status", assistant.pattern_check("C(=O)[OH]"))
        both = assistant.pattern_check("[CX3](=O)[NX3].[SX4](=O)(=O)[NX3]")
        self.assertEqual(both["matches_reference_groups"]["[SX4](=O)(=O)[NX3]"], ["sulfonamide"])
        self.assertIn("amide, urea, carbamate", both["warning"])
        self.assertEqual(assistant.pattern_check("c1ccccc1")["matches_reference_groups"], [])   # Rings do not "match" phenol.

    def test_names_match_without_hyphens_spaces_or_commas(self):
        # ChEMBL stores "Sr-144528" and "Win-552122"; people write SR144528 and WIN 55,212-2.
        for name in ("SR144528", "sr 144528", "WIN 55,212-2"):
            result = json.loads(assistant.run_tool(DATA, "lookup_compound", {"identifier": name})[0])
            self.assertTrue(result["removed_measurements"], name)
        requests = assistant.prefetch_requests(DATA, "Compare SR144528 and WIN 55,212-2")
        self.assertIn(("lookup_compound", {"identifier": "CHEMBL381689"}), requests)
        self.assertIn(("lookup_compound", {"identifier": "CHEMBL188"}), requests)

    def test_a_near_miss_smiles_points_to_the_recorded_molecule(self):
        # SR144528 written from memory with its chlorine and methyl swapped: no exact match, but its isomer is offered.
        swapped = "Cc1ccc(Cn2nc(C(=O)NC3C4(C)CCC(C4)C3(C)C)cc2-c2ccc(C)c(Cl)c2)cc1"
        result = json.loads(assistant.run_tool(DATA, "lookup_compound", {"identifier": swapped})[0])
        self.assertFalse(result["found"])
        first = result["possible_intended_molecules"][0]
        self.assertEqual((first["chembl_ids"], first["same_formula"], first["in_modeling_dataset"]), ("CHEMBL381689", True, False))
        self.assertEqual(first["removal_reasons"], ["pki_range_ge_1.0"])
        self.assertTrue(first["difference_from_query"].startswith("same formula"))
        # Asked by name, a missing name points to a SMILES retry; a homologue is offered with its difference shown.
        self.assertIn("call lookup_compound again with its SMILES",
                      json.loads(assistant.run_tool(DATA, "lookup_compound", {"identifier": "AM630"})[0])["message"])
        jwh_073 = json.loads(assistant.run_tool(DATA, "describe_molecule", {"smiles": "CCCCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21"})[0])
        self.assertEqual(jwh_073["possible_intended_molecules"][0]["difference_from_query"], "+CH2 (a different formula)")

    def test_group_composition_reports_shares_not_one_example(self):
        result = json.loads(assistant.run_tool(DATA, "group_composition", {"substructure": "[a]-[CX4H1]-c1ccccc1"})[0])
        self.assertEqual(result["n_molecules"], 73)
        self.assertAlmostEqual(result["largest_group_share"], 20 / 73, places=3)
        self.assertLess(result["largest_group_share"], 0.5)     # Mixed, so "dominated by" would be wrong.
        self.assertAlmostEqual(sum(g["n"] for g in result["top_groups"]), 73, delta=73)
        self.assertEqual(result["top_groups"][0]["n"], 20)

    def test_query_table_defaults_to_a_compact_set_of_columns(self):
        default = json.loads(assistant.run_tool(DATA, "query_table", {"limit": 20})[0])
        self.assertEqual(list(default["rows"][0]), assistant.DEFAULT_COLUMNS["compounds"])
        self.assertIn("logp", default["columns_not_shown"])
        asked = json.loads(assistant.run_tool(DATA, "query_table", {"limit": 20, "columns": ["smiles", "logp"]})[0])
        self.assertEqual(list(asked["rows"][0]), ["smiles", "logp"])

    def test_system_prompt_carries_the_schema(self):
        # The assistant should not need a tool call to learn the tables, columns or substructure names.
        for expected in ["lipophilic_efficiency", "measurements:", "morpholine", "benzo[c]chromene"]:
            self.assertIn(expected, assistant.SYSTEM_PROMPT)

    def test_images_go_to_the_person_not_the_model(self):
        text, images = assistant.run_tool(DATA, "draw_molecules", {"smiles_list": ["CCO"]})
        self.assertEqual(len(images), 1)
        self.assertTrue(images[0].startswith(b"\x89PNG"))
        self.assertNotIn("_images", text)


class ChatLoopTests(unittest.TestCase):
    def test_tool_round_trip_keeps_reasoning_content(self):
        client = FakeClient([reply(tool_calls=[("dataset_overview", {})], reasoning="plan"),
                             reply("There are 3,576 molecules.")])
        messages = assistant.new_conversation() + [{"role": "user", "content": "How many molecules?"}]
        usage, seen = {}, []
        answer = assistant.chat_turn(client, DATA, messages, on_tool=lambda *call: seen.append(call[0]), usage=usage)
        self.assertEqual(answer, "There are 3,576 molecules.")
        self.assertEqual(seen, ["dataset_overview"])
        # The second request must carry the first reply's reasoning_content and the tool result.
        second = client.requests[1]
        self.assertEqual(second[2]["reasoning_content"], "plan")
        self.assertEqual(second[3]["role"], "tool")
        self.assertEqual(second[3]["tool_call_id"], "call_0")
        self.assertEqual(json.loads(second[3]["content"])["counts"]["compounds"], len(DATA.compounds))
        self.assertEqual(usage["input_tokens"], 2000)
        self.assertGreater(usage["cost_usd"], 0)

    def test_a_local_endpoint_needs_no_key_and_sends_no_deepseek_fields(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            self.assertTrue(assistant.has_api_key("http://localhost:11434/v1"))
            self.assertFalse(assistant.has_api_key(assistant.DEEPSEEK_BASE_URL))
            self.assertEqual(assistant.make_client(base_url="http://localhost:11434/v1").api_key, "local")
        local = FakeClient([reply("answer", reasoning=None)], base_url="http://localhost:11434/v1")
        messages = assistant.new_conversation() + [{"role": "user", "content": "?"}]
        self.assertEqual(assistant.chat_turn(local, DATA, messages), "answer")
        # "thinking" is a DeepSeek parameter; a local runner gets its context window raised instead,
        # because Ollama's 4,096-token default is smaller than the prompt plus tool schemas.
        local_body = local.calls[0]["extra_body"]
        self.assertNotIn("thinking", local_body)
        self.assertEqual(local_body["options"]["num_ctx"], assistant.LOCAL_CONTEXT_TOKENS)
        self.assertGreater(assistant.LOCAL_CONTEXT_TOKENS,
                           (len(assistant.SYSTEM_PROMPT) + len(json.dumps(assistant.TOOL_SCHEMAS))) // 4)
        hosted = FakeClient([reply("answer")])
        assistant.chat_turn(hosted, DATA, assistant.new_conversation() + [{"role": "user", "content": "?"}])
        self.assertIn("thinking", hosted.calls[0]["extra_body"])
        self.assertNotIn("options", hosted.calls[0]["extra_body"])

    def test_local_endpoints_get_the_compact_profile(self):
        full_prompt, full_tools = assistant.conversation_profile(assistant.DEEPSEEK_BASE_URL)
        small_prompt, small_tools = assistant.conversation_profile("http://localhost:11434/v1")
        self.assertIs(full_prompt, assistant.SYSTEM_PROMPT)
        self.assertIs(small_prompt, assistant.COMPACT_SYSTEM_PROMPT)
        self.assertLess(len(small_tools), len(full_tools))
        # Measured limit: Gemma 4 26B loops instead of answering once the brief passes ~1,100 tokens.
        self.assertLess(len(small_prompt) // 4, 550)
        self.assertTrue({s["function"]["name"] for s in small_tools} <= {s["function"]["name"] for s in full_tools})
        # The switch can be forced either way, for like-for-like comparisons between models.
        self.assertIs(assistant.conversation_profile("http://localhost:11434/v1", compact=False)[0],
                      assistant.SYSTEM_PROMPT)
        local = FakeClient([reply("answer", reasoning=None)], base_url="http://localhost:11434/v1")
        assistant.chat_turn(local, DATA, assistant.new_conversation("http://localhost:11434/v1"))
        self.assertEqual(len(local.calls[0]["tools"]), len(small_tools))

    def test_local_models_are_costed_at_zero(self):
        totals = {}
        usage = reply("x").usage
        assistant.add_usage(totals, usage, "gemma4:26b")
        self.assertEqual(totals["cost_usd"], 0.0)
        self.assertEqual(totals["input_tokens"], 1000)
        assistant.add_usage(totals, usage, "deepseek-v4-pro")
        self.assertGreater(totals["cost_usd"], 0.0)

    def test_tool_limit_forces_a_final_answer(self):
        looping = [reply(tool_calls=[("dataset_overview", {})]) for _ in range(2)]
        client = FakeClient(looping + [reply("Best answer so far.")])
        messages = assistant.new_conversation() + [{"role": "user", "content": "?"}]
        self.assertEqual(assistant.chat_turn(client, DATA, messages, max_tool_rounds=2), "Best answer so far.")

    def test_stereo_labels_come_from_rdkit(self):
        def labels(smiles):
            result = json.loads(assistant.run_tool(DATA, "describe_molecule", {"smiles": smiles})[0])["stereochemistry"]
            return [c["label"] for c in result["stereocentres"]], [b["label"] for b in result["double_bonds"]]
        # (R)-methanandamide, with anandamide's four cis (Z) double bonds.
        self.assertEqual(labels("CCCCC/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCC(=O)N[C@H](C)CO"), (["R"], ["Z"] * 4))
        # Taranabant is (S,S), listed in the order the stereocentres appear in the SMILES.
        self.assertEqual(labels("C[C@H](NC(=O)C(C)(C)Oc1ccc(C(F)(F)F)cn1)[C@@H](Cc1ccc(Cl)cc1)c1cccc(C#N)c1")[0], ["S", "S"])
        # Nabilone is recorded as a racemate, so its stereocentres are unspecified rather than guessed,
        # and a note says not to resolve them (the A/B test found DeepSeek otherwise tries, and errs).
        self.assertEqual(labels("CCCCCCC(C)(C)c1cc(O)c2c(c1)OC(C)(C)C1CCC(=O)CC21")[0], ["unspecified"] * 2)
        def note(smiles):
            return json.loads(assistant.run_tool(DATA, "describe_molecule", {"smiles": smiles})[0])["stereochemistry"].get("note")
        self.assertEqual(note("CCCCCCC(C)(C)c1cc(O)c2c(c1)OC(C)(C)C1CCC(=O)CC21"), assistant.UNSPECIFIED_STEREO_NOTE)
        self.assertIsNone(note("CCCCC/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCC(=O)N[C@H](C)CO"))   # Fully defined: no note.


class PrefetchTests(unittest.TestCase):
    def test_molecules_in_a_question_are_recognised_by_code(self):
        smiles = "CCCCCCC(C)(C)c1cc(O)c2c(c1)OC(C)(C)C1CCC(=O)CC21"
        self.assertEqual(assistant.prefetch_requests(DATA, f"What is the formula of {smiles}?"),
                         [("describe_molecule", {"smiles": smiles})])
        # Names and IDs are looked up by ChEMBL ID, which is exact.
        self.assertEqual(assistant.prefetch_requests(DATA, "What is the pKi of nabilone?"),
                         [("lookup_compound", {"identifier": "CHEMBL2218896"})])
        self.assertEqual(assistant.prefetch_requests(DATA, "Tell me about chembl600647"),
                         [("lookup_compound", {"identifier": "CHEMBL600647"})])
        # "Methanandamide" contains "anandamide", and "Rac-Ibipinabant" contains "Ibipinabant":
        # each must be read as the one molecule actually named.
        self.assertEqual(assistant.prefetch_requests(DATA, "Is methanandamide stable?"),
                         [("lookup_compound", {"identifier": "CHEMBL120526"})])
        self.assertEqual(assistant.prefetch_requests(DATA, "What about Rac-Ibipinabant?"),
                         [("lookup_compound", {"identifier": "CHEMBL158784"})])
        # Ordinary words and short fragments are not structures, and dataset-wide questions need nothing.
        self.assertEqual(assistant.prefetch_requests(DATA, "How many molecules have pKi of 9 or higher? See CCO, CO2, SVR."), [])
        self.assertLessEqual(len(assistant.prefetch_requests(DATA, " ".join(["nabilone anandamide cannabidiol",
                                                                             "CHEMBL600647 CHEMBL3410832 honokiol"]))),
                             assistant.MAX_PREFETCH)

    def test_exact_names_win_over_partial_matches(self):
        for name, chembl_id in [("Ibipinabant", "CHEMBL412262"), ("Rac-Ibipinabant", "CHEMBL158784")]:
            result = json.loads(assistant.run_tool(DATA, "lookup_compound", {"identifier": name})[0])
            self.assertEqual(result["compound"]["chembl_ids"], chembl_id)

    def test_results_are_attached_to_the_question_and_shown_as_pre_fetched(self):
        client = FakeClient([reply("Nabilone's curated pKi is 7.98.")])
        messages = assistant.new_conversation() + [{"role": "user", "content": "What is the pKi of nabilone?"}]
        seen = []
        assistant.chat_turn(client, DATA, messages, on_tool=lambda name, *rest: seen.append(name))
        sent = client.requests[0][-1]["content"]
        self.assertTrue(sent.startswith("What is the pKi of nabilone?"))
        self.assertIn(assistant.PREFETCH_HEADER, sent)
        self.assertIn('"chembl_ids": "CHEMBL2218896"', sent)   # The same JSON run_tool gives the LLM.
        self.assertEqual(seen, ["lookup_compound (pre-fetched)"])
        # A question is pre-fetched once, and the switch turns it off.
        assistant.chat_turn(FakeClient([reply("again")]), DATA, messages[:-1])
        self.assertEqual(messages[-2]["content"].count(assistant.PREFETCH_HEADER), 1)
        off = FakeClient([reply("ok")])
        assistant.chat_turn(off, DATA, assistant.new_conversation() + [{"role": "user", "content": "pKi of nabilone?"}],
                            prefetch_tools=False)
        self.assertNotIn(assistant.PREFETCH_HEADER, off.requests[0][-1]["content"])


if __name__ == "__main__":
    unittest.main()
