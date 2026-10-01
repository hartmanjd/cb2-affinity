"""Check the chemistry-knowledge evaluation's answers and scoring without calling DeepSeek."""
from pathlib import Path
import collections
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chemistry_evaluation as evaluation  # noqa: E402
from openai.types.chat import ChatCompletionChunk  # noqa: E402
from test_research_assistant import DATA, FakeClient, reply  # noqa: E402

QUESTIONS = evaluation.build_questions(DATA)


def chunk(reasoning=None, content=None, usage=False):
    """One streamed piece of a DeepSeek reply."""
    delta = {"role": "assistant", "content": content}
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return ChatCompletionChunk.model_validate({
        "id": "x", "object": "chat.completion.chunk", "created": 0, "model": "deepseek-v4-pro",
        "choices": [] if usage else [{"index": 0, "delta": delta, "finish_reason": None}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150} if usage else None})


class FakeStream(list):
    closed = False

    def close(self):
        self.closed = True


class TimedClient(FakeClient):
    """FakeClient plus with_options (used for time limits) and streamed replies."""
    def with_options(self, **options):
        self.options = getattr(self, "options", []) + [options]
        return self

    def create(self, **request):
        result = super().create(**request)
        return FakeStream(result) if request.get("stream") else result


def find(category, text):
    return next(q for q in QUESTIONS if q["category"] == category and text in q["question"])


class AnswerTests(unittest.TestCase):
    def test_known_answers_match_published_values(self):
        # Values checkable against outside references, so a mistake in the RDKit code would show.
        self.assertEqual(find("structure_reading", "c2c(c1)OC(C)(C)C1CCC(=O)CC21")["expected"], "C24H36O3")  # nabilone
        self.assertEqual(find("cip_label", "N[C@H](C)CO")["expected"], "R")        # (R)-methanandamide
        self.assertEqual(find("cip_label", "O=C(O[C@@H]1")["expected"], "R,R")     # (-)-EGCG is 2R,3R
        self.assertAlmostEqual(find("calculation", "Cheng-Prusoff")["expected"], 33.33, places=2)
        self.assertAlmostEqual(find("calculation", "ligand efficiency")["expected"], 0.387, places=3)

    def test_every_category_is_classified_and_letters_are_balanced(self):
        self.assertTrue({q["category"] for q in QUESTIONS} <= set(evaluation.TOOL_COULD_ANSWER))
        letters = collections.Counter(q["expected"] for q in QUESTIONS
                                      if q["category"] in ("pharmacology", "medchem_concepts"))
        # No letter should be right for much more than a quarter of the questions.
        self.assertLessEqual(max(letters.values()) / sum(letters.values()), 0.4)


class ScoringTests(unittest.TestCase):
    def test_structures_compare_by_connectivity_and_record_stereo(self):
        row = find("name_to_structure", "delta-8")
        no_stereo = "CCCCCc1cc(O)c2c(c1)OC(C)(C)C1CC=C(C)CC21"
        self.assertEqual(evaluation.score(no_stereo, row), "correct")
        self.assertEqual(evaluation.same_structure(no_stereo, row["expected"]), (True, False))
        self.assertEqual(evaluation.same_structure(row["expected"], row["expected"]), (True, True))
        # Delta-9 differs from delta-8 only in the double bond position, so it must fail.
        self.assertEqual(evaluation.score("CCCCCc1cc(O)c2c(c1)OC(C)(C)C1CCC(C)=CC21", row), "wrong")
        self.assertEqual(evaluation.score("not a smiles", row), "wrong")

    def test_lower_case_aromatic_smiles_survive_answer_extraction(self):
        text = "Here it is.\nFINAL ANSWER: C=CCc1ccc(O)c(-c2cc(CC=C)ccc2O)c1"
        self.assertEqual(evaluation.raw_final_answer(text), "C=CCc1ccc(O)c(-c2cc(CC=C)ccc2O)c1")

    def test_names_reject_near_misses(self):
        honokiol = find("structure_to_name", "c(-c2ccc(O)c(CC=C)c2)")
        self.assertEqual(evaluation.score("HONOKIOL", honokiol), "correct")
        self.assertEqual(evaluation.score("4-O-METHYLHONOKIOL", honokiol), "wrong")
        delta8 = find("structure_to_name", "CC=C(C)C[C@@H]21")
        self.assertEqual(evaluation.score("Δ8-THC", delta8), "correct")
        self.assertEqual(evaluation.score("DELTA-9-TETRAHYDROCANNABINOL", delta8), "wrong")

    def test_numbers_letters_formulas_and_unknown(self):
        ki = find("calculation", "pKi 7.98")
        self.assertEqual(evaluation.score("10.5", ki), "correct")
        self.assertEqual(evaluation.score("10.5 NM", ki), "correct")
        self.assertEqual(evaluation.score("12", ki), "wrong")
        free_energy = find("calculation", "free energy")
        self.assertEqual(evaluation.score("−10.9", free_energy), "correct")   # Unicode minus sign.
        self.assertEqual(evaluation.score("10.9", free_energy), "wrong")
        letter = QUESTIONS[[q["category"] for q in QUESTIONS].index("pharmacology")]
        self.assertEqual(evaluation.score(f"{letter['expected']})", letter), "correct")
        self.assertEqual(evaluation.score("UNKNOWN", letter), "unknown")
        self.assertEqual(evaluation.score("", letter), "wrong")
        formula = find("structure_reading", "molecular formula")
        self.assertEqual(evaluation.score("C₂₄H₃₆O₃", formula), "correct")


class RunTests(unittest.TestCase):
    def test_bare_mode_sends_no_tools_and_assistant_mode_uses_them(self):
        question = find("calculation", "pKi 7.98")
        bare = [chunk(reasoning="10^(9-7.98)"), chunk(content="FINAL ANSWER: 10.47"), chunk(usage=True)]
        client = TimedClient([bare, reply(tool_calls=[("dataset_overview", {})]), reply("FINAL ANSWER: UNKNOWN")])
        frame = evaluation.run_evaluation(client, DATA, questions=[question], progress=lambda *_: None)
        self.assertNotIn("tools", client.calls[0])
        self.assertTrue(client.calls[0]["stream"])
        self.assertEqual(client.calls[0]["messages"][0]["content"], evaluation.BARE_SYSTEM_PROMPT)
        self.assertIn("tools", client.calls[1])
        self.assertEqual(list(frame["outcome"]), ["correct", "unknown"])
        self.assertEqual(list(frame["n_tool_calls"]), [0, 1])
        self.assertEqual(list(frame["forced_by_time_limit"]), [False, False])
        # Every request carries the time limit as a timeout, and is never silently retried.
        self.assertTrue(all(o["max_retries"] == 0 for o in client.options))

    def test_bare_answer_is_forced_from_the_reasoning_so_far(self):
        question = find("calculation", "pKi 7.98")
        stream = [chunk(reasoning="Ki = 10^(9 - 7.98) nM, which is about 10.5"), chunk(reasoning=" ... still checking")]
        client = TimedClient([stream, reply("From my reasoning.\nFINAL ANSWER: 10.5")])
        frame = evaluation.run_evaluation(client, DATA, modes=("bare",), questions=[question], time_limit=0,
                                          progress=lambda *_: None)
        forced_request = client.calls[1]
        # The forced request hands back the partial reasoning, does not think again, and does not stream.
        self.assertIn("about 10.5", client.requests[1][-1]["content"])
        self.assertEqual(forced_request["extra_body"]["thinking"]["type"], "disabled")
        self.assertNotIn("stream", forced_request)
        self.assertEqual(frame.loc[0, "outcome"], "correct")
        self.assertTrue(frame.loc[0, "forced_by_time_limit"])
        self.assertGreater(frame.loc[0, "cost_usd"], 0)   # The abandoned stream is still costed.

    def test_assistant_answer_is_forced_from_tool_results_at_the_deadline(self):
        question = find("structure_reading", "heavy")
        client = TimedClient([reply("FINAL ANSWER: 35")])
        frame = evaluation.run_evaluation(client, DATA, modes=("assistant",), questions=[question], time_limit=0,
                                          progress=lambda *_: None)
        # With no time left, no tool round starts; the only request is the forced answer.
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["tool_choice"], "none")
        self.assertEqual(client.calls[0]["extra_body"]["thinking"]["type"], "disabled")
        self.assertTrue(client.requests[0][-1]["content"].startswith("Time limit reached"))
        self.assertTrue(frame.loc[0, "forced_by_time_limit"])

    def test_answers_are_saved_as_they_arrive_and_a_run_resumes(self):
        import tempfile
        questions = [find("calculation", "pKi 7.98"), find("calculation", "Cheng-Prusoff")]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run.csv"
            first = TimedClient([reply("FINAL ANSWER: 10.47")])
            evaluation.run_evaluation(first, DATA, modes=("assistant",), questions=questions[:1], output=output,
                                      progress=lambda *_: None)
            self.assertEqual(len(pd.read_csv(output)), 1)
            second = TimedClient([reply("FINAL ANSWER: 33.3")])
            evaluation.run_evaluation(second, DATA, modes=("assistant",), questions=questions, output=output,
                                      progress=lambda *_: None)
            self.assertEqual(len(second.calls), 1)   # Question 1 was already saved, so only question 2 was asked.
            self.assertEqual(list(pd.read_csv(output)["correct"]), [True, True])

    def test_decision_separates_tool_gaps_from_knowledge_gaps(self):
        frame = pd.DataFrame([
            {"category": "calculation", "mode": "assistant", "outcome": "wrong", "correct": False},
            {"category": "pharmacology", "mode": "assistant", "outcome": "wrong", "correct": False},
            {"category": "medchem_concepts", "mode": "assistant", "outcome": "correct", "correct": True}])
        verdicts = {category: verdict for category, _, _, verdict in evaluation.decide(frame)}
        self.assertEqual(verdicts["calculation"], "gap: fix with a tool, not a model")
        self.assertEqual(verdicts["pharmacology"], "gap: knowledge, a specialist candidate")
        self.assertEqual(verdicts["medchem_concepts"], "no gap")


if __name__ == "__main__":
    unittest.main()
