"""Check the green/yellow number highlighting: what counts as a tool number, a model number, or neither."""
from pathlib import Path
import json
import os
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import number_check  # noqa: E402

TOOL = [json.dumps({"pki": 7.4318, "median_difference": 0.399, "share": 0.109, "p": 1.02e-5, "n": 211,
                    "ki_nm": 10000.0, "difference": -2.42, "year": 2019, "smiles": "CC1CCC(C)CC1"})]


def kinds(answer, given=()):
    """Each number in an answer with its kind, for readable assertions."""
    return {answer[start:end]: kind for start, end, kind in
            number_check.classify(answer, number_check.tool_values(TOOL), given)}


class ClassifyTests(unittest.TestCase):
    def test_tool_numbers_match_at_the_precision_written(self):
        found = kinds("median 7.43, also 7.4318 and 7.4; n = 211; Ki 10,000 nM; p = 1.02×10⁻⁵; in 2019")
        self.assertEqual(set(found.values()), {"tool"})

    def test_percent_and_dropped_sign(self):
        self.assertEqual(kinds("the largest share is 11%")["11%"], "tool")        # 0.109 as a percentage
        self.assertEqual(kinds("a drop of 2.42 pKi")["2.42"], "tool")            # the tool's -2.42

    def test_the_models_own_numbers_are_yellow(self):
        found = kinds("about 2.5-fold, roughly 28-fold, and 17% of papers")
        self.assertEqual(found, {"2.5": "model", "28": "model", "17%": "model"})

    def test_identifiers_code_small_counts_list_markers_and_the_question_are_left_plain(self):
        self.assertEqual(kinds("CHEMBL600647 and C24H36O3"), {})
        self.assertEqual(kinds("2 pairs and 3 caveats"), {"2": None, "3": None})
        self.assertEqual(kinds("10. Tenth point")["10"], None)
        self.assertEqual(kinds("pKi of 9 or 12.5", given=["Which have pKi 9 or 12.5?"]), {"9": None, "12.5": None})
        highlighted, counts = number_check.highlight("Code `pki_range_ge_1.0` and ```\n12.5\n```", TOOL)
        self.assertEqual(counts, {"tool": 0, "model": 0})
        self.assertNotIn("<span", highlighted)

    def test_conventions_are_left_plain_and_x_means_times(self):
        found = kinds("the 95% CI, at the 95% level, 10^0.31 is 2.5x weaker; 95% of papers")
        self.assertEqual(found, {"95%": "model", "10": None, "0.31": "model", "2.5": "model"})
        self.assertEqual([kind for _, _, kind in number_check.classify("95% CI and 95% of papers", [])], [None, "model"])

    def test_highlight_marks_both_kinds_and_labelled_general_knowledge(self):
        highlighted, counts = number_check.highlight(
            "The median is 7.43. From general knowledge, CB2 has 360 amino acids. It is 2.5-fold.", TOOL)
        self.assertEqual(counts, {"tool": 1, "model": 2})
        self.assertIn(f"background-color:{number_check.GREEN};border-radius:3px;padding:0 2px\">7.43</span>", highlighted)
        self.assertIn("2.5</span>", highlighted)
        # The labelled sentence is tinted as a whole, and the decimal point in 7.43 did not end a sentence.
        self.assertEqual(highlighted.count(f"background-color:{number_check.YELLOW};border-radius:3px\">"), 1)
        self.assertIn("From general knowledge, CB2 has", highlighted)

    def test_model_html_is_escaped(self):
        highlighted, _ = number_check.highlight("<script>alert(1)</script> p < 0.05", TOOL)
        self.assertNotIn("<script>", highlighted)
        self.assertIn("&lt;script>", highlighted)


class AppTests(unittest.TestCase):
    def test_answers_are_highlighted_in_the_app(self):
        from streamlit.testing.v1 import AppTest
        import research_assistant as assistant

        def fake_chat_turn(client, data, messages, usage=None, on_tool=None, **options):
            on_tool("lookup_compound", "{}", TOOL[0], [])
            messages.append({"role": "assistant", "content": "Its pKi is 7.43, about 2.5-fold stronger."})
            return "Its pKi is 7.43, about 2.5-fold stronger."

        with patch.dict(os.environ, {"ASSISTANT_PUBLIC_DEMO": "", "DEEPSEEK_API_KEY": "sk-test"}), \
                patch.object(assistant, "chat_turn", fake_chat_turn), \
                patch.object(assistant, "make_client", lambda key=None: object()):
            app = AppTest.from_file(str(ROOT / "scripts" / "assistant_app.py"), default_timeout=120)
            app.run()
            app.chat_input[0].set_value("What is the pKi of nabilone?").run()
            self.assertEqual(len(app.exception), 0)
            answer = next(m.value for m in app.markdown if "7.43" in m.value and "<span" in m.value)
            self.assertIn(f"{number_check.GREEN};border-radius:3px;padding:0 2px\">7.43", answer)
            self.assertIn(f"{number_check.YELLOW};border-radius:3px;padding:0 2px\">2.5", answer)
            self.assertTrue(any("numbers match a tool result" in c.value for c in app.caption))


if __name__ == "__main__":
    unittest.main()
