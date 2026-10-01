"""Check the app's home page: its numbers match the project's files, and the page draws."""
from pathlib import Path
import json
import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import home_page  # noqa: E402
from test_research_assistant import DATA  # noqa: E402


class FactTests(unittest.TestCase):
    def test_numbers_match_the_project_files(self):
        facts = home_page.facts(DATA, ROOT)
        curated = pd.read_csv(ROOT / "provenance/curation/curated_structures.csv")
        self.assertEqual(facts["molecules"], len(curated))
        self.assertEqual(facts["funnel"][-1][1], len(curated))
        counts = [count for _, count in facts["funnel"]]
        self.assertEqual(counts, sorted(counts, reverse=True))   # A funnel only ever narrows.
        self.assertAlmostEqual(facts["strongest"]["pki"], curated["pki_target"].max())
        # The swimming-pool fact: Ki (molar) x 2.5 million litres x molecular weight, in milligrams.
        strongest = facts["strongest"]
        weight = DATA.compounds.loc[DATA.compounds["pki"].idxmax(), "mol_weight"]
        self.assertAlmostEqual(strongest["pool_mg"], 10 ** -strongest["pki"] * 2.5e6 * weight * 1000)
        # The best model and the noise floor agree with the saved model comparison.
        self.assertEqual(facts["models"][0]["model"], "Support vector regression")
        self.assertLess(facts["noise_floor"], facts["models"][0]["rmse_pki"])
        self.assertGreater(facts["dummy_rmse"], facts["models"][-1]["rmse_pki"])


class PageTests(unittest.TestCase):
    def test_home_page_draws_and_folds_away_after_a_question(self):
        from streamlit.testing.v1 import AppTest
        import research_assistant as assistant

        def fake_chat_turn(client, data, messages, usage=None, **options):
            messages.append({"role": "assistant", "content": "An answer."})
            return "An answer."

        with patch.dict(os.environ, {"ASSISTANT_PUBLIC_DEMO": "", "DEEPSEEK_API_KEY": "sk-test"}), \
                patch.object(assistant, "chat_turn", fake_chat_turn), \
                patch.object(assistant, "make_client", lambda key=None: object()):
            app = AppTest.from_file(str(ROOT / "scripts" / "assistant_app.py"), default_timeout=120)
            app.run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn("Can a model beat the lab?", [s.value for s in app.subheader])
            self.assertEqual([m.label for m in app.metric][0], "Molecules")
            self.assertEqual(len(app.expander), 1)      # Only "Show the numbers" on the tour itself.
            app.chat_input[0].set_value("How many molecules?").run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn("About this project", [e.label for e in app.expander])


if __name__ == "__main__":
    unittest.main()
