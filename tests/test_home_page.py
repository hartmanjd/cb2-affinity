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
        counts = [count for _, count, _ in facts["funnel"]]
        self.assertEqual(counts, sorted(counts, reverse=True))   # A funnel only ever narrows.
        self.assertAlmostEqual(facts["strongest"]["pki"], curated["pki_target"].max())
        # The strongest and weakest binders, each with five neighbours that are not the molecule itself.
        self.assertAlmostEqual(facts["weakest"]["pki"], curated["pki_target"].min())
        for key in ["strongest", "weakest"]:
            neighbours = facts[key]["neighbours"]
            self.assertEqual(len(neighbours), 5)
            self.assertNotIn(facts[key]["chembl_id"], [n["label"] for n in neighbours])
            similarities = [n["similarity"] for n in neighbours]
            self.assertEqual(similarities, sorted(similarities, reverse=True))
        # The best model and the noise floor agree with the saved model comparison.
        self.assertEqual(facts["models"][0]["model"], "Support vector regression")
        self.assertLess(facts["noise_floor"], facts["models"][0]["rmse_pki"])
        self.assertGreater(facts["dummy_rmse"], facts["models"][-1]["rmse_pki"])


class NeighbourClickTests(unittest.TestCase):
    def test_a_click_picks_that_neighbour_and_no_click_picks_none(self):
        molecule = home_page.facts(DATA, ROOT)["strongest"]
        first = molecule["neighbours"][0]
        self.assertEqual(home_page.picked_neighbour({"selection": {"pick": [{"chembl_id": first["chembl_id"]}]}},
                                                    molecule), first)
        for nothing in [None, {}, {"selection": {}}, {"selection": {"pick": []}}]:
            self.assertIsNone(home_page.picked_neighbour(nothing, molecule))

    def test_bars_are_clickable_and_only_a_clicked_bar_is_outlined(self):
        molecule = home_page.facts(DATA, ROOT)["strongest"]
        spec = home_page.neighbour_chart(molecule, home_page.COLOURS["light"])
        self.assertEqual([p["name"] for p in spec["params"]], ["pick"])
        self.assertEqual(len(spec["params"][0]["views"]), 2)   # Bars and names are both clickable.
        self.assertIs(spec["layer"][0]["encoding"]["strokeWidth"]["condition"]["empty"], False)


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
            self.assertIn("Can a model support the lab?", [s.value for s in app.subheader])
            self.assertIn("The weakest binder", [s.value for s in app.subheader])
            self.assertEqual([m.label for m in app.metric][0], "Molecules")
            self.assertEqual(len(app.expander), 0)      # The tour is shown open on arrival.
            self.assertEqual(app.title[0].value, "Affinity, Audited")
            # A click on a neighbour's bar (simulated: the test runner cannot click charts) shows that
            # neighbour; "Back" returns to the strongest binder.
            neighbour = home_page.facts(DATA, ROOT)["strongest"]["neighbours"][0]
            app.session_state["strongest_neighbours_0"] = {"selection": {"pick": [{"chembl_id": neighbour["chembl_id"]}]}}
            app.run()
            back = next(b for b in app.button if b.label.startswith("Back to CHEMBL600647"))
            self.assertFalse(back.disabled)
            back.click().run()
            back = next(b for b in app.button if b.label.startswith("Back to CHEMBL600647"))
            self.assertTrue(back.disabled)   # Nothing picked any more: the binder itself is shown.
            app.chat_input[0].set_value("How many molecules?").run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn("About this project", [e.label for e in app.expander])


if __name__ == "__main__":
    unittest.main()
