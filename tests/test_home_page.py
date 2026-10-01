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


class ChartTests(unittest.TestCase):
    def charts(self):
        facts = home_page.facts(DATA, ROOT)
        colours = home_page.COLOURS["light"]
        return {"pyramid": home_page.pyramid_chart(facts["funnel"], colours),
                "strongest": home_page.neighbour_chart(facts["strongest"], colours),
                "weakest": home_page.neighbour_chart(facts["weakest"], colours)}

    def test_each_selection_lives_on_one_layer(self):
        # Vega-Lite rejects a selection shared by two layers ("Duplicate signal name"), and the browser then
        # draws nothing at all, silently. That is how the neighbour chart once vanished.
        for name, spec in self.charts().items():
            for parameter in spec.get("params", []):
                self.assertEqual(len(parameter.get("views", [])), 1, name)

    def test_charts_render(self):
        # A full render with the Vega-Lite engine, when it is installed (it is not a project dependency).
        try:
            import vl_convert
        except ImportError:
            self.skipTest("vl-convert not installed")
        for name, spec in self.charts().items():
            self.assertTrue(vl_convert.vegalite_to_svg({**spec, "width": 600}).startswith("<svg"), name)

    def test_no_raw_hover_pop_ups(self):
        # Every layer's pop-up is switched off explicitly, except the neighbour charts' click strips.
        for name, spec in self.charts().items():
            for index, layer in enumerate(spec["layer"]):
                if name != "pyramid" and index == len(spec["layer"]) - 1:
                    self.assertIn("tooltip", layer["encoding"], name)
                else:
                    self.assertIsNone(layer["mark"]["tooltip"], f"{name} layer {index}")

    def test_pyramid_layers_narrow_steadily_and_readably(self):
        widths = home_page.layer_widths([22523, 5825, 4576, 4081, 3905, 3576])
        self.assertEqual(widths, sorted(widths, reverse=True))
        self.assertAlmostEqual(widths[0], 1.0)
        self.assertGreaterEqual(min(widths), 0.3)                       # Even the last layer has room for its count.
        self.assertTrue(all(a - b > 0.04 for a, b in zip(widths, widths[1:])))   # Every step visibly narrows.


class NameTests(unittest.TestCase):
    def test_named_molecules_are_shown_by_name(self):
        # A clicked neighbour is captioned with its ChEMBL name when it has one (EGCG, next to the weakest binder).
        weakest = home_page.facts(DATA, ROOT)["weakest"]
        egcg = next(n for n in weakest["neighbours"] if n["chembl_id"] == "CHEMBL297453")
        self.assertEqual(home_page.display_name(egcg), "Epigalocatechin Gallate (CHEMBL297453)")
        self.assertEqual(home_page.display_name({"name": None, "chembl_id": "CHEMBL600647"}), "CHEMBL600647")


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
        self.assertEqual(spec["layer"][-1]["mark"]["opacity"], 0.001)   # The near-invisible click strips, on top.
        bars = next(layer for layer in spec["layer"] if "strokeWidth" in layer.get("encoding", {}))
        self.assertIs(bars["encoding"]["strokeWidth"]["condition"]["empty"], False)


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
            # The audit opens in the page: a tab per workbook sheet, and the scores link to it.
            self.assertIn("The audit", [s.value for s in app.subheader])
            self.assertEqual([tab.label for tab in app.tabs][:3], ["Controls", "Dataset questions", "Chemistry questions"])
            self.assertTrue(any(f"](#{home_page.AUDIT_ANCHOR})" in m.value for m in app.markdown))
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
