"""Check dataset handoffs and review serialization without downloads or model fits."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from provenance_support import (
    current_selection, current_curation, current_preparation, current_mlp_baseline,
    latest_acquisition, save_review_record, load_review_record, sha256,
)


class DatasetHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = "provenance/raw/snapshot_1/acquisition_manifest.json"
        self.selection = "provenance/selection/snapshot_1/selection_manifest.json"
        self.curation = "provenance/curation/manifest.json"
        self.preparation = "provenance/preparation/manifest.json"
        self.write(self.raw, {"retrieved_at_utc": "2026-01-01", "completed_at_utc": "2026-01-01"})
        self.write(self.selection, {"source_manifest": self.raw})
        self.write(self.curation, {"completed_at_utc": "2026-01-01", "selection_manifest": self.selection,
                                 "file_sha256": {self.selection: sha256(self.root / self.selection)}})
        self.write(self.preparation, {"completed_at_utc": "2026-01-01", "source_curation_manifest": self.curation,
                                    "source_sha256": {self.curation: sha256(self.root / self.curation)}})

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_consolidated_existing_stages_are_usable(self):
        self.assertEqual(current_curation(self.root), Path(self.curation))
        self.assertEqual(current_preparation(self.root), Path(self.preparation))

    def test_incomplete_acquisition_does_not_displace_completed_download(self):
        self.write("provenance/raw/snapshot_2/acquisition_manifest.json", {"retrieved_at_utc": "2026-02-01"})
        self.assertEqual(latest_acquisition(self.root), Path(self.raw))

    def test_new_download_requires_its_own_selection(self):
        self.write("provenance/raw/snapshot_2/acquisition_manifest.json",
                   {"retrieved_at_utc": "2026-02-01", "completed_at_utc": "2026-02-01"})
        with self.assertRaisesRegex(FileNotFoundError, "notebook 01"):
            current_selection(self.root)

    def test_repeated_selection_uses_completed_candidate_without_overwriting_sources(self):
        next_selection = "provenance/selection/snapshot_1/run_01/selection_manifest.json"
        self.write(next_selection, {"source_manifest": self.raw, "completed_at_utc": "2026-02-01"})
        self.write("provenance/selection/snapshot_1/run_02/selection_manifest.json", {"source_manifest": self.raw})
        self.assertEqual(current_selection(self.root), Path(next_selection))
        self.assertTrue((self.root / self.selection).is_file())
        with self.assertRaises(FileNotFoundError):
            current_curation(self.root)

    def test_changed_upstream_bytes_cannot_reuse_old_curation(self):
        self.write(self.selection, {"source_manifest": self.raw, "changed": True})
        with self.assertRaises(FileNotFoundError):
            current_curation(self.root)

    def test_new_matching_curation_does_not_reuse_old_preparation(self):
        new = "provenance/curation/run_01/manifest.json"
        self.write(new, {"completed_at_utc": "2026-02-01", "selection_manifest": self.selection,
                         "file_sha256": {self.selection: sha256(self.root / self.selection)}})
        self.assertEqual(current_curation(self.root), Path(new))
        with self.assertRaisesRegex(FileNotFoundError, "notebook 03"):
            current_preparation(self.root)

    def test_mlp_ignores_wrong_preparation_and_nested_tuning(self):
        parent = "provenance/models/multilayer_perceptron/baseline/"
        correct = {"completed_at_utc": "2026-02-01", "source_preparation_manifest": self.preparation,
                   "source_preparation_manifest_sha256": sha256(self.root / self.preparation)}
        self.write(parent + "run_01/manifest.json", correct)
        self.write(parent + "run_02/manifest.json", {**correct, "completed_at_utc": "2026-03-01",
                                                    "source_preparation_manifest_sha256": "0"*64})
        self.write(parent + "run_01/tuning/manifest.json", {**correct, "completed_at_utc": "2026-04-01"})
        self.assertEqual(current_mlp_baseline(self.root), Path(parent + "run_01/manifest.json"))

    def test_saved_models_remain_visible_but_are_labeled_when_pipeline_advances(self):
        import build_project_reference as reporting
        from unittest.mock import patch
        curation = json.loads((self.root / self.curation).read_text())
        curation["source_raw_folder"] = "provenance/raw/snapshot_1"
        self.write(self.curation, curation)
        self.write("provenance/models/random_forest/files/baseline/manifest.json", {
            "completed_at_utc": "2026-02-01", "source_preparation_manifest": self.preparation,
            "source_preparation_manifest_sha256": sha256(self.root / self.preparation),
        })
        with patch.object(reporting, "ROOT", self.root):
            current = reporting.latest_models({"preparation": self.preparation}, [])
            waiting = reporting.latest_models({"preparation": None}, [])
        self.assertEqual(current[0]["pipeline_status"], "current preparation")
        self.assertEqual(waiting[0]["pipeline_status"], "previous preparation; rerun pending")
        self.assertEqual(waiting[0]["manifest"], current[0]["manifest"])


class ReviewRecordTests(unittest.TestCase):
    def test_json_roundtrip_preserves_review_values_and_literal_relations(self):
        from openpyxl import Workbook
        with tempfile.TemporaryDirectory() as directory:
            book = Workbook()
            sheet = book.active
            sheet.title = "Review"
            sheet.append(["activity_id", "relation", "ki_nm", "reason"])
            sheet.append([123, "=", 0.0125, None])
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = "A1:D2"
            sheet["B2"].data_type = "s"
            path = Path(directory) / "review.json"
            save_review_record(book, path)
            loaded = load_review_record(path)
            self.assertEqual(list(loaded["Review"].values), list(sheet.values))
            self.assertEqual(loaded["Review"]["B2"].data_type, "s")
            self.assertEqual(loaded["Review"].freeze_panes, "A2")
            self.assertEqual(loaded["Review"].auto_filter.ref, "A1:D2")
            self.assertEqual(list(Path(directory).glob("*.xlsx")), [])
            loaded.close()
            book.close()


if __name__ == "__main__":
    unittest.main()
