"""Data and information-boundary contracts; fixtures are not PHM observations."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from phmforge_bench.pdmb_fault import prepare_pdmb_fault
from phmforge_bench.runner import _femto_score, _recording_metrics, _xjtu_score
from phmforge_bench.ldm import _numeric_csv, prepare_ldm
import real_rul_model as rul
from verified_phmforge import VerifiedForge


class AdditionalDatasetContracts(unittest.TestCase):
    def test_bearing_rul_scoring_checks_all_identity_fields_and_balances_units(self):
        gold = [{"sample_id": "a", "unit_id": 1, "scenario_id": 2, "actual_rul": 5},
                {"sample_id": "b", "unit_id": 1, "scenario_id": 2, "actual_rul": 5},
                {"sample_id": "c", "unit_id": 3, "scenario_id": 1, "actual_rul": 5}]
        pred = [{**{k: row[k] for k in ("sample_id", "unit_id", "scenario_id")},
                 "predicted_rul": value} for row, value in zip(gold, (7, 7, 9))]
        score = _xjtu_score(pred, gold)
        self.assertAlmostEqual(score["mae"], 8 / 3)
        self.assertEqual(score["mean_bearing_mae"], 3)
        with self.assertRaisesRegex(ValueError, "unit or scenario identity"):
            _xjtu_score([{**pred[0], "scenario_id": 1}, *pred[1:]], gold)
        with self.assertRaisesRegex(ValueError, "Sample coverage differs"):
            _xjtu_score(pred[:-1], gold)

    def test_ldm_rejects_incomplete_or_tampered_author_files(self):
        with tempfile.TemporaryDirectory(prefix="phmforge-ldm-binding-") as folder:
            root = Path(folder)
            path = root / "source/fft/B02.npy"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"wrong source")
            pin = {"revision": "fixture", "files": [{"path": "fft/B02.npy", "url": "https://example.invalid",
                   "bytes": path.stat().st_size, "sha256": "0" * 64}]}
            with patch("phmforge_bench.ldm.resource_json", return_value=pin):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    prepare_ldm(root / "data", root / "source")
            self.assertFalse((root / "data/READY.json").exists())

    def test_ldm_csv_rejects_invalid_row_width(self):
        with tempfile.TemporaryDirectory(prefix="phmforge-ldm-csv-") as folder:
            path = Path(folder) / "data.csv"
            path.write_text("Time,a,b\nfirst,1\n")
            with self.assertRaisesRegex(ValueError, "CSV row"):
                _numeric_csv(path, 2)

    def test_mixed_source_groups_remain_disjoint_and_test_labels_are_private(self):
        with tempfile.TemporaryDirectory(prefix="phmforge-source-contract-") as folder:
            base = Path(folder)
            source = base / "fixture.csv"
            with source.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["group", "label", "signal"])
                writer.writeheader()
                for group in range(8):
                    for label in (0, 1):
                        writer.writerow({"group": str(group), "label": label,
                                         "signal": "[" + ",".join(str(n + label) for n in range(32)) + "]"})
            pin = {"dataset": "FIXTURE", "source_name": "fixture", "revision": "fixture",
                   "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "bytes": source.stat().st_size,
                   "format": "waveforms", "channels": ["signal"], "group_field": "group",
                   "task_id": "fixture_task", "group_definition": "Software fixture groups"}
            with patch("phmforge_bench.pdmb_fault.resource_json", return_value={"fixture": pin}):
                result = prepare_pdmb_fault(base / "prepared", "fixture", source)
            self.assertFalse(result["group_overlap"])
            root = base / "prepared"
            data = root / "public/datasets/FIXTURE"
            with np.load(data / "train.npz", allow_pickle=False) as train, \
                 np.load(data / "val.npz", allow_pickle=False) as val, \
                 np.load(root / "holdout/test.npz", allow_pickle=False) as held:
                gold = json.loads((root / "private/labels.json").read_text())
                train_groups, val_groups = set(train["groups"]), set(val["groups"])
                test_groups = {item["group"] for item in gold}
                self.assertFalse(train_groups & val_groups or train_groups & test_groups or val_groups & test_groups)
                self.assertFalse(set(held.files) & {"labels", "groups"})
                self.assertEqual(set(held["sample_ids"]), {item["sample_id"] for item in gold})
                self.assertEqual({item["label"] for item in gold}, {0, 1})
            self.assertFalse((root / "public/private").exists())

    def test_mixed_labels_do_not_produce_a_spurious_group_majority_score(self):
        gold = [{"sample_id": "a", "label": 0, "group": "g"},
                {"sample_id": "b", "label": 1, "group": "g"}]
        pred = [{"sample_id": "a", "predicted_label": 0},
                {"sample_id": "b", "predicted_label": 1}]
        metrics = _recording_metrics(pred, gold)
        self.assertIsNone(metrics["recording_accuracy"])
        self.assertEqual(metrics["recordings"][0]["window_accuracy"], 1)

    def test_femto_evaluation_checks_identity_and_reports_ambiguous_annotation(self):
        gold = [{"sample_id": "a", "unit_id": 1, "scenario_id": 1, "actual_rul": 3,
                 "ambiguous_challenge_annotation": False},
                {"sample_id": "b", "unit_id": 2, "scenario_id": 2, "actual_rul": 5,
                 "ambiguous_challenge_annotation": True}]
        pred = [{"sample_id": "b", "unit_id": 2, "scenario_id": 2, "predicted_rul": 4},
                {"sample_id": "a", "unit_id": 1, "scenario_id": 1, "predicted_rul": 3}]
        score = _femto_score(pred, gold)
        self.assertEqual(score["n"], 2)
        self.assertEqual(score["sensitivity_excluding_ambiguous_annotation"]["n"], 1)
        with self.assertRaisesRegex(ValueError, "unit or scenario identity"):
            _femto_score([{**pred[0], "scenario_id": 1}, pred[1]], gold)

    def test_all_cmapss_development_observations_are_label_free(self):
        with tempfile.TemporaryDirectory(prefix="phmforge-cmapss-contract-") as folder:
            root = Path(folder)
            public = root / "public"
            registry = {}
            for index in range(1, 5):
                name = f"CMAPSS_FD00{index}"
                data = public / "datasets" / name
                data.mkdir(parents=True)
                raw = np.ones((60, 26))
                raw[:, 1] = np.arange(1, 61)
                np.savetxt(data / "train.txt", raw)
                (data / "training_config.json").write_text(json.dumps({
                    "window": 30, "validation_engine_ids": [1], "validation_prefix_fractions": [0.5, 0.8]}))
                registry[name] = {"task": "rul", "path": f"datasets/{name}", "available_splits": ["train", "val"]}
            (public / "dataset_registry.json").write_text(json.dumps(registry))
            backend = VerifiedForge(public, root / "output")
            for name in registry:
                with self.subTest(dataset=name):
                    answer = backend.call("load_dataset", {"dataset_name": name, "split": "val"})
                    with np.load(answer["data_path"], allow_pickle=False) as observations:
                        self.assertEqual(observations["dataset"].item(), name)
                        self.assertFalse(set(observations.files) & {"labels", "actual_rul", "groups"})
                        self.assertEqual(len(observations["sample_ids"]), 2)

    def test_cmapss_checkpoint_rejects_a_different_source_subset(self):
        with tempfile.TemporaryDirectory(prefix="phmforge-cmapss-binding-") as folder:
            root = Path(folder)
            model = rul.RULNetwork("mlp", 30, len(rul.SENSORS))
            checkpoint = root / "model.pt"
            torch.save({"dataset": "CMAPSS_FD001", "architecture": "mlp", "window": 30,
                        "sensors": rul.SENSORS, "state_dict": model.state_dict(),
                        "mean": torch.zeros(len(rul.SENSORS)), "scale": torch.ones(len(rul.SENSORS)),
                        "rul_cap": 125}, checkpoint)
            np.savez_compressed(root / "input.npz", dataset=np.asarray("CMAPSS_FD002"),
                                windows=np.ones((1, 30, len(rul.SENSORS)), dtype=np.float32),
                                unit_ids=np.asarray([1]), sample_ids=np.asarray(["sample"]),
                                endpoint_cycles=np.asarray([30]))
            with self.assertRaisesRegex(ValueError, "dataset identities differ"):
                rul.infer(checkpoint, root / "input.npz", root / "out.json")


if __name__ == "__main__":
    unittest.main()
