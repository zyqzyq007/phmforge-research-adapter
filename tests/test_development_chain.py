"""Cross-tool regression test using software fixtures, not PHM research data."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import real_rul_model as rul
from verified_phmforge import VerifiedForge


class DevelopmentToolChain(unittest.TestCase):
    def test_loaded_rul_prefixes_can_be_predicted_and_verified(self):
        with tempfile.TemporaryDirectory(prefix="phmforge-contract-") as folder:
            root = Path(folder)
            public, output = root / "public", root / "output"
            data = public / "datasets" / "CMAPSS_FD001"
            data.mkdir(parents=True)
            (public / "dataset_registry.json").write_text(json.dumps({
                "CMAPSS_FD001": {"task": "rul", "path": "datasets/CMAPSS_FD001",
                                  "available_splits": ["train", "val"],
                                  "test_inputs_visible": False}
            }))
            raw = np.ones((60, 26))
            raw[:, 1] = np.arange(1, 61)
            np.savetxt(data / "train.txt", raw)
            (data / "training_config.json").write_text(json.dumps({
                "window": 30, "validation_engine_ids": [1],
                "validation_prefix_fractions": [0.5, 0.8]
            }))
            backend = VerifiedForge(public, output)
            observations = backend.call("load_dataset", {
                "dataset_name": "CMAPSS_FD001", "split": "val"})
            with np.load(observations["data_path"], allow_pickle=False) as values:
                self.assertFalse(set(values.files) & {"labels", "actual_rul", "groups"})

            # An initialized network suffices for an interface test. This is
            # deliberately not a training test or reported research result.
            model = rul.RULNetwork("mlp", 30, len(rul.SENSORS))
            checkpoint = output / "interface_fixture.pt"
            torch.save({"architecture": "mlp", "window": 30,
                        "sensors": rul.SENSORS, "state_dict": model.state_dict(),
                        "mean": torch.zeros(len(rul.SENSORS)),
                        "scale": torch.ones(len(rul.SENSORS)), "rul_cap": 125}, checkpoint)
            prediction = backend.call("predict_rul", {
                "model_path": str(checkpoint), "test_data": observations["data_path"],
                "unit_id": 0})
            truth = backend.call("load_ground_truth", {
                "dataset_name": "CMAPSS_FD001", "file": "val"})
            verified = backend.call("verify_ground_truth", {
                "predictions": prediction["artifact"], "ground_truth": truth["artifact"]})
            self.assertEqual(verified["status"], "success")
            self.assertEqual(verified["metrics"]["n"], 2)
            self.assertEqual({row["sample_id"] for row in prediction["predictions"]},
                             {"1:30", "1:48"})


if __name__ == "__main__":
    unittest.main()
