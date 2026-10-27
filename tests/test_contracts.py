"""Small software-contract checks; fixtures are not PHM research data."""
import unittest

from phmforge_bench.data import resource_json
from phmforge_bench.runner import _recording_metrics
from verified_benchmark_scoring import classification, regression


class EvaluationContracts(unittest.TestCase):
    def test_classification_uses_identity_not_input_order(self):
        gold = [{"sample_id": "a", "label": 0, "group": "g1", "load_lb": 100},
                {"sample_id": "b", "label": 1, "group": "g2", "load_lb": 200}]
        pred = [{"sample_id": "b", "predicted_label": 1}, {"sample_id": "a", "predicted_label": 0}]
        self.assertEqual(classification(pred, gold, declared_classes=[0, 1, 2])["accuracy"], 1.0)
        self.assertEqual(_recording_metrics(pred, gold)["recording_accuracy"], 1.0)

    def test_duplicate_missing_and_unknown_labels_are_rejected(self):
        gold = [{"sample_id": "a", "label": 0, "group": "g1"}, {"sample_id": "b", "label": 1, "group": "g2"}]
        for pred in ([{"sample_id": "a", "predicted_label": 0}],
                     [{"sample_id": "a", "predicted_label": 0}, {"sample_id": "a", "predicted_label": 0}],
                     [{"sample_id": "a", "predicted_label": 9}, {"sample_id": "b", "predicted_label": 1}]):
            with self.subTest(pred=pred), self.assertRaises(ValueError):
                classification(pred, gold, declared_classes=[0, 1, 2])

    def test_regression_rejects_nonfinite_or_missing_id(self):
        gold = [{"unit_id": 1, "actual_rul": 2.0}, {"unit_id": 2, "actual_rul": 3.0}]
        for pred in ([{"unit_id": 1, "predicted_rul": float("nan")}, {"unit_id": 2, "predicted_rul": 3}],
                     [{"unit_id": 2, "predicted_rul": 3}]):
            with self.subTest(pred=pred), self.assertRaises(ValueError):
                regression(pred, gold)

    def test_original_denominator_is_not_release_coverage(self):
        inv = resource_json("scenario_inventory.json")
        self.assertEqual(len(inv["tasks"]), 75)
        self.assertEqual(sum(x["release_status"] == "integrated_research_adaptation" for x in inv["tasks"]), 2)
        self.assertEqual(len({x["id"] for x in inv["tasks"]}), 75)

    def test_cmapss_train_validation_engines_disjoint(self):
        cfg = resource_json("cmapss_fd001_config.json")
        self.assertEqual(len(cfg["train_engine_ids"]), 80)
        self.assertEqual(len(cfg["validation_engine_ids"]), 20)
        self.assertFalse(set(cfg["train_engine_ids"]) & set(cfg["validation_engine_ids"]))


if __name__ == "__main__":
    unittest.main()
