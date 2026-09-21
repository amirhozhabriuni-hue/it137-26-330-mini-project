import json
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from data_loader import load_dataset  # noqa: E402
from features import FEATURE_COLUMNS, extract_features  # noqa: E402


class CounterfactualTests(unittest.TestCase):
    def setUp(self):
        df = load_dataset()
        feature_df = extract_features(df)

        X = feature_df[FEATURE_COLUMNS].values
        y = feature_df["label"].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

        self.clf = DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
            class_weight="balanced",
        )
        self.clf.fit(self.X_train, self.y_train)

    def test_saved_counterfactuals_are_valid(self):
        path = ROOT / "results" / "counterfactual_examples.json"
        self.assertTrue(path.exists(), "Day 3 results file is missing")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        examples = data.get("all_generated_counterfactuals", [])
        self.assertGreaterEqual(len(examples), 3)

        for example in examples:
            cf = np.asarray(example["counterfactual"], dtype=float)
            predicted = int(self.clf.predict(cf.reshape(1, -1))[0])
            self.assertEqual(predicted, int(example["target_prediction"]))
            self.assertTrue(example["validity"])

    def test_counterfactual_metrics_match_saved_examples(self):
        path = ROOT / "results" / "counterfactual_examples.json"
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        examples = data["all_generated_counterfactuals"]
        valid_count = sum(bool(e["validity"]) for e in examples)

        self.assertEqual(valid_count, data["valid_counterfactuals"])
        self.assertEqual(data["counterfactuals_generated"], len(examples))

        representatives = data.get("representative_examples", [])
        self.assertEqual(len(representatives), 3)
        self.assertEqual(
            len({example["rule"] for example in representatives}),
            3,
        )


if __name__ == "__main__":
    unittest.main()
