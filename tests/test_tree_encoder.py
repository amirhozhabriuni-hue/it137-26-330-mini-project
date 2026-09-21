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
from tree_encoder import DecisionTreeSMTEncoder  # noqa: E402
from verifier import (  # noqa: E402
    encoded_prediction,
    get_feature_domain,
    random_valid_vector,
)


class TreeEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = load_dataset()
        feature_df = extract_features(df)

        X = feature_df[FEATURE_COLUMNS].values
        y = feature_df["label"].values

        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

        cls.clf = DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
            class_weight="balanced",
        )
        cls.clf.fit(cls.X_train, cls.y_train)
        cls.encoder = DecisionTreeSMTEncoder(
            cls.clf,
            FEATURE_COLUMNS,
        )

        cls.domain = get_feature_domain(cls.X_train)
        cls.domain_constraints = []

        import z3

        for name in FEATURE_COLUMNS:
            info = cls.domain[name]
            variable = cls.encoder.variables[name]
            cls.domain_constraints.extend(
                [
                    variable >= z3.RealVal(str(float(info["min"]))),
                    variable <= z3.RealVal(str(float(info["max"]))),
                ]
            )

            if info["type"] in {"integer", "binary"}:
                cls.domain_constraints.append(z3.IsInt(variable))

    def test_tree_structure_is_preserved(self):
        self.assertEqual(self.clf.get_depth(), 5)
        self.assertEqual(self.clf.get_n_leaves(), 24)
        self.assertEqual(self.encoder.leaf_count(), 24)

    def test_random_encoded_predictions_match(self):
        rng = np.random.default_rng(42)

        for _ in range(100):
            vector = random_valid_vector(self.domain, rng)
            sklearn_prediction = int(
                self.clf.predict(vector.reshape(1, -1))[0]
            )
            smt_prediction = encoded_prediction(
                self.encoder,
                vector,
                self.domain_constraints,
            )
            self.assertEqual(smt_prediction, sklearn_prediction)


if __name__ == "__main__":
    unittest.main()
