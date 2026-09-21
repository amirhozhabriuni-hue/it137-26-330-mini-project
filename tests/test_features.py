import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from data_loader import load_dataset  # noqa: E402
from features import FEATURE_COLUMNS, extract_features  # noqa: E402


class FeatureExtractionTests(unittest.TestCase):
    def test_dataset_and_binary_labels(self):
        df = load_dataset()
        self.assertEqual(len(df), 969)
        self.assertEqual(int((df["label"] == 0).sum()), 444)
        self.assertEqual(int((df["label"] == 1).sum()), 525)

    def test_fixed_feature_space(self):
        df = load_dataset()
        feature_df = extract_features(df)

        self.assertEqual(len(FEATURE_COLUMNS), 12)
        self.assertEqual(
            list(feature_df[FEATURE_COLUMNS].columns),
            FEATURE_COLUMNS,
        )
        self.assertEqual(len(feature_df), len(df))
        self.assertTrue(feature_df[FEATURE_COLUMNS].notna().all().all())

    def test_binary_feature_ranges(self):
        df = load_dataset()
        feature_df = extract_features(df)

        binary_features = [
            "has_digit",
            "has_unit",
            "has_shall",
            "has_must",
            "passive_indicator",
            "has_list_marker",
        ]

        for name in binary_features:
            self.assertTrue(
                set(feature_df[name].unique()).issubset({0, 1}),
                msg=f"Unexpected values in {name}",
            )

        self.assertTrue((feature_df["uppercase_ratio"] >= 0).all())
        self.assertTrue((feature_df["uppercase_ratio"] <= 1).all())


if __name__ == "__main__":
    unittest.main()
