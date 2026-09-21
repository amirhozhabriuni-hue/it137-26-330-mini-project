"""
Interpretable feature extraction for FR/NFR requirements.

The feature space is intentionally small and explicit so that
counterfactual generation and later SMT verification can use
the same feature representation.
"""

import re
import pandas as pd


# Words/phrases that can appear in functional requirements.
FR_CUES = [
    "shall",
    "must",
    "will",
    "can",
    "may",
    "the system",
    "the software",
    "user shall",
]


# Quality-attribute terms commonly associated with NFRs.
NFR_CUES = [
    "performance",
    "reliability",
    "security",
    "usability",
    "maintainability",
    "portability",
    "scalability",
    "availability",
    "efficiency",
    "response time",
    "accuracy",
    "robustness",
    "testability",
]


NEGATION_TERMS = [
    "not",
    "no",
    "never",
    "without",
    "cannot",
    "don't",
    "doesn't",
]


def count_terms(text: str, terms: list[str]) -> int:
    """Count occurrences of predefined terms."""
    text_lower = text.lower()
    return sum(text_lower.count(term) for term in terms)


def has_digit(text: str) -> int:
    """Return 1 if the requirement contains at least one digit."""
    return int(bool(re.search(r"\d", text)))


def has_unit(text: str) -> int:
    """Return 1 if the requirement contains a quantitative unit."""
    units = [
        "ms",
        "sec",
        "min",
        "hour",
        "gb",
        "mb",
        "%",
        "seconds",
        "milliseconds",
    ]

    text_lower = text.lower()
    return int(any(unit in text_lower for unit in units))


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert requirement text into the fixed 12-feature representation.

    Input:
        DataFrame with:
            requirement
            label

    Output:
        DataFrame containing 12 numeric features + label.
    """

    texts = df["requirement"].astype(str)

    features = pd.DataFrame(index=df.index)

    # 1. Number of words
    features["length_words"] = texts.apply(
        lambda text: len(text.split())
    )

    # 2. Approximate sentence count
    features["sentence_count"] = texts.apply(
        lambda text: max(1, text.count(".") + text.count(";"))
    )

    # 3. Functional-requirement cue count
    features["fr_cue_count"] = texts.apply(
        lambda text: count_terms(text, FR_CUES)
    )

    # 4. Non-functional-requirement cue count
    features["nfr_cue_count"] = texts.apply(
        lambda text: count_terms(text, NFR_CUES)
    )

    # 5. Negation count
    features["negation_count"] = texts.apply(
        lambda text: count_terms(text, NEGATION_TERMS)
    )

    # 6. Contains a digit
    features["has_digit"] = texts.apply(has_digit)

    # 7. Contains a quantitative unit
    features["has_unit"] = texts.apply(has_unit)

    # 8. Contains "shall"
    features["has_shall"] = texts.apply(
        lambda text: int("shall" in text.lower())
    )

    # 9. Contains "must"
    features["has_must"] = texts.apply(
        lambda text: int("must" in text.lower())
    )

    # 10. Simple passive-voice indicator
    features["passive_indicator"] = texts.apply(
        lambda text: int(
            bool(
                re.search(
                    r"\b(be|is|are|was|were)\s+\w+ed\b",
                    text.lower(),
                )
            )
        )
    )

    # 11. Contains a list marker / colon / bullet
    features["has_list_marker"] = texts.apply(
        lambda text: int(
            bool(re.search(r"[:\-•]|\b\d+\.", text))
        )
    )

    # 12. Ratio of uppercase words
    features["uppercase_ratio"] = texts.apply(
        lambda text: (
            sum(1 for word in text.split() if word.isupper())
            / max(1, len(text.split()))
        )
    )

    # Preserve the binary target label
    features["label"] = df["label"].values

    return features


FEATURE_COLUMNS = [
    "length_words",
    "sentence_count",
    "fr_cue_count",
    "nfr_cue_count",
    "negation_count",
    "has_digit",
    "has_unit",
    "has_shall",
    "has_must",
    "passive_indicator",
    "has_list_marker",
    "uppercase_ratio",
]


if __name__ == "__main__":
    from data_loader import load_dataset

    df = load_dataset()
    feature_df = extract_features(df)

    print("=== Feature Extraction Successful ===")
    print(f"Dataset rows: {len(feature_df)}")
    print(f"Number of features: {len(FEATURE_COLUMNS)}")

    print("\n=== Feature Names ===")
    for i, feature in enumerate(FEATURE_COLUMNS, start=1):
        print(f"{i}. {feature}")

    print("\n=== First 5 Feature Vectors ===")
    print(feature_df.head().to_string(index=False))