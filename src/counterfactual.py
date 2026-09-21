"""
Day 3 - Counterfactual Generation

This module searches the valid feature space for minimally
changed instances that flip the Decision Tree prediction.

Important:
The counterfactual is defined in feature space.
It is not a natural-language rewrite of the requirement.
"""

import json
from itertools import combinations, product
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from data_loader import load_dataset
from features import extract_features, FEATURE_COLUMNS


# ---------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_DEPTH = 5

# Number of test cases used for coverage evaluation
N_SELECTED_CASES = 30

# Number of examples stored as representative examples
N_REPRESENTATIVE_EXAMPLES = 3


# ---------------------------------------------------------
# Feature types
# ---------------------------------------------------------

BINARY_FEATURES = {
    "has_digit",
    "has_unit",
    "has_shall",
    "has_must",
    "passive_indicator",
    "has_list_marker",
}

INTEGER_FEATURES = {
    "length_words",
    "sentence_count",
    "fr_cue_count",
    "nfr_cue_count",
    "negation_count",
}

CONTINUOUS_FEATURE = "uppercase_ratio"


# ---------------------------------------------------------
# Data preparation
# ---------------------------------------------------------

def prepare_data():
    """Load data and reproduce the Day 1/Day 2 split."""

    df = load_dataset()

    feature_df = extract_features(df)

    X = feature_df[FEATURE_COLUMNS].values
    y = feature_df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return (
        df,
        feature_df,
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

def train_model(X_train, y_train):
    """Train the same Decision Tree used in previous days."""

    clf = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    clf.fit(X_train, y_train)

    return clf


# ---------------------------------------------------------
# Feature domains
# ---------------------------------------------------------

def build_feature_domains(X_train):
    """
    Define valid feature domains.

    Binary features:
        0 or 1

    Integer features:
        observed training-data range

    uppercase_ratio:
        0.00 to 1.00 in increments of 0.05
    """

    domains = {}

    for index, feature_name in enumerate(FEATURE_COLUMNS):

        if feature_name in BINARY_FEATURES:

            domains[feature_name] = [0, 1]

        elif feature_name in INTEGER_FEATURES:

            minimum = int(
                np.floor(np.min(X_train[:, index]))
            )

            maximum = int(
                np.ceil(np.max(X_train[:, index]))
            )

            domains[feature_name] = list(
                range(minimum, maximum + 1)
            )

        elif feature_name == CONTINUOUS_FEATURE:

            domains[feature_name] = [
                round(value, 2)
                for value in np.arange(
                    0.0,
                    1.0001,
                    0.05,
                )
            ]

    return domains


# ---------------------------------------------------------
# Feature ranges for normalized magnitude
# ---------------------------------------------------------

def build_feature_ranges(X_train):
    """
    Build normalization ranges from the training data.
    """

    ranges = {}

    for index, feature_name in enumerate(FEATURE_COLUMNS):

        if feature_name == CONTINUOUS_FEATURE:

            ranges[feature_name] = 1.0

        else:

            observed_range = (
                np.max(X_train[:, index])
                - np.min(X_train[:, index])
            )

            if observed_range > 0:

                ranges[feature_name] = float(
                    observed_range
                )

            else:

                ranges[feature_name] = 1.0

    return ranges


def normalized_change(
    original,
    candidate,
    feature_ranges,
):
    """
    Compute normalized L1-style feature change.
    """

    total = 0.0

    for index, feature_name in enumerate(FEATURE_COLUMNS):

        change = abs(
            float(candidate[index])
            - float(original[index])
        )

        total += (
            change
            / feature_ranges[feature_name]
        )

    return float(total)


# ---------------------------------------------------------
# Candidate values
# ---------------------------------------------------------

def candidate_values_for_feature(
    original_value,
    feature_name,
    domains,
):
    """
    Generate small valid changes around the original value.
    """

    values = domains[feature_name]

    # Binary feature: flip 0 <-> 1
    if feature_name in BINARY_FEATURES:

        return [
            value
            for value in values
            if value != original_value
        ]

    # Integer feature: change by exactly +/-1
    if feature_name in INTEGER_FEATURES:

        return [
            value
            for value in values
            if abs(value - original_value) == 1
        ]

    # Continuous feature:
    # consider all values defined in the domain
    if feature_name == CONTINUOUS_FEATURE:

        return [
            value
            for value in values
            if not np.isclose(
                value,
                original_value,
            )
        ]

    return []


# ---------------------------------------------------------
# Counterfactual search
# ---------------------------------------------------------

def find_counterfactual(
    clf,
    x,
    target_class,
    domains,
    feature_ranges,
):
    """
    Find a counterfactual with minimum sparsity.

    Optimization priority:
        1. Minimum number of changed features
        2. Minimum normalized change
    """

    original_prediction = int(
        clf.predict(
            x.reshape(1, -1)
        )[0]
    )

    # No search needed if already target class
    if original_prediction == target_class:
        return None

    number_of_features = len(
        FEATURE_COLUMNS
    )

    # Search sparsity from 1 upward
    for sparsity in range(
        1,
        number_of_features + 1,
    ):

        candidates_for_sparsity = []

        # Choose which features will change
        for changed_indices in combinations(
            range(number_of_features),
            sparsity,
        ):

            value_options = []
            possible = True

            for index in changed_indices:

                feature_name = FEATURE_COLUMNS[index]

                values = candidate_values_for_feature(
                    x[index],
                    feature_name,
                    domains,
                )

                if not values:

                    possible = False
                    break

                value_options.append(values)

            if not possible:
                continue

            # Try all combinations of replacement values
            for replacement_values in product(
                *value_options
            ):

                candidate = x.copy()

                for index, value in zip(
                    changed_indices,
                    replacement_values,
                ):

                    candidate[index] = value

                prediction = int(
                    clf.predict(
                        candidate.reshape(1, -1)
                    )[0]
                )

                if prediction == target_class:

                    magnitude = normalized_change(
                        x,
                        candidate,
                        feature_ranges,
                    )

                    candidates_for_sparsity.append(
                        (
                            magnitude,
                            candidate.copy(),
                            changed_indices,
                        )
                    )

        # If any valid candidate exists at this sparsity,
        # choose the one with minimum normalized magnitude.
        if candidates_for_sparsity:

            candidates_for_sparsity.sort(
                key=lambda item: item[0]
            )

            (
                magnitude,
                best_candidate,
                changed_indices,
            ) = candidates_for_sparsity[0]

            changed_features = []

            for index in changed_indices:

                changed_features.append({
                    "feature": FEATURE_COLUMNS[index],
                    "original": float(x[index]),
                    "counterfactual": float(
                        best_candidate[index]
                    ),
                })

            return {
                "original_prediction": int(
                    original_prediction
                ),
                "target_prediction": int(
                    target_class
                ),
                "changed_features": changed_features,
                "sparsity": int(sparsity),
                "normalized_magnitude": float(
                    magnitude
                ),
                "counterfactual": (
                    best_candidate.tolist()
                ),
            }

    return None


# ---------------------------------------------------------
# Rule construction
# ---------------------------------------------------------

def rule_from_counterfactual(result):
    """
    Represent the point counterfactual as an exact-value rule.
    """

    conditions = []

    for item in result["changed_features"]:

        feature = item["feature"]
        value = item["counterfactual"]

        condition = (
            f"{feature} == {value:.4f}"
        )

        conditions.append(condition)

    if not conditions:

        return "TRUE"

    return " AND ".join(conditions)


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main():

    print("=== DAY 3: COUNTERFACTUAL GENERATION ===")

    # -----------------------------------------------------
    # 1. Prepare data
    # -----------------------------------------------------

    (
        df,
        feature_df,
        X_train,
        X_test,
        y_train,
        y_test,
    ) = prepare_data()

    print(
        f"Total samples : {len(df)}"
    )

    print(
        f"Training      : {len(X_train)}"
    )

    print(
        f"Testing       : {len(X_test)}"
    )

    # -----------------------------------------------------
    # 2. Train model
    # -----------------------------------------------------

    clf = train_model(
        X_train,
        y_train,
    )

    # -----------------------------------------------------
    # 3. Build valid feature domains
    # -----------------------------------------------------

    domains = build_feature_domains(
        X_train
    )

    feature_ranges = build_feature_ranges(
        X_train
    )

    # -----------------------------------------------------
    # 4. Select deterministic test cases
    # -----------------------------------------------------

    selected_count = min(
        N_SELECTED_CASES,
        len(X_test),
    )

    selected_indices = list(
        range(selected_count)
    )

    print(
        f"\nSelected test cases for coverage: "
        f"{selected_count}"
    )

    # -----------------------------------------------------
    # 5. Generate counterfactuals
    # -----------------------------------------------------

    predictions = clf.predict(
        X_test
    )

    all_generated = []

    target_class = {
        0: 1,
        1: 0,
    }

    for index in selected_indices:

        x = X_test[index]

        original_prediction = int(
            predictions[index]
        )

        desired_class = target_class[
            original_prediction
        ]

        result = find_counterfactual(
            clf,
            x,
            desired_class,
            domains,
            feature_ranges,
        )

        if result is None:

            continue

        result["test_index"] = int(
            index
        )

        result["actual_label"] = int(
            y_test[index]
        )

        result["original_actual_match"] = (
            int(
                original_prediction
                == y_test[index]
            )
        )

        result["rule"] = (
            rule_from_counterfactual(
                result
            )
        )

        # Explicitly verify the generated CF
        # against the original classifier.
        cf_array = np.asarray(
            result["counterfactual"],
            dtype=float,
        )

        verified_prediction = int(
            clf.predict(
                cf_array.reshape(1, -1)
            )[0]
        )

        result["verified_prediction"] = (
            verified_prediction
        )

        result["validity"] = (
            verified_prediction
            == result["target_prediction"]
        )

        all_generated.append(
            result
        )

    # -----------------------------------------------------
    # 6. Calculate metrics
    # -----------------------------------------------------

    generated_count = len(
        all_generated
    )

    valid_count = sum(
        1
        for result in all_generated
        if result["validity"]
    )

    # Validity:
    # among generated counterfactuals,
    # how many actually reach the target class?
    counterfactual_validity = (
        valid_count / generated_count
        if generated_count > 0
        else 0.0
    )

    # Coverage:
    # among the selected test cases,
    # how many received a valid counterfactual?
    counterfactual_coverage = (
        valid_count / selected_count
        if selected_count > 0
        else 0.0
    )

    if generated_count > 0:

        average_sparsity = float(
            np.mean(
                [
                    result["sparsity"]
                    for result in all_generated
                ]
            )
        )

        average_magnitude = float(
            np.mean(
                [
                    result[
                        "normalized_magnitude"
                    ]
                    for result in all_generated
                ]
            )
        )

    else:

        average_sparsity = None
        average_magnitude = None

    # -----------------------------------------------------
    # 7. Choose representative examples
    # -----------------------------------------------------

    # Select representative examples deterministically by unique rule.
    # This keeps the saved examples diverse and makes the report reproducible.
    representative_examples = []
    seen_rules = set()

    for result in all_generated:
        rule = result["rule"]

        if rule not in seen_rules:
            representative_examples.append(result)
            seen_rules.add(rule)

        if len(representative_examples) >= N_REPRESENTATIVE_EXAMPLES:
            break

    # Fall back to additional generated examples if fewer than three
    # unique rules exist in the selected cases.
    if len(representative_examples) < N_REPRESENTATIVE_EXAMPLES:
        for result in all_generated:
            if result not in representative_examples:
                representative_examples.append(result)

            if len(representative_examples) >= N_REPRESENTATIVE_EXAMPLES:
                break

    # -----------------------------------------------------
    # 8. Print results
    # -----------------------------------------------------

    print(
        "\n=== COUNTERFACTUAL RESULTS ==="
    )

    print(
        f"Selected test cases : "
        f"{selected_count}"
    )

    print(
        f"Counterfactuals generated : "
        f"{generated_count}"
    )

    print(
        f"Valid counterfactuals : "
        f"{valid_count}"
    )

    print(
        f"\nCounterfactual validity : "
        f"{counterfactual_validity:.4f}"
    )

    print(
        f"Counterfactual coverage : "
        f"{counterfactual_coverage:.4f}"
    )

    print(
        f"Average sparsity : "
        f"{average_sparsity}"
    )

    print(
        f"Average normalized magnitude : "
        f"{average_magnitude}"
    )

    # -----------------------------------------------------
    # 9. Print three representative examples
    # -----------------------------------------------------

    for i, result in enumerate(
        representative_examples,
        start=1,
    ):

        print(
            f"\n=== Representative Example {i} ==="
        )

        print(
            "Test index:",
            result["test_index"],
        )

        print(
            "Original prediction:",
            result["original_prediction"],
        )

        print(
            "Actual label:",
            result["actual_label"],
        )

        print(
            "Target prediction:",
            result["target_prediction"],
        )

        print(
            "Changed features:"
        )

        for item in result[
            "changed_features"
        ]:

            print(
                f"  {item['feature']}: "
                f"{item['original']} -> "
                f"{item['counterfactual']}"
            )

        print(
            "Sparsity:",
            result["sparsity"],
        )

        print(
            "Normalized magnitude:",
            f"{result['normalized_magnitude']:.4f}",
        )

        print(
            "Rule:",
            result["rule"],
        )

        print(
            "Verified prediction:",
            result["verified_prediction"],
        )

        print(
            "Validity:",
            result["validity"],
        )

    # -----------------------------------------------------
    # 10. Save counterfactual results
    # -----------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parent.parent
    )

    results_dir = (
        project_root / "results"
    )

    results_dir.mkdir(
        exist_ok=True
    )

    examples_path = (
        results_dir
        / "counterfactual_examples.json"
    )

    output = {

        "selected_test_cases":
            selected_count,

        "counterfactuals_generated":
            generated_count,

        "valid_counterfactuals":
            valid_count,

        "counterfactual_validity":
            counterfactual_validity,

        "counterfactual_coverage":
            counterfactual_coverage,

        "average_sparsity":
            average_sparsity,

        "average_normalized_magnitude":
            average_magnitude,

        "representative_examples":
            representative_examples,

        "all_generated_counterfactuals":
            all_generated,
    }

    with open(
        examples_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=4,
        )

    print(
        f"\nExamples saved to: "
        f"{examples_path}"
    )

    # -----------------------------------------------------
    # 11. Update metrics.json
    # -----------------------------------------------------

    metrics_path = (
        results_dir
        / "metrics.json"
    )

    if metrics_path.exists():

        with open(
            metrics_path,
            "r",
            encoding="utf-8",
        ) as f:

            metrics = json.load(f)

    else:

        metrics = {}

    metrics[
        "counterfactual_selected_cases"
    ] = int(selected_count)

    metrics[
        "counterfactuals_generated"
    ] = int(generated_count)

    metrics[
        "counterfactual_validity"
    ] = float(
        counterfactual_validity
    )

    metrics[
        "average_sparsity"
    ] = average_sparsity

    metrics[
        "average_normalized_magnitude"
    ] = average_magnitude

    metrics[
        "counterfactual_coverage"
    ] = float(
        counterfactual_coverage
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )

    print(
        f"Metrics updated: {metrics_path}"
    )

    print(
        "\n=== DAY 3 COMPLETED ==="
    )


if __name__ == "__main__":
    main()