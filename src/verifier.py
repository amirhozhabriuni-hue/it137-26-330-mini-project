"""
Day 4 - Formal Verification with Z3

This module performs three tasks:

1. Validate the SMT encoding against the original
   scikit-learn Decision Tree.

2. Test important tree boundary cases.

3. Verify counterfactual rules using SAT/UNSAT queries.

Important:
UNSAT means that no counterexample was found within
the explicitly encoded feature domain.

It does NOT prove clinical safety or correctness of
the dataset, labels, or feature extraction.
"""

import json
import time
from pathlib import Path

import numpy as np
import z3

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from data_loader import load_dataset
from features import extract_features, FEATURE_COLUMNS
from tree_encoder import DecisionTreeSMTEncoder


# ---------------------------------------------------------
# Fixed experiment settings
# ---------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_DEPTH = 5

N_RANDOM_VALIDATION_CASES = 500


# ---------------------------------------------------------
# Feature definitions
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
# Train exactly the same tree
# ---------------------------------------------------------

def train_baseline():
    """Reproduce the exact Day 1 model."""

    df = load_dataset()

    feature_df = extract_features(df)

    X = feature_df[
        FEATURE_COLUMNS
    ].values

    y = feature_df[
        "label"
    ].values

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    clf = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    clf.fit(
        X_train,
        y_train,
    )

    return (
        clf,
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ---------------------------------------------------------
# Domain definition
# ---------------------------------------------------------

def get_feature_domain(X_train):
    """
    Define the feature-space domain used for verification.

    Binary features:
        0 or 1

    Integer/count features:
        training-data minimum to maximum

    uppercase_ratio:
        0.0 to 1.0
    """

    domain = {}

    for index, name in enumerate(
        FEATURE_COLUMNS
    ):

        if name in BINARY_FEATURES:

            domain[name] = {
                "type": "binary",
                "min": 0,
                "max": 1,
            }

        elif name in INTEGER_FEATURES:

            minimum = int(
                np.floor(
                    np.min(
                        X_train[:, index]
                    )
                )
            )

            maximum = int(
                np.ceil(
                    np.max(
                        X_train[:, index]
                    )
                )
            )

            domain[name] = {
                "type": "integer",
                "min": minimum,
                "max": maximum,
            }

        elif name == CONTINUOUS_FEATURE:

            domain[name] = {
                "type": "continuous",
                "min": 0.0,
                "max": 1.0,
            }

    return domain


# ---------------------------------------------------------
# Generate valid random vectors
# ---------------------------------------------------------

def random_valid_vector(
    domain,
    rng,
):
    """
    Generate a valid feature-space vector.
    """

    values = []

    for name in FEATURE_COLUMNS:

        info = domain[name]

        if info["type"] == "binary":

            value = rng.integers(
                0,
                2,
            )

        elif info["type"] == "integer":

            value = rng.integers(
                info["min"],
                info["max"] + 1,
            )

        elif info["type"] == "continuous":

            # Keep two decimal places to make the
            # generated feature interpretable.
            value = round(
                float(
                    rng.uniform(
                        info["min"],
                        info["max"],
                    )
                ),
                2,
            )

        else:

            raise ValueError(
                f"Unknown domain type for {name}"
            )

        values.append(
            float(value)
        )

    return np.asarray(
        values,
        dtype=float,
    )


# ---------------------------------------------------------
# Convert a numerical value to Z3
# ---------------------------------------------------------

def z3_value(value):
    """Convert a Python numerical value to exact Z3 RealVal."""

    return z3.RealVal(
        str(float(value))
    )


# ---------------------------------------------------------
# Fix a vector in the SMT model
# ---------------------------------------------------------

def vector_constraints(
    encoder,
    vector,
):
    """
    Constrain every SMT feature variable to the
    supplied feature vector.
    """

    constraints = []

    for index, name in enumerate(
        FEATURE_COLUMNS
    ):

        variable = encoder.variables[
            name
        ]

        constraints.append(
            variable == z3_value(
                vector[index]
            )
        )

    return constraints


# ---------------------------------------------------------
# Ask Z3 which class is predicted
# ---------------------------------------------------------

def encoded_prediction(
    encoder,
    vector,
    domain_constraints,
):
    """
    Determine the class predicted by the SMT encoding.

    Returns:
        0 or 1
    """

    for class_id in [0, 1]:

        solver = z3.Solver()

        solver.add(
            domain_constraints
        )

        solver.add(
            vector_constraints(
                encoder,
                vector,
            )
        )

        solver.add(
            encoder.prediction_formula(
                class_id
            )
        )

        result = solver.check()

        if result == z3.sat:

            return class_id

    return None


# ---------------------------------------------------------
# Encoder validation
# ---------------------------------------------------------

def validate_encoder(
    clf,
    encoder,
    domain_constraints,
    domain,
    X_train,
):
    """
    Compare sklearn predictions with SMT predictions
    on random valid vectors.
    """

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    mismatches = []

    for i in range(
        N_RANDOM_VALIDATION_CASES
    ):

        vector = random_valid_vector(
            domain,
            rng,
        )

        sklearn_prediction = int(
            clf.predict(
                vector.reshape(1, -1)
            )[0]
        )

        smt_prediction = (
            encoded_prediction(
                encoder,
                vector,
                domain_constraints,
            )
        )

        if (
            smt_prediction
            != sklearn_prediction
        ):

            mismatches.append({
                "case": int(i),
                "vector": vector.tolist(),
                "sklearn_prediction":
                    sklearn_prediction,
                "smt_prediction":
                    smt_prediction,
            })

    return mismatches


# ---------------------------------------------------------
# Boundary cases
# ---------------------------------------------------------

def get_tree_thresholds(clf):
    """
    Extract all internal Decision Tree thresholds.
    """

    tree = clf.tree_

    thresholds = []

    for node_id in range(
        tree.node_count
    ):

        feature_index = int(
            tree.feature[node_id]
        )

        threshold = float(
            tree.threshold[node_id]
        )

        if feature_index >= 0:

            thresholds.append({
                "node_id": int(node_id),
                "feature_index":
                    feature_index,
                "feature_name":
                    FEATURE_COLUMNS[
                        feature_index
                    ],
                "threshold":
                    threshold,
            })

    return thresholds


def make_midpoint_vector(
    domain,
):
    """
    Make one deterministic valid vector
    to use for boundary tests.
    """

    vector = []

    for name in FEATURE_COLUMNS:

        info = domain[name]

        if info["type"] == "binary":

            value = 0

        elif info["type"] == "integer":

            value = (
                info["min"]
                + info["max"]
            ) / 2

            value = int(
                round(value)
            )

        else:

            value = 0.5

        vector.append(
            float(value)
        )

    return np.asarray(
        vector,
        dtype=float,
    )


def validate_boundaries(
    clf,
    encoder,
    domain_constraints,
    domain,
):
    """
    Test Decision Tree thresholds using valid feature values.

    For continuous features:
        test the exact learned threshold.

    For integer/binary features:
        if the threshold itself is a valid value, test it;
        otherwise test the nearest valid values on both sides.

    This avoids testing impossible feature vectors such as
    an integer feature equal to 0.5.
    """

    thresholds = get_tree_thresholds(clf)

    boundary_results = []

    for item in thresholds:

        feature_index = item["feature_index"]
        feature_name = item["feature_name"]
        threshold = item["threshold"]

        info = domain[feature_name]

        # -------------------------------------------------
        # Continuous feature
        # -------------------------------------------------

        if info["type"] == "continuous":

            if (
                threshold < info["min"]
                or threshold > info["max"]
            ):
                continue

            test_values = [threshold]

        # -------------------------------------------------
        # Integer feature
        # -------------------------------------------------

        elif info["type"] == "integer":

            candidate_values = set()

            floor_value = int(
                np.floor(threshold)
            )

            ceil_value = int(
                np.ceil(threshold)
            )

            candidate_values.add(
                floor_value
            )

            candidate_values.add(
                ceil_value
            )

            # Keep only valid domain values
            test_values = [
                float(value)
                for value in sorted(
                    candidate_values
                )
                if (
                    value >= info["min"]
                    and value <= info["max"]
                )
            ]

        # -------------------------------------------------
        # Binary feature
        # -------------------------------------------------

        elif info["type"] == "binary":

            test_values = [
                0.0,
                1.0,
            ]

        else:
            continue

        # -------------------------------------------------
        # Test valid boundary vectors
        # -------------------------------------------------

        for test_value in test_values:

            vector = make_midpoint_vector(
                domain
            )

            vector[feature_index] = (
                test_value
            )

            sklearn_prediction = int(
                clf.predict(
                    vector.reshape(1, -1)
                )[0]
            )

            smt_prediction = (
                encoded_prediction(
                    encoder,
                    vector,
                    domain_constraints,
                )
            )

            boundary_results.append({
                "node_id":
                    item["node_id"],

                "feature":
                    feature_name,

                "learned_threshold":
                    threshold,

                "tested_value":
                    test_value,

                "sklearn_prediction":
                    sklearn_prediction,

                "smt_prediction":
                    smt_prediction,

                "match":
                    (
                        sklearn_prediction
                        == smt_prediction
                    ),
            })

    return boundary_results


# ---------------------------------------------------------
# Leaf coverage
# ---------------------------------------------------------

def test_leaf_coverage(
    encoder,
    domain_constraints,
):
    """
    Check whether every valid input is covered
    by at least one leaf.

    We search for:
        Domain AND NOT(OR(all leaf paths))

    UNSAT means the tree encoding covers the domain.
    """

    solver = z3.Solver()

    solver.add(
        domain_constraints
    )

    solver.add(
        z3.Not(
            encoder.all_leaf_formula()
        )
    )

    start = time.perf_counter()

    result = solver.check()

    elapsed = (
        time.perf_counter()
        - start
    )

    return {
        "result":
            str(result),
        "coverage_verified":
            result == z3.unsat,
        "time_seconds":
            elapsed,
    }


# ---------------------------------------------------------
# Leaf exclusivity
# ---------------------------------------------------------

def test_leaf_exclusivity(
    encoder,
    domain_constraints,
):
    """
    Check that no two incompatible leaves can
    hold simultaneously inside the valid domain.
    """

    violations = []

    leaves = encoder.leaf_paths

    for i in range(
        len(leaves)
    ):

        for j in range(
            i + 1,
            len(leaves)
        ):

            leaf_i = leaves[i]
            leaf_j = leaves[j]

            solver = z3.Solver()

            solver.add(
                domain_constraints
            )

            solver.add(
                encoder.leaf_formula(
                    leaf_i
                )
            )

            solver.add(
                encoder.leaf_formula(
                    leaf_j
                )
            )

            result = solver.check()

            if result != z3.unsat:

                violations.append({
                    "leaf_a":
                        leaf_i["node_id"],
                    "leaf_b":
                        leaf_j["node_id"],
                    "result":
                        str(result),
                })

    return violations


# ---------------------------------------------------------
# Counterfactual rules
# ---------------------------------------------------------

def load_counterfactuals():
    """Load Day 3 counterfactual results."""

    project_root = (
        Path(__file__)
        .resolve()
        .parent.parent
    )

    path = (
        project_root
        / "results"
        / "counterfactual_examples.json"
    )

    if not path.exists():

        raise FileNotFoundError(
            "counterfactual_examples.json "
            "was not found."
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def verify_point_counterfactual(
    encoder,
    vector,
    target_class,
    domain_constraints,
):
    """
    Verify the exact counterfactual point.

    Property:
        point AND NOT(target class)

    UNSAT means the exact point cannot receive
    another class under the encoded model.
    """

    solver = z3.Solver()

    solver.add(
        domain_constraints
    )

    solver.add(
        vector_constraints(
            encoder,
            vector,
        )
    )

    solver.add(
        z3.Not(
            encoder.prediction_formula(
                target_class
            )
        )
    )

    start = time.perf_counter()

    result = solver.check()

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        str(result),
        elapsed,
        None
        if result == z3.unsat
        else str(
            solver.model()
        ),
    )


def verify_changed_feature_rule(
    encoder,
    result,
    target_class,
    domain_constraints,
):
    """
    Verify the generalized rule consisting only of
    changed features.

    Other features remain unconstrained.

    This can legitimately return SAT because the
    generalized rule may describe more than one region.
    """

    solver = z3.Solver()

    solver.add(
        domain_constraints
    )

    for item in result[
        "changed_features"
    ]:

        feature = item[
            "feature"
        ]

        value = item[
            "counterfactual"
        ]

        solver.add(
            encoder.variables[
                feature
            ]
            == z3_value(value)
        )

    solver.add(
        z3.Not(
            encoder.prediction_formula(
                target_class
            )
        )
    )

    start = time.perf_counter()

    result_status = solver.check()

    elapsed = (
        time.perf_counter()
        - start
    )

    witness = None

    if result_status == z3.sat:

        model = solver.model()

        witness = {}

        for feature_name in FEATURE_COLUMNS:

            value = model.eval(
                encoder.variables[
                    feature_name
                ],
                model_completion=True,
            )

            witness[
                feature_name
            ] = str(value)

    return (
        str(result_status),
        elapsed,
        witness,
    )


# ---------------------------------------------------------
# Main verification
# ---------------------------------------------------------

def main():

    print(
        "=== DAY 4: FORMAL VERIFICATION ==="
    )

    # -----------------------------------------------------
    # 1. Train model
    # -----------------------------------------------------

    (
        clf,
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_baseline()

    # -----------------------------------------------------
    # 2. Build encoder
    # -----------------------------------------------------

    encoder = DecisionTreeSMTEncoder(
        clf,
        FEATURE_COLUMNS,
    )

    print(
        f"Original tree depth: "
        f"{clf.get_depth()}"
    )

    print(
        f"Original leaf count: "
        f"{clf.get_n_leaves()}"
    )

    print(
        f"Encoded leaf count: "
        f"{encoder.leaf_count()}"
    )

    # -----------------------------------------------------
    # 3. Domain
    # -----------------------------------------------------

    domain = get_feature_domain(
        X_train
    )

    # Use the same explicit bounds for the SMT model.
    # The tree_encoder currently provides the training-range
    # version. We therefore construct the verification bounds
    # here explicitly to match the declared project domain.

    domain_constraints = []

    for feature_name in FEATURE_COLUMNS:

        info = domain[
            feature_name
        ]

        variable = encoder.variables[
            feature_name
        ]

        domain_constraints.append(
            variable
            >= z3_value(
                info["min"]
            )
        )

        domain_constraints.append(
            variable
            <= z3_value(
                info["max"]
            )
        )

        # Integer features must be integers.
        if info[
            "type"
        ] in {
            "integer",
            "binary",
        }:

            domain_constraints.append(
                z3.IsInt(
                    variable
                )
            )

    # -----------------------------------------------------
    # 4. Random encoder validation
    # -----------------------------------------------------

    print(
        "\n=== RANDOM ENCODER VALIDATION ==="
    )

    mismatches = validate_encoder(
        clf,
        encoder,
        domain_constraints,
        domain,
        X_train,
    )

    print(
        f"Cases checked: "
        f"{N_RANDOM_VALIDATION_CASES}"
    )

    print(
        f"Mismatch count: "
        f"{len(mismatches)}"
    )

    # -----------------------------------------------------
    # 5. Boundary tests
    # -----------------------------------------------------

    print(
        "\n=== BOUNDARY TESTS ==="
    )

    boundary_results = validate_boundaries(
        clf,
        encoder,
        domain_constraints,
        domain,
    )

    boundary_mismatches = [
        item
        for item in boundary_results
        if not item["match"]
    ]

    print(
        f"Boundary cases checked: "
        f"{len(boundary_results)}"
    )

    print(
        f"Boundary mismatches: "
        f"{len(boundary_mismatches)}"
    )

    # -----------------------------------------------------
    # 6. Leaf coverage
    # -----------------------------------------------------

    print(
        "\n=== LEAF COVERAGE ==="
    )

    leaf_coverage = test_leaf_coverage(
        encoder,
        domain_constraints,
    )

    print(
        f"Result: "
        f"{leaf_coverage['result']}"
    )

    print(
        f"Coverage verified: "
        f"{leaf_coverage['coverage_verified']}"
    )

    # -----------------------------------------------------
    # 7. Leaf exclusivity
    # -----------------------------------------------------

    print(
        "\n=== LEAF EXCLUSIVITY ==="
    )

    exclusivity_violations = (
        test_leaf_exclusivity(
            encoder,
            domain_constraints,
        )
    )

    print(
        f"Exclusivity violations: "
        f"{len(exclusivity_violations)}"
    )

    # -----------------------------------------------------
    # 8. Counterfactual verification
    # -----------------------------------------------------

    print(
        "\n=== COUNTERFACTUAL VERIFICATION ==="
    )

    cf_data = load_counterfactuals()

    all_counterfactuals = (
        cf_data.get(
            "all_generated_counterfactuals",
            [],
        )
    )

    point_results = []
    generalized_results = []

    for index, cf in enumerate(
        all_counterfactuals
    ):

        vector = np.asarray(
            cf["counterfactual"],
            dtype=float,
        )

        target_class = int(
            cf["target_prediction"]
        )

        # Exact point verification
        (
            point_status,
            point_time,
            point_witness,
        ) = verify_point_counterfactual(
            encoder,
            vector,
            target_class,
            domain_constraints,
        )

        # Generalized changed-feature rule
        (
            generalized_status,
            generalized_time,
            generalized_witness,
        ) = verify_changed_feature_rule(
            encoder,
            cf,
            target_class,
            domain_constraints,
        )

        point_results.append({
            "index":
                int(index),
            "target_class":
                target_class,
            "status":
                point_status,
            "time_seconds":
                point_time,
            "witness":
                point_witness,
        })

        generalized_results.append({
            "index":
                int(index),
            "target_class":
                target_class,
            "status":
                generalized_status,
            "time_seconds":
                generalized_time,
            "witness":
                generalized_witness,
            "rule":
                cf["rule"],
        })

    # -----------------------------------------------------
    # 9. Counters
    # -----------------------------------------------------

    point_unsat = sum(
        1
        for item in point_results
        if item["status"]
        == "unsat"
    )

    point_sat = sum(
        1
        for item in point_results
        if item["status"]
        == "sat"
    )

    generalized_unsat = sum(
        1
        for item in generalized_results
        if item["status"]
        == "unsat"
    )

    generalized_sat = sum(
        1
        for item in generalized_results
        if item["status"]
        == "sat"
    )

    all_times = [
        item["time_seconds"]
        for item in point_results
    ]

    if all_times:

        median_verification_time = float(
            np.median(all_times)
        )

    else:

        median_verification_time = None

    # -----------------------------------------------------
    # 10. Print CF summary
    # -----------------------------------------------------

    print(
        f"Counterfactual rules checked: "
        f"{len(point_results)}"
    )

    print(
        f"Exact-point UNSAT: "
        f"{point_unsat}"
    )

    print(
        f"Exact-point SAT: "
        f"{point_sat}"
    )

    print(
        f"Changed-feature-rule UNSAT: "
        f"{generalized_unsat}"
    )

    print(
        f"Changed-feature-rule SAT: "
        f"{generalized_sat}"
    )

    # -----------------------------------------------------
    # 11. Save verification cases
    # -----------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parent.parent
    )

    results_dir = (
        project_root
        / "results"
    )

    results_dir.mkdir(
        exist_ok=True
    )

    verification_path = (
        results_dir
        / "verification_cases.json"
    )

    verification_output = {

        "random_encoder_validation": {
            "cases_checked":
                N_RANDOM_VALIDATION_CASES,
            "mismatch_count":
                len(mismatches),
            "mismatches":
                mismatches,
        },

        "boundary_validation": {
            "cases_checked":
                len(boundary_results),
            "mismatch_count":
                len(boundary_mismatches),
            "results":
                boundary_results,
        },

        "leaf_coverage":
            leaf_coverage,

        "leaf_exclusivity": {
            "violation_count":
                len(
                    exclusivity_violations
                ),
            "violations":
                exclusivity_violations,
        },

        "counterfactual_exact_point": {
            "rules_checked":
                len(point_results),
            "unsat":
                point_unsat,
            "sat":
                point_sat,
            "results":
                point_results,
        },

        "counterfactual_changed_feature_rule": {
            "rules_checked":
                len(generalized_results),
            "unsat":
                generalized_unsat,
            "sat":
                generalized_sat,
            "results":
                generalized_results,
        },

        "median_verification_time_seconds":
            median_verification_time,
    }

    with open(
        verification_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            verification_output,
            f,
            indent=4,
        )

    # -----------------------------------------------------
    # 12. Update metrics.json
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
        "verification_rules_checked"
    ] = int(
        len(point_results)
    )

    metrics[
        "verification_unsat"
    ] = int(
        point_unsat
    )

    metrics[
        "verification_sat"
    ] = int(
        point_sat
    )

    metrics[
        "generalized_rule_unsat"
    ] = int(
        generalized_unsat
    )

    metrics[
        "generalized_rule_sat"
    ] = int(
        generalized_sat
    )

    metrics[
        "encoder_mismatch_count"
    ] = int(
        len(mismatches)
    )

    metrics[
        "boundary_cases_checked"
    ] = int(
        len(boundary_results)
    )

    metrics[
        "boundary_mismatch_count"
    ] = int(
        len(boundary_mismatches)
    )

    metrics[
        "leaf_coverage_verified"
    ] = bool(
        leaf_coverage[
            "coverage_verified"
        ]
    )

    metrics[
        "leaf_exclusivity_violations"
    ] = int(
        len(exclusivity_violations)
    )

    metrics[
        "median_verification_time"
    ] = median_verification_time

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
        f"\nVerification results saved to: "
        f"{verification_path}"
    )

    print(
        f"Metrics updated: "
        f"{metrics_path}"
    )

    print(
        "\n=== DAY 4 VERIFICATION COMPLETED ==="
    )


if __name__ == "__main__":
    main()