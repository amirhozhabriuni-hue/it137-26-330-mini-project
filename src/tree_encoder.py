"""
Day 4 - Exact Decision Tree -> SMT Encoder

The trained scikit-learn Decision Tree is represented as
root-to-leaf logical paths.

Each leaf becomes:

    condition_1 AND condition_2 AND ... AND condition_k

and is associated with a predicted class.

The encoder is then used by Z3 to reason about the model.
"""

from pathlib import Path

import numpy as np
import z3

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from data_loader import load_dataset
from features import extract_features, FEATURE_COLUMNS


RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_DEPTH = 5


class DecisionTreeSMTEncoder:
    """
    Exact logical representation of a trained Decision Tree.
    """

    def __init__(self, clf, feature_names):
        self.clf = clf
        self.feature_names = feature_names

        # One real-valued Z3 variable for each feature.
        self.variables = {
            name: z3.Real(name)
            for name in feature_names
        }

        self.leaf_paths = []

        self._extract_leaf_paths()

    # -----------------------------------------------------
    # Extract root-to-leaf paths
    # -----------------------------------------------------

    def _extract_leaf_paths(self):
        """
        Traverse the sklearn tree and collect every leaf path.
        """

        tree = self.clf.tree_

        def visit(node_id, conditions):
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]

            # Leaf node
            if left_child == right_child:

                values = tree.value[node_id][0]

                # sklearn's prediction at the leaf
                predicted_class = int(
                    np.argmax(values)
                )

                self.leaf_paths.append({
                    "node_id": int(node_id),
                    "conditions": list(conditions),
                    "predicted_class": predicted_class,
                })

                return

            feature_index = int(
                tree.feature[node_id]
            )

            threshold = float(
                tree.threshold[node_id]
            )

            feature_name = self.feature_names[
                feature_index
            ]

            # Left branch:
            # feature <= threshold
            left_condition = (
                feature_name,
                "<=",
                threshold,
            )

            visit(
                left_child,
                conditions + [left_condition],
            )

            # Right branch:
            # feature > threshold
            right_condition = (
                feature_name,
                ">",
                threshold,
            )

            visit(
                right_child,
                conditions + [right_condition],
            )

        visit(0, [])

    # -----------------------------------------------------
    # Convert one condition to Z3
    # -----------------------------------------------------

    def condition_to_z3(self, condition):
        """
        Convert an extracted tree condition into a Z3 expression.
        """

        feature_name, operator, threshold = condition

        variable = self.variables[
            feature_name
        ]

        threshold_real = z3.RealVal(
            str(threshold)
        )

        if operator == "<=":
            return variable <= threshold_real

        if operator == ">":
            return variable > threshold_real

        raise ValueError(
            f"Unknown operator: {operator}"
        )

    # -----------------------------------------------------
    # Encode one leaf
    # -----------------------------------------------------

    def leaf_formula(self, leaf):
        """
        Return the conjunction of all path conditions
        leading to one leaf.
        """

        conditions = [
            self.condition_to_z3(condition)
            for condition in leaf["conditions"]
        ]

        if not conditions:
            return z3.BoolVal(True)

        return z3.And(conditions)

    # -----------------------------------------------------
    # Formula for a predicted class
    # -----------------------------------------------------

    def class_formula(self, class_id):
        """
        Return:

            OR(all leaf paths predicting class_id)
        """

        paths = [
            self.leaf_formula(leaf)
            for leaf in self.leaf_paths
            if leaf["predicted_class"] == class_id
        ]

        if not paths:
            return z3.BoolVal(False)

        return z3.Or(paths)

    # -----------------------------------------------------
    # Complete model formula
    # -----------------------------------------------------

    def prediction_formula(self, class_id):
        """
        Logical condition under which the encoded tree
        predicts class_id.
        """

        return self.class_formula(
            class_id
        )

    # -----------------------------------------------------
    # Bounds
    # -----------------------------------------------------

    def domain_constraints(
        self,
        X_reference,
    ):
        """
        Constrain each feature to its observed reference
        range.

        This defines the explicit feature domain used by Z3.
        """

        constraints = []

        for index, feature_name in enumerate(
            self.feature_names
        ):

            minimum = float(
                np.min(
                    X_reference[:, index]
                )
            )

            maximum = float(
                np.max(
                    X_reference[:, index]
                )
            )

            variable = self.variables[
                feature_name
            ]

            constraints.append(
                variable >= z3.RealVal(
                    str(minimum)
                )
            )

            constraints.append(
                variable <= z3.RealVal(
                    str(maximum)
                )
            )

        return constraints

    # -----------------------------------------------------
    # Full model description
    # -----------------------------------------------------

    def all_leaf_formula(self):
        """
        Return the disjunction of all leaf paths.

        For a valid tree this should cover every input in
        the corresponding feature domain.
        """

        paths = [
            self.leaf_formula(leaf)
            for leaf in self.leaf_paths
        ]

        return z3.Or(paths)

    def leaf_count(self):
        return len(
            self.leaf_paths
        )

    def describe(self):
        """
        Print a readable summary of the encoded tree.
        """

        print(
            f"Encoded tree depth: "
            f"{self.clf.get_depth()}"
        )

        print(
            f"Encoded leaf count: "
            f"{self.leaf_count()}"
        )

        print("\n=== LEAF PATHS ===")

        for leaf in self.leaf_paths:

            print(
                f"\nLeaf node "
                f"{leaf['node_id']}"
            )

            print(
                f"Predicted class: "
                f"{leaf['predicted_class']}"
            )

            if not leaf["conditions"]:

                print("  TRUE")

            else:

                for condition in leaf[
                    "conditions"
                ]:

                    feature, operator, threshold = condition

                    print(
                        f"  {feature} "
                        f"{operator} "
                        f"{threshold:.6f}"
                    )


# ---------------------------------------------------------
# Train the exact same tree
# ---------------------------------------------------------

def train_baseline():

    df = load_dataset()

    feature_df = extract_features(
        df
    )

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
# Main test
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        "=== DAY 4: DECISION TREE SMT ENCODING ==="
    )

    (
        clf,
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_baseline()

    print(
        f"Original tree depth: "
        f"{clf.get_depth()}"
    )

    print(
        f"Original tree leaves: "
        f"{clf.get_n_leaves()}"
    )

    encoder = DecisionTreeSMTEncoder(
        clf,
        FEATURE_COLUMNS,
    )

    print(
        f"\nEncoded leaf count: "
        f"{encoder.leaf_count()}"
    )

    encoder.describe()

    # Simple symbolic sanity check:
    # create a solver and assert the model coverage.
    solver = z3.Solver()

    solver.add(
        encoder.all_leaf_formula()
    )

    result = solver.check()

    print(
        f"\nLeaf coverage satisfiability: "
        f"{result}"
    )

    print(
        "\n=== TREE ENCODING COMPLETED ==="
    )