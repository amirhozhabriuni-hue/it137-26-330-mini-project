"""
Baseline Decision Tree classifier for FR/NFR requirements.

Day 1:
- Load dataset
- Extract 12 interpretable features
- Split into train/test sets
- Train Decision Tree
- Evaluate baseline performance
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from data_loader import load_dataset
from features import extract_features, FEATURE_COLUMNS


RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_DEPTH = 5


def train_baseline():
    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------
    df = load_dataset()

    # ---------------------------------------------------------
    # 2. Extract interpretable features
    # ---------------------------------------------------------
    feature_df = extract_features(df)

    X = feature_df[FEATURE_COLUMNS].values
    y = feature_df["label"].values

    # ---------------------------------------------------------
    # 3. Stratified 80/20 train-test split
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # ---------------------------------------------------------
    # 4. Train interpretable Decision Tree
    # ---------------------------------------------------------
    clf = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    clf.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 5. Predictions
    # ---------------------------------------------------------
    y_pred = clf.predict(X_test)
    probabilities = clf.predict_proba(X_test)

    # ---------------------------------------------------------
    # 6. Evaluation metrics
    # ---------------------------------------------------------
    metrics = {
        "n_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "max_depth": MAX_DEPTH,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(
            f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "precision_macro": float(
            precision_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred,
        ).tolist(),
        "class_distribution": {
            "train": np.bincount(y_train).tolist(),
            "test": np.bincount(y_test).tolist(),
        },
    }

    # ---------------------------------------------------------
    # 7. Print results
    # ---------------------------------------------------------
    print("\n=== BASELINE RESULTS ===")

    print(f"Total samples : {metrics['n_total']}")
    print(f"Training      : {metrics['n_train']}")
    print(f"Testing       : {metrics['n_test']}")

    print(f"\nAccuracy      : {metrics['accuracy']:.4f}")
    print(f"Macro-F1      : {metrics['macro_f1']:.4f}")
    print(f"Macro-Precision: {metrics['precision_macro']:.4f}")
    print(f"Macro-Recall   : {metrics['recall_macro']:.4f}")

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Classification Report ===")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["FR", "NFR"],
            zero_division=0,
        )
    )

    # ---------------------------------------------------------
    # 8. Print Decision Tree
    # ---------------------------------------------------------
    print("\n=== DECISION TREE ===")
    print(
        export_text(
            clf,
            feature_names=FEATURE_COLUMNS,
        )
    )

    print(f"Tree depth: {clf.get_depth()}")
    print(f"Number of leaves: {clf.get_n_leaves()}")

    return (
        clf,
        X_train,
        X_test,
        y_train,
        y_test,
        probabilities,
        metrics,
    )


def save_metrics(metrics):
    """Save evaluation metrics to results/metrics.json."""

    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / "metrics.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    (
        clf,
        X_train,
        X_test,
        y_train,
        y_test,
        probabilities,
        metrics,
    ) = train_baseline()

    save_metrics(metrics)