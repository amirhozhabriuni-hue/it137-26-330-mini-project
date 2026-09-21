"""
Day 2 - Confidence, Calibration, and Local Stability

This module:
1. Retrains the same baseline Decision Tree.
2. Computes probability-based confidence.
3. Computes local stability using valid feature perturbations.
4. Computes ECE and Brier score.
5. Creates a calibration plot.
6. Updates results/metrics.json.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from data_loader import load_dataset
from features import extract_features, FEATURE_COLUMNS


# ---------------------------------------------------------
# Fixed experiment settings
# ---------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_DEPTH = 5

N_ECE_BINS = 10


# ---------------------------------------------------------
# Train the same baseline model
# ---------------------------------------------------------

def prepare_data():
    """Load dataset, extract features, and create the same 80/20 split."""

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

    return df, X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train the same Decision Tree used in Day 1."""

    clf = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    clf.fit(X_train, y_train)

    return clf


# ---------------------------------------------------------
# Probability-based confidence
# ---------------------------------------------------------

def compute_confidence(clf, X_test):
    """
    Confidence = maximum predicted class probability.

    This is a model output, not a formal guarantee.
    """

    probabilities = clf.predict_proba(X_test)
    predictions = clf.predict(X_test)

    confidence = np.max(probabilities, axis=1)

    return predictions, probabilities, confidence


# ---------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------

def expected_calibration_error(y_true, predictions, confidence, n_bins=10):
    """
    Compute Expected Calibration Error.

    For each confidence bin:
        gap = |average confidence - empirical accuracy|

    ECE is the weighted average of these gaps.
    """

    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)
    confidence = np.asarray(confidence)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    bin_results = []

    for i in range(n_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)

        count = int(np.sum(mask))

        if count == 0:
            bin_results.append({
                "bin": i + 1,
                "lower": float(lower),
                "upper": float(upper),
                "count": 0,
                "accuracy": None,
                "confidence": None,
                "gap": None,
            })
            continue

        bin_accuracy = np.mean(
            predictions[mask] == y_true[mask]
        )

        bin_confidence = np.mean(confidence[mask])

        gap = abs(bin_accuracy - bin_confidence)

        ece += (count / len(y_true)) * gap

        bin_results.append({
            "bin": i + 1,
            "lower": float(lower),
            "upper": float(upper),
            "count": count,
            "accuracy": float(bin_accuracy),
            "confidence": float(bin_confidence),
            "gap": float(gap),
        })

    return float(ece), bin_results


# ---------------------------------------------------------
# Local stability
# ---------------------------------------------------------

def generate_neighbors(x, X_train):
    """
    Generate deterministic local feature perturbations.

    Rules:
    - Binary features: flip 0 <-> 1
    - Count/integer features: change by +/- 1
    - uppercase_ratio: change by +/- 0.05

    Values are clipped to the observed training-data domain.
    """

    feature_names = FEATURE_COLUMNS

    neighbors = []

    train_min = np.min(X_train, axis=0)
    train_max = np.max(X_train, axis=0)

    binary_features = {
        "has_digit",
        "has_unit",
        "has_shall",
        "has_must",
        "passive_indicator",
        "has_list_marker",
    }

    continuous_feature = "uppercase_ratio"

    for i, feature_name in enumerate(feature_names):

        candidate_values = []

        if feature_name in binary_features:

            current = int(round(x[i]))
            candidate_values = [1 - current]

        elif feature_name == continuous_feature:

            candidate_values = [
                x[i] - 0.05,
                x[i] + 0.05,
            ]

        else:

            candidate_values = [
                x[i] - 1,
                x[i] + 1,
            ]

        for new_value in candidate_values:

            new_value = np.clip(
                new_value,
                train_min[i],
                train_max[i],
            )

            # Avoid duplicate neighbor equal to original
            if np.isclose(new_value, x[i]):
                continue

            neighbor = x.copy()
            neighbor[i] = new_value

            neighbors.append(neighbor)

    return neighbors


def local_stability(clf, X_test, X_train):
    """
    Local stability =
        1 - (number of neighbor label flips / number of neighbors)

    If a point has no valid neighbors, stability is recorded as 1.0.
    """

    stability_scores = []
    flip_counts = []
    neighbor_counts = []

    original_predictions = clf.predict(X_test)

    for x, original_prediction in zip(X_test, original_predictions):

        neighbors = generate_neighbors(x, X_train)

        if len(neighbors) == 0:
            stability_scores.append(1.0)
            flip_counts.append(0)
            neighbor_counts.append(0)
            continue

        neighbor_array = np.asarray(neighbors)

        neighbor_predictions = clf.predict(neighbor_array)

        flips = int(
            np.sum(neighbor_predictions != original_prediction)
        )

        total_neighbors = len(neighbors)

        stability = 1.0 - (flips / total_neighbors)

        stability_scores.append(stability)
        flip_counts.append(flips)
        neighbor_counts.append(total_neighbors)

    return (
        np.asarray(stability_scores),
        np.asarray(flip_counts),
        np.asarray(neighbor_counts),
    )


# ---------------------------------------------------------
# Calibration plot
# ---------------------------------------------------------

def plot_calibration(bin_results, output_path):
    """Save reliability/calibration plot."""

    accuracies = []
    confidences = []
    counts = []

    for item in bin_results:

        if item["count"] == 0:
            continue

        accuracies.append(item["accuracy"])
        confidences.append(item["confidence"])
        counts.append(item["count"])

    plt.figure(figsize=(7, 6))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.plot(
        confidences,
        accuracies,
        marker="o",
        label="Model",
    )

    plt.xlabel("Mean confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Calibration Plot")

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ---------------------------------------------------------
# Save Day 2 results
# ---------------------------------------------------------

def save_results(
    existing_metrics,
    ece,
    brier,
    mean_confidence,
    mean_local_stability,
    accuracy_by_confidence,
    bin_results,
):
    """Update metrics.json with Day 2 results."""

    project_root = Path(__file__).resolve().parent.parent

    results_dir = project_root / "results"
    figures_dir = results_dir / "figures"

    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    existing_metrics["mean_confidence"] = float(mean_confidence)
    existing_metrics["mean_local_stability"] = float(
        mean_local_stability
    )
    existing_metrics["ece"] = float(ece)
    existing_metrics["brier_score"] = float(brier)

    existing_metrics["accuracy_by_confidence"] = accuracy_by_confidence

    existing_metrics["ece_bins"] = bin_results

    metrics_path = results_dir / "metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            existing_metrics,
            f,
            indent=4,
        )

    print(f"\nUpdated metrics saved to: {metrics_path}")

    calibration_path = figures_dir / "calibration_plot.png"

    plot_calibration(
        bin_results,
        calibration_path,
    )

    print(
        f"Calibration plot saved to: {calibration_path}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=== DAY 2: CONFIDENCE + CALIBRATION ===")

    # 1. Prepare data
    (
        df,
        X_train,
        X_test,
        y_train,
        y_test,
    ) = prepare_data()

    print(f"Total samples : {len(df)}")
    print(f"Training      : {len(X_train)}")
    print(f"Testing       : {len(X_test)}")

    # 2. Train model
    clf = train_model(
        X_train,
        y_train,
    )

    # 3. Predictions and confidence
    (
        predictions,
        probabilities,
        confidence,
    ) = compute_confidence(
        clf,
        X_test,
    )

    # 4. Correct/incorrect analysis
    correctness = (
        predictions == y_test
    ).astype(int)

    mean_confidence = np.mean(confidence)

    print(
        f"\nMean confidence: "
        f"{mean_confidence:.4f}"
    )

    print(
        f"Accuracy of test predictions: "
        f"{np.mean(correctness):.4f}"
    )

    # 5. ECE
    ece, bin_results = expected_calibration_error(
        y_test,
        predictions,
        confidence,
        n_bins=N_ECE_BINS,
    )

    print(
        f"\nECE: {ece:.4f}"
    )

    # 6. Brier score
    positive_class_probability = probabilities[:, 1]

    brier = brier_score_loss(
        y_test,
        positive_class_probability,
    )

    print(
        f"Brier score: {brier:.4f}"
    )

    # 7. Local stability
    (
        stability_scores,
        flip_counts,
        neighbor_counts,
    ) = local_stability(
        clf,
        X_test,
        X_train,
    )

    mean_local_stability = np.mean(
        stability_scores
    )

    print(
        f"Mean local stability: "
        f"{mean_local_stability:.4f}"
    )

    # 8. Accuracy by confidence level
    high_confidence = confidence >= 0.80
    medium_confidence = (
        (confidence >= 0.60)
        & (confidence < 0.80)
    )
    low_confidence = confidence < 0.60

    accuracy_by_confidence = {
        "low_<0.60": {
            "count": int(np.sum(low_confidence)),
            "accuracy": (
                float(np.mean(correctness[low_confidence]))
                if np.any(low_confidence)
                else None
            ),
        },
        "medium_0.60-0.79": {
            "count": int(np.sum(medium_confidence)),
            "accuracy": (
                float(np.mean(correctness[medium_confidence]))
                if np.any(medium_confidence)
                else None
            ),
        },
        "high_>=0.80": {
            "count": int(np.sum(high_confidence)),
            "accuracy": (
                float(np.mean(correctness[high_confidence]))
                if np.any(high_confidence)
                else None
            ),
        },
    }

    print("\n=== ACCURACY BY CONFIDENCE ===")

    for level, values in accuracy_by_confidence.items():

        print(
            f"{level}: "
            f"count={values['count']}, "
            f"accuracy={values['accuracy']}"
        )

    # 9. Print ECE bins
    print("\n=== ECE BINS ===")

    for item in bin_results:

        if item["count"] == 0:
            continue

        print(
            f"Bin {item['bin']}: "
            f"count={item['count']}, "
            f"confidence={item['confidence']:.4f}, "
            f"accuracy={item['accuracy']:.4f}"
        )

    # 10. Update metrics.json and save plot

    project_root = Path(__file__).resolve().parent.parent

    metrics_path = (
        project_root
        / "results"
        / "metrics.json"
    )

    if metrics_path.exists():

        with open(
            metrics_path,
            "r",
            encoding="utf-8",
        ) as f:
            existing_metrics = json.load(f)

    else:

        existing_metrics = {}

    save_results(
        existing_metrics,
        ece,
        brier,
        mean_confidence,
        mean_local_stability,
        accuracy_by_confidence,
        bin_results,
    )

    print("\n=== DAY 2 COMPLETED ===")