"""
Day 5 - Full End-to-End Pipeline

Runs all project stages in the correct order:

1. Baseline classifier
2. Confidence + calibration + local stability
3. Counterfactual generation
4. Decision Tree SMT encoding
5. Formal verification

All modules use the same:
- dataset
- feature definitions
- train/test split
- random seed
- Decision Tree configuration
"""

import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"


# ---------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------

STAGES = [
    ("Day 1 - Baseline Classifier", "classifier.py"),
    ("Day 2 - Confidence and Calibration", "confidence.py"),
    ("Day 3 - Counterfactual Generation", "counterfactual.py"),
    ("Day 4 - Decision Tree SMT Encoding", "tree_encoder.py"),
    ("Day 4 - Formal Verification", "verifier.py"),
]


def run_stage(stage_name, filename):
    """
    Run one project module using the current Python interpreter.
    """

    print("\n")
    print("=" * 70)
    print(stage_name)
    print("=" * 70)

    script_path = SRC_DIR / filename

    if not script_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {script_path}"
        )

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        print(
            f"\nERROR: {filename} failed "
            f"with exit code {result.returncode}"
        )
        return False

    print(
        f"\nCompleted successfully: {filename}"
    )

    return True


def main():
    """
    Run the complete Mini-Project pipeline.
    """

    print("=" * 70)
    print("EXPLAINABLE AND VERIFIABLE ML - FULL PIPELINE")
    print("=" * 70)

    print(
        f"\nProject root:\n{PROJECT_ROOT}"
    )

    print(
        f"\nPython executable:\n{sys.executable}"
    )

    # -----------------------------------------------------
    # Check important project files
    # -----------------------------------------------------

    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "requirements_dataset.csv"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            "Dataset not found:\n"
            f"{dataset_path}"
        )

    print(
        "\nDataset found:"
        f"\n{dataset_path}"
    )

    # -----------------------------------------------------
    # Run all stages
    # -----------------------------------------------------

    for stage_name, filename in STAGES:

        success = run_stage(
            stage_name,
            filename,
        )

        if not success:

            print(
                "\nPIPELINE STOPPED."
            )

            sys.exit(1)

    # -----------------------------------------------------
    # Final output check
    # -----------------------------------------------------

    results_dir = (
        PROJECT_ROOT
        / "results"
    )

    expected_outputs = [
        results_dir / "metrics.json",
        results_dir / "counterfactual_examples.json",
        results_dir / "verification_cases.json",
        results_dir
        / "figures"
        / "calibration_plot.png",
    ]

    print("\n")
    print("=" * 70)
    print("FINAL OUTPUT CHECK")
    print("=" * 70)

    all_outputs_exist = True

    for output_path in expected_outputs:

        exists = output_path.exists()

        status = "OK" if exists else "MISSING"

        print(
            f"{status:8} {output_path}"
        )

        if not exists:
            all_outputs_exist = False

    # -----------------------------------------------------
    # Final status
    # -----------------------------------------------------

    print("\n")

    if all_outputs_exist:

        print(
            "=" * 70
        )

        print(
            "FULL PIPELINE COMPLETED SUCCESSFULLY"
        )

        print(
            "=" * 70
        )

    else:

        print(
            "PIPELINE FINISHED, "
            "BUT SOME OUTPUT FILES ARE MISSING."
        )

        sys.exit(1)


if __name__ == "__main__":
    main()