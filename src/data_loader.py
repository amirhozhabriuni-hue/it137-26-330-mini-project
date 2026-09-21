"""
Data loader for the PROMISE requirements classification dataset.

Dataset columns:
- ProjectID
- RequirementText
- _class_

Final binary labels:
- 0 = Functional Requirement (FR)
- 1 = Non-Functional Requirement (NFR)
"""

from pathlib import Path
import pandas as pd


# Project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset location:
DATA_PATH = PROJECT_ROOT / "data" / "requirements_dataset.csv"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load and normalize the requirements dataset.

    Returns a DataFrame with:
        requirement : requirement text
        label       : 0 = FR, 1 = NFR
    """

    # Load CSV
    df = pd.read_csv(path)

    # Check that the expected columns exist
    required_columns = {"ProjectID", "RequirementText", "_class_"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Keep only the columns needed for this project
    df = df[["RequirementText", "_class_"]].copy()

    # Rename columns to our internal names
    df = df.rename(
        columns={
            "RequirementText": "requirement",
            "_class_": "label_raw",
        }
    )

    # Remove rows with missing text or missing labels
    df["requirement"] = df["requirement"].astype(str).str.strip()
    df["label_raw"] = df["label_raw"].astype(str).str.strip().str.upper()

    df = df[
        (df["requirement"] != "")
        & (df["label_raw"] != "")
        & (df["requirement"].str.lower() != "nan")
        & (df["label_raw"].str.lower() != "nan")
    ].copy()

    # Binary mapping:
    # F = Functional Requirement
    # Everything else = Non-Functional Requirement
    df["label"] = (df["label_raw"] != "F").astype(int)

    # Return only the fields needed by the ML pipeline
    return df[["requirement", "label"]].reset_index(drop=True)


if __name__ == "__main__":
    df = load_dataset()

    print("=== Dataset Loaded Successfully ===")
    print(f"Total requirements: {len(df)}")

    print("\n=== Binary Class Distribution ===")
    print(
        df["label"]
        .map({0: "FR", 1: "NFR"})
        .value_counts()
        .sort_index()
    )

    print("\n=== First 5 Requirements ===")
    print(df.head().to_string(index=False))