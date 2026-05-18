from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
from ucimlrepo import fetch_ucirepo


DATASET_ID = 519
SOURCE_URL = "https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records"

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = Path("reports")

RAW_CSV_PATH = RAW_DIR / "heart_failure.csv"
METADATA_PATH = RAW_DIR / "metadata.json"
QUALITY_REPORT_PATH = REPORTS_DIR / "data_quality_report.json"
PROCESSED_PARQUET_PATH = PROCESSED_DIR / "heart_failure_features.parquet"
DUCKDB_PATH = DATA_DIR / "heart_failure.duckdb"

BINARY_COLUMNS = [
    "anaemia",
    "diabetes",
    "high_blood_pressure",
    "sex",
    "smoking",
    "death_event",
]


def normalize_column_name(column: str) -> str:
    return column.strip().lower().replace(" ", "_")


def fetch_heart_failure_data() -> pd.DataFrame:
    dataset = fetch_ucirepo(id=DATASET_ID)
    features = dataset.data.features
    targets = dataset.data.targets
    df = pd.concat([features, targets], axis=1)
    df.columns = [normalize_column_name(column) for column in df.columns]
    return df


def dataframe_sha256(df: pd.DataFrame) -> str:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def validate_data(df: pd.DataFrame) -> dict:
    checks = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "missing_values": int(df.isna().sum().sum()),
        "required_columns_present": all(
            column in df.columns
            for column in [
                "age",
                "anaemia",
                "creatinine_phosphokinase",
                "diabetes",
                "ejection_fraction",
                "high_blood_pressure",
                "platelets",
                "serum_creatinine",
                "serum_sodium",
                "sex",
                "smoking",
                "time",
                "death_event",
            ]
        ),
        "death_event_binary": set(df["death_event"].dropna().unique()).issubset({0, 1}),
        "binary_columns_valid": all(
            set(df[column].dropna().unique()).issubset({0, 1}) for column in BINARY_COLUMNS
        ),
        "age_range_valid": bool(df["age"].between(0, 120).all()),
        "ejection_fraction_range_valid": bool(df["ejection_fraction"].between(0, 100).all()),
        "serum_creatinine_positive": bool((df["serum_creatinine"] > 0).all()),
        "serum_sodium_positive": bool((df["serum_sodium"] > 0).all()),
        "time_non_negative": bool((df["time"] >= 0).all()),
    }
    checks["passed"] = all(
        value
        for key, value in checks.items()
        if key not in {"row_count", "column_count", "missing_values"}
    ) and checks["missing_values"] == 0
    return checks


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    features["age_group"] = pd.cut(
        features["age"],
        bins=[0, 40, 55, 70, 120],
        labels=["under_40", "40_to_55", "56_to_70", "over_70"],
        include_lowest=True,
    ).astype("string")
    features["creatinine_to_sodium_ratio"] = (
        features["serum_creatinine"] / features["serum_sodium"]
    )

    label_maps = {
        "anaemia": {0: "no_anaemia", 1: "anaemia"},
        "diabetes": {0: "no_diabetes", 1: "diabetes"},
        "high_blood_pressure": {0: "no_hypertension", 1: "hypertension"},
        "sex": {0: "woman", 1: "man"},
        "smoking": {0: "non_smoker", 1: "smoker"},
        "death_event": {0: "survived_follow_up", 1: "died_during_follow_up"},
    }

    for column, mapping in label_maps.items():
        features[f"{column}_label"] = features[column].map(mapping)

    return features


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_duckdb(raw_df: pd.DataFrame, feature_df: pd.DataFrame) -> None:
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(DUCKDB_PATH)) as conn:
        conn.register("raw_df", raw_df)
        conn.register("feature_df", feature_df)
        conn.execute("CREATE OR REPLACE TABLE heart_failure_raw AS SELECT * FROM raw_df")
        conn.execute("CREATE OR REPLACE TABLE heart_failure_features AS SELECT * FROM feature_df")


def run_pipeline() -> None:
    raw_df = fetch_heart_failure_data()
    raw_hash = dataframe_sha256(raw_df)
    quality_report = validate_data(raw_df)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df.to_csv(RAW_CSV_PATH, index=False)
    write_json(
        METADATA_PATH,
        {
            "source_url": SOURCE_URL,
            "uci_dataset_id": DATASET_ID,
            "raw_csv_path": str(RAW_CSV_PATH),
            "duckdb_path": str(DUCKDB_PATH),
            "row_count": int(len(raw_df)),
            "column_count": int(len(raw_df.columns)),
            "sha256": raw_hash,
            "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_json(QUALITY_REPORT_PATH, quality_report)

    feature_df = build_feature_table(raw_df)
    feature_df.to_parquet(PROCESSED_PARQUET_PATH, index=False)
    load_duckdb(raw_df, feature_df)

    if not quality_report["passed"]:
        raise SystemExit(f"Data quality checks failed. See {QUALITY_REPORT_PATH}")

    print(f"Ingested {len(raw_df)} rows from UCI dataset {DATASET_ID}.")
    print(f"Raw CSV: {RAW_CSV_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Quality report: {QUALITY_REPORT_PATH}")
    print(f"Feature parquet: {PROCESSED_PARQUET_PATH}")
    print(f"DuckDB database: {DUCKDB_PATH}")


if __name__ == "__main__":
    run_pipeline()
