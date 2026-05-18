from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ingest import METADATA_PATH, RAW_CSV_PATH, SOURCE_URL, run_pipeline


DOCS_DIR = Path("docs")
SCHEMA_DOC_PATH = DOCS_DIR / "dataset_schema.md"
SUMMARY_DOC_PATH = DOCS_DIR / "project_summary.md"


COLUMN_DESCRIPTIONS = {
    "age": "Age of the patient in years.",
    "anaemia": "Whether the patient has decreased red blood cells or hemoglobin.",
    "creatinine_phosphokinase": "Level of the CPK enzyme in the blood, mcg/L.",
    "diabetes": "Whether the patient has diabetes.",
    "ejection_fraction": "Percentage of blood leaving the heart at each contraction.",
    "high_blood_pressure": "Whether the patient has hypertension.",
    "platelets": "Platelet count in the blood, kiloplatelets/mL.",
    "serum_creatinine": "Level of serum creatinine in the blood, mg/dL.",
    "serum_sodium": "Level of serum sodium in the blood, mEq/L.",
    "sex": "Patient sex encoded as 0 or 1 in the source dataset.",
    "smoking": "Whether the patient smokes.",
    "time": "Follow-up period in days.",
    "death_event": "Target variable: 1 if the patient died during follow-up, otherwise 0.",
}


def markdown_table(rows: list[dict[str, str]]) -> str:
    header = "| Column | Type | Missing | Description |\n|---|---:|---:|---|"
    body = "\n".join(
        f"| `{row['column']}` | `{row['type']}` | {row['missing']} | {row['description']} |"
        for row in rows
    )
    return f"{header}\n{body}"


def main() -> None:
    if not RAW_CSV_PATH.exists() or not METADATA_PATH.exists():
        run_pipeline()

    df = pd.read_csv(RAW_CSV_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "column": column,
            "type": str(df[column].dtype),
            "missing": str(int(df[column].isna().sum())),
            "description": COLUMN_DESCRIPTIONS.get(column, ""),
        }
        for column in df.columns
    ]

    SCHEMA_DOC_PATH.write_text(
        "\n".join(
            [
                "# Dataset Schema",
                "",
                f"Source: [{SOURCE_URL}]({SOURCE_URL})",
                "",
                f"Rows: {metadata['row_count']}",
                f"Columns: {metadata['column_count']}",
                f"Raw SHA256: `{metadata['sha256']}`",
                "",
                markdown_table(rows),
                "",
            ]
        ),
        encoding="utf-8",
    )

    SUMMARY_DOC_PATH.write_text(
        "\n".join(
            [
                "# Project Summary",
                "",
                "This repository implements the 13 teaching problems for the Heart Failure Clinical Records dataset.",
                "",
                "Implemented assets:",
                "",
                "- Ingestion: `ingest.py`",
                "- Data validation: `validate.py`",
                "- Classification, imbalance handling, regression, clustering, and feature importance: `train_models.py`",
                "- Survival analysis: `survival_analysis.py`",
                "- Lineage and dataset versioning: `lineage.py`",
                "- REST model serving: `serve_api.py`",
                "- Workflow orchestration: `run_all.py`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Schema docs: {SCHEMA_DOC_PATH}")
    print(f"Summary docs: {SUMMARY_DOC_PATH}")


if __name__ == "__main__":
    main()
