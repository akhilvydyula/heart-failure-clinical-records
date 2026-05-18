from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ingest import QUALITY_REPORT_PATH, RAW_CSV_PATH, validate_data


def main() -> None:
    if not RAW_CSV_PATH.exists():
        raise SystemExit(f"Raw data not found at {RAW_CSV_PATH}. Run `python ingest.py` first.")

    df = pd.read_csv(RAW_CSV_PATH)
    report = validate_data(df)

    QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Validated {len(df)} rows.")
    print(f"Quality report: {Path(QUALITY_REPORT_PATH)}")

    if not report["passed"]:
        raise SystemExit("Data quality checks failed.")


if __name__ == "__main__":
    main()
