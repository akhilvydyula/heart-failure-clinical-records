from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone

from ingest import METADATA_PATH, RAW_CSV_PATH, REPORTS_DIR, SOURCE_URL, run_pipeline


LINEAGE_REPORT_PATH = REPORTS_DIR / "lineage.json"
DVC_LOCK_PATH = "dvc.lock"


def main() -> None:
    if not RAW_CSV_PATH.exists() or not METADATA_PATH.exists():
        run_pipeline()

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    raw_bytes = RAW_CSV_PATH.read_bytes()
    raw_md5 = hashlib.md5(raw_bytes).hexdigest()
    lineage = {
        "dataset": "heart_failure_clinical_records",
        "source": SOURCE_URL,
        "raw_asset": str(RAW_CSV_PATH),
        "sha256": metadata["sha256"],
        "md5": raw_md5,
        "row_count": metadata["row_count"],
        "column_count": metadata["column_count"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "downstream_assets": [
            "data/processed/heart_failure_features.parquet",
            "data/heart_failure.duckdb",
            "reports/data_quality_report.json",
            "reports/model_metrics.json",
            "reports/survival_analysis.json",
        ],
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LINEAGE_REPORT_PATH.write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")

    dvc_lock = f"""schema: '2.0'
stages:
  ingest:
    cmd: python ingest.py
    outs:
      - path: {RAW_CSV_PATH.as_posix()}
        hash: md5
        md5: {raw_md5}
        size: {RAW_CSV_PATH.stat().st_size}
"""
    with open(DVC_LOCK_PATH, "w", encoding="utf-8") as file:
        file.write(dvc_lock)

    print(f"Lineage report: {LINEAGE_REPORT_PATH}")
    print(f"DVC-style lock file: {DVC_LOCK_PATH}")


if __name__ == "__main__":
    main()
