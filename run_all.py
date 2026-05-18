from __future__ import annotations

import json
from pathlib import Path

import generate_docs
import lineage
import survival_analysis
import train_models
from ingest import REPORTS_DIR, run_pipeline
from validate import main as validate_main


WEEKLY_SUMMARY_PATH = REPORTS_DIR / "weekly_model_summary.md"


def write_weekly_summary() -> None:
    metrics_path = train_models.MODEL_REPORT_PATH
    survival_path = survival_analysis.SURVIVAL_REPORT_PATH
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    survival = json.loads(survival_path.read_text(encoding="utf-8"))
    selected_model = metrics["selected_model"]
    selected_metrics = metrics[selected_model]

    WEEKLY_SUMMARY_PATH.write_text(
        "\n".join(
            [
                "# Weekly Model Performance Summary",
                "",
                f"Selected classification model: `{selected_model}`",
                f"ROC AUC: {selected_metrics['roc_auc']:.3f}",
                f"F1 score: {selected_metrics['f1']:.3f}",
                f"Best threshold for F1: {selected_metrics['best_threshold_for_f1']:.3f}",
                "",
                f"Kaplan-Meier survival at 100 days: {survival['survival_probability_at_100_days']:.3f}",
                f"Kaplan-Meier survival at 200 days: {survival['survival_probability_at_200_days']:.3f}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    run_pipeline()
    validate_main()
    train_models.main()
    survival_analysis.main()
    lineage.main()
    generate_docs.main()
    write_weekly_summary()
    print(f"Weekly summary: {WEEKLY_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
