# Heart Failure Clinical Records Project

This repository implements teaching solutions for the UCI Heart Failure Clinical Records dataset.

Dataset source: [UCI Machine Learning Repository, dataset 519](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)

The project covers data ingestion, validation, versioning, feature engineering, mortality classification, survival analysis, interpretability, regression, clustering, imbalance handling, orchestration, API serving, and documentation.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Invoke Task Runner

This project uses [Invoke](https://www.pyinvoke.org/) as a Pythonic Make alternative. Tasks live in `tasks.py`, so the workflow can be orchestrated in Python instead of shell-specific syntax.

List available tasks:

```bash
invoke --list
```

## Run Everything

```bash
invoke all
```

This runs ingestion, validation, feature generation, model training, survival analysis, lineage/version metadata, docs generation, and a weekly model summary.

## Problem Solutions

| # | Problem | Implementation |
|---:|---|---|
| 1 | Binary classification: predict `death_event` | `train_models.py` trains logistic regression, random forest, and gradient boosting models. |
| 2 | Survival analysis | `survival_analysis.py` creates Kaplan-Meier curves and survival probabilities at 100 and 200 days. |
| 3 | Feature importance and clinical interpretability | `train_models.py` writes permutation importance to `reports/feature_importance.csv`. |
| 4 | Regression: predict follow-up time | `train_models.py` trains a random forest regressor and writes `reports/regression_metrics.json`. |
| 5 | Clustering: identify patient subgroups | `train_models.py` runs K-means and writes `reports/cluster_summary.csv`. |
| 6 | Imbalanced learning | Classification models use class weighting and threshold tuning, reported in `reports/model_metrics.json`. |
| 7 | Data ingestion pipeline | `ingest.py` fetches UCI ID 519, writes raw CSV, metadata, Parquet, and DuckDB tables. |
| 8 | Data quality and validation | `validate.py` checks missing values, binary fields, and plausible clinical ranges. |
| 9 | Data versioning and lineage | `lineage.py` writes `reports/lineage.json` and a DVC-style `dvc.lock`. |
| 10 | Feature engineering pipeline | `ingest.py` creates age groups, creatinine/sodium ratio, and readable labels. |
| 11 | Reproducible workflow orchestration | `run_all.py` acts as a local DAG and writes `reports/weekly_model_summary.md`. |
| 12 | App/API wrapper for model serving | `streamlit_app.py` provides an interactive dashboard; `serve_api.py` exposes `/health` and `/predict` with FastAPI. |
| 13 | Data lineage and documentation | `generate_docs.py` writes schema and project docs under `docs/`. |

## Individual Commands

Ingest and transform:

```bash
invoke ingest
```

Validate raw data:

```bash
invoke validate
```

Train ML models, generate feature importance, run regression, and cluster patients:

```bash
invoke train
```

Run survival analysis:

```bash
invoke survival
```

Generate lineage and DVC-style lock metadata:

```bash
invoke lineage
```

Generate docs:

```bash
invoke docs
```

Start the Streamlit dashboard:

```bash
invoke streamlit
```

Start the prediction API:

```bash
invoke api
```

The Streamlit app is the easiest way to use the model interactively. It opens a local dashboard with patient input controls, risk probability, feature importance, model metrics, and cluster summaries.

Example prediction request:

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  --data @examples/predict_payload.json
```

## Generated Outputs

Generated files are ignored by git except `dvc.lock`.

- `data/raw/heart_failure.csv`
- `data/raw/metadata.json`
- `data/processed/heart_failure_features.parquet`
- `data/heart_failure.duckdb`
- `reports/data_quality_report.json`
- `reports/model_metrics.json`
- `reports/feature_importance.csv`
- `reports/regression_metrics.json`
- `reports/cluster_summary.csv`
- `reports/survival_analysis.json`
- `reports/kaplan_meier_curve.csv`
- `reports/kaplan_meier_by_risk_group.csv`
- `reports/lineage.json`
- `reports/weekly_model_summary.md`
- `docs/dataset_schema.md`
- `docs/project_summary.md`
- `models/death_event_model.joblib`

## Teaching Notes

This is a compact local implementation meant for curriculum use. The dataset is small, so the focus is on workflow design and reproducibility rather than production-scale infrastructure.

For clinical use, survival modeling and classification would need stronger validation, calibration, uncertainty analysis, and external data.
