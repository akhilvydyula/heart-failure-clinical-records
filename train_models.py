from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ingest import RAW_CSV_PATH, REPORTS_DIR, run_pipeline, write_json


MODELS_DIR = Path("models")
MODEL_REPORT_PATH = REPORTS_DIR / "model_metrics.json"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
CLUSTER_REPORT_PATH = REPORTS_DIR / "cluster_summary.csv"
REGRESSION_REPORT_PATH = REPORTS_DIR / "regression_metrics.json"
BEST_MODEL_PATH = MODELS_DIR / "death_event_model.joblib"

TARGET = "death_event"
TIME_TARGET = "time"
FEATURE_COLUMNS = [
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
]
CLINICAL_FEATURE_COLUMNS = [column for column in FEATURE_COLUMNS if column != "time"]


def load_raw_data() -> pd.DataFrame:
    if not RAW_CSV_PATH.exists():
        run_pipeline()
    return pd.read_csv(RAW_CSV_PATH)


def classification_models(random_state: int = 42) -> dict[str, Pipeline]:
    scaler = ColumnTransformer(
        [("scaled_numeric", StandardScaler(), FEATURE_COLUMNS)],
        remainder="drop",
    )
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", scaler),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        min_samples_leaf=3,
                        random_state=random_state,
                    ),
                )
            ]
        ),
        "gradient_boosting": Pipeline(
            [("model", GradientBoostingClassifier(random_state=random_state))]
        ),
    }


def tune_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    best_index = int(np.nanargmax(f1_scores[:-1]))
    return float(thresholds[best_index]), float(f1_scores[best_index])


def evaluate_classifiers(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    results: dict[str, dict] = {}
    trained_models: dict[str, Pipeline] = {}
    for name, model in classification_models().items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = model.predict(X_test)
        threshold, threshold_f1 = tune_threshold(y_test, probabilities)
        tuned_predictions = (probabilities >= threshold).astype(int)

        results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_recall_auc": float(average_precision_score(y_test, probabilities)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "f1": float(f1_score(y_test, predictions)),
            "best_threshold_for_f1": threshold,
            "best_threshold_f1": threshold_f1,
            "tuned_confusion_matrix": confusion_matrix(y_test, tuned_predictions).tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0,
            ),
        }
        trained_models[name] = model

    best_model_name = max(results, key=lambda model_name: results[model_name]["roc_auc"])
    results["selected_model"] = best_model_name
    return trained_models[best_model_name], results


def write_feature_importance(model: Pipeline, df: pd.DataFrame) -> None:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=25,
        random_state=42,
        scoring="roc_auc",
    )
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    FEATURE_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)


def evaluate_follow_up_regression(df: pd.DataFrame) -> dict:
    X = df[CLINICAL_FEATURE_COLUMNS]
    y = df[TIME_TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=42,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "target": TIME_TARGET,
        "note": "Follow-up time is right-censored for survivors, so this is a teaching baseline rather than a clinical survival model.",
        "mean_absolute_error_days": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }


def cluster_patients(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[CLINICAL_FEATURE_COLUMNS])
    clusters = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42).fit_predict(
        scaled_features
    )
    clustered = df.assign(cluster=clusters)
    return (
        clustered.groupby("cluster")
        .agg(
            patients=("death_event", "size"),
            mortality_rate=("death_event", "mean"),
            average_age=("age", "mean"),
            average_ejection_fraction=("ejection_fraction", "mean"),
            average_serum_creatinine=("serum_creatinine", "mean"),
            diabetes_rate=("diabetes", "mean"),
            smoking_rate=("smoking", "mean"),
        )
        .reset_index()
        .sort_values("mortality_rate", ascending=False)
    )


def main() -> None:
    df = load_raw_data()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_model, classification_results = evaluate_classifiers(df)
    joblib.dump(best_model, BEST_MODEL_PATH)
    write_json(MODEL_REPORT_PATH, classification_results)
    write_feature_importance(best_model, df)

    regression_results = evaluate_follow_up_regression(df)
    write_json(REGRESSION_REPORT_PATH, regression_results)

    cluster_summary = cluster_patients(df)
    cluster_summary.to_csv(CLUSTER_REPORT_PATH, index=False)

    print(f"Saved selected mortality model: {BEST_MODEL_PATH}")
    print(f"Classification metrics: {MODEL_REPORT_PATH}")
    print(f"Feature importance: {FEATURE_IMPORTANCE_PATH}")
    print(f"Regression metrics: {REGRESSION_REPORT_PATH}")
    print(f"Cluster summary: {CLUSTER_REPORT_PATH}")


if __name__ == "__main__":
    main()
