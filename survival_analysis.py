from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ingest import RAW_CSV_PATH, REPORTS_DIR, run_pipeline


SURVIVAL_REPORT_PATH = REPORTS_DIR / "survival_analysis.json"
SURVIVAL_CURVE_PATH = REPORTS_DIR / "kaplan_meier_curve.csv"
STRATIFIED_CURVE_PATH = REPORTS_DIR / "kaplan_meier_by_risk_group.csv"


def load_raw_data() -> pd.DataFrame:
    if not RAW_CSV_PATH.exists():
        run_pipeline()
    return pd.read_csv(RAW_CSV_PATH)


def kaplan_meier_curve(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    survival_probability = 1.0
    at_risk = len(df)

    for day, group in df.sort_values("time").groupby("time"):
        deaths = int(group["death_event"].sum())
        censored = int((group["death_event"] == 0).sum())
        if at_risk > 0 and deaths > 0:
            survival_probability *= 1 - deaths / at_risk
        rows.append(
            {
                "time": int(day),
                "at_risk": int(at_risk),
                "deaths": deaths,
                "censored": censored,
                "survival_probability": survival_probability,
            }
        )
        at_risk -= deaths + censored

    return pd.DataFrame(rows)


def survival_at(curve: pd.DataFrame, day: int) -> float:
    eligible = curve[curve["time"] <= day]
    if eligible.empty:
        return 1.0
    return float(eligible.iloc[-1]["survival_probability"])


def add_risk_group(df: pd.DataFrame) -> pd.DataFrame:
    creatinine_high = df["serum_creatinine"] >= df["serum_creatinine"].median()
    ejection_low = df["ejection_fraction"] <= df["ejection_fraction"].median()
    risk_score = creatinine_high.astype(int) + ejection_low.astype(int)
    return df.assign(
        clinical_risk_group=risk_score.map(
            {
                0: "lower_risk",
                1: "mixed_risk",
                2: "higher_risk",
            }
        )
    )


def stratified_survival_curves(df: pd.DataFrame) -> pd.DataFrame:
    curves = []
    for group_name, group_df in add_risk_group(df).groupby("clinical_risk_group"):
        curve = kaplan_meier_curve(group_df)
        curve.insert(0, "clinical_risk_group", group_name)
        curves.append(curve)
    return pd.concat(curves, ignore_index=True)


def main() -> None:
    df = load_raw_data()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    overall_curve = kaplan_meier_curve(df)
    stratified_curve = stratified_survival_curves(df)

    overall_curve.to_csv(SURVIVAL_CURVE_PATH, index=False)
    stratified_curve.to_csv(STRATIFIED_CURVE_PATH, index=False)

    grouped = add_risk_group(df).groupby("clinical_risk_group")["death_event"].agg(["size", "mean"])
    report = {
        "method": "Kaplan-Meier estimator",
        "censoring_note": "Rows with death_event=0 are treated as right-censored at their follow-up time.",
        "survival_probability_at_100_days": survival_at(overall_curve, 100),
        "survival_probability_at_200_days": survival_at(overall_curve, 200),
        "risk_group_mortality": {
            index: {"patients": int(row["size"]), "mortality_rate": float(row["mean"])}
            for index, row in grouped.iterrows()
        },
    }
    SURVIVAL_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Survival report: {SURVIVAL_REPORT_PATH}")
    print(f"Overall Kaplan-Meier curve: {SURVIVAL_CURVE_PATH}")
    print(f"Stratified Kaplan-Meier curve: {STRATIFIED_CURVE_PATH}")


if __name__ == "__main__":
    main()
