# Dataset Schema

Source: [https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)

Rows: 299
Columns: 13
Raw SHA256: `2524f43d9c6dc1d86d9feca3293886ab74f7f54308b89fa475e44deaef89a543`

| Column | Type | Missing | Description |
|---|---:|---:|---|
| `age` | `float64` | 0 | Age of the patient in years. |
| `anaemia` | `int64` | 0 | Whether the patient has decreased red blood cells or hemoglobin. |
| `creatinine_phosphokinase` | `int64` | 0 | Level of the CPK enzyme in the blood, mcg/L. |
| `diabetes` | `int64` | 0 | Whether the patient has diabetes. |
| `ejection_fraction` | `int64` | 0 | Percentage of blood leaving the heart at each contraction. |
| `high_blood_pressure` | `int64` | 0 | Whether the patient has hypertension. |
| `platelets` | `float64` | 0 | Platelet count in the blood, kiloplatelets/mL. |
| `serum_creatinine` | `float64` | 0 | Level of serum creatinine in the blood, mg/dL. |
| `serum_sodium` | `int64` | 0 | Level of serum sodium in the blood, mEq/L. |
| `sex` | `int64` | 0 | Patient sex encoded as 0 or 1 in the source dataset. |
| `smoking` | `int64` | 0 | Whether the patient smokes. |
| `time` | `int64` | 0 | Follow-up period in days. |
| `death_event` | `int64` | 0 | Target variable: 1 if the patient died during follow-up, otherwise 0. |
