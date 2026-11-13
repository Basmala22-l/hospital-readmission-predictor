"""Feature engineering used at training time, reused by the app (same logic, no leakage)."""
import pandas as pd

DRUG_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]


def icd_group(code):
    if pd.isna(code):
        return "Missing"
    c = str(code)
    if c.startswith(("V", "E")):
        return "Other"
    try:
        n = float(c)
    except ValueError:
        return "Other"
    i = int(n)
    if 390 <= n < 460 or i == 785:
        return "Circulatory"
    if 460 <= n < 520 or i == 786:
        return "Respiratory"
    if 520 <= n < 580 or i == 787:
        return "Digestive"
    if i == 250:
        return "Diabetes"
    if 800 <= n < 1000:
        return "Injury"
    if 710 <= n < 740:
        return "Musculoskeletal"
    if 580 <= n < 630 or i == 788:
        return "Genitourinary"
    if 140 <= n < 240:
        return "Neoplasms"
    return "Other"


def add_features(d, near_const, age_map):
    d = d.copy()
    d["age_num"] = d["age"].map(age_map)
    for c in ["diag_1", "diag_2", "diag_3"]:
        d[c] = d[c].apply(icd_group)
    d["total_prior_visits"] = (d["number_outpatient"] + d["number_emergency"]
                               + d["number_inpatient"])
    d["n_drugs_taken"] = (d[DRUG_COLS] != "No").sum(axis=1)
    d["n_drugs_changed"] = d[DRUG_COLS].isin(["Up", "Down"]).sum(axis=1)
    for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        d[c] = d[c].astype(str)
    return d.drop(columns=near_const)
