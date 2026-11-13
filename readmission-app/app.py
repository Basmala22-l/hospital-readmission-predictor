import joblib
import pandas as pd
import streamlit as st

from features import add_features

st.set_page_config(page_title="30-Day Readmission Risk", page_icon="🏥", layout="centered")


@st.cache_resource
def load_artifact():
    return joblib.load("readmission_model.joblib")


art = load_artifact()
PIPE = art["pipeline"]
T_MED = art["threshold_medium"]
T_HIGH = art["threshold_high"]
NEAR_CONST = art["near_const"]
AGE_MAP = art["age_map"]
DEFAULTS = art["defaults"]

# Representative ICD-9 codes for each diagnosis group (the model was trained on grouped codes)
DIAG_CODES = {
    "Circulatory (heart, blood pressure)": "428",
    "Respiratory": "486",
    "Digestive": "560",
    "Diabetes": "250.8",
    "Injury": "820",
    "Genitourinary": "599",
    "Musculoskeletal": "715",
    "Neoplasms (cancer)": "153",
    "Other": "V57",
}
DISCHARGE = {
    "Home": 1,
    "Short-term hospital": 2,
    "Skilled nursing facility": 3,
    "Home with health service": 6,
    "Rehabilitation facility": 22,
    "Other / not mapped": 18,
}
ADMISSION_SOURCE = {"Physician referral": 1, "Emergency room": 7, "Transfer from hospital": 4}
ADMISSION_TYPE = {"Emergency": 1, "Urgent": 2, "Elective": 3}
AGE_GROUPS = list(AGE_MAP.keys())

LABELS = {
    "age": "Age group",
    "number_inpatient": "Prior inpatient stays",
    "number_outpatient": "Prior outpatient visits",
    "number_emergency": "Prior emergency visits",
    "discharge_disposition_id": "Discharge destination",
    "time_in_hospital": "Length of stay",
    "num_medications": "Number of medications",
    "num_lab_procedures": "Number of lab tests",
    "num_procedures": "Number of procedures",
    "number_diagnoses": "Number of diagnoses",
    "diag_1": "Primary diagnosis",
    "insulin": "Insulin",
    "metformin": "Metformin",
    "A1Cresult": "A1C result",
    "change": "Medication change",
    "diabetesMed": "Diabetes medication",
    "admission_source_id": "Admission source",
    "admission_type_id": "Admission type",
    "max_glu_serum": "Glucose serum test",
    "gender": "Gender",
}


def predict(row: dict) -> float:
    X = add_features(pd.DataFrame([row]), NEAR_CONST, AGE_MAP)
    return float(PIPE.predict_proba(X)[0, 1])


st.title("🏥 30-Day Readmission Risk")
st.caption("Decision-support tool for care teams. It estimates risk from a patient's discharge "
           "data. **It is not a diagnosis and does not replace clinical judgment.**")

with st.form("patient"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.selectbox("Age group", AGE_GROUPS, index=AGE_GROUPS.index("[70-80)"))
        gender = st.selectbox("Gender", ["Female", "Male"])
        stay = st.number_input("Length of stay (days)", 1, 14, 4)
        n_inp = st.number_input("Prior inpatient stays (last year)", 0, 21, 0)
        n_out = st.number_input("Prior outpatient visits (last year)", 0, 40, 0)
        n_emr = st.number_input("Prior emergency visits (last year)", 0, 76, 0)
        diag = st.selectbox("Primary diagnosis", list(DIAG_CODES.keys()))
        n_diag = st.number_input("Number of diagnoses", 1, 16, 7)
    with c2:
        disch = st.selectbox("Discharge destination", list(DISCHARGE.keys()))
        src = st.selectbox("Admission source", list(ADMISSION_SOURCE.keys()), index=1)
        atype = st.selectbox("Admission type", list(ADMISSION_TYPE.keys()))
        n_med = st.number_input("Number of medications", 1, 81, 15)
        n_lab = st.number_input("Number of lab tests", 1, 132, 43)
        n_proc = st.number_input("Number of procedures", 0, 6, 1)
        insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
        metformin = st.selectbox("Metformin", ["No", "Steady", "Up", "Down"])
        a1c = st.selectbox("A1C result", ["None", "Norm", ">7", ">8"])
        change = st.selectbox("Diabetes medication changed during stay?", ["No", "Ch"],
                              format_func=lambda x: "Yes" if x == "Ch" else "No")
        diabetes_med = st.selectbox("On diabetes medication?", ["Yes", "No"])
    submitted = st.form_submit_button("Estimate risk")

if submitted:
    user = {
        "age": age, "gender": gender, "time_in_hospital": int(stay),
        "number_inpatient": int(n_inp), "number_outpatient": int(n_out),
        "number_emergency": int(n_emr), "diag_1": DIAG_CODES[diag],
        "number_diagnoses": int(n_diag),
        "discharge_disposition_id": DISCHARGE[disch],
        "admission_source_id": ADMISSION_SOURCE[src],
        "admission_type_id": ADMISSION_TYPE[atype],
        "num_medications": int(n_med), "num_lab_procedures": int(n_lab),
        "num_procedures": int(n_proc), "insulin": insulin, "metformin": metformin,
        "A1Cresult": a1c, "change": change, "diabetesMed": diabetes_med,
    }
    row = {**DEFAULTS, **user}

    try:
        p = predict(row)
    except Exception as e:  # never crash on bad input
        st.error(f"Could not compute the risk for these inputs: {e}")
        st.stop()

    if p >= T_HIGH:
        tier, color, action = "HIGH", "red", "Follow-up call **and** a post-discharge visit."
    elif p >= T_MED:
        tier, color, action = "MEDIUM", "orange", "Follow-up phone call after discharge."
    else:
        tier, color, action = "LOW", "green", "Standard discharge care."

    st.subheader(f"Risk level: :{color}[{tier}]")
    st.metric("Model score (relative risk indicator)", f"{p:.2f}")
    st.write(f"**Suggested action:** {action}")
    st.caption("The score ranks patients by risk; it is not a literal probability. "
               "In our test set: High ≈ 24% readmitted, Medium ≈ 12%, Low ≈ 7% "
               "(overall average ≈ 11%).")

    # Local explanation: how much does the score drop if each input is reset to the typical value?
    effects = []
    for key in user:
        if key not in DEFAULTS or user[key] == DEFAULTS[key]:
            continue
        alt = dict(row)
        alt[key] = DEFAULTS[key]
        effects.append((key, p - predict(alt)))
    effects.sort(key=lambda x: abs(x[1]), reverse=True)
    top = [e for e in effects if abs(e[1]) > 0.005][:4]

    st.markdown("#### Main factors behind this result")
    if not top:
        st.write("This patient is close to the typical profile; no single factor stands out.")
    for key, delta in top:
        arrow = "⬆️ raises" if delta > 0 else "⬇️ lowers"
        st.write(f"- **{LABELS.get(key, key)}** {arrow} the risk score")

st.divider()
st.caption("Model trained on the Diabetes 130-US Hospitals dataset (1999–2008, UCI). "
           "It misses roughly a third of readmissions, so it should support, not replace, "
           "clinical decisions. Dataset: https://archive.ics.uci.edu/dataset/296")
