import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hospital Readmission Risk", layout="wide")

@st.cache_resource
def load_model():
    with open('models/xgb_calibrated.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

st.title("30-Day Hospital Readmission Risk Predictor")
st.markdown("Enter patient details at discharge to predict readmission risk.")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 5, 95, 65, step=10)
    time_in_hospital = st.slider("Length of stay (days)", 1, 14, 4)
    num_medications = st.slider("Number of medications", 1, 30, 10)
    num_lab_procedures = st.slider("Lab procedures", 1, 100, 40)
    num_procedures = st.slider("Procedures", 0, 6, 1)

with col2:
    number_inpatient = st.slider("Prior inpatient visits", 0, 10, 0)
    number_emergency = st.slider("Prior emergency visits", 0, 10, 0)
    number_outpatient = st.slider("Prior outpatient visits", 0, 10, 0)
    number_diagnoses = st.slider("Number of diagnoses", 1, 16, 5)
    discharge_disposition_id = st.selectbox(
        "Discharge disposition",
        options=[1, 2, 3, 4, 5, 6],
        format_func=lambda x: {
            1: "Home", 2: "Short-term hospital", 3: "Skilled nursing",
            4: "ICF", 5: "Another type of facility", 6: "Home health"
        }[x]
    )

with col3:
    admission_type_id = st.selectbox("Admission type",
        options=[1,2,3,4,5,6],
        format_func=lambda x: {
            1:"Emergency",2:"Urgent",3:"Elective",
            4:"Newborn",5:"Not available",6:"NULL"
        }[x])
    admission_source_id = st.selectbox("Admission source",
        options=[1,2,3,4,5,7],
        format_func=lambda x: {
            1:"Physician referral",2:"Clinic referral",3:"HMO referral",
            4:"Transfer from hospital",5:"Transfer from SNF",7:"Emergency room"
        }[x])
    diabetesMed = st.selectbox("On diabetes medication?", [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No")
    metformin = st.selectbox("Metformin", [0,1,2,3],
        format_func=lambda x: ["No","Steady","Up","Down"][x])
    insulin = st.selectbox("Insulin", [0,1,2,3],
        format_func=lambda x: ["No","Steady","Up","Down"][x])
    change = st.selectbox("Medication change?", [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No")

# Build input dataframe with all 46 features
input_dict = {
    'admission_type_id': admission_type_id,
    'discharge_disposition_id': discharge_disposition_id,
    'admission_source_id': admission_source_id,
    'time_in_hospital': time_in_hospital,
    'num_lab_procedures': num_lab_procedures,
    'num_procedures': num_procedures,
    'num_medications': num_medications,
    'number_outpatient': number_outpatient,
    'number_emergency': number_emergency,
    'number_inpatient': number_inpatient,
    'number_diagnoses': number_diagnoses,
    'metformin': metformin,
    'repaglinide': 0, 'nateglinide': 0, 'chlorpropamide': 0,
    'glimepiride': 0, 'acetohexamide': 0, 'glipizide': 0,
    'glyburide': 0, 'tolbutamide': 0, 'pioglitazone': 0,
    'rosiglitazone': 0, 'acarbose': 0, 'miglitol': 0,
    'troglitazone': 0, 'tolazamide': 0, 'examide': 0,
    'citoglipton': 0, 'insulin': insulin,
    'glyburide-metformin': 0, 'glipizide-metformin': 0,
    'glimepiride-pioglitazone': 0, 'metformin-rosiglitazone': 0,
    'metformin-pioglitazone': 0,
    'change': change, 'diabetesMed': diabetesMed,
    'age': age, 'gender': 1, 'diag_1': 4, 'diag_2': 4, 'diag_3': 0,
    'race_Asian': 0, 'race_Caucasian': 1,
    'race_Hispanic': 0, 'race_Other': 0, 'race_Unknown': 0,
}

input_df = pd.DataFrame([input_dict])

# Reorder columns to match training data exactly
expected_cols = ['gender','age','admission_type_id','discharge_disposition_id',
    'admission_source_id','time_in_hospital','num_lab_procedures','num_procedures',
    'num_medications','number_outpatient','number_emergency','number_inpatient',
    'diag_1','diag_2','diag_3','number_diagnoses','metformin','repaglinide',
    'nateglinide','chlorpropamide','glimepiride','acetohexamide','glipizide',
    'glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol',
    'troglitazone','tolazamide','examide','citoglipton','insulin',
    'glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone',
    'metformin-rosiglitazone','metformin-pioglitazone','change','diabetesMed',
    'race_Asian','race_Caucasian','race_Hispanic','race_Other','race_Unknown']

input_df = input_df[expected_cols]

if st.button("Predict Readmission Risk", type="primary"):
    prob = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    col_result, col_gauge = st.columns([1, 2])

    with col_result:
        risk_pct = round(prob * 100, 1)
        if prob >= 0.5:
            st.error(f"High Risk: {risk_pct}% probability of readmission")
        elif prob >= 0.25:
            st.warning(f"Moderate Risk: {risk_pct}% probability of readmission")
        else:
            st.success(f"Low Risk: {risk_pct}% probability of readmission")

        st.metric("Readmission Probability", f"{risk_pct}%")
        st.caption("Baseline rate in dataset: 11.2%")

    st.markdown("SHAP Explainability: ")
    with col_gauge:
        # SHAP waterfall
        with open('models/xgb_model.pkl', 'rb') as f:
            base_model = pickle.load(f)

        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(input_df)

        shap_exp = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.values[0],
            feature_names=input_df.columns.tolist()
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(shap_exp, show=False)
        st.pyplot(plt.gcf())
        plt.close()