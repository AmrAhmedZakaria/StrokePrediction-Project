import streamlit as st
import pandas as pd
import pickle

model_path = "final.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

st.title("Stroke Prediction App 🩺")
st.write("Enter the following details to predict the likelihood of stroke.")

gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
hypertension = st.radio("Hypertension (0 = No, 1 = Yes)", options=[0, 1])
heart_disease = st.radio("Heart Disease (0 = No, 1 = Yes)", options=[0, 1])
ever_married = st.radio("Ever Married (0 = No, 1 = Yes)", options=[0, 1])
residence_type = st.radio("Residence Type (0 = Rural, 1 = Urban)", options=[0, 1])
smoking_status = st.selectbox(
    "Smoking Status", options=["Unknown", "Formerly smoked", "Never smoked", "Smokes"]
)
work_type = st.selectbox(
    "Work Type",
    options=["Govt_job", "Never_worked", "Private", "Self-employed", "Children"],
)
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=85.0)
bmi = st.number_input("BMI", min_value=0.0, value=23.5)
age = st.number_input("Age", min_value=0, max_value=120, value=45)

scaling_params = {
    "avg_glucose_level": {"mean": 105.23, "std": 45.12},
    "bmi": {"mean": 28.15, "std": 7.45},
    "age": {"mean": 42.34, "std": 15.67},
}


def scale_feature(value, feature):
    mean = scaling_params[feature]["mean"]
    std = scaling_params[feature]["std"]
    return (value - mean) / std


avg_glucose_level_scaled = scale_feature(avg_glucose_level, "avg_glucose_level")
bmi_scaled = scale_feature(bmi, "bmi")
age_scaled = scale_feature(age, "age")

smoking_status_encoded = [
    1 if smoking_status == "Unknown" else 0,
    1 if smoking_status == "Formerly smoked" else 0,
    1 if smoking_status == "Never smoked" else 0,
    1 if smoking_status == "Smokes" else 0,
]

work_type_encoded = [
    1 if work_type == "Govt_job" else 0,
    1 if work_type == "Never_worked" else 0,
    1 if work_type == "Private" else 0,
    1 if work_type == "Self-employed" else 0,
    1 if work_type == "Children" else 0,
]

input_data = [
    1 if gender == "Male" else 0,  # Gender
    hypertension,
    heart_disease,
    ever_married,
    residence_type,
    *smoking_status_encoded,
    *work_type_encoded,
    avg_glucose_level_scaled,
    bmi_scaled,
    age_scaled,
]

columns = [
    "gender",
    "hypertension",
    "heart_disease",
    "ever_married",
    "Residence_type",
    "smoking_status_Unknown",
    "smoking_status_formerly smoked",
    "smoking_status_never smoked",
    "smoking_status_smokes",
    "work_type_Govt_job",
    "work_type_Never_worked",
    "work_type_Private",
    "work_type_Self-employed",
    "work_type_children",
    "avg_glucose_level",
    "bmi",
    "age",
]

input_df = pd.DataFrame([input_data], columns=columns)

st.write("Input Data for Prediction (after scaling):")
st.dataframe(input_df)

if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.error(
            "The model predicts a high likelihood of stroke. Please consult a healthcare professional."
        )
    else:
        st.success("The model predicts a low likelihood of stroke.")

    st.write("Prediction Probability:", prediction_proba)
