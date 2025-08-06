import streamlit as st
import pandas as pd
import joblib
import json

# Load model and feature list
model = joblib.load(r"rf_model.pkl")
with open(r"model_features.json") as f:
    model_features = json.load(f)

st.title("🎓 Student Performance Predictor")

# Collect user inputs
age = st.slider("Age", 15, 22, 17)
Medu = st.slider("Mother's Education (0-4)", 0, 4, 2)
Fedu = st.slider("Father's Education (0-4)", 0, 4, 2)
studytime = st.slider("Study Time (1-4)", 1, 4, 2)
failures = st.slider("Past Failures (0-3)", 0, 3, 0)
absences = st.slider("Absences", 0, 30, 4)

# Initialize input data with all features set to 0
input_data = pd.DataFrame([0] * len(model_features), index=model_features).T

# Set numerical features (only if they exist)
for col, val in [('age', age), ('Medu', Medu), ('Fedu', Fedu), ('absences', absences)]:
    if col in input_data.columns:
        input_data[col] = val

# Set one-hot encoded values
study_col = f"studytime_{studytime}"
fail_col = f"failures_{failures}"
if study_col in input_data.columns:
    input_data[study_col] = 1
if fail_col in input_data.columns:
    input_data[fail_col] = 1

# Common categorical dummy flags (update if needed)
for flag in ['sex_M', 'school_MS', 'address_U', 'famsize_LE3', 'Pstatus_T']:
    if flag in input_data.columns:
        input_data[flag] = 1

# Predict
if st.button("Predict Grade"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Final Grade (G3): {prediction:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
