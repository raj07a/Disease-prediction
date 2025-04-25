import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# ---------------- Load Datasets ----------------
symptom_df = pd.read_csv("DiseaseAndSymptoms.csv")
severity_df = pd.read_csv("Symptom-severity.csv")
description_df = pd.read_csv("symptom_Description.csv")     # Make sure this file exists
precaution_df = pd.read_csv("symptom_precaution.csv")       # Make sure this file exists

# ---------------- Preprocessing ----------------
symptom_cols = [col for col in symptom_df.columns if col.startswith("Symptom")]
symptom_df[symptom_cols] = symptom_df[symptom_cols].fillna("")

symptom_df["all_symptoms"] = symptom_df[symptom_cols].values.tolist()
symptom_df["all_symptoms"] = symptom_df["all_symptoms"].apply(
    lambda x: sorted(set(sym.strip().lower() for sym in x if sym.strip() != ""))
)

symptom_df["all_symptoms_str"] = symptom_df["all_symptoms"].apply(lambda x: ",".join(x))
symptom_df.drop_duplicates(subset=["Disease", "all_symptoms_str"], inplace=True)

# Balance dataset
min_count = symptom_df["Disease"].value_counts().min()
balanced_df = pd.concat([
    group.sample(min_count, random_state=42)
    for _, group in symptom_df.groupby("Disease")
])

# ---------------- Vectorize Symptoms ----------------
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(balanced_df["all_symptoms"])
symptom_list = list(mlb.classes_)

le = LabelEncoder()
y = le.fit_transform(balanced_df["Disease"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- Model Training ----------------
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', verbose=0)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# ---------------- Save Model ----------------
joblib.dump(best_model, "disease_model.pkl")
joblib.dump(mlb, "symptom_encoder.pkl")
joblib.dump(le, "disease_encoder.pkl")

# ---------------- Prediction Utilities ----------------
def predict_multiple_diseases(symptom_list, top_n=3):
    model = joblib.load("disease_model.pkl")
    mlb = joblib.load("symptom_encoder.pkl")
    le = joblib.load("disease_encoder.pkl")

    processed_input = sorted(set(sym.strip().lower() for sym in symptom_list if sym.strip() != ""))
    input_vector = mlb.transform([processed_input])
    pred_proba = model.predict_proba(input_vector)[0]
    top_indices = np.argsort(pred_proba)[::-1][:top_n]
    predictions = [(le.inverse_transform([i])[0], round(pred_proba[i] * 100, 2)) for i in top_indices if pred_proba[i] > 0]
    return predictions

def get_precautions(disease):
    row = precaution_df[precaution_df['Disease'].str.lower() == disease.lower()]
    if row.empty:
        return ["No precautions available."]
    return row.iloc[0, 1:].dropna().tolist()

def get_description(disease):
    row = description_df[description_df['Disease'].str.lower() == disease.lower()]
    if row.empty:
        return "Description not available."
    return row.iloc[0]['Description']

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="AI Disease Predictor", layout="centered")
st.markdown("<h1 style='text-align:center; color:#2E86C1;'>🤖 AI Disease Predictor</h1>", unsafe_allow_html=True)

valid_symptoms = list(mlb.classes_)
user_symptoms = st.multiselect("🩺 Select your symptoms:", options=valid_symptoms)
age_group = st.selectbox("👤 Age Group:", ["<18", "18-35", "36-60", "60+"])
gender = st.selectbox("⚧ Gender:", ["Male", "Female", "Other"])

def get_risk_level(sym_count):
    return "High" if sym_count >= 6 else "Moderate" if sym_count >= 3 else "Low"

if st.button("🔍 Predict Disease"):
    if user_symptoms:
        predictions = predict_multiple_diseases(user_symptoms)
        risk = get_risk_level(len(user_symptoms))

        for disease, confidence in predictions:
            description = get_description(disease)
            precautions = get_precautions(disease)

            st.markdown(f"<h3 style='color:green;'>🧠 Predicted Disease: <b>{disease}</b></h3>", unsafe_allow_html=True)
            st.progress(confidence / 100.0)
            st.markdown(f"📈 Confidence: **{confidence}%**")
            st.markdown(f"🔶 Risk Level: **{risk}**")
            st.markdown(f"👤 Age Group: **{age_group}**")
            st.markdown(f"⚧ Gender: **{gender}**")

            with st.expander("📘 Disease Description"):
                st.write(description)

            with st.expander("⚠️ Recommended Precautions"):
                for i, p in enumerate(precautions, 1):
                    st.markdown(f"{i}. {p}")
    else:
        st.warning("⚠️ Please select at least one symptom to proceed.")

# ---------------- Visualization ----------------
st.markdown("### 📊 Top 15 Disease Occurrences")
top15 = symptom_df["Disease"].value_counts().nlargest(15)
fig, ax = plt.subplots(figsize=(10, 6))
top15.plot(kind='barh', ax=ax, color='skyblue')
ax.set_title("Most Common Diseases")
ax.set_xlabel("Count")
st.pyplot(fig)

# ---------------- Footer ----------------
st.markdown("""
---
<div style='font-size:16px; text-align:center;'>
Created by <b>RAJA RAWAT</b> | Research Intern @ <b>SPSU</b>
</div>
""", unsafe_allow_html=True)
