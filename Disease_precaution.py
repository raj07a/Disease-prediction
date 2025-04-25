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
import seaborn as sns

# ---------------- Step 1: Load Datasets ----------------
symptom_df = pd.read_csv("Cleaned_DiseaseAndSymptoms.csv")
description_df = pd.read_csv("symptom_Description.csv")

# ---------------- Step 2: Preprocessing ----------------
symptom_cols = [col for col in symptom_df.columns if col.startswith("Symptom")]
symptom_df[symptom_cols] = symptom_df[symptom_cols].fillna("")

symptom_df["all_symptoms"] = symptom_df[symptom_cols].values.tolist()
symptom_df["all_symptoms"] = symptom_df["all_symptoms"].apply(
    lambda x: sorted(set(sym.strip().lower() for sym in x if sym.strip() != ""))
)

# Drop duplicates
symptom_df["all_symptoms_str"] = symptom_df["all_symptoms"].apply(lambda x: ",".join(x))
symptom_df.drop_duplicates(subset=["Disease", "all_symptoms_str"], inplace=True)

# Balance dataset
min_count = symptom_df["Disease"].value_counts().min()
balanced_df = pd.concat([
    group.sample(min_count, random_state=42)
    for _, group in symptom_df.groupby("Disease")
])

# ---------------- Step 3: Feature Engineering ----------------
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(balanced_df["all_symptoms"])

le = LabelEncoder()
y = le.fit_transform(balanced_df["Disease"])

# ---------------- Step 4: Model Training ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
model = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring="accuracy", n_jobs=-1)
model.fit(X_train, y_train)
best_model = model.best_estimator_

# ---------------- Step 5: Evaluation ----------------
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, labels=np.unique(y), target_names=le.inverse_transform(np.unique(y)), zero_division=0))

# ---------------- Step 6: Save ----------------
joblib.dump(best_model, "disease_model.pkl")
joblib.dump(mlb, "symptom_encoder.pkl")
joblib.dump(le, "disease_encoder.pkl")

# ---------------- Step 7: Utility Functions ----------------
def predict_multiple_diseases(symptom_list, top_n=3):
    model = joblib.load("disease_model.pkl")
    mlb = joblib.load("symptom_encoder.pkl")
    le = joblib.load("disease_encoder.pkl")
    processed_input = sorted(set(sym.strip().lower() for sym in symptom_list if sym.strip() != ""))
    input_vector = mlb.transform([processed_input])
    proba = model.predict_proba(input_vector)[0]
    top_indices = np.argsort(proba)[::-1][:top_n]
    return [(le.inverse_transform([i])[0], round(proba[i]*100, 2)) for i in top_indices if proba[i] > 0]

def get_description(disease):
    row = description_df[description_df['Disease'].str.lower() == disease.lower()]
    if row.empty:
        return "Description not available."
    return row.iloc[0]['Description']

# ---------------- Step 8: Streamlit UI ----------------
st.set_page_config(page_title="Disease Predictor", layout="centered")
st.markdown("<h1 style='text-align: center; color:#2E86C1;'>🤖 AI-Powered Disease Predictor</h1>", unsafe_allow_html=True)

valid_symptoms = list(mlb.classes_)
user_symptoms = st.multiselect("🩺 Select Symptoms:", options=valid_symptoms)
age_group = st.selectbox("👤 Age Group:", ["< 18", "18 - 35", "36 - 60", "60+"])
gender = st.selectbox("⚧ Gender:", ["Male", "Female", "Other"])

def get_risk_level(n): return "High" if n >= 6 else "Moderate" if n >= 3 else "Low"

if st.button("🔍 Predict Disease"):
    if user_symptoms:
        preds = predict_multiple_diseases(user_symptoms)
        risk = get_risk_level(len(user_symptoms))
        for disease, conf in preds:
            st.markdown(f"<h3 style='color:green;'>🧠 Predicted Disease: <b>{disease}</b></h3>", unsafe_allow_html=True)
            st.progress(conf / 100.0)
            st.markdown(f"<div style='font-size:18px;'>📈 Confidence: <b>{conf}%</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:18px;'>🔶 Risk Level: <b>{risk}</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:18px;'>👤 Age Group: <b>{age_group}</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:18px;'>⚧ Gender: <b>{gender}</b></div>", unsafe_allow_html=True)
            with st.expander("📘 Disease Description"):
                st.write(get_description(disease))
    else:
        st.warning("⚠️ Please select at least one symptom.")

# ---------------- Step 9: Chart ----------------
st.markdown("### 📊 Top 15 Diseases in Dataset")
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(data=symptom_df, y="Disease", order=symptom_df["Disease"].value_counts().index[:15], palette="coolwarm", ax=ax)
ax.set_title("Disease Frequency")
st.pyplot(fig)

# ---------------- Footer ----------------
st.markdown("""
---
<div style='font-size:16px;'>
Author: <b>RAJA RAWAT</b> | Research Intern @ <b>SPSU</b>
</div>
""", unsafe_allow_html=True)
