# 🔧 metadata_builder.py
import pandas as pd
import joblib
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load datasets
old_symptom_df = pd.read_csv("DiseaseAndSymptoms.csv")
old_precaution_df = pd.read_csv("Disease precaution.csv")
diet_df = pd.read_csv("diets.csv")
med_df = pd.read_csv("medications.csv")
workout_df = pd.read_csv("workout_df.csv")
description_df = pd.read_csv("description.csv")
new_precaution_df = pd.read_csv("precautions_df.csv")

# Clean disease names
def clean_disease_names(df, column):
    df[column] = df[column].str.strip().str.lower()
    return df

old_symptom_df = clean_disease_names(old_symptom_df, "Disease")
old_precaution_df = clean_disease_names(old_precaution_df, "Disease")
diet_df = clean_disease_names(diet_df, "Disease")
med_df = clean_disease_names(med_df, "Disease")
workout_df = clean_disease_names(workout_df, "disease")
description_df = clean_disease_names(description_df, "Disease")
new_precaution_df = clean_disease_names(new_precaution_df, "Disease")

# Build metadata dict
disease_metadata = {}

for disease in old_symptom_df["Disease"].unique():
    dis = disease.lower().strip()
    disease_metadata[dis] = {
        "precautions": (
            old_precaution_df[old_precaution_df["Disease"] == dis].iloc[:, 1:].dropna(axis=1).values.tolist()[0]
            if dis in old_precaution_df["Disease"].values else
            new_precaution_df[new_precaution_df["Disease"] == dis].iloc[:, 1:].dropna(axis=1).values.tolist()[0]
            if dis in new_precaution_df["Disease"].values else []
        ),
        "medications": (
            med_df[med_df["Disease"] == dis].iloc[:, 1:].dropna(axis=1).values.tolist()[0]
            if dis in med_df["Disease"].values else []
        ),
        "diet": (
            diet_df[diet_df["Disease"] == dis].iloc[:, 1:].dropna(axis=1).values.tolist()[0]
            if dis in diet_df["Disease"].values else []
        ),
        "workout": (
            workout_df[workout_df["disease"] == dis].iloc[:, 1:].dropna(axis=1).values.tolist()[0]
            if dis in workout_df["disease"].values else []
        ),
        "description": (
            description_df[description_df["Disease"] == dis]["Description"].values[0]
            if dis in description_df["Disease"].values else ""
        )
    }

# Save metadata
joblib.dump(disease_metadata, "disease_metadata.pkl")

# 🔧 train_model.py
# Load and preprocess training data
symptom_df = pd.read_csv("DiseaseAndSymptoms.csv")
symptom_cols = [col for col in symptom_df.columns if col.startswith("Symptom")]

symptom_df[symptom_cols] = symptom_df[symptom_cols].fillna("")
symptom_df["all_symptoms"] = symptom_df[symptom_cols].values.tolist()
symptom_df["all_symptoms"] = symptom_df["all_symptoms"].apply(lambda x: sorted(set(s.strip().lower() for s in x if s.strip() != "")))
symptom_df["all_symptoms_str"] = symptom_df["all_symptoms"].apply(lambda x: ",".join(x))
symptom_df.drop_duplicates(subset=["Disease", "all_symptoms_str"], inplace=True)

# Balance dataset
min_count = symptom_df["Disease"].value_counts().min()
balanced_df = pd.concat([
    group.sample(min_count, random_state=42)
    for _, group in symptom_df.groupby("Disease")
])

# Encode
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(balanced_df["all_symptoms"])
le = LabelEncoder()
y = le.fit_transform(balanced_df["Disease"])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test), labels=np.unique(y_test), target_names=le.inverse_transform(np.unique(y_test)), zero_division=0))

# Save model & encoders
joblib.dump(model, "disease_model.pkl")
joblib.dump(mlb, "symptom_encoder.pkl")
joblib.dump(le, "disease_encoder.pkl")
import streamlit as st
# Load components
try:
    disease_model = joblib.load("disease_model.pkl")
    symptom_encoder = joblib.load("symptom_encoder.pkl")
    label_encoder = joblib.load("disease_encoder.pkl")
    disease_metadata = joblib.load("disease_metadata.pkl")
    freq_df = pd.read_csv("DiseaseAndSymptoms.csv")
except Exception as e:
    st.error(f"❌ Required file missing or unreadable: {e}")
    st.stop()

# UI config
st.set_page_config("AI Disease Predictor", layout="wide")
st.title("🧠 AI Disease Predictor & Medical Assistant")
st.markdown("Select symptoms to predict a disease and view treatment suggestions.")

symptoms = list(symptom_encoder.classes_)
selected_symptoms = st.multiselect("🩺 Select Symptoms:", symptoms)

if st.button("🔍 Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select symptoms.")
    else:
        input_vector = symptom_encoder.transform([selected_symptoms])
        proba = disease_model.predict_proba(input_vector)[0]
        pred_idx = np.argmax(proba)
        pred_disease = label_encoder.inverse_transform([pred_idx])[0].lower()
        confidence = round(proba[pred_idx] * 100, 2)

        st.success(f"✅ Predicted Disease: {pred_disease.title()}")
        st.info(f"📊 Confidence Score: {confidence:.2f}%")

        info = disease_metadata.get(pred_disease, {})

        st.markdown("### 📘 Description")
        st.markdown(f"<div class='content-block'>{info.get('description', 'No description available.')}</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ⚠️ Precautions")
            for i, p in enumerate(info.get("precautions", []), 1):
                st.markdown(f"{i}. {p}")
        with col2:
            st.markdown("### 💊 Medications")
            for m in info.get("medications", []):
                st.markdown(f"- {m}")

        st.markdown("### 🥗 Diet Plan")
        for d in info.get("diet", []):
            st.markdown(f"- {d}")
        st.markdown("### 🏃 Workout")
        for w in info.get("workout", []):
            st.markdown(f"- {w}")

        st.markdown("### 📊 Disease Frequency Chart")
        try:
            disease_counts = freq_df["Disease"].value_counts()
            fig, ax = plt.subplots(figsize=(10, 3))
            disease_counts.plot(kind="bar", ax=ax, color="#66b3ff")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        except:
            st.warning("Chart error.")

st.markdown("""
<style>
.content-block {
    background-color: #f8f9fa;
    padding: 16px;
    border-radius: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)
