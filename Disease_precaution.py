# disease_prediction.py (Finalized Version)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Step 1: Load Datasets --------------------
symptom_df = pd.read_csv("Diseaseandsymptoms.csv")
precaution_df = pd.read_csv("Disease precaution.csv")

# -------------------- Step 2: Preprocessing --------------------
symptom_cols = [col for col in symptom_df.columns if col.startswith("Symptom")]
symptom_df[symptom_cols] = symptom_df[symptom_cols].fillna("")
symptom_df["all_symptoms"] = symptom_df[symptom_cols].values.tolist()
symptom_df["all_symptoms"] = symptom_df["all_symptoms"].apply(
    lambda x: sorted(set(sym.strip().lower() for sym in x if sym.strip() != ""))
)

# Handle duplicates
symptom_df["all_symptoms_str"] = symptom_df["all_symptoms"].apply(lambda x: ",".join(x))
symptom_df.drop_duplicates(subset=["Disease", "all_symptoms_str"], inplace=True)

# -------------------- Step 3: Balance the Dataset --------------------
from sklearn.utils import resample
min_count = symptom_df["Disease"].value_counts().min()
balanced_df = pd.concat([
    group.sample(min_count, random_state=42)
    for _, group in symptom_df.groupby("Disease")
])

# -------------------- Step 4: Feature Encoding --------------------
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(balanced_df["all_symptoms"])
le = LabelEncoder()
y = le.fit_transform(balanced_df["Disease"])

# -------------------- Step 5: Model Training with GridSearch --------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}

base_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy', verbose=0, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best Params:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, best_model.predict(X_test)))
print(classification_report(y_test, best_model.predict(X_test),
      labels=np.unique(y),
      target_names=le.inverse_transform(np.unique(y)),
      zero_division=0))

# -------------------- Step 6: Save Model & Encoders --------------------
joblib.dump(best_model, "disease_model.pkl")
joblib.dump(mlb, "symptom_encoder.pkl")
joblib.dump(le, "disease_encoder.pkl")

# -------------------- Step 7: Utility Functions --------------------
def predict_disease(symptom_list):
    model = joblib.load("disease_model.pkl")
    mlb = joblib.load("symptom_encoder.pkl")
    le = joblib.load("disease_encoder.pkl")

    processed_input = sorted(set(sym.strip().lower() for sym in symptom_list if sym.strip() != ""))
    input_vector = mlb.transform([processed_input])
    proba = model.predict_proba(input_vector)[0]
    pred_index = proba.argmax()
    disease = le.inverse_transform([pred_index])[0]
    confidence = round(proba[pred_index] * 100, 2)
    return disease, confidence

def get_precautions(disease_name):
    row = precaution_df[precaution_df["Disease"].str.lower() == disease_name.lower()]
    if row.empty:
        return ["No precautions found."]
    return row.iloc[0, 1:].dropna().tolist()

# -------------------- Step 8: Streamlit UI --------------------
st.set_page_config(page_title="Disease Predictor AI", layout="centered")
st.title("🤖 AI-Powered Disease Predictor")

st.markdown("Select your symptoms below:")

valid_symptoms = list(mlb.classes_)
user_symptoms = st.multiselect("🩺 Select Symptoms:", options=valid_symptoms)

# Age and gender dropdowns
col1, col2 = st.columns(2)
with col1:
    age_group = st.selectbox("👤 Select Age Group:", ["< 18", "18 - 35", "36 - 60", "60+"])
with col2:
    gender = st.selectbox("⚧ Select Gender:", ["Male", "Female", "Other"])

if st.button("🔍 Predict Disease"):
    if user_symptoms:
        disease, confidence = predict_disease(user_symptoms)
        precautions = get_precautions(disease)

        st.success(f"🧠 Predicted Disease: {disease}")
        st.progress(confidence / 100.0)
        st.info(f"📈 Confidence Level: {confidence}%")
        st.markdown(f"🔶 Risk Level: {'High' if len(user_symptoms) >= 6 else 'Moderate' if len(user_symptoms) >= 3 else 'Low'}")
        st.markdown(f"👤 Age Group: **{age_group}**")
        st.markdown(f"⚧ Gender: **{gender}**")

        with st.expander("⚠️ Recommended Precautions"):
            for i, p in enumerate(precautions, 1):
                st.markdown(f"{i}. {p}")
    else:
        st.warning("⚠️ Please select at least one symptom.")

# -------------------- Step 9: Disease Frequency Chart --------------------
st.subheader("📊 Top 15 Disease Occurrences")
fig, ax = plt.subplots(figsize=(12, 5))
symptom_df["Disease"].value_counts().head(15).plot(kind="barh", color="skyblue", ax=ax)
ax.set_xlabel("Count")
ax.set_ylabel("Disease")
ax.set_title("Top 15 Diseases")
st.pyplot(fig)

# -------------------- Footer --------------------
st.markdown("""
---
<div style='font-size:16px;'>
Developed with ❤️ using <b>Streamlit</b>  
<br>Author: <b>RAJA RAWAT</b> | Research Intern @ <b>SPSU</b>
</div>
""", unsafe_allow_html=True)
