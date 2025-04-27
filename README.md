
# 🤖 AI-Powered Intelligent Disease Prediction System

## 📚 Introduction
Early detection of diseases can drastically improve patient outcomes. However, symptom overlap often leads to misdiagnosis. This project leverages Machine Learning (ML) techniques to build an intelligent disease prediction system that predicts possible diseases based on user-input symptoms.

## 🚀 Problem Statement
Traditional diagnosis methods are time-consuming and prone to human error. There is a pressing need for an AI-powered tool that can assist users in early-stage disease prediction.

## 🎯 Objective
- Predict top possible diseases based on symptoms.
- Recommend precautions to users.
- Enhance diagnosis speed and accuracy using Machine Learning.

## 🏗️ System Architecture

```plaintext
User Inputs (Symptoms, Age, Gender)
    ↓
Preprocessing & Symptom Encoding
    ↓
Trained Random Forest Classifier
    ↓
Predicted Disease + Confidence + Risk
    ↓
Display UI (Precautions, Description)
**
🛠️ Tech Stack**
Python 3.10

Streamlit (Frontend)

Scikit-learn (ML Modeling)

Pandas & Numpy (Data Handling)

Matplotlib & Seaborn (Visualization)

#🔥 Features
Intelligent multi-disease prediction (Top 3 diseases).

Dynamic UI for Age Group, Gender, and Symptoms.

Risk Level Analysis (Low, Moderate, High).

Disease Precautions & Description.

📂 Project Structure
bash
Copy
Edit
/app/
  ├── disease_prediction_app.py
/models/
  ├── disease_model.pkl
  ├── symptom_encoder.pkl
  ├── disease_encoder.pkl
/datasets/
  ├── DiseaseAndSymptoms.csv
  ├── Disease precaution.csv
/screenshots/
  ├── ui_homepage.png
  ├── prediction_result.png
requirements.txt
README.md
📥 Installation
bash
Copy
Edit
git clone https://github.com/raj07a/Disease-prediction.git
cd Disease-prediction
pip install -r requirements.txt
🧠 Usage
bash
Copy
Edit
streamlit run app/disease_prediction_app.py
🎯 Results
Model Accuracy: 90%

Macro F1 Score: 0.63

Weighted F1 Score: 0.90

Prediction Speed: <1 second

👥 Contributors
Raja Rawat - Research Intern @ SPSU

📜 License
This project is licensed under the MIT License.
