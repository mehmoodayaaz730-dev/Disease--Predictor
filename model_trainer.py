"""
model_trainer.py
Trains Logistic Regression models for Diabetes, Heart Disease, and Liver Disease
and saves them as .pkl files so the Streamlit app can load and use them.
"""

import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────
# 1. DIABETES  (based on Pima Indians dataset)
# ─────────────────────────────────────────────
def train_diabetes_model():
    np.random.seed(42)
    n = 800
    # Features: Pregnancies, Glucose, BloodPressure, SkinThickness,
    #           Insulin, BMI, DiabetesPedigreeFunction, Age
    X = np.column_stack([
        np.random.randint(0, 17, n),                      # Pregnancies
        np.random.normal(120, 30, n).clip(50, 200),       # Glucose
        np.random.normal(70, 12, n).clip(40, 120),        # BloodPressure
        np.random.normal(25, 10, n).clip(0, 60),          # SkinThickness
        np.random.normal(80, 100, n).clip(0, 500),        # Insulin
        np.random.normal(32, 7, n).clip(18, 60),          # BMI
        np.random.exponential(0.5, n).clip(0.08, 2.5),    # Pedigree
        np.random.randint(21, 81, n).astype(float),       # Age
    ])
    # Label: high glucose + high BMI → more likely diabetic
    prob = 1 / (1 + np.exp(-(
        -8 + 0.04*X[:,1] + 0.07*X[:,5] + 0.02*X[:,7] + 0.3*X[:,6]
    )))
    y = (np.random.rand(n) < prob).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_s))
    print(f"Diabetes Model Accuracy: {acc*100:.1f}%")
    return model, scaler


# ─────────────────────────────────────────────
# 2. HEART DISEASE  (Cleveland-style)
# ─────────────────────────────────────────────
def train_heart_model():
    np.random.seed(7)
    n = 800
    # Features: Age, Sex(0/1), ChestPainType(0-3), RestingBP, Cholesterol,
    #           FastingBS, RestECG(0-2), MaxHR, ExerciseAngina(0/1),
    #           Oldpeak, Slope(0-2), NumVessels(0-3), Thal(0-2)
    X = np.column_stack([
        np.random.randint(29, 78, n).astype(float),          # Age
        np.random.randint(0, 2, n).astype(float),            # Sex
        np.random.randint(0, 4, n).astype(float),            # ChestPainType
        np.random.normal(130, 20, n).clip(90, 200),          # RestingBP
        np.random.normal(245, 50, n).clip(120, 400),         # Cholesterol
        np.random.randint(0, 2, n).astype(float),            # FastingBS
        np.random.randint(0, 3, n).astype(float),            # RestECG
        np.random.normal(150, 22, n).clip(70, 210),          # MaxHR
        np.random.randint(0, 2, n).astype(float),            # ExerciseAngina
        np.random.exponential(1.0, n).clip(0, 6.2),          # Oldpeak
        np.random.randint(0, 3, n).astype(float),            # Slope
        np.random.randint(0, 4, n).astype(float),            # NumVessels
        np.random.randint(0, 3, n).astype(float),            # Thal
    ])
    prob = 1 / (1 + np.exp(-(
        -5 + 0.04*X[:,0] + 0.5*X[:,2] + 0.01*X[:,4] - 0.02*X[:,7]
        + 0.6*X[:,8] + 0.3*X[:,9] + 0.4*X[:,11]
    )))
    y = (np.random.rand(n) < prob).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=7)
    model.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_s))
    print(f"Heart Disease Model Accuracy: {acc*100:.1f}%")
    return model, scaler


# ─────────────────────────────────────────────
# 3. LIVER DISEASE  (ILPD-style)
# ─────────────────────────────────────────────
def train_liver_model():
    np.random.seed(13)
    n = 800
    # Features: Age, Gender(0/1), TotalBilirubin, DirectBilirubin,
    #           AlkalinePhosphotase, AlamineAminotransferase,
    #           AspartateAminotransferase, TotalProteins, Albumin,
    #           AlbuminGlobulinRatio
    X = np.column_stack([
        np.random.randint(4, 90, n).astype(float),           # Age
        np.random.randint(0, 2, n).astype(float),            # Gender
        np.random.exponential(1.5, n).clip(0.4, 30),         # TotalBilirubin
        np.random.exponential(0.5, n).clip(0.1, 15),         # DirectBilirubin
        np.random.normal(220, 120, n).clip(60, 800),         # AlkalinePhosphotase
        np.random.exponential(30, n).clip(7, 700),           # AlamineAT
        np.random.exponential(35, n).clip(10, 800),          # AspartateAT
        np.random.normal(6.5, 1.0, n).clip(2.7, 9.6),       # TotalProteins
        np.random.normal(3.2, 0.6, n).clip(0.9, 5.5),       # Albumin
        np.random.normal(1.0, 0.3, n).clip(0.3, 2.8),       # A/G Ratio
    ])
    prob = 1 / (1 + np.exp(-(
        -3 + 0.02*X[:,0] + 0.15*X[:,2] + 0.003*X[:,4]
        + 0.002*X[:,5] + 0.001*X[:,6] - 0.3*X[:,9]
    )))
    y = (np.random.rand(n) < prob).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=13)
    model.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_s))
    print(f"Liver Disease Model Accuracy: {acc*100:.1f}%")
    return model, scaler


# ─────────────────────────────────────────────
# SAVE ALL MODELS
# ─────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    dm, ds = train_diabetes_model()
    pickle.dump((dm, ds), open("models/diabetes_model.pkl", "wb"))

    hm, hs = train_heart_model()
    pickle.dump((hm, hs), open("models/heart_model.pkl", "wb"))

    lm, ls = train_liver_model()
    pickle.dump((lm, ls), open("models/liver_model.pkl", "wb"))

    print("\n✅ All models trained and saved in /models folder!")
