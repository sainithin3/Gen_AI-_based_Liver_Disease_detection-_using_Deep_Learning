from flask import Flask, request, jsonify, render_template, flash
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import os
from datetime import datetime
from dataclasses import dataclass
from typing import List

app = Flask(__name__)
app.secret_key = "AIzaSyCTQIiyLlojCvxKt-DKlIa3TJgQ4YTJ-Ug"

MODEL_DIR = "saved_model"

# -------------------------
# Data Classes
# -------------------------

@dataclass
class DetoxTask:
    title: str
    description: str
    duration: str
    difficulty: str
    benefits: List[str]


@dataclass
class Recipe:
    name: str
    ingredients: List[str]
    instructions: List[str]
    health_benefits: List[str]
    preparation_time: str
    difficulty: str


# -------------------------
# Load ML Model
# -------------------------

def load_saved_model():
    global liver_model, scaler, label_encoder, feature_names

    liver_model = load_model(os.path.join(MODEL_DIR, "liver_disease_model.h5"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))

    print("Model and preprocessors loaded successfully")


# -------------------------
# Feature Mapping (IMPORTANT)
# These names must match training dataset exactly
# -------------------------

FEATURE_MAPPING = {
    "gender": "Gender of the patient",
    "age": "Age of the patient",
    "total_bilirubin": "Total Bilirubin",
    "direct_bilirubin": "Direct Bilirubin",
    "alkaline_phosphotase": "\xa0Alkphos Alkaline Phosphotase",
    "alamine_aminotransferase": "\xa0Sgpt Alamine Aminotransferase",
    "aspartate_aminotransferase": "Sgot Aspartate Aminotransferase",
    "total_proteins": "Total Protiens",
    "albumin": "\xa0ALB Albumin",
    "albumin_globulin_ratio": "A/G Ratio Albumin and Globulin Ratio"
}


# -------------------------
# Static Health Suggestions
# -------------------------

def generate_detox_challenge(result, age):

    return DetoxTask(
        title="Daily Liver Detox Walk",
        description="Take a brisk 15 minute walk and drink warm lemon water.",
        duration="15 minutes",
        difficulty="Easy",
        benefits=[
            "Improves liver metabolism",
            "Boosts digestion",
            "Removes toxins"
        ]
    )


def generate_recipe():

    return Recipe(
        name="Liver Detox Salad",
        ingredients=[
            "Spinach",
            "Garlic",
            "Olive Oil",
            "Lemon"
        ],
        instructions=[
            "Wash spinach",
            "Chop garlic",
            "Mix olive oil and lemon",
            "Combine everything"
        ],
        health_benefits=[
            "Supports liver detox",
            "Improves digestion",
            "Rich in antioxidants"
        ],
        preparation_time="10 minutes",
        difficulty="Easy"
    )


# -------------------------
# Routes
# -------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("500.html"), 500


# -------------------------
# Prediction Route
# -------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        form_data = {}

        for form_name, feature_name in FEATURE_MAPPING.items():

            value = request.form[form_name]

            if form_name == "gender":
                form_data[feature_name] = value
            else:
                form_data[feature_name] = float(value)

        input_df = pd.DataFrame([form_data])

        # Encode gender
        input_df["Gender of the patient"] = label_encoder.transform(
            input_df["Gender of the patient"]
        )

        # Ensure same column order used during training
        input_df = input_df.reindex(columns=feature_names)

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = liver_model.predict(input_scaled)

        probability = float(prediction[0][0])

        result = "Positive (Liver Disease)" if probability < 0.5 else "Negative (Healthy)"

        probability_display = (1 - probability) * 100 if probability < 0.5 else probability * 100

        age = int(form_data["Age of the patient"])

        detox = generate_detox_challenge(result, age)
        recipe = generate_recipe()

        return render_template(
            "result.html",
            prediction=result,
            probability=probability_display,
            detox_challenge=detox,
            recipe=recipe,
            timestamp=datetime.now()
        )

    except Exception as e:

        flash(str(e), "error")

        return render_template("index.html")


# -------------------------
# Run Server
# -------------------------

if __name__ == "__main__":

    load_saved_model()

    app.run(debug=True)