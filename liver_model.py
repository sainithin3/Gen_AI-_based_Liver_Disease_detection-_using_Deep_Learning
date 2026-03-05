# liver_model.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models
import joblib
import json
import os

class LiverDiseaseModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
    
    def create_model(self):
        """Create the neural network model"""
        self.model = models.Sequential([
            keras.layers.Input(shape=(11,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def load_saved_model(self, base_path=''):
        """Load the saved model and associated files"""
        try:
            # Load the Keras model
            model_path = os.path.join(base_path, 'liver_disease_model.keras')
            self.model = tf.keras.models.load_model(model_path)
            
            # Load the scaler
            self.scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
            
            # Load feature names
            with open(os.path.join(base_path, 'feature_names.json'), 'r') as f:
                self.feature_names = json.load(f)
            
            print("Model and associated files loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Creating new model...")
            self.create_model()
    
    def save_model(self, base_path='model/'):
        """Save the model and associated files"""
        os.makedirs(base_path, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(base_path, 'liver_disease_model')
        tf.keras.models.save_model(self.model, model_path, save_format='tf')
        
        # Save the scaler
        joblib.dump(self.scaler, os.path.join(base_path, 'scaler.pkl'))
        
        # Save feature names
        with open(os.path.join(base_path, 'feature_names.json'), 'w') as f:
            json.dump(self.feature_names, f)
    
    def predict(self, input_data):
        """Make predictions using the model"""
        # Convert input to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Create a new DataFrame with the correct column names
        formatted_input = pd.DataFrame(columns=self.feature_names)
        
        # Map the input columns to the correct feature names
        column_mapping = {
            'Age of the patient': 'Age of the patient',
            'Total Bilirubin': 'Total Bilirubin',
            'Direct Bilirubin': 'Direct Bilirubin',
            'Alkaline Phosphotase': 'Alkphos Alkaline Phosphotase',
            'Alamine Aminotransferase': 'Sgpt Alamine Aminotransferase',
            'Aspartate Aminotransferase': 'Sgot Aspartate Aminotransferase',
            'Total Proteins': 'Total Protiens',  # Note the spelling in original dataset
            'Albumin': 'ALB Albumin',
            'A/G Ratio Albumin and Globulin Ratio': 'A/G Ratio Albumin and Globulin Ratio',
            'Gender_Female': 'Gender_Female',
            'Gender_Male': 'Gender_Male'
        }
        
        # Map input data to correct column names
        for new_col, old_col in column_mapping.items():
            if new_col in input_data:
                formatted_input[old_col] = input_data[new_col]
        
        # Scale the features
        scaled_data = self.scaler.transform(formatted_input)
        
        # Make prediction
        prediction_prob = self.model.predict(scaled_data)
        predictions = (prediction_prob > 0.5).astype(int)
        
        # Prepare results
        results = []
        for prob, pred in zip(prediction_prob, predictions):
            results.append({
                'probability': float(prob[0]),
                'prediction': int(pred[0]),
                'diagnosis': 'Liver Disease' if pred[0] == 1 else 'No Liver Disease'
            })
        
        return results