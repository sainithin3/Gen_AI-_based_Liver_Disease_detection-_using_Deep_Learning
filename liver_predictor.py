import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import pandas as pd
import joblib

class LiverPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        
    def load_model(self, model_path='liver_disease_model.keras', scaler_path='scaler.pkl'):
        """Load the saved model and scaler"""
        self.model = models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        print("Model and scaler loaded successfully")
        
    def prepare_input(self, data):
        """Prepare input data for prediction"""
        # Create DataFrame with expected column order
        input_df = pd.DataFrame([data])
        
        # Scale the features using the loaded scaler
        scaled_data = self.scaler.transform(input_df)
        return scaled_data
    
    def predict(self, input_data):
        """Make prediction using the loaded model"""
        # Prepare input
        scaled_data = self.prepare_input(input_data)
        
        # Make prediction
        prediction_prob = self.model.predict(scaled_data)
        prediction = (prediction_prob > 0.5).astype(int)
        
        return {
            'probability': float(prediction_prob[0][0]),
            'prediction': int(prediction[0][0]),
            'diagnosis': 'Liver Disease' if prediction[0][0] == 1 else 'No Liver Disease'
        }