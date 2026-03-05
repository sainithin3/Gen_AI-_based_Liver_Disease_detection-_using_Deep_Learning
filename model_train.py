# model_training.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class LiverDiseasePredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_names = None
        
    def load_and_preprocess_data(self, file_path):
        try:
            # Read the data
            data = pd.read_csv(file_path, encoding='unicode_escape')
            self.feature_names = data.columns[:-1]  # Store feature names
            
            print("Dataset shape:", data.shape)
            print("\nFeatures:", list(self.feature_names))
            
            # Handle missing values more robustly
            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
            categorical_columns = data.select_dtypes(include=['object']).columns
            
            # Fill numeric missing values with median
            for col in numeric_columns:
                data[col] = data[col].fillna(data[col].median())
                
            # Fill categorical missing values with mode
            for col in categorical_columns:
                data[col] = data[col].fillna(data[col].mode()[0])
            
            # Encode categorical variables
            data['Gender of the patient'] = self.label_encoder.fit_transform(data['Gender of the patient'])
            
            # Extract features and target
            X = data.drop('Result', axis=1)
            y = data['Result'].map({1: 0, 2: 1})  # Map to binary classification (0: healthy, 1: disease)
            
            # Print class distribution
            print("\nClass distribution:")
            print(y.value_counts(normalize=True))
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            
            return X_scaled, y
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def create_model(self, input_shape):
        model = Sequential([
            # First dense layer with input
            Dense(256, activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.4),
            
            # Second dense layer
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Third dense layer
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Fourth dense layer
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer,
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        print("\nModel Summary:")
        model.summary()
        
        return model
    
    def train_model(self, X, y, validation_split=0.2, epochs=150):
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"\nTraining set shape: {X_train.shape}")
            print(f"Testing set shape: {X_test.shape}")
            
            # Create callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=10,
                min_lr=0.00001,
                verbose=1
            )
            
            # Create and train model
            self.model = self.create_model(X.shape[1])
            
            history = self.model.fit(
                X_train, y_train,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=32,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            # Evaluate model
            self.evaluate_model(X_test, y_test)
            
            # Save the model and preprocessors
            self.save_model()
            
            return history, (X_train, X_test, y_train, y_test)
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
    
    def evaluate_model(self, X_test, y_test):
        try:
            y_pred = (self.model.predict(X_test) > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            print("\nModel Evaluation Metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            
            print("\nConfusion Matrix:")
            print(conf_matrix)
            
            # Calculate and print additional metrics
            tn, fp, fn, tp = conf_matrix.ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)
            
            print("\nAdditional Metrics:")
            print(f"Specificity: {specificity:.4f}")
            print(f"Sensitivity: {sensitivity:.4f}")
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise
    
    def plot_training_history(self, history):
        try:
            plt.figure(figsize=(15, 5))
            
            # Accuracy plot
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
            # Loss plot
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting training history: {str(e)}")
            raise
    
    def save_model(self, model_dir='saved_model'):
        """Save the model and preprocessors"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)
            
            # Save Keras model
            self.model.save(os.path.join(model_dir, 'liver_disease_model.h5'))
            
            # Save preprocessors
            joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.joblib'))
            joblib.dump(self.label_encoder, os.path.join(model_dir, 'label_encoder.joblib'))
            
            # Save feature names
            joblib.dump(self.feature_names, os.path.join(model_dir, 'feature_names.joblib'))
            
            print(f"\nModel and preprocessors saved to {model_dir}")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise

    def predict_single_case(self, input_data):
        try:
            # Convert input to DataFrame if it's a dictionary
            if isinstance(input_data, dict):
                input_data = pd.DataFrame([input_data])
            
            # Encode gender if present
            if 'Gender of the patient' in input_data.columns:
                input_data['Gender of the patient'] = self.label_encoder.transform(
                    input_data['Gender of the patient']
                )
            
            # Scale the input data
            input_scaled = self.scaler.transform(input_data)
            
            # Make prediction
            prediction = self.model.predict(input_scaled)
            probability = prediction[0][0]
            result = "Negative (Healthy)" if probability < 0.5 else "Positive (Liver Disease)"
            
            return result, probability
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise

def main():
    try:
        print("=== Liver Disease Prediction System ===")
        
        # Initialize predictor
        predictor = LiverDiseasePredictor()
        
        # Load and preprocess data
        print("\nLoading and preprocessing data...")
        X, y = predictor.load_and_preprocess_data("Liver Patient Dataset (LPD)_train.csv")
        
        # Train model
        print("\nTraining model...")
        history, (X_train, X_test, y_train, y_test) = predictor.train_model(X, y)
        
        # Plot training history
        print("\nPlotting training history...")
        predictor.plot_training_history(history)
        
        print("\nModel training completed successfully!")
        
    except Exception as e:
        print(f"An error occurred in the main program: {str(e)}")

if __name__ == "__main__":
    main()