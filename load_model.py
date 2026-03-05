import tensorflow as tf
import numpy as np
import h5py
import os

def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def extract_weights_from_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        # Get model_weights group
        model_weights = f['model_weights']
        
        # Extract weights for each layer
        extracted_weights = []
        for layer_name in model_weights.keys():
            if 'dense' in layer_name or 'batch_normalization' in layer_name:
                layer = model_weights[layer_name]
                weight_names = layer.attrs['weight_names']
                
                layer_weights = []
                for weight_name in weight_names:
                    weight_value = layer[weight_name][()]
                    layer_weights.append(weight_value)
                
                extracted_weights.append(layer_weights)
        
        return extracted_weights

def set_model_weights(model, extracted_weights):
    for layer, weights in zip(model.layers[1:], extracted_weights):  # Skip input layer
        layer.set_weights(weights)

def main():
    model_path = 'liver_disease_model.keras'
    input_shape = (11,)
    
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return
    
    try:
        # Extract weights from the h5 file
        extracted_weights = extract_weights_from_h5(model_path)
        
        # Create a new model with the expected architecture
        new_model = create_model(input_shape)
        
        # Set the extracted weights to the new model
        set_model_weights(new_model, extracted_weights)
        
        print("Model rebuilt and weights loaded successfully!")
        new_model.summary()
        
        # Test the model with sample data
        sample_input = np.random.rand(1, 11)
        prediction = new_model.predict(sample_input)
        print(f"Sample prediction: {prediction[0][0]}")
        
        # Save the rebuilt model
        new_model.save('rebuilt_liver_disease_model.keras')
        print("Rebuilt model saved successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    print(f"TensorFlow version: {tf.__version__}")

if __name__ == "__main__":
    main()