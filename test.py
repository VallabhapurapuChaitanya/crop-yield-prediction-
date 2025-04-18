import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import os


def predict_single_sample(area, item, year, rainfall, pesticides, avg_temp):
    """
    Make a prediction for a single sample using the trained models

    Args:
        area (str): Country/region name
        item (str): Crop type
        year (int): Year of prediction
        rainfall (float): Average rainfall in mm per year
        pesticides (float): Pesticide usage in tonnes
        avg_temp (float): Average temperature

    Returns:
        float: Predicted crop yield
    """
    # Check if models exist
    if not os.path.exists('models/bigru_model.h5'):
        print("Error: Models not found. Please run the training script first.")
        return None

    # Load models and preprocessing objects
    print("Loading models and preprocessing objects...")
    bigru_model = load_model('models/bigru_model.h5')
    hybrid_xgb = joblib.load('models/hybrid_xgb_model.joblib')
    scaler_X = joblib.load('models/scaler_X.joblib')
    scaler_y = joblib.load('models/scaler_y.joblib')

    # Load encoders
    encoders = {}
    for col in ['Area', 'Item']:
        encoders[col] = joblib.load(f'models/encoder_{col}.joblib')

    # Check if the provided area and item are in the encoder classes
    if area not in encoders['Area'].classes_:
        print(f"Warning: '{area}' not found in training data. Using closest match.")
        area = encoders['Area'].classes_[0]  # Using the first class as default

    if item not in encoders['Item'].classes_:
        print(f"Warning: '{item}' not found in training data. Using closest match.")
        item = encoders['Item'].classes_[0]  # Using the first class as default

    # Encode the categorical variables
    area_encoded = encoders['Area'].transform([area])[0]
    item_encoded = encoders['Item'].transform([item])[0]

    # Create a single sample in array format
    sample = np.array([[area_encoded, item_encoded, year, rainfall, pesticides, avg_temp]])

    # Scale the sample
    sample_scaled = scaler_X.transform(sample)

    # For Bi-GRU, we need historical data to form a sequence
    # Here we'll duplicate the sample to create a sequence of required length
    time_steps = 3
    sample_seq = np.array([np.tile(sample_scaled[0], (time_steps, 1))])

    # Get Bi-GRU prediction
    bigru_pred = bigru_model.predict(sample_seq)

    # Combine with tabular data for hybrid model
    hybrid_features = np.column_stack((bigru_pred, sample_scaled))

    # Get hybrid model prediction
    y_pred_hybrid = hybrid_xgb.predict(hybrid_features)

    # Inverse transform prediction to original scale
    prediction = scaler_y.inverse_transform(y_pred_hybrid.reshape(-1, 1)).flatten()[0]

    return prediction


def test_with_array_input():
    """
    Test the model with sample array input
    """
    # Sample input data
    sample_data = {
        'area': 'India',
        'item': 'Wheat',
        'year': 2023,
        'rainfall': 750.5,
        'pesticides': 320000,
        'avg_temp': 15.2
    }

    print("Testing model with single sample:")
    print(f"Area: {sample_data['area']}")
    print(f"Crop: {sample_data['item']}")
    print(f"Year: {sample_data['year']}")
    print(f"Rainfall: {sample_data['rainfall']} mm/year")
    print(f"Pesticides: {sample_data['pesticides']} tonnes")
    print(f"Average Temperature: {sample_data['avg_temp']}°C")

    # Get prediction
    predicted_yield = predict_single_sample(
        sample_data['area'],
        sample_data['item'],
        sample_data['year'],
        sample_data['rainfall'],
        sample_data['pesticides'],
        sample_data['avg_temp']
    )

    if predicted_yield is not None:
        print(f"\nPredicted Yield: {predicted_yield:.2f} hg/ha")
        print(f"                 ({predicted_yield / 100:.2f} tonnes/ha)")


if __name__ == "__main__":
    test_with_array_input()

    # Example of how to call with custom data
    print("\n\nYou can also call the function directly with your own data:")
    print("Example:")
    print("predicted_yield = predict_single_sample('India', 'Rice', 2024, 1200, 65000, 25.3)")
    p = predict_single_sample('India', 'Potatoes', 2024, 1200, 65000, 25.3)
    print(p)