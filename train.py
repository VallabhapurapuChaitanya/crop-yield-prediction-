import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Bidirectional, GRU, Input, Concatenate
from tensorflow.keras.losses import MeanSquaredError
import joblib
import os

# Create directories for saving models and plots
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# Function to load and preprocess the dataset
def load_and_preprocess_data(file_path):
    # Load the dataset
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # Display basic information about the dataset
    print("\nDataset Overview:")
    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}: {df[col].dtype}")

    print("\nSample Data:")
    print(df.head())

    print("\nChecking for missing values:")
    print(df.isnull().sum())

    # Encoding categorical variables
    label_encoders = {}
    for col in ['Area', 'Item']:
        label_encoders[col] = LabelEncoder()
        df[col + '_encoded'] = label_encoders[col].fit_transform(df[col])

    # Create feature and target variables
    X = df[['Area_encoded', 'Item_encoded', 'Year',
            'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']]
    y = df['hg/ha_yield']

    # Scale the features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Scale the target variable
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

    return X_scaled, y_scaled, scaler_X, scaler_y, label_encoders, df


# Function to create sequences for GRU
def create_sequences(X, y, time_steps=3):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)


# Function to build the Bi-GRU model
def build_bigru_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Bidirectional(GRU(64, return_sequences=True))(inputs)
    x = Bidirectional(GRU(32))(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)

    # Fix: Use MeanSquaredError() instead of 'mse' string
    model.compile(optimizer='adam', loss=MeanSquaredError())
    return model


# Function to build the hybrid model
def build_hybrid_model(bigru_model, X_train_seq, X_train_tab, y_train):
    # Get Bi-GRU predictions
    bigru_preds = bigru_model.predict(X_train_seq)

    # Combine Bi-GRU predictions with tabular features
    features_for_xgb = np.column_stack((bigru_preds, X_train_tab))

    # Train XGBoost on the combined features
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(features_for_xgb, y_train)

    return xgb_model


# Function to evaluate models and plot results
def evaluate_and_plot(model_name, y_true, y_pred, scaler_y=None):
    if scaler_y is not None:
        y_true = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} Evaluation Metrics:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Plot actual vs predicted
    axs[0, 0].scatter(y_true, y_pred, alpha=0.5)
    axs[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    axs[0, 0].set_xlabel('Actual Yield')
    axs[0, 0].set_ylabel('Predicted Yield')
    axs[0, 0].set_title('Actual vs Predicted Yield')

    # Plot residuals
    residuals = y_true - y_pred
    axs[0, 1].scatter(y_pred, residuals, alpha=0.5)
    axs[0, 1].axhline(y=0, color='r', linestyle='--')
    axs[0, 1].set_xlabel('Predicted Yield')
    axs[0, 1].set_ylabel('Residuals')
    axs[0, 1].set_title('Residual Plot (Bias Analysis)')

    # Plot distribution of residuals
    sns.histplot(residuals, kde=True, ax=axs[1, 0])
    axs[1, 0].set_xlabel('Residuals')
    axs[1, 0].set_title('Distribution of Residuals')

    # Plot prediction error distribution
    error_percentage = abs(residuals / y_true) * 100
    sns.histplot(error_percentage, kde=True, ax=axs[1, 1])
    axs[1, 1].set_xlabel('Percentage Error')
    axs[1, 1].set_title('Prediction Error Distribution')

    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_evaluation.png')
    plt.close()

    # Return metrics dictionary
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }


def main():
    # Load and preprocess the dataset
    X_scaled, y_scaled, scaler_X, scaler_y, label_encoders, df = load_and_preprocess_data('dataset.csv')

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Create sequences for Bi-GRU
    time_steps = 3
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)

    # Save the non-sequence versions for XGBoost
    X_train_tab = X_train[time_steps:]
    X_test_tab = X_test[time_steps:]

    # Build and train the Bi-GRU model
    print("\nTraining Bi-GRU model...")
    bigru_model = build_bigru_model((time_steps, X_train.shape[1]))
    bigru_history = bigru_model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32,
                                    validation_split=0.2, verbose=1)

    # Save the Bi-GRU model
    bigru_model.save('models/bigru_model.h5')

    # Plot Bi-GRU training history
    plt.figure(figsize=(10, 6))
    plt.plot(bigru_history.history['loss'], label='Training Loss')
    plt.plot(bigru_history.history['val_loss'], label='Validation Loss')
    plt.title('Bi-GRU Model Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plots/bigru_training_history.png')
    plt.close()

    # Evaluate Bi-GRU model
    y_pred_bigru = bigru_model.predict(X_test_seq)
    bigru_metrics = evaluate_and_plot('Bi-GRU', y_test_seq, y_pred_bigru, scaler_y)

    # Train XGBoost model separately
    print("\nTraining standalone XGBoost model...")
    xgb_standalone = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_standalone.fit(X_train_tab, y_train[time_steps:])

    # Save the standalone XGBoost model
    joblib.dump(xgb_standalone, 'models/xgb_standalone.joblib')

    # Evaluate standalone XGBoost model
    y_pred_xgb = xgb_standalone.predict(X_test_tab)
    xgb_metrics = evaluate_and_plot('XGBoost', y_test[time_steps:], y_pred_xgb, scaler_y)

    # Plot feature importance for XGBoost
    plt.figure(figsize=(10, 6))
    xgb_feature_cols = ['Area_encoded', 'Item_encoded', 'Year',
                        'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
    importances = xgb_standalone.feature_importances_
    indices = np.argsort(importances)
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [xgb_feature_cols[i] for i in indices])
    plt.title('XGBoost Feature Importance')
    plt.savefig('plots/xgb_feature_importance.png')
    plt.close()

    # Build the hybrid model (Bi-GRU + XGBoost)
    print("\nTraining hybrid model (Bi-GRU + XGBoost)...")
    hybrid_xgb = build_hybrid_model(bigru_model, X_train_seq, X_train_tab, y_train_seq)

    # Save the XGBoost part of the hybrid model
    joblib.dump(hybrid_xgb, 'models/hybrid_xgb_model.joblib')

    # Save scalers and encoders for inference
    joblib.dump(scaler_X, 'models/scaler_X.joblib')
    joblib.dump(scaler_y, 'models/scaler_y.joblib')
    for col, encoder in label_encoders.items():
        joblib.dump(encoder, f'models/encoder_{col}.joblib')

    # Evaluate the hybrid model
    bigru_preds_test = bigru_model.predict(X_test_seq)
    hybrid_features_test = np.column_stack((bigru_preds_test, X_test_tab))
    y_pred_hybrid = hybrid_xgb.predict(hybrid_features_test)
    hybrid_metrics = evaluate_and_plot('Hybrid', y_test_seq, y_pred_hybrid, scaler_y)
    plot_hybrid_model_accuracy(y_test_seq, y_pred_hybrid, scaler_y)

    # Compare models
    models = ['Bi-GRU', 'XGBoost', 'Hybrid']
    metrics = [bigru_metrics, xgb_metrics, hybrid_metrics]

    # Plot comparison of models
    plt.figure(figsize=(12, 8))

    # R² comparison
    plt.subplot(3, 1, 1)
    plt.bar(models, [m['r2'] for m in metrics])
    plt.title('R² Score Comparison')
    plt.ylim(0, 1)

    # RMSE comparison
    plt.subplot(3, 1, 2)
    plt.bar(models, [m['rmse'] for m in metrics])
    plt.title('RMSE Comparison')

    # MAE comparison
    plt.subplot(3, 1, 3)
    plt.bar(models, [m['mae'] for m in metrics])
    plt.title('MAE Comparison')

    plt.tight_layout()
    plt.savefig('plots/model_comparison.png')
    plt.close()

    print("\nAll models have been trained, evaluated, and saved.")
    print("Check the 'models' directory for saved models and 'plots' directory for evaluation plots.")


def plot_hybrid_model_accuracy(y_true, y_pred, scaler_y=None):
    """
    Generate and save an accuracy graph specifically for the hybrid model.

    Parameters:
    -----------
    y_true : array-like
        The actual target values
    y_pred : array-like
        The predicted target values from the hybrid model
    scaler_y : StandardScaler, optional
        Scaler used to inverse transform the target values
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import os

    os.makedirs('plots', exist_ok=True)

    # Inverse transform if scaler is provided
    if scaler_y is not None:
        y_true = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Calculate accuracy metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Calculate percentage accuracy (using R² as indicator)
    accuracy_percentage = r2 * 100 if r2 > 0 else 0

    # Calculate percent error for each prediction
    percent_error = np.abs((y_true - y_pred) / y_true) * 100
    mean_percent_error = np.mean(percent_error)

    # Create figure with 2 rows, 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Hybrid Model (Bi-GRU + XGBoost) Accuracy Analysis', fontsize=16)

    # 1. Actual vs Predicted scatter plot with perfect prediction line
    axs[0, 0].scatter(y_true, y_pred, alpha=0.5, color='blue')
    # Add diagonal line representing perfect predictions
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axs[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
    axs[0, 0].set_xlabel('Actual Yield (hg/ha)')
    axs[0, 0].set_ylabel('Predicted Yield (hg/ha)')
    axs[0, 0].set_title(f'Actual vs Predicted Yield (R² = {r2:.4f})')
    axs[0, 0].grid(True, alpha=0.3)

    # Add text box with metrics
    metrics_text = f'R² Score: {r2:.4f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nAccuracy: {accuracy_percentage:.2f}%'
    axs[0, 0].text(0.05, 0.95, metrics_text, transform=axs[0, 0].transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2. Prediction error distribution
    sns.histplot(percent_error, kde=True, ax=axs[0, 1], color='green')
    axs[0, 1].axvline(mean_percent_error, color='red', linestyle='--',
                      label=f'Mean Error: {mean_percent_error:.2f}%')
    axs[0, 1].set_xlabel('Percentage Error (%)')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].set_title('Distribution of Prediction Error Percentage')
    axs[0, 1].legend()

    # 3. Prediction accuracy over different yield ranges
    # Create yield range bins
    bins = np.linspace(y_true.min(), y_true.max(), 10)
    bin_indices = np.digitize(y_true, bins)

    # Calculate average accuracy (1 - error) for each bin
    bin_accuracies = []
    bin_centers = []

    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.sum(mask) > 0:  # Only calculate if there are points in the bin
            bin_error = np.mean(percent_error[mask])
            bin_accuracy = 100 - bin_error
            bin_accuracies.append(bin_accuracy)
            bin_center = (bins[i - 1] + bins[i]) / 2
            bin_centers.append(bin_center)

    # Plot the average accuracy for each yield range
    axs[1, 0].bar(bin_centers, bin_accuracies, width=(bins[1] - bins[0]) * 0.8, alpha=0.7, color='purple')
    axs[1, 0].set_xlabel('Yield Range (hg/ha)')
    axs[1, 0].set_ylabel('Average Accuracy (%)')
    axs[1, 0].set_title('Prediction Accuracy Across Different Yield Ranges')
    axs[1, 0].set_ylim(0, 100)
    axs[1, 0].grid(True, alpha=0.3)

    # 4. Time series plot of actual vs predicted (assuming sequential order)
    indices = np.arange(len(y_true))
    axs[1, 1].plot(indices, y_true, 'b-', label='Actual Yield')
    axs[1, 1].plot(indices, y_pred, 'r--', label='Predicted Yield')
    axs[1, 1].fill_between(indices, y_true, y_pred, alpha=0.3, color='gray', label='Difference')
    axs[1, 1].set_xlabel('Sample Index')
    axs[1, 1].set_ylabel('Yield (hg/ha)')
    axs[1, 1].set_title('Actual vs Predicted Yield (Sequential View)')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.savefig('plots/hybrid_model_accuracy_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nHybrid Model Accuracy Analysis:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"Mean Percentage Error: {mean_percent_error:.2f}%")
    print(f"Model Accuracy: {accuracy_percentage:.2f}%")
    print(f"Accuracy graph saved to plots/hybrid_model_accuracy_analysis.png")

    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'accuracy': accuracy_percentage,
        'mean_percent_error': mean_percent_error
    }


# Function to load models and predict on new data
def predict_crop_yield(data_file):
    # Load the models and preprocessing objects
    print("\nLoading models and preprocessing objects...")
    bigru_model = load_model('models/bigru_model.h5')
    hybrid_xgb = joblib.load('models/hybrid_xgb_model.joblib')
    scaler_X = joblib.load('models/scaler_X.joblib')
    scaler_y = joblib.load('models/scaler_y.joblib')

    # Load encoders
    label_encoders = {}
    for col in ['Area', 'Item']:
        label_encoders[col] = joblib.load(f'models/encoder_{col}.joblib')

    # Load and preprocess the test data
    print(f"\nProcessing test data from {data_file}...")
    test_df = pd.read_csv(data_file)

    # Display test data
    print("\nTest Data:")
    print(test_df.head())

    # Encode categorical variables
    for col in ['Area', 'Item']:
        test_df[col + '_encoded'] = label_encoders[col].transform(test_df[col])

    # Create features
    X_test = test_df[['Area_encoded', 'Item_encoded', 'Year',
                      'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']]

    # Scale features
    X_test_scaled = scaler_X.transform(X_test)

    # Create sequences for Bi-GRU
    time_steps = 3
    if len(X_test_scaled) >= time_steps:
        # Use sequences if we have enough data
        X_test_seq = []
        for i in range(len(X_test_scaled) - time_steps + 1):
            X_test_seq.append(X_test_scaled[i:i + time_steps])
        X_test_seq = np.array(X_test_seq)

        # Get Bi-GRU predictions
        bigru_preds = bigru_model.predict(X_test_seq)

        # Prepare data for hybrid model
        X_test_tab = X_test_scaled[time_steps - 1:]
        hybrid_features = np.column_stack((bigru_preds, X_test_tab))

        # Get hybrid model predictions
        y_pred_hybrid = hybrid_xgb.predict(hybrid_features)

        # Inverse transform predictions
        predictions = scaler_y.inverse_transform(y_pred_hybrid.reshape(-1, 1)).flatten()

        # Add predictions to dataframe
        result_df = test_df.iloc[time_steps - 1:].copy()
        result_df['predicted_yield'] = predictions
    else:
        print("Warning: Test data has fewer samples than required time steps for sequence modeling.")
        print("Using only XGBoost for prediction.")

        # Load standalone XGBoost model
        xgb_standalone = joblib.load('models/xgb_standalone.joblib')

        # Get XGBoost predictions
        y_pred_xgb = xgb_standalone.predict(X_test_scaled)

        # Inverse transform predictions
        predictions = scaler_y.inverse_transform(y_pred_xgb.reshape(-1, 1)).flatten()

        # Add predictions to dataframe
        result_df = test_df.copy()
        result_df['predicted_yield'] = predictions

    print("\nPrediction Results:")
    print(result_df[['Area', 'Item', 'Year', 'predicted_yield']])

    # Save predictions to CSV
    result_df.to_csv('crop_yield_predictions.csv', index=False)
    print("\nPredictions saved to 'crop_yield_predictions.csv'")

    return result_df


if __name__ == "__main__":
    # Train the models
    main()

    # Example of using the model for real-time prediction
    # Comment out the line below if you want to skip prediction for now
    # predict_crop_yield('test_data.csv')