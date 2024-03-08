# inference.py
import os
import joblib
import numpy as np
import pandas as pd

def load_model_and_transformers(models_folder):
    """Load model and transformers."""
    model_path = os.path.join(models_folder, 'model.joblib')
    numeric_imputer_path = os.path.join(models_folder, 'numeric_imputer.joblib')
    categorical_imputer_path = os.path.join(models_folder, 'categorical_imputer.joblib')
    encoder_path = os.path.join(models_folder, 'ordinal_encoder.joblib')
    scaler_path = os.path.join(models_folder, 'scaler.joblib')

    regressor = joblib.load(model_path)
    numeric_imputer = joblib.load(numeric_imputer_path)
    categorical_imputer = joblib.load(categorical_imputer_path)
    ordinal_encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)

    return regressor, numeric_imputer, categorical_imputer, ordinal_encoder, scaler

def preprocess_new_data(new_data, ordinal_features, numeric_features, numeric_imputer, categorical_imputer, ordinal_encoder, scaler):
    """Preprocess new data."""
    new_data[ordinal_features] = categorical_imputer.transform(new_data[ordinal_features])
    new_data[numeric_features] = numeric_imputer.transform(new_data[numeric_features])
    new_data[ordinal_features] = ordinal_encoder.transform(new_data[ordinal_features])
    new_data[numeric_features] = scaler.transform(new_data[numeric_features])
    return new_data

def make_prediction(data_test, models_folder, selected_columns, ordinal_features, numeric_features):
    """Make predictions."""
    regressor, numeric_imputer, categorical_imputer, ordinal_encoder, scaler = load_model_and_transformers(models_folder)
    X_test = data_test[selected_columns].copy()
    X_test_transformed = preprocess_new_data(X_test, ordinal_features, numeric_features, 
                                             numeric_imputer, categorical_imputer, ordinal_encoder, scaler)
    
    predictions = regressor.predict(X_test_transformed)
    return np.exp(predictions)
