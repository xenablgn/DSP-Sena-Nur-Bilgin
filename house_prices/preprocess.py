# preprocess.py
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def train_test_split_data(data: pd.DataFrame, selected_columns: list, target_variable: str, test_size: float = 0.25, random_state: int = 42):
    """Split the dataset into train and test sets."""
    X = data[selected_columns]
    y = data[target_variable]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_valid, y_train, y_valid

def filter_outliers(X, y, numeric_features, z_score_threshold=3):
    """Filter outliers."""
    z_scores = X[numeric_features].apply(lambda x: np.abs((x - x.mean()) / x.std()))
    outliers = z_scores.max(axis=1) > z_score_threshold
    return X[~outliers], y[~outliers]

def preprocess_features(data, features, strategy='mean'):
    """Preprocess features."""
    imputer = SimpleImputer(strategy=strategy)
    imputer.fit(data[features])
    return imputer

def engineer_features(data, ordinal_features, numeric_features):
    """Engineer features."""
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit(data[ordinal_features])

    scaler = StandardScaler()
    scaler.fit(data[numeric_features])

    return ordinal_encoder, scaler

def transform_data(data_train, data_valid, ordinal_features, numeric_features):
    """Transform data."""
    categorical_imputer = preprocess_features(data_train, ordinal_features, strategy='most_frequent')
    data_train[ordinal_features] = categorical_imputer.transform(data_train[ordinal_features])
    data_valid[ordinal_features] = categorical_imputer.transform(data_valid[ordinal_features])
    numeric_imputer = preprocess_features(data_train, numeric_features)
    data_train[numeric_features] = numeric_imputer.transform(data_train[numeric_features])
    data_valid[numeric_features] = numeric_imputer.transform(data_valid[numeric_features])
    ordinal_encoder, scaler = engineer_features(data_train, ordinal_features, numeric_features)
    data_train[ordinal_features] = ordinal_encoder.transform(data_train[ordinal_features])
    data_valid[ordinal_features] = ordinal_encoder.transform(data_valid[ordinal_features])
    data_train[numeric_features] = scaler.transform(data_train[numeric_features])
    data_valid[numeric_features] = scaler.transform(data_valid[numeric_features])
    return data_train, data_valid

def save_model_and_transformers(regressor, numeric_imputer, categorical_imputer, ordinal_encoder, scaler, models_folder):
    """Save model and transformers."""
    model_path = os.path.join(models_folder, 'model.joblib')
    numeric_imputer_path = os.path.join(models_folder, 'numeric_imputer.joblib')
    categorical_imputer_path = os.path.join(models_folder, 'categorical_imputer.joblib')
    encoder_path = os.path.join(models_folder, 'ordinal_encoder.joblib')
    scaler_path = os.path.join(models_folder, 'scaler.joblib')

    joblib.dump(regressor, model_path)
    joblib.dump(numeric_imputer, numeric_imputer_path)
    joblib.dump(categorical_imputer, categorical_imputer_path)
    joblib.dump(ordinal_encoder, encoder_path)
    joblib.dump(scaler, scaler_path)
