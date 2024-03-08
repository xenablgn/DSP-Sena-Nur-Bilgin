# train.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error, r2_score
from preprocess import filter_outliers, train_test_split_data, filter_outliers, transform_data,save_model_and_transformers,preprocess_features,engineer_features

# Define or import numeric_features and ordinal_features here

def evaluate_model(y_true, y_pred):
    """Evaluate model."""
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        'RMSLE': round(rmsle, 3),
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2-score': round(r2, 2)
    }

def build_model(data: pd.DataFrame, target_variable: str,models_folder:str,ordinal_features:list,numeric_features:list):
    """Build model."""
    X_train, X_valid, y_train, y_valid = train_test_split_data(data, target_variable)
    X_train, y_train = filter_outliers(X_train, y_train, numeric_features)
    X_valid, y_valid = filter_outliers(X_valid, y_valid, numeric_features)
    transformed_X_train, transformed_X_valid = transform_data(X_train.copy(), X_valid.copy(), ordinal_features, numeric_features)
    y_train_log, y_valid_log = np.log(y_train), np.log(y_valid)
    model = LinearRegression()
    model.fit(transformed_X_train, y_train_log)
    y_valid_pred_log = model.predict(transformed_X_valid)
    y_valid_pred = np.exp(y_valid_pred_log)
    evaluation_results = evaluate_model(y_valid, y_valid_pred)
    save_model_and_transformers(model, preprocess_features(data, numeric_features),  # Pass data instead of data[numeric_features]
                                preprocess_features(data, ordinal_features, strategy='most_frequent'),  # Pass data and strategy
                                *engineer_features(data, ordinal_features, numeric_features), 
                                models_folder)
    return evaluation_results
