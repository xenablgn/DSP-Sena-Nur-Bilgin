from typing import Dict

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_log_error, mean_squared_error,
    mean_absolute_error, r2_score
)

from house_prices.preprocess import preprocess_train_data
from house_prices import MODEL_PATH


def evaluate_model(y_true, y_pred) -> Dict[str, str]:
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        'RMSLE': str(round(rmsle, 3)),
        'MSE': str(mse),
        'RMSE': str(rmse),
        'MAE': str(mae),
        'R2-score': str(round(r2, 2))
    }


def build_model(data: pd.DataFrame) -> Dict[str, str]:
    X_train_process, y_train_process_lg, X_test_process, y_test_process_lg = \
        preprocess_train_data(data)
    model = LinearRegression()
    model.fit(X_train_process, y_train_process_lg)
    y_test_pred_log = model.predict(X_test_process)
    y_test_pred = np.exp(y_test_pred_log)
    y_test_true = np.exp(y_test_process_lg)
    evaluation_results = evaluate_model(y_test_true, y_test_pred)
    joblib.dump(model, MODEL_PATH)

    return evaluation_results
