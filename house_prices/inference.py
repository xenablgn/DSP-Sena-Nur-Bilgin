# inference.py

import joblib
import numpy as np
import pandas as pd

from house_prices.preprocess import preprocess_test_data
from house_prices.__init__ import MODEL_PATH


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    model = joblib.load(MODEL_PATH)
    X_test_processed = preprocess_test_data(input_data)
    y_test_pred_log = model.predict(X_test_processed)
    y_test_pred = np.exp(y_test_pred_log)

    return y_test_pred
