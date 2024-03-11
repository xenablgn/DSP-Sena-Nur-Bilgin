import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from typing import Tuple, Optional, List

from house_prices.__init__ import (
    ORD_FEAT, NUM_FEAT, SELEC_COLUMNS, TARGET,
    NUMERIC_IMPUTER_PATH, CAT_IMPUTER_PATH, ORDINAL_ENCODER_PATH, SCALER_PATH
)


def train_test_split_data(
        data: pd.DataFrame, test_size: float = 0.25,
        random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = data[SELEC_COLUMNS]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def filter_outliers(
        X: pd.DataFrame, y: Optional[pd.Series] = None,
        NUM_FEAT: Optional[List[str]] = None,
        z_score_threshold: float = 3
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    if NUM_FEAT is None:
        raise ValueError("Numeric features must be provided.")
    z_scores = X[NUM_FEAT].apply(
        lambda x: np.abs((x - x.mean()) / x.std())
    )
    outliers = z_scores.max(axis=1) > z_score_threshold
    return X[~outliers], y[~outliers] if y is not None else None


def fit_encoders(X: pd.DataFrame) -> Tuple[SimpleImputer, SimpleImputer,
                                           OrdinalEncoder, StandardScaler]:
    ordinal_imputer = SimpleImputer(strategy='most_frequent')
    ordinal_imputer.fit(X[ORD_FEAT])
    numeric_imputer = SimpleImputer(strategy='mean')
    numeric_imputer.fit(X[NUM_FEAT])

    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit(X[ORD_FEAT])

    scaler = StandardScaler()
    scaler.fit(X[NUM_FEAT])

    return ordinal_imputer, numeric_imputer, ordinal_encoder, scaler


def save_objects(*objects: Tuple) -> None:
    for obj, path in objects:
        joblib.dump(obj, path)


def load_objects(*paths: str) -> List:
    return [joblib.load(path) for path in paths]


def transform_data(X: pd.DataFrame, ordinal_imputer: SimpleImputer,
                   numeric_imputer: SimpleImputer,
                   ordinal_encoder: OrdinalEncoder,
                   scaler: StandardScaler, fit: bool = True
                   ) -> pd.DataFrame:
    if fit:
        X.loc[:, ORD_FEAT] = ordinal_imputer.transform(X.loc[:, ORD_FEAT])
        X.loc[:, ORD_FEAT] = ordinal_encoder.transform(X.loc[:, ORD_FEAT])
        X.loc[:, NUM_FEAT] = numeric_imputer.transform(X.loc[:, NUM_FEAT])

        numeric_transformed = scaler.transform(
            X.loc[:, NUM_FEAT])
        numeric_transformed = numeric_transformed.astype(int)
        X.loc[:, NUM_FEAT] = numeric_transformed

    return X


def preprocess_train_data(data: pd.DataFrame) -> Tuple[pd.DataFrame,
                                                       pd.DataFrame,
                                                       pd.Series,
                                                       pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split_data(data)

    X_train.drop_duplicates(inplace=False)
    y_train.drop_duplicates(inplace=False)

    X_train, y_train = filter_outliers(X_train, y_train, NUM_FEAT)
    X_test, y_test = filter_outliers(X_test, y_test, NUM_FEAT)

    encoders = fit_encoders(X_train)
    save_objects(*zip(encoders, [NUMERIC_IMPUTER_PATH,
                                 CAT_IMPUTER_PATH,
                                 ORDINAL_ENCODER_PATH,
                                 SCALER_PATH]))

    X_train = transform_data(X_train, *encoders)
    X_test = transform_data(X_test, *encoders)

    y_train_log = np.log(y_train)
    y_test_log = np.log(y_test)

    return X_train, y_train_log, X_test, y_test_log


def preprocess_test_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data[SELEC_COLUMNS]

    data.drop_duplicates(inplace=False)

    data, _ = filter_outliers(data, None, NUM_FEAT)

    encoders = load_objects(NUMERIC_IMPUTER_PATH, CAT_IMPUTER_PATH,
                            ORDINAL_ENCODER_PATH, SCALER_PATH)

    data = transform_data(data, *encoders)

    return data
