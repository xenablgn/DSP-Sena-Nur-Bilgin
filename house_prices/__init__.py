NUM_FEAT = [
    'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'TotalBsmtSF',
    '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd',
    'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea'
]
ORD_FEAT = ["OverallQual", "KitchenQual", "GarageFinish"]
SELEC_COLUMNS = NUM_FEAT + ORD_FEAT
TARGET = 'SalePrice'

MODEL_PATH = '../Models/model.joblib'
NUMERIC_IMPUTER_PATH = '../Models/numeric_imputer.joblib'
CAT_IMPUTER_PATH = '../Models/cat_imputer.joblib'
ORDINAL_ENCODER_PATH = '../Models/ordinal_encoder.joblib'
SCALER_PATH = '../Models/scaler.joblib'
