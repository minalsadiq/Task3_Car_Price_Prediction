"""
config.py
Configuration constants for the Car Price Prediction project.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

DATASET_PATH = os.path.join(DATASET_DIR, "car_data.csv")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COLUMN = "selling_price"

NUMERIC_FEATURES = [
    "year",
    "present_price",
    "kms_driven",
    "horsepower",
    "mileage",
    "engine_cc",
    "owner_count",
]
CATEGORICAL_FEATURES = [
    "brand",
    "fuel_type",
    "seller_type",
    "transmission",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

for directory in (DATASET_DIR, RESULTS_DIR, MODELS_DIR):
    os.makedirs(directory, exist_ok=True)
