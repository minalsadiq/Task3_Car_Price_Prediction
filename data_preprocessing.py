"""
data_preprocessing.py
Handles dataset acquisition, cleaning, feature engineering, and
preprocessing for the Car Price Prediction project.
"""

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import config
from utils import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Encapsulates all data loading, cleaning, and preprocessing logic."""

    def __init__(self, dataset_path: str = config.DATASET_PATH):
        self.dataset_path = dataset_path
        self.preprocessor = self._build_preprocessor()

    # ------------------------------------------------------------------
    # Dataset acquisition
    # ------------------------------------------------------------------
    def load_or_create_dataset(self) -> pd.DataFrame:
        """
        Load the car price dataset from disk, generating a realistic
        synthetic dataset on first run if no local copy exists.
        """
        if os.path.exists(self.dataset_path):
            logger.info("Loading existing dataset from %s", self.dataset_path)
            df = pd.read_csv(self.dataset_path)
        else:
            logger.info("Dataset not found locally. Generating synthetic dataset.")
            df = self._generate_synthetic_dataset()
            df.to_csv(self.dataset_path, index=False)
            logger.info("Dataset saved to %s", self.dataset_path)
        return df

    @staticmethod
    def _generate_synthetic_dataset(n_samples: int = 1500) -> pd.DataFrame:
        """Generate a realistic synthetic car-price dataset."""
        rng = np.random.default_rng(config.RANDOM_STATE)

        brands = ["Toyota", "Honda", "Ford", "BMW", "Hyundai", "Suzuki", "Tata", "Audi"]
        brand_premium = {
            "Toyota": 1.0, "Honda": 0.95, "Ford": 0.9, "BMW": 1.8,
            "Hyundai": 0.85, "Suzuki": 0.7, "Tata": 0.65, "Audi": 1.9,
        }
        fuel_types = ["Petrol", "Diesel", "CNG", "Electric"]
        seller_types = ["Dealer", "Individual"]
        transmissions = ["Manual", "Automatic"]

        records = []
        for _ in range(n_samples):
            brand = rng.choice(brands)
            year = int(rng.integers(2005, 2024))
            age = 2024 - year
            horsepower = float(rng.uniform(60, 400))
            mileage = float(rng.uniform(8, 25))
            engine_cc = float(rng.uniform(800, 4000))
            kms_driven = float(max(rng.normal(age * 12000, 8000), 500))
            owner_count = int(rng.integers(1, 4))
            fuel_type = rng.choice(fuel_types)
            seller_type = rng.choice(seller_types)
            transmission = rng.choice(transmissions, p=[0.7, 0.3])

            base_price = (
                5000
                + horsepower * 120
                + engine_cc * 5
                - age * 900
                - kms_driven * 0.02
                - owner_count * 1500
            )
            base_price *= brand_premium[brand]
            if transmission == "Automatic":
                base_price *= 1.1
            if fuel_type == "Electric":
                base_price *= 1.25
            elif fuel_type == "Diesel":
                base_price *= 1.05

            base_price = max(base_price, 1000)
            noise = rng.normal(0, base_price * 0.07)
            selling_price = max(base_price + noise, 1000)
            present_price = selling_price * rng.uniform(1.05, 1.4)

            records.append(
                {
                    "brand": brand,
                    "year": year,
                    "present_price": round(present_price / 1000, 2),
                    "kms_driven": round(kms_driven, 0),
                    "horsepower": round(horsepower, 1),
                    "mileage": round(mileage, 2),
                    "engine_cc": round(engine_cc, 0),
                    "owner_count": owner_count,
                    "fuel_type": fuel_type,
                    "seller_type": seller_type,
                    "transmission": transmission,
                    config.TARGET_COLUMN: round(selling_price / 1000, 2),
                }
            )

        df = pd.DataFrame(records)
        # Inject a handful of missing values to simulate real-world messiness
        missing_idx = rng.choice(df.index, size=int(len(df) * 0.015), replace=False)
        df.loc[missing_idx, "mileage"] = np.nan
        return df

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicates, impute missing values, and remove outliers."""
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info("Removed %d duplicate rows", before - len(df))

        for col in config.NUMERIC_FEATURES:
            if df[col].isnull().sum() > 0:
                n_missing = df[col].isnull().sum()
                df[col] = df[col].fillna(df[col].median())
                logger.info("Imputed %d missing values in '%s' with median", n_missing, col)

        # Outlier removal via IQR on the target variable
        q1 = df[config.TARGET_COLUMN].quantile(0.25)
        q3 = df[config.TARGET_COLUMN].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        before_outlier = len(df)
        df = df[(df[config.TARGET_COLUMN] >= lower) & (df[config.TARGET_COLUMN] <= upper)]
        df = df.reset_index(drop=True)
        logger.info(
            "Removed %d outlier rows based on target IQR bounds [%.2f, %.2f]",
            before_outlier - len(df), lower, upper,
        )
        return df

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features such as car age."""
        df = df.copy()
        df["car_age"] = 2024 - df["year"]
        if "car_age" not in config.NUMERIC_FEATURES:
            config.NUMERIC_FEATURES.append("car_age")
            config.FEATURE_COLUMNS.append("car_age")
        return df

    # ------------------------------------------------------------------
    # Preprocessing pipeline (encoding + scaling)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_preprocessor() -> ColumnTransformer:
        """Build a ColumnTransformer that scales numeric and encodes categoricals."""
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        categorical_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, config.NUMERIC_FEATURES),
                ("cat", categorical_transformer, config.CATEGORICAL_FEATURES),
            ]
        )
        return preprocessor

    def split_and_transform(self, df: pd.DataFrame):
        """Split into train/test and fit-transform features via the preprocessor."""
        X = df[config.FEATURE_COLUMNS]
        y = df[config.TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )

        X_train_t = self.preprocessor.fit_transform(X_train)
        X_test_t = self.preprocessor.transform(X_test)

        logger.info("Split data: %d train rows, %d test rows", len(X_train), len(X_test))
        return X_train_t, X_test_t, y_train, y_test

    def run_pipeline(self):
        """Execute the full preprocessing pipeline end-to-end."""
        df = self.load_or_create_dataset()
        df = self.clean_data(df)
        df = self.engineer_features(df)
        self.preprocessor = self._build_preprocessor()
        X_train, X_test, y_train, y_test = self.split_and_transform(df)
        return df, X_train, X_test, y_train, y_test
