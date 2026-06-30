"""
model.py
Regression model training logic for the Car Price Prediction project.
Trains and compares Linear Regression, Random Forest Regressor, and
XGBoost (falling back to GradientBoostingRegressor if XGBoost is not
installed).
"""

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

import config
from utils import get_logger

logger = get_logger(__name__)

try:
    from xgboost import XGBRegressor

    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
    logger.warning(
        "xgboost not installed; using GradientBoostingRegressor as a fallback."
    )


class ModelTrainer:
    """Trains, compares, and persists regression models."""

    def __init__(self):
        if _XGBOOST_AVAILABLE:
            boosted_model = XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                random_state=config.RANDOM_STATE,
                verbosity=0,
            )
            boosted_name = "XGBoost"
        else:
            boosted_model = GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=4,
                random_state=config.RANDOM_STATE,
            )
            boosted_name = "GradientBoosting (XGBoost fallback)"

        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=300, random_state=config.RANDOM_STATE
            ),
            boosted_name: boosted_model,
        }
        self.trained_models = {}
        self.cv_scores = {}

    def train_all(self, X_train, y_train):
        """Fit every candidate model and record 5-fold CV R^2."""
        for name, model in self.models.items():
            logger.info("Training %s ...", name)
            model.fit(X_train, y_train)
            self.trained_models[name] = model

            scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
            self.cv_scores[name] = scores.mean()
            logger.info("%s -> CV R^2: %.4f", name, scores.mean())
        return self.trained_models

    def get_best_model(self):
        """Select the model with the highest cross-validated R^2."""
        best_name = max(self.cv_scores, key=self.cv_scores.get)
        best_model = self.trained_models[best_name]
        logger.info(
            "Best model selected: %s (CV R^2 = %.4f)",
            best_name,
            self.cv_scores[best_name],
        )
        return best_name, best_model

    @staticmethod
    def save_model(model, path: str = config.BEST_MODEL_PATH):
        joblib.dump(model, path)
        logger.info("Model saved to %s", path)

    @staticmethod
    def save_preprocessor(preprocessor, path: str = config.PREPROCESSOR_PATH):
        joblib.dump(preprocessor, path)
        logger.info("Preprocessor saved to %s", path)
