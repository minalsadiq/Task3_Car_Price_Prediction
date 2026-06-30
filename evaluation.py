"""
evaluation.py
Regression model evaluation utilities: MAE, RMSE, R^2.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config
from utils import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluates regression models and stores reports."""

    def __init__(self, results_dir: str = config.RESULTS_DIR):
        self.results_dir = results_dir

    def evaluate(self, model, model_name: str, X_test, y_test) -> dict:
        """
        Evaluate a single trained regression model on the held-out test set.

        Returns
        -------
        dict
            Dictionary containing MAE, RMSE, R^2 and predictions.
        """
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        logger.info(
            "[%s] MAE=%.3f, RMSE=%.3f, R^2=%.4f", model_name, mae, rmse, r2
        )

        report_path = os.path.join(
            self.results_dir, f"{model_name.replace(' ', '_').lower()}_report.txt"
        )
        with open(report_path, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"MAE  : {mae:.4f}\n")
            f.write(f"RMSE : {rmse:.4f}\n")
            f.write(f"R^2  : {r2:.4f}\n")
        logger.info("Evaluation report saved to %s", report_path)

        return {
            "model_name": model_name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "y_pred": y_pred,
            "y_test": y_test,
        }

    def compare_models(self, results: list) -> pd.DataFrame:
        """Build a comparison table of all evaluated models."""
        comparison = pd.DataFrame(
            [
                {
                    "Model": r["model_name"],
                    "MAE": r["mae"],
                    "RMSE": r["rmse"],
                    "R2": r["r2"],
                }
                for r in results
            ]
        ).sort_values("R2", ascending=False).reset_index(drop=True)

        comparison_path = os.path.join(self.results_dir, "model_comparison.csv")
        comparison.to_csv(comparison_path, index=False)
        logger.info("Model comparison table saved to %s", comparison_path)
        return comparison
