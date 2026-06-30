"""
visualization.py
Visualizations for the Car Price Prediction project. All plots are
saved automatically to the results/ directory.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config
from utils import get_logger

logger = get_logger(__name__)
sns.set_theme(style="whitegrid")


class Visualizer:
    """Generates and saves all charts for the project."""

    def __init__(self, results_dir: str = config.RESULTS_DIR):
        self.results_dir = results_dir

    def _save(self, fig, filename: str):
        path = os.path.join(self.results_dir, filename)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info("Saved figure: %s", path)

    def plot_price_distribution(self, df):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df[config.TARGET_COLUMN], kde=True, ax=ax, color="darkorange")
        ax.set_title("Selling Price Distribution")
        ax.set_xlabel("Selling Price (in thousands)")
        self._save(fig, "price_distribution.png")

    def plot_correlation_heatmap(self, df):
        fig, ax = plt.subplots(figsize=(8, 7))
        corr = df[config.NUMERIC_FEATURES + [config.TARGET_COLUMN]].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        self._save(fig, "correlation_heatmap.png")

    def plot_brand_price(self, df):
        fig, ax = plt.subplots(figsize=(10, 6))
        order = df.groupby("brand")[config.TARGET_COLUMN].median().sort_values(
            ascending=False
        ).index
        sns.boxplot(data=df, x="brand", y=config.TARGET_COLUMN, order=order, ax=ax)
        ax.set_title("Selling Price by Brand")
        ax.tick_params(axis="x", rotation=30)
        self._save(fig, "price_by_brand.png")

    def plot_actual_vs_predicted(self, y_test, y_pred, model_name: str):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(y_test, y_pred, alpha=0.5, color="teal")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction")
        ax.set_xlabel("Actual Selling Price")
        ax.set_ylabel("Predicted Selling Price")
        ax.set_title(f"Actual vs Predicted - {model_name}")
        ax.legend()
        filename = f"actual_vs_predicted_{model_name.replace(' ', '_').lower()}.png"
        self._save(fig, filename)

    def plot_model_comparison(self, comparison_df):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.barplot(data=comparison_df, x="Model", y="R2", ax=axes[0], palette="crest")
        axes[0].set_title("Model R^2 Comparison")
        axes[0].tick_params(axis="x", rotation=20)

        sns.barplot(data=comparison_df, x="Model", y="RMSE", ax=axes[1], palette="flare")
        axes[1].set_title("Model RMSE Comparison")
        axes[1].tick_params(axis="x", rotation=20)
        fig.tight_layout()
        self._save(fig, "model_comparison.png")

    def plot_feature_importance(self, model, feature_names, model_name: str):
        if not hasattr(model, "feature_importances_"):
            logger.info("%s has no feature_importances_ attribute; skipping.", model_name)
            return
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:15]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(
            x=importances[idx], y=np.array(feature_names)[idx], ax=ax, palette="magma"
        )
        ax.set_title(f"Top Feature Importances - {model_name}")
        self._save(fig, f"feature_importance_{model_name.replace(' ', '_').lower()}.png")
