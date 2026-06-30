"""
main.py
Entry point for the Car Price Prediction project.
Orchestrates preprocessing, training, evaluation, and visualization.
"""

import sys

import config
from data_preprocessing import DataPreprocessor
from model import ModelTrainer
from evaluation import ModelEvaluator
from visualization import Visualizer
from utils import get_logger, print_section

logger = get_logger(__name__)


def main():
    try:
        print_section("STEP 1: DATA PREPROCESSING")
        preprocessor = DataPreprocessor()
        df, X_train, X_test, y_train, y_test = preprocessor.run_pipeline()
        print(df.describe())

        print_section("STEP 2: EXPLORATORY DATA ANALYSIS")
        viz = Visualizer()
        viz.plot_price_distribution(df)
        viz.plot_correlation_heatmap(df)
        viz.plot_brand_price(df)

        print_section("STEP 3: MODEL TRAINING")
        trainer = ModelTrainer()
        trained_models = trainer.train_all(X_train, y_train)

        print_section("STEP 4: MODEL EVALUATION")
        evaluator = ModelEvaluator()
        results = []
        feature_names = preprocessor.preprocessor.get_feature_names_out()
        for name, model in trained_models.items():
            result = evaluator.evaluate(model, name, X_test, y_test)
            results.append(result)
            viz.plot_actual_vs_predicted(y_test, result["y_pred"], name)
            viz.plot_feature_importance(model, feature_names, name)

        comparison_df = evaluator.compare_models(results)
        print(comparison_df)
        viz.plot_model_comparison(comparison_df)

        print_section("STEP 5: SAVE BEST MODEL")
        best_name, best_model = trainer.get_best_model()
        trainer.save_model(best_model)
        trainer.save_preprocessor(preprocessor.preprocessor)

        print_section("PROJECT COMPLETE")
        logger.info("Best performing model: %s", best_name)
        logger.info("All results saved under: %s", config.RESULTS_DIR)

    except Exception as exc:
        logger.exception("Pipeline failed with error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
