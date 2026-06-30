# Task 3 — Car Price Prediction with Machine Learning

## Overview
This project builds a regression pipeline to predict used car selling
prices from vehicle characteristics such as brand, age, horsepower,
mileage, engine size, ownership history, fuel type, seller type, and
transmission.

## Objective
Train, evaluate, and compare **Linear Regression**, **Random Forest
Regressor**, and **XGBoost** (with an automatic fallback to
`GradientBoostingRegressor` if XGBoost is not installed) to predict
selling price, and persist the best-performing model.

## Project Structure
```
Task3_Car_Price_Prediction/
├── main.py                 # Orchestrates the full pipeline
├── data_preprocessing.py   # Loading, cleaning, feature engineering, encoding
├── model.py                  # Model definitions, training, model selection
├── evaluation.py             # MAE, RMSE, R^2 evaluation
├── visualization.py          # EDA & evaluation plots
├── config.py                  # Paths and hyperparameters
├── utils.py                    # Logging helpers
├── requirements.txt
├── .gitignore
├── datasets/                  # Dataset + instructions
├── models/                     # Saved model + preprocessor (created at runtime)
└── results/                    # Generated plots and reports (created at runtime)
```

## How to Run
```bash
pip install -r requirements.txt
python main.py
```

## Methodology
1. **Data Preprocessing** — load (or synthetically generate) the car
   dataset, drop duplicates, impute missing numeric values with the
   median, and remove price outliers using the IQR method.
2. **Feature Engineering** — derive `car_age` from `year`; numeric
   features are standardized and categorical features
   (`brand`, `fuel_type`, `seller_type`, `transmission`) are one-hot
   encoded via a `ColumnTransformer`/`Pipeline`.
3. **Model Training** — Linear Regression, Random Forest Regressor,
   and XGBoost (or its fallback) are trained and evaluated with 5-fold
   cross-validation (R²).
4. **Evaluation** — MAE, RMSE, and R² are computed on a held-out test
   set for every model, with actual-vs-predicted scatter plots.
5. **Model Selection** — the model with the highest cross-validated R²
   is selected, saved to `models/best_model.joblib`, along with the
   fitted preprocessing pipeline (`models/preprocessor.joblib`).

## Results
All generated artifacts are saved automatically to `results/`:
- `price_distribution.png`, `correlation_heatmap.png`, `price_by_brand.png`
- `actual_vs_predicted_<model>.png` for each model
- `feature_importance_<model>.png` for tree-based models
- `model_comparison.png` / `model_comparison.csv`
- Per-model text reports (`*_report.txt`)

## Key Learnings
- Vehicle age, horsepower, and engine size are strong price drivers.
- Tree-based ensembles capture non-linear brand/feature interactions
  better than plain linear regression.
- Outlier handling materially improves regression stability on
  real-world pricing data.
