# 🚗 Car Price Prediction using Machine Learning

## Overview

Car Price Prediction is a Machine Learning regression project developed to estimate vehicle prices based on multiple input features.

The purpose of this project is to build predictive models capable of learning pricing patterns from historical vehicle data and generating accurate price predictions.

This project follows a complete machine learning pipeline including:

- Data Collection
- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training
- Model Evaluation
- Performance Comparison
- Visualization

The project also compares multiple regression algorithms to determine which model performs better for price estimation.

---

# 📌 Problem Statement

Car prices are influenced by multiple factors such as brand, model specifications, engine characteristics, and vehicle attributes.

Estimating prices manually can be difficult and inconsistent.

This project aims to automate price prediction using Machine Learning and identify the most influential features affecting vehicle prices.

---

# 🎯 Objectives

The objectives of this project are:

✔ Understand vehicle pricing patterns  
✔ Prepare and clean the dataset  
✔ Build regression models  
✔ Predict car prices accurately  
✔ Compare model performance  
✔ Visualize prediction results  
✔ Extract business insights from data  

---

# 📂 Project Structure

```plaintext
Task3_Car_Price_Prediction/

│
├── datasets/
│   └── car_data.csv
│
├── models/
│
├── results/
│   ├── actual_vs_predicted_linear_regression.png
│   ├── actual_vs_predicted_random_forest.png
│   ├── actual_vs_predicted_xgboost.png
│   ├── correlation_heatmap.png
│   ├── feature_importance_random_forest.png
│   ├── feature_importance_xgboost.png
│   ├── linear_regression_report.txt
│   ├── random_forest_regressor_report.txt
│   ├── xgboost_report.txt
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── price_by_brand.png
│   └── price_distribution.png
│
├── config.py
├── data_preprocessing.py
├── evaluation.py
├── main.py
├── model.py
├── visualization.py
├── utils.py
├── requirements.txt
└── README.md
```

---

# 📊 Dataset Description

The dataset contains vehicle-related information used to predict selling prices.

Example attributes include:

| Feature | Description |
|----------|-------------|
| Brand | Vehicle manufacturer |
| Year | Manufacturing year |
| Fuel Type | Type of fuel |
| Transmission | Transmission system |
| Mileage | Distance covered |
| Engine | Engine characteristics |
| Power | Vehicle power |
| Seats | Number of seats |

### Target Variable

Car Price

---

# 🧹 Data Preprocessing

Before training the models, preprocessing was performed.

Steps included:

### 1. Data Loading
Imported dataset into Python.

### 2. Data Cleaning
Handled missing values and removed inconsistencies.

### 3. Feature Selection
Selected relevant variables for prediction.

### 4. Encoding
Converted categorical features into numerical values.

### 5. Dataset Splitting
Separated training and testing datasets.

### 6. Scaling (if required)
Prepared data for better model learning.

---

# 📈 Exploratory Data Analysis (EDA)

Data visualization was performed to understand patterns.

Generated outputs:

### Price Distribution
Shows overall spread of car prices.

### Price by Brand
Compares pricing across brands.

### Correlation Heatmap
Measures relationships among variables.

### Actual vs Predicted Graphs
Visual comparison of model predictions.

### Feature Importance Graphs
Shows which variables influence predictions most.

---

# 🤖 Machine Learning Models

Three regression models were implemented and evaluated.

---

## 1. Linear Regression

Baseline regression model.

Features:
- Fast
- Simple
- Interpretable

---

## 2. Random Forest Regressor

Tree-based ensemble model.

Features:
- Handles nonlinear relationships
- Strong predictive capability
- Reduced overfitting

---

## 3. XGBoost Regressor

Gradient boosting model.

Features:
- High prediction performance
- Efficient learning
- Advanced optimization

---

# 📉 Model Evaluation

Model performance was measured using regression evaluation metrics.

Evaluation Metrics:

- R² Score
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Prediction Accuracy Analysis

Generated Results:

```plaintext
Linear Regression Report

Random Forest Report

XGBoost Report

Actual vs Predicted Graphs

Model Comparison Graph
```

---

# 📊 Results & Findings

Key findings from the project:

- Multiple regression models were successfully trained.
- Model performance was compared visually.
- Feature importance analysis identified major price drivers.
- Actual vs predicted analysis helped evaluate prediction quality.
- Model comparison provided understanding of algorithm performance.

The final model demonstrated strong capability for predicting vehicle prices based on historical data.

---

# 📷 Generated Outputs

✔ Price Distribution Graph  
✔ Brand-wise Price Analysis  
✔ Correlation Heatmap  
✔ Feature Importance Visualizations  
✔ Actual vs Predicted Results  
✔ Model Comparison Dashboard  
✔ Performance Reports  

<img width="1536" height="1024" alt="Dashboard" src="https://github.com/user-attachments/assets/e4947027-258f-45eb-873f-684ca4b963b0" />

---

# 🛠 Technologies Used

| Category | Tools |
|----------|-------|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost |
| Development | VS Code |

---

# 🚀 Installation

Clone repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

Move into project:

```bash
cd Task3_Car_Price_Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run project:

```bash
python main.py
```

---

# 📚 Learning Outcomes

This project improved my understanding of:

- Regression Modeling
- Predictive Analytics
- Data Cleaning
- Feature Engineering
- Model Evaluation
- Data Visualization
- Performance Comparison

---

# 🔮 Future Improvements

Possible future enhancements:

- Hyperparameter Tuning
- More Advanced Models
- Real-Time Price Prediction
- Web Deployment
- Interactive Dashboard

---

# 👩‍💻 Author

**Minal**  **Sadiq**
Data Science Student  
Machine Learning Enthusiast

---

⭐ If you found this project useful, consider giving it a star.
