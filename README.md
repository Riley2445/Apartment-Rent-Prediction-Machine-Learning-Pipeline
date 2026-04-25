# 🏠 Apartment Rent Prediction: Machine Learning Pipeline

This repository contains a complete machine learning workflow to predict apartment rental prices across the US. The project demonstrates data cleaning, exploratory data analysis (EDA), and the implementation of a robust scikit-learn pipeline for model comparison.

## 📊 Dataset Overview
The analysis is performed on the `apartments_for_rent.csv` dataset, which contains 10,000 instances. 
* **Target Variable:** `price`
* **Key Features:** Latitude, Longitude, Bathrooms, Bedrooms, Square Feet, and Categorical features (Pets Allowed, Has Photo).

## 🛠 Project Workflow

### 1. Data Cleaning & Preparation
* Dropped non-pertinent columns (`id` and `fee`).
* Split data into **80% Training** and **20% Testing** sets using `train_test_split` (random_state=42).

### 2. Automated Preprocessing Pipeline
To prevent data leakage and ensure reproducibility, I utilized `ColumnTransformer` to route data through specialized pipelines:
* **Numerical Pipeline:** Handles missing values using **Median Imputation** and scales features using **StandardScaler**.
* **Categorical Pipeline:** Handles missing values with a **Constant Imputer** ("No_Pets") and applies **OneHotEncoder** (dropping the first category).

### 3. Model Selection & Evaluation
I evaluated two regression algorithms using **10-fold Cross-Validation** with the Root Mean Squared Error (RMSE) metric:
* **Linear Regression:** Served as the baseline model.
* **Random Forest Regressor:** Achieved a significant reduction in error compared to the baseline.

## 📈 Final Results
The final model performance was evaluated on the unseen test set:
* **Final RMSE:** Approximately **$464.09**
* The Random Forest model demonstrated superior performance in capturing the non-linear relationships within the rental data.

---

### 🚀 How to Run
1. Ensure `apartments_for_rent.csv` is in the root directory.
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib`
3. Run the script or Jupyter Notebook to view the full analysis and performance metrics.
