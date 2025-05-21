# Cardiovascular Disease Risk Prediction  
**Eric Sun, Grace Deng, Seth Sukboontip**

## Overview

This project investigates predictive modeling for cardiovascular disease (CVD) using a behavioral health dataset derived from the CDC's Behavioral Risk Factor Surveillance System (BRFSS). We leverage logistic regression, random forests, and neural networks to identify at-risk individuals based on self-reported lifestyle and health factors.

We implement advanced feature engineering, address class imbalance with SMOTEENN, and evaluate model performance with appropriate metrics for imbalanced classification.

## Dataset

We use a cleaned subset of the [BRFSS dataset](https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset), which contains 19 behavior- and health-related variables for U.S. adults. The target variable is whether a participant has been diagnosed with heart disease.

Notable challenges:
- **Class imbalance**: Only ~8% of participants report heart disease.
- **Mixed data types**: Includes binary, ordinal, and nominal categorical variables.

## Key Steps

### 1. Data Preprocessing
- Ordinal and one-hot encoding for categorical variables.
- Custom handling of multi-class categorical columns (e.g., diabetes types).
- Feature normalization and correlation analysis to reduce redundancy.

### 2. Exploratory Data Analysis
- Visualized target class imbalance.
- Analyzed relationships between features (e.g., BMI, diet, comorbidities).

### 3. Feature Engineering
- Used K-means clustering (k=6) to discover patient segments.
- One-hot encoded cluster labels as additional features.
- Dropped highly correlated features (e.g., BMI vs. weight/height).

### 4. Modeling

#### Logistic Regression
- Baseline model with and without SMOTEENN.
- Highlights limitations of accuracy in imbalanced settings.

#### Random Forest
- Used `RandomizedSearchCV` for hyperparameter tuning.
- Tuned class weights and optimized classification thresholds.
- Evaluated feature importances and model performance via F1 score.

#### Neural Network
- Fully connected feedforward ANN trained with weighted BCE loss.
- Hyperparameter search across architecture, learning rate, weight decay, and classification threshold.
- Evaluated using precision, recall, and F1 on validation and test sets.

## Results

| Model                         | Positive F1 Score |
|------------------------------|-------------------|
| Logistic Regression (raw)    | 0.11              |
| Logistic Regression (SMOTEENN) | 0.29            |
| Random Forest (SMOTEENN)     | 0.36              |
| Random Forest + Threshold Tuning | 0.37          |
| Neural Network               | 0.37              |

### Top Predictive Features
- `Age_Lower`, `General_Health`, `Weight_(kg)`, `Fruit_Consumption`, and `Cluster_4` from K-means showed strong importance in tree-based models.
- Neural networks improved recall for CVD-positive cases with appropriate threshold tuning and class-weighting.

## Usage

1. Download the dataset from Kaggle and place `CVD_cleaned.csv` in your Google Drive.
2. Open the notebook in Google Colab.
3. Run all cells to perform data analysis and train models.

## Future Work

- Explore additional resampling techniques like SMOTETomek.
- Incorporate explainability tools like SHAP for tree/NN models.
- Consider ensemble approaches (e.g., XGBoost, LightGBM).
- Evaluate clinical implications of precision vs. recall tradeoffs.
