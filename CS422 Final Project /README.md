# CS422 Final Project - Blind Data Classification

A comprehensive **machine learning classification project** built for a _blind_ dataset (no feature metadata or domain knowledge). This repository demonstrates end-to-end skills in **data preprocessing, exploratory data analysis (EDA), imbalance handling, dimensionality reduction, feature selection, model training & evaluation, and production-ready model export (ONNX)**. It is designed to be recruiter-friendly and keyword-rich for quick technical screening.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Key Achievements](#key-achievements)  
- [Methodology](#methodology)  
  - [1. Data Exploration & Quality Assessment](#1-data-exploration--quality-assessment)  
  - [2. Feature Relationship Analysis](#2-feature-relationship-analysis)  
  - [3. Dimensionality Reduction (PCA)](#3-dimensionality-reduction-pca)  
  - [4. Class Imbalance Handling](#4-class-imbalance-handling)  
  - [5. Model Selection & Evaluation](#5-model-selection--evaluation)  
  - [6. Feature Selection Strategy](#6-feature-selection-strategy)  
- [Results & Metrics](#results--metrics)  
- [Technical Implementation & Pipeline](#technical-implementation--pipeline)  
  - [Pipeline Architecture](#pipeline-architecture)  
  - [Pipeline Rationale](#pipeline-rationale)  
  - [Libraries & Tools](#libraries--tools)  
- [Deployment](#deployment)  
  - [ONNX Export & Validation Example](#onnx-export--validation-example)  
- [Key Learnings & Insights](#key-learnings--insights)  
- [Future Enhancements](#future-enhancements)  
- [References & Resources](#references--resources)  
- [Author & Acknowledgments](#author--acknowledgments)  
- [License](#license)

---

## Project Overview

**Problem Statement:** Build an accurate, robust, and interpretable classifier for a blind dataset (no feature descriptions). The dataset poses multiple realistic challenges:  
- **Blind Dataset:** No metadata or domain knowledge for features.  
- **Class Imbalance:** Three classes with distribution approximately 16.7%, 49.9%, 33.4%.  
- **High Dimensionality & Multicollinearity:** 15 numeric features with severe correlation and VIFs indicating redundancy.  
- **Large Scale:** ~1.2 million samples requiring scalable and efficient solutions.  

This project demonstrates practical decision-making for real-world constraints: interpretability, speed, reproducibility, and deployability.

## Key Achievements
- Achieved ~51% multiclass accuracy on a balanced dataset using NearMiss undersampling and a Decision Tree classifier.  
- Reduced 15 features to 2 PCA components, preserving 98.29% of variance.  
- Built a concise, interpretable Decision Tree (max_depth=3) that generalized well and was easy to explain to stakeholders.  
- Exported the final model to ONNX format for cross-platform deployment and fast inference.  
- Performed extensive analysis of feature relationships (correlation matrices, VIF) and produced reproducible pipeline code and notebook.  
- Demonstrated end-to-end production readiness: preprocessing → modeling → validation → ONNX export.  

## Methodology

### 1. Data Exploration & Quality Assessment
**What was done:**  
- Inspected dataset shape and types: 1,200,000 × 16 (15 numeric features + 1 target).  
- Checked for missing values, NaNs, and duplicates — none were present.  
- Confirmed column dtypes (float64) and validated target class distribution.  
- Performed basic descriptive statistics and visualized distributions for skewness and outliers.  

**Why:** Ensures data is ready for ML pipelines, reduces preprocessing complexity, and informs sampling strategy for class imbalance.

### 2. Feature Relationship Analysis
**What was done:**  
- Computed correlation matrix and visualized with heatmaps.  
- Calculated Variance Inflation Factor (VIF) for multicollinearity diagnosis.  
- Flagged features with correlation > 0.90 and VIFs > 10 (some > 500).  

**Why:** High multicollinearity inflates variance of coefficients and reduces interpretability. Addressing redundancy improves stability and runtime efficiency.

### 3. Dimensionality Reduction (PCA)
**What was done:**  
- Standardized features (z-score) and applied PCA.  
- Reduced 15 features to 2 principal components while retaining 98.29% variance.  

**Why:** PCA extracts orthogonal directions of maximum variance, reduces noise, computational cost, and aids qualitative model validation.

### 4. Class Imbalance Handling
**What was tested:** Random Undersampling, NearMiss Undersampling (imblearn), Class weighting in model loss functions  

**Why NearMiss was chosen:**  
- Random undersampling lost informative boundary samples (accuracy dropped to ~33%).  
- NearMiss preserves informative edge cases, producing better performance (~51% accuracy).  
- Balanced fairness and accuracy effectively.

### 5. Model Selection & Evaluation
**Models evaluated:** Decision Tree (max_depth=3), Logistic Regression, Random Forest, XGBoost, KNN, Gaussian Naive Bayes, SGD Classifier  

**Why Decision Tree:**  
- Comparable top-level accuracy with far greater explainability.  
- Low inference cost and robust to PCA-transformed features.  
- Simple decision rules for non-technical stakeholders.  

**Evaluation approach:** Stratified train/test splits, cross-validation, confusion matrices, classification reports, and pairwise binary-class experiments.

### 6. Feature Selection Strategy
**Methods tested:** L1-based selection (LinearSVC), SelectKBest (ANOVA F-test), Recursive Feature Elimination (RFE — discarded due to runtime)  

**Final choice:** SelectKBest for statistical grounding, speed on 1.2M samples, and consistency with L1 selection. RFE was impractical at this scale.

## Results & Metrics

**Multiclass Performance Summary**

| Dataset Type           | Accuracy | Model         | Notes                                  |
|------------------------|---------|---------------|----------------------------------------|
| Original (Imbalanced)   | ~49.9%  | Multiple tied | Biased toward majority class           |
| Random Undersampling    | ~33.3%  | Decision Tree | Aggressive sampling lost signal        |
| NearMiss Undersampling  | ~51.0%  | Decision Tree | Best trade-off: fairness & accuracy   |

**Binary Classification Results (pairwise)**  
- Class 1 vs 2: 74.9%  
- Class 1 vs 3: 66.7%  
- Class 2 vs 3: 59.9%  

*Interpretation:* Pairwise tasks simplified decision boundaries and improved accuracy.

## Technical Implementation & Pipeline

### Pipeline Architecture
- Preprocessing → PCA → Feature Selection → Model Training → ONNX Export

### Pipeline Rationale
- **StandardScaler:** Required for PCA scaling.  
- **PCA:** Handles multicollinearity and reduces dimensionality.  
- **SelectKBest:** Retains the most discriminative features.  
- **Decision Tree:** Provides interpretable, robust predictions.

### Libraries & Tools
- **Data Processing & Analysis:** pandas, numpy, statsmodels  
- **Visualization:** matplotlib, seaborn  
- **Machine Learning Models:** scikit-learn, xgboost  
- **Imbalanced Data Handling:** imblearn (NearMiss)  
- **Dimensionality Reduction & Feature Selection:** PCA, SelectKBest  
- **Model Deployment:** skl2onnx, onnxruntime  
- **Other:** warnings suppression

### Deployment
Exported final model to ONNX for cross-platform deployment, fast inference, and hardware flexibility.  

**ONNX Model Validation Example:**  
- Predicted class distribution: `{2: 239991, 3: 9}`  
- Prediction correctness: `{False: 120155, True: 119845}`

## Key Learnings & Insights
- **PCA Effectiveness:** Reduces multicollinearity while preserving variance.  
- **NearMiss Sampling:** Preserves boundary samples for better class balance.  
- **Decision Tree Interpretability:** Actionable insights for unknown features.  
- **Pipeline Approach:** Prevents data leakage and improves reproducibility.  
- **Scalability:** Handles ~1.2M samples efficiently.

## Future Enhancements
- Explore ensemble methods (Random Forest, XGBoost, Gradient Boosting).  
- Implement domain-guided feature engineering.  
- Experiment with deep learning for complex non-linear patterns.  
- Test advanced oversampling (SMOTE, ADASYN) and hybrid techniques (SMOTE + Tomek).  
- Integrate model explainability tools (SHAP, LIME) on ONNX model.


## References & Resources
- [Scikit-learn Documentation](https://scikit-learn.org/)  
- [Imbalanced-learn Library](https://imbalanced-learn.org/)  
- [ONNX Runtime](https://onnxruntime.ai/)  
- [Principal Component Analysis Guide](https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c/)  

*Links above included for quick reference — full bibliography and citations are in the project notebook.*

## Author & Acknowledgments
- **Author:** Fatima Vahora  
- **Course:** CS422 — Data Mining  
- **Date:** July 2025  
- **Final Grade:** 100 / 100 Points  

Special thanks to my course instructor for guidance and a hands-on curriculum that enabled me to build practical, production-ready solutions. This project highlights my capabilities in data preprocessing, feature engineering, model evaluation, and deployment, and demonstrates my ability to solve complex classification problems efficiently and effectively.
