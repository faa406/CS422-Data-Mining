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
- [Project Structure](#project-structure)  
- [References & Resources](#references--resources)  
- [Author & Acknowledgments](#author--acknowledgments)  
- [License](#license)

---

## Project Overview

**Problem statement:** Build an accurate, robust, and interpretable classifier for a blind dataset (no feature descriptions). The dataset poses multiple realistic challenges:

- **Blind Dataset:** No metadata or domain knowledge for features.  
- **Class Imbalance:** Three classes with distribution approximately **16.7%**, **49.9%**, **33.4%**.  
- **High Dimensionality & Multicollinearity:** 15 numeric features with severe correlation and VIFs indicating redundancy.  
- **Large Scale:** ~**1.2 million** samples requiring scalable and efficient solutions.

This project demonstrates practical decision-making for real-world constraints: interpretability, speed, reproducibility, and deployability.

---

## Key Achievements

- Achieved **~51% multiclass accuracy** on a balanced dataset using **NearMiss undersampling** and a Decision Tree classifier.  
- Reduced 15 features to **2 PCA components**, preserving **98.29%** of variance.  
- Built a concise, interpretable **Decision Tree (max_depth=3)** that generalized well and was easy to explain to stakeholders.  
- Exported the final model to **ONNX** format for cross-platform deployment and fast inference.  
- Performed an extensive analysis of feature relationships (correlation matrices, VIF) and produced reproducible pipeline code and notebook.  
- Demonstrated end-to-end production readiness: preprocessing → modeling → validation → ONNX export.

---

## Methodology

### 1. Data Exploration & Quality Assessment
**What was done**
- Inspected dataset shape and types: **1,200,000 × 16** (15 numeric features + 1 target).  
- Checked for missing values, NaNs, and duplicates — none were present.  
- Confirmed column dtypes (`float64`) and validated target class distribution.  
- Performed basic descriptive statistics and visualized distributions for skewness and outliers.

**Why**
- Ensures data is ready for ML pipelines (no imputation required), reduces preprocessing complexity, and informs the sampling strategy for class imbalance.

---

### 2. Feature Relationship Analysis
**What was done**
- Computed correlation matrix and visualized with heatmaps.  
- Calculated Variance Inflation Factor (VIF) for multicollinearity diagnosis.  
- Flagged features with correlation > 0.90 and VIFs > 10 (some > 500).

**Why**
- High multicollinearity inflates variance of coefficients and reduces interpretability. Identifying and addressing redundancy improves stability and runtime efficiency.

---

### 3. Dimensionality Reduction (PCA)
**What was done**
- Standardized features (z-score) and applied **PCA**.  
- Reduced 15 features to **2 principal components** while retaining **98.29% variance**.

**Why**
- PCA handles severe feature correlation by extracting orthogonal directions of maximum variance, reducing noise and computational cost while retaining predictive power. Visualizing 2 components also aided qualitative model validation.

---

### 4. Class Imbalance Handling
**What was tested**
- Random Undersampling  
- NearMiss Undersampling (imblearn)  
- Class weighting in model loss functions

**Why NearMiss was chosen**
- Random undersampling was too aggressive and lost informative boundary samples (accuracy dropped to ~33%).  
- **NearMiss**, which selects samples near class boundaries, preserved informative edge cases and produced substantially better performance (~51% accuracy).  
- NearMiss balanced fairness and accuracy for this use case.

---

### 5. Model Selection & Evaluation
**Models evaluated**
- Decision Tree (max_depth=3), Logistic Regression, Random Forest, XGBoost, KNN, Gaussian Naive Bayes, SGD Classifier

**Why Decision Tree**
- Comparable top-level accuracy with far greater explainability.  
- Low inference cost and robust to the type of transformed features (PCA outputs).  
- Easy to present to non-technical stakeholders via simple decision rules.

**Evaluation approach**
- Stratified train/test splits, cross-validation, confusion matrices, classification reports, and binary-class pair experiments to measure separability between classes.

---

### 6. Feature Selection Strategy
**Methods tested**
- L1-based selection (LinearSVC)  
- SelectKBest (ANOVA F-test)  
- Recursive Feature Elimination (RFE — discarded due to runtime)

**Final choice**
- **SelectKBest** for its statistical grounding (ANOVA F-test), speed on 1.2M samples, and consistent results with L1 selection. RFE was impractical at this scale.

---

## Results & Metrics

### Multiclass Performance Summary
| Dataset Type | Accuracy | Model | Notes |
|--------------|----------|-------|-------|
| Original (Imbalanced) | ~49.9% | Multiple tied | Biased toward majority class |
| Random Undersampling | ~33.3% | Decision Tree | Aggressive sampling lost signal |
| **NearMiss Undersampling** | **~51.0%** | **Decision Tree** | Best trade-off: fairness & accuracy |

### Binary Classification Results (pairwise)
- Class 1 vs 2: **74.9%**  
- Class 1 vs 3: **66.7%**  
- Class 2 vs 3: **59.9%**

**Interpretation:** Pairwise binary tasks simplified decision boundaries and improved accuracy because the classifier focuses on separating two distributions rather than handling three overlapping regions simultaneously.

---

## Pipeline Rationale
- **StandardScaler:** Required for PCA scaling to ensure components reflect relative variance across standardized dimensions.  
- **PCA:** Handles multicollinearity and reduces dimensionality while preserving the majority of variance.  
- **SelectKBest:** Retains the most discriminative features/components based on statistical tests (ANOVA F-test), ensuring the classifier receives strong signals.  
- **Decision Tree:** Provides interpretable, robust predictions with low compute overhead and clear decision rules suitable for stakeholders.  

## Libraries & Tools
- **Data Processing & Analysis:** pandas, numpy, statsmodels  
- **Visualization:** matplotlib, seaborn  
- **Machine Learning Models:** scikit-learn, xgboost  
- **Imbalanced Data Handling:** imblearn (NearMiss)  
- **Dimensionality Reduction & Feature Selection:** PCA, SelectKBest  
- **Model Deployment:** skl2onnx, onnxruntime  
- **Other:** warnings suppression for cleaner notebook outputs  

## Deployment
Exported the final model to ONNX to enable cross-platform, production-ready deployment. ONNX ensures:
- Consistent predictions across environments (training vs. production)  
- Fast inference with optimized runtimes (onnxruntime)  
- Hardware flexibility — can be deployed on CPUs, GPUs, and embedded devices  

**ONNX Model Validation Example**
- Predicted class distribution: `{2: 239991, 3: 9}`  
- Prediction correctness: `{False: 120155, True: 119845}`  

## Key Learnings & Insights
- **PCA Effectiveness:** Reduces multicollinearity while preserving variance and improving model stability.  
- **NearMiss Sampling:** Focuses on boundary samples to improve class balance without discarding critical information.  
- **Decision Tree Interpretability:** Provides actionable insights for datasets with unknown feature meaning.  
- **Pipeline Approach:** Combines preprocessing, transformation, selection, and classification to prevent data leakage and improve reproducibility.  
- **Scalability:** Choices like SelectKBest, PCA, and NearMiss allow handling ~1.2M samples efficiently.  

## Future Enhancements
- Explore ensemble methods (Random Forest, XGBoost, Gradient Boosting) for accuracy while balancing interpretability.  
- Implement domain-guided feature engineering if feature semantics become available.  
- Experiment with deep learning architectures for complex non-linear relationships.  
- Test advanced oversampling techniques (SMOTE, ADASYN) and hybrid sampling (SMOTE + Tomek).  
- Integrate model explainability tools (SHAP, LIME) on the deployed ONNX model.  

## Author & Acknowledgments
- **Author:** Fatima Vahora  
- **Course:** CS422 — Data Mining  
- **Date:** July 2025  
- **Final Grade:** 100 / 100 Points


Special thanks to my course instructor for guidance and a hands-on curriculum that enabled me to build practical, production-ready solutions. This project highlights my capabilities in data preprocessing, feature engineering, model evaluation, and deployment, and demonstrates my ability to solve complex classification problems efficiently and effectively.

