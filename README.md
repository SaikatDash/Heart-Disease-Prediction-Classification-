# 🩺 Heart Disease Prediction

A machine learning project that predicts heart disease presence and severity using medical diagnostic data. This project demonstrates a complete classification workflow with multiple models, comprehensive evaluation metrics, and both automated and manual preprocessing approaches.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)
![Status](https://img.shields.io/badge/Status-Complete-success)

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Key Features](#key-features)
- [Models & Results](#models--results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Highlights](#project-highlights)
- [Technologies Used](#technologies-used)

## 🎯 Overview

This project builds and evaluates multiple classification models to predict heart disease based on 14 medical attributes. The analysis includes:

- **Complete Exploratory Data Analysis (EDA)** with visualizations
- **4 Classification Models**: Logistic Regression, Random Forest, SVM, KNN
- **Two Preprocessing Approaches**: Automated pipelines and manual step-by-step preprocessing
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Feature Importance Analysis** to identify key medical indicators

**Problem Type**: Multi-class Classification (5 classes: No Disease + 4 Severity Levels)  
**Best Model**: Random Forest Classifier with ~99% accuracy

## 📊 Dataset

**Source**: [Heart Disease UCI Dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

**Description**: Medical diagnostic data from 920 patients containing 14 clinical attributes for predicting heart disease presence and severity.

**Features** (14 attributes):
| Feature | Description | Type |
|---------|-------------|------|
| `age` | Age in years | Numerical |
| `sex` | Sex (1 = male, 0 = female) | Categorical |
| `cp` | Chest pain type (0-3) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Numerical |
| `chol` | Serum cholesterol (mg/dl) | Numerical |
| `fbs` | Fasting blood sugar > 120 mg/dl | Categorical |
| `restecg` | Resting ECG results (0-2) | Categorical |
| `thalach` | Maximum heart rate achieved | Numerical |
| `exang` | Exercise induced angina | Categorical |
| `oldpeak` | ST depression induced by exercise | Numerical |
| `slope` | Slope of peak exercise ST segment | Categorical |
| `ca` | Number of major vessels (0-3) | Numerical |
| `thal` | Thalassemia (3 = normal, 6 = fixed, 7 = reversible) | Categorical |

**Target Variable**: `num`
- `0` = No heart disease
- `1-4` = Heart disease with increasing severity

## ✨ Key Features

- 📊 **Comprehensive EDA**: Distribution analysis, correlation heatmaps, and feature visualizations
- 🤖 **Multiple Models**: Comparison of 4 different classification algorithms
- 🔄 **Two Preprocessing Methods**:
  - Automated: Scikit-Learn Pipelines for production-ready code
  - Manual: Step-by-step preprocessing for educational understanding
- 📈� Models & Results

### Models Evaluated

1. **Logistic Regression** - Baseline linear model
2. **Random Forest Classifier** - Ensemble method with 10,000 trees
3. **Support Vector Machine (SVM)** - Kernel-based classifier
4. **K-Nearest Neighbors (KNN)** - Instance-based learning

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** 🏆 | **~99%** | **~99%** | **~99%** | **~99%** |
| SVM | ~90% | ~90% | ~90% | ~90% |
| Logistic Regression | ~86% | ~86% | ~86% | ~86% |
| KNN | ~85% | ~85% | ~85% | ~85% |
| Logistic Regression (Manual) | ~86% | ~86% | ~86% | ~86% |

### Key Insights

**Top 5 Most Important Features** (from Random Forest):
1. 🩺 **Chest Pain Type (cp)** - Most predictive indicator
2. 💉 **Number of Major Vessels (ca)** - Strong correlation with disease
3. ❤️ **Maximum Heart Rate (thalach)** - Key cardiovascular metric
4. 📉 **ST Depression (oldpeak)** - Exercise-induced changes
5. 🔬 **Thalassemia (thal)** - Blood disorder indicator

**Medical Context**:
- **High Recall Priority**: Minimizing false negatives is critical in healthcare
- **Model Interpretability**: Feature importance helps validate clinical knowledge
- **Balanced Dataset**: Similar number of examples across classes

## 🔧 Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- 2GB RAM minimuAnalysis

Open `heat_disease_prediction.ipynb` and run cells sequentially:

**Notebook Structure**:
1. **Setup**: Import libraries and load data
2. **EDA**: Exploratory analysis with visualizations
3. **Preprocessing**: Feature engineering and data preparation
4. **Model Training**: Train 4 different classifiers
5. **Evaluation**: Compare models with multiple metrics
6. **Manual Preprocessing**: Step-by-step preprocessing demonstration
7. **Feature Importance**: Analyze most predictive features

### Example Code Snippet

```python
# Using the trained Random Forest model (best performer)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Create pipeline with preprocessing and model
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=10000, random_state=42))
])

# Train
rf_pipeline.fit(X_train, y_train)

# Predict
predictions = rf_pipeline.predict(X_test
# Prepare features and target
X = df.drop(['num', 'id', 'dataset'], axis=1)
y = df['num']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model (see notebook for complete preprocessing steps)
# ... preprocessing code ...
model = LogisticRegression(random_state=42)
model.fit(X_train_preprocessed, y_train)
```

## 📁 Project Structure

```
heart-disease-prediction/
│
├── heat_disease_prediction.ipynb    # Main Jupyter notebook
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
│🎯 Project Highlights

### What Makes This Project Special

**1. Dual Preprocessing Approach**
- **Pipeline Method**: Production-ready automated preprocessing
- **Manual Method**: Educational step-by-step demonstration
- Both achieve similar results, validating the methodology

**2. Healthcare-Focused Analysis**
- Emphasis on recall to minimize false negatives
- Feature importance aligned with medical knowledge
- Confusion matrix analysis for clinical relevance

**3. Comprehensive Documentation**
- Detailed explanations of classification concepts
- Clear commentary throughout the notebook
- Theoretical foundations for each algorithm

**4. Model Comparison**
- 4 different algorithms evaluated
- Multiple evaluation metrics beyond accuracy
- Visual comparison of performance

### Project Workflow

```
Data Loading → EDA → Preprocessing → Model Training → Evaluation → Insights
    ↓            ↓         ↓              ↓              ↓           ↓
KaggleHub   Visualize  Pipeline/    4 Models    Metrics +      Feature
Download    Features   Manual       Trained     Confusion      Importance
                                               Matrix
```

## 🛠️ Technologies Used

**Core Libraries**:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `scikit-learn` - Machine learning models and preprocessing
- `matplotlib` & `seaborn` - Data visualization
- `kagglehub` - Dataset download

**Models Implemented**:
- Logistic Regression
- Random Forest Classifier (10,000 estimators)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

**Preprocessing Techniques**:
- SimpleImputer (mean/mode imputation)
- StandardScaler (feature scaling)
- OneHotEncoder (categorical encoding)
- ColumnTransformer (feature transformation)

---

## 📝 Notes

- Dataset automatically downloaded via KaggleHub API
- No missing values after imputation
- Stratified train-test split (80/20) maintains class balance
- Random state (42) ensures reproducibility

**Project Status**: ✅ Complete  
**Last Updated**
---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

**Project Status**: ✅ Complete | Last Updated: February 2026
