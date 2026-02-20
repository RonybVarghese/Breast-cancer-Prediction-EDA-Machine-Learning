# 🧠 Breast Cancer Classification – Machine Learning Project

## 📖 Project Overview

This project focuses on building a Machine Learning model to classify breast cancer tumors as **Malignant (M)** or **Benign (B)** using diagnostic measurement features.

The goal is to:
- Perform data cleaning and preprocessing
- Conduct Exploratory Data Analysis (EDA)
- Handle class imbalance
- Apply feature scaling
- Train multiple classification models
- Compare models using F1-score
- Select the best performing model

---

## 📂 Dataset Information

The dataset contains medical measurements computed from digitized images of breast mass.

### Features:
- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry
- Fractal Dimension
- and more...

### Target Variable:
- `M` → Malignant (Cancerous)
- `B` → Benign (Non-Cancerous)

---

## 🔍 Project Workflow

### 1️⃣ Import Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

### 2️⃣ Data Preprocessing

- Removed `Unnamed: 32` column (fully null)
- Checked for duplicates
- Checked for missing values
- Converted target variable into numeric format
- Handled imbalanced data
- Applied **StandardScaler** to normalize features

---

### 3️⃣ Exploratory Data Analysis (EDA)

- Checked class distribution
- Identified skewness and outliers
- Visualized correlations
- Analyzed feature importance patterns

---

### 4️⃣ Train-Test Split

- Split dataset into training and testing sets
- Used `train_test_split` from sklearn

---

### 5️⃣ Model Building

Four models were trained and evaluated:

#### 🔹 1. Logistic Regression (Baseline Model)
- Training F1-score: **98%**
- Testing F1-score: **98%**

#### 🔹 2. K-Nearest Neighbors (KNN)
- Lower performance compared to baseline

#### 🔹 3. Decision Tree
- Performed well but slightly below Logistic Regression

#### 🔹 4. Random Forest (Best Model)
- Training F1-score: **99%**
- Testing F1-score: **98%**

---

## 📊 Why F1 Score Instead of Accuracy?

The dataset is slightly imbalanced.  
Accuracy alone may give misleading results.

F1-score balances:
- Precision
- Recall

Making it more reliable for medical diagnosis problems.

---

## 🏆 Final Model

The **Random Forest Classifier** performed the best overall with:
- High F1-score
- Strong generalization on test data
- Reduced overfitting compared to Decision Tree

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
