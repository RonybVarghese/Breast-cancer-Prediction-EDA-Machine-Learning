# 🧠 Breast Cancer Diagnosis Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting whether a tumor is **Benign (B)** or **Malignant (M)** using machine learning classification algorithms.

The dataset contains medical measurements of breast tumors, and the goal is to build and compare multiple classification models to determine the most accurate one.

---

## 📊 Problem Statement

Early detection of breast cancer is critical for improving survival rates.  
This project builds predictive models to classify tumors based on their diagnostic features.

---

## 🗂 Dataset Information

- Target Variable: `diagnosis`
    - M → Malignant
    - B → Benign
- Features: Various numerical features derived from digitized images of fine needle aspirate (FNA) of breast mass.

---

## ⚙️ Project Workflow

### 1️⃣ Import Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### 2️⃣ Data Preprocessing
- Removed unnecessary columns
- Checked missing values
- Converted categorical target into numeric format (if required)
- Feature scaling using:
  - StandardScaler
  - MinMaxScaler
  - RobustScaler

### 3️⃣ Exploratory Data Analysis (EDA)
- Target distribution visualization
- Correlation heatmap
- Feature distribution plots

### 4️⃣ Train-Test Split
- Split dataset into training and testing sets using:
```python
train_test_split()                                                  

### 5️⃣ Model Building

The following models were implemented and compared:

Logistic Regression

K-Nearest Neighbors (KNN)

Decision Tree Classifier

Random Forest Classifier

6️⃣ Model Evaluation

Models were evaluated using:

Accuracy Score

Confusion Matrix

Classification Report

📈 Results

The Random Forest model achieved the highest accuracy among all models tested.

Model	Performance
Logistic Regression	                       High
KNN	                                       Moderate
Decision Tree	                           Good
Random Forest	                           Best

🛠 Technologies Used

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook

🚀 How to Run This Project

Clone the repository:

git clone https://github.com/your-username/your-repo-name.git

Install required libraries:

pip install -r requirements.txt

Open the notebook:

jupyter notebook MLProjects.ipynb
🎯 Future Improvements

Hyperparameter tuning (GridSearchCV)

Cross-validation

ROC-AUC comparison

Feature importance analysis

Deployment using Streamlit or Flask

👨‍💻 Author

Rony
Data Science Enthusiast


