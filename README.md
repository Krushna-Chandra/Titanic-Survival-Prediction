# 🚢 Titanic Survival Prediction

**A Complete End-to-End Machine Learning Project**  
Predict whether a passenger survived the Titanic disaster using Python, data analysis, and machine learning.

---

## 🧠 Project Overview

This project builds a classification model that predicts the likelihood of a passenger’s survival based on features such as age, sex, fare, and passenger class. It includes data cleaning, visualization, feature engineering, model building, and evaluation.

---

## 📍 Table of Contents

- [🔍 Dataset](#-dataset)  
- [📦 Technologies Used](#-technologies-used)  
- [🚀 Project Features](#-project-features)  
- [📊 Visualizations](#-visualizations)  
- [🛠️ How It Works](#%EF%B8%8F-how-it-works)  
- [📈 Model Training & Evaluation](#-model-training--evaluation)  
- [📁 File Structure](#-file-structure)  
- [📌 Conclusion](#-conclusion)  
- [📌 Future Improvements](#-future-improvements)  

---

## 🔍 Dataset

The dataset used in this project is from **Kaggle’s Titanic Machine Learning Competition**:

- `train.csv`: Training data with survival labels  
- `test.csv`: Test data without survival labels

📌 You can download the dataset here:  
https://www.kaggle.com/c/titanic

---

## 📦 Technologies Used

- Python  
- Pandas  
- NumPy  
- Matplotlib & Seaborn  
- Scikit-Learn  
- Jupyter Notebook

---

## 🚀 Project Features

✔ Exploratory Data Analysis (EDA)  
✔ Data cleaning & preprocessing  
✔ Feature engineering  
✔ Model building (Logistic Regression)  
✔ Model evaluation and reports  
✔ Professional visualizations

---

## 📊 Visualizations

The project includes multiple visual insights, such as:

- Survival counts
   <img width="876" height="549" alt="image" src="https://github.com/user-attachments/assets/a31f67f9-50e4-4d7a-9c29-9d97db355aa0" />

- Survival by gender & class
  <img width="882" height="566" alt="image" src="https://github.com/user-attachments/assets/7fd9f2e5-5421-49b5-be88-1401fcc049c7" />
  
- Age distribution
  <img width="883" height="556" alt="image" src="https://github.com/user-attachments/assets/3f53962e-6644-4e51-9bde-0016c00a0d1d" />

- Fare comparison
  <img width="889" height="546" alt="image" src="https://github.com/user-attachments/assets/57713854-4f3c-41ce-9cd6-8a67909af896" />

- Servival Rate by Passenger Class
  <img width="883" height="547" alt="image" src="https://github.com/user-attachments/assets/fc960d3e-3487-445f-9ef0-685a09777a9b" />


These help to understand patterns in the data and how different features affect passenger survival.

---

## 🛠️ How It Works

### 1. Load the dataset

```python
df = pd.read_csv("train.csv")
df.head()
```

### 2. Clean missing values

```python
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)
```

### 3. Feature encoding

```python
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
```

### 4. Prepare features & labels

```python
X = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_Q','Embarked_S']]
y = df['Survived']
```

### 5. Train-test split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 📈 Model Training & Evaluation

The model uses **Logistic Regression** as a baseline classification algorithm:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

Evaluate performance:

```python
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## 📁 File Structure

```
Titanic-Survival-Prediction/
├─ README.md
├─ train.csv
├─ test.csv
├─ Titanic_Survival_Documented.ipynb
├─ model_training.ipynb
├─ visuals.py
```

---

## 📌 Conclusion

This project demonstrates a structured end-to-end machine learning workflow including data preprocessing, visualization, model training, and evaluation. It serves as a foundational data science portfolio piece.

---

## 📌 Future Improvements

✔ Improve model using RandomForest, XGBoost
✔ Deploy as a **Streamlit app**
✔ Add **Hyperparameter tuning**
✔ Create interactive dashboards

---
