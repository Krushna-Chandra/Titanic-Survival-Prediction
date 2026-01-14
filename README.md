```
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
- [📜 License](#-license)

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
- Survival by gender & class  
- Age distribution  
- Fare comparison  
- Correlation heatmap

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

## 📜 License

This project is licensed under the **MIT License** 👨‍💻

```
