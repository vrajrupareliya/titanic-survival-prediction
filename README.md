#  Titanic Survival Prediction using Machine Learning

Predicting survival chances on the **Titanic dataset** using **Logistic Regression** — one of the most classic beginner projects in machine learning.  
This project demonstrates data cleaning, exploratory data analysis (EDA), feature engineering, model building, and evaluation using scikit-learn.

---

##  Project Overview

The goal of this project is to predict whether a passenger survived or not on the Titanic based on features such as age, sex, passenger class, fare, and family relationships.

We build a **classification model** using **Logistic Regression** and evaluate its performance using accuracy, confusion matrix, classification report, and cross-validation.

---

##  Dataset

- **Source:** [Kaggle - Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)
- **Files Used:** `train.csv`
- **Target Variable:** `Survived` (1 = Survived, 0 = Did Not Survive)

---

##  Data Preprocessing

- Handled missing values:
  - `Age` → filled with median  
  - `Embarked` → filled with mode  
  - `Cabin` → dropped (too many missing values)
- Converted categorical variables:
  - `Sex` → numeric encoding (male = 0, female = 1)
  - `Embarked` → one-hot encoding
- Created new engineered features:
  - `FamilySize = SibSp + Parch + 1`
  - `IsAlone = 1 if FamilySize == 1 else 0`
  - `Title` extracted from the passenger’s name (e.g., Mr, Mrs, Miss)
- Scaled numeric features using `StandardScaler`

---

##  Exploratory Data Analysis (EDA)

A detailed EDA was performed to understand survival patterns:

| Visualization | Insights |
|----------------|-----------|
|  Survival by Gender | Females had a much higher survival rate |
|  Survival by Passenger Class | 1st Class passengers had the best survival rate |
|  Age Distribution | Most passengers were between 20–40 years |
|  Fare vs. Survival | Higher fare correlated with better survival |
|  Family Size | Small families had higher survival chances |

Example visualizations:

```python
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Passenger Class')
plt.show()
```

##  Model Building

- Model used: Logistic Regression

- from sklearn.linear_model import LogisticRegression
- model = LogisticRegression(max_iter=1000)
- model.fit(X_train, y_train)

##  Model Evaluation

- Hold-out Test Accuracy: `81.0%`

- Confusion Matrix:

- `[[90 15]
[19 55]]`


Classification Report:

| Metric           | Precision | Recall | F1-Score |
| ---------------- | --------- | ------ | -------- |
| Not Survived (0) | 0.83      | 0.86   | 0.84     |
| Survived (1)     | 0.79      | 0.74   | 0.76     |


- 5-Fold Cross Validation Accuracy: `0.791`

 Tech Stack

- Language: Python

- Libraries:

- pandas

- numpy

- matplotlib / seaborn

- scikit-learn (LogisticRegression, train_test_split, metrics)

##  Results & Insights

- Logistic Regression achieved `~81%` accuracy

- Cross-validation confirmed stable performance `(~79%)`

- Gender, Class, Fare, and Title were the most influential features