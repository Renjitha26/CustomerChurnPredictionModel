# Problem Statement : Predict whether a customer will leave (churn) or stay

# Step 1: Import Libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix

# Step 2: Load Data
df = pd.read_csv("churn.csv")
print(df.head())

# Step 3: Explore Data
print(df.info())
print(df.describe())
print(df.columns)

# Step 4: Data Cleaning
# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Remove missing values
df.dropna(inplace=True)
# Drop unnecessary column
df.drop('customerID', axis=1, inplace=True)

# Step 5: Encode Categorical Data
df = pd.get_dummies(df, drop_first=True)

# Step 6: Define Features & Target
X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42 )

# Step 8: Feature Scaling 
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 9: Train Multiple Models

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

# SVM
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Step 10: Evaluation Function
def evaluate_model(y_test, pred, name):
    print(f"\n{name}")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

# Step 11: Compare Models
evaluate_model(y_test, lr_pred, "Logistic Regression")
evaluate_model(y_test, knn_pred, "KNN")
evaluate_model(y_test, svm_pred, "SVM")
evaluate_model(y_test, dt_pred, "Decision Tree")
evaluate_model(y_test, rf_pred, "Random Forest")

 # or
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    print(f"\n{name}")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))