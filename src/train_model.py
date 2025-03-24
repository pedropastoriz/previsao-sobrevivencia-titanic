# src/train_model.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Carregar e preparar os dados
df = pd.read_csv('data/titanic_data.csv')
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

def insert_age(cols):
    Age, Pclass = cols
    if pd.isnull(Age):
        return {1: 38, 2: 30, 3: 25}.get(Pclass, 25)
    return Age

df['Age'] = df[['Age', 'Pclass']].apply(insert_age, axis=1)
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'].fillna(df['Embarked'].value_counts().idxmax(), inplace=True)

sex = pd.get_dummies(df['Sex'], drop_first=True)
embark = pd.get_dummies(df['Embarked'], drop_first=True)
df.drop(['Sex', 'Embarked'], axis=1, inplace=True)
df = pd.concat([df, sex, embark], axis=1)

X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Salvar modelo treinado
joblib.dump(model, 'src/titanic_model.pkl')
print("Modelo treinado e salvo. Bora descobrir se tu vai sobreviver ao Titanic?")
