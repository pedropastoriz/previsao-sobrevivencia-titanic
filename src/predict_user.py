# src/predict_user.py

import numpy as np
import joblib

print("=== Você sobreviveria ao Titanic? Descubra! ===")

pclass = int(input("Classe (1 = 1ª classe, 2 = 2ª, 3 = 3ª): "))
age = int(input("Idade: "))
sibsp = int(input("Número de irmãos/cônjuges a bordo: "))
parch = int(input("Número de pais/filhos a bordo: "))
fare = float(input("Preço pago pela passagem (lembre que estamos em 1912): "))
sex = input("Sexo (male/female): ").strip().lower()
embarked = input("Porto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton): ").strip().upper()

# Features: [Pclass, Age, SibSp, Parch, Fare, male, Q, S]
male = 1 if sex == 'male' else 0
Q = 1 if embarked == 'Q' else 0
S = 1 if embarked == 'S' else 0

user_data = np.array([pclass, age, sibsp, parch, fare, male, Q, S]).reshape(1, -1)

model = joblib.load('src/titanic_model.pkl')
prediction = model.predict(user_data)[0]

resultado = "Você ganhou mais um dia de vida, aproveite! Sobreviveu ao Titanic!" if prediction == 1 else "Você virou uma camiseta de saudades eternas. Não sobreviveu."
print(f"\nResultado: {resultado}")
