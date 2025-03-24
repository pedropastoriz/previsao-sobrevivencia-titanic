# src/predict_user.py

import numpy as np
import joblib

# Códigos ANSI para cor
BLUE = '\033[94m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'

# Titanic em ASCII
titanic_ascii = f"""{BLUE}
          |    |    |
         )_)  )_)  )_)
        )___))___))___)\\
       )____)____)_____)\\\\
     _____|____|____|____\\\\\\__
-----\                   /-----
  ^^^^^ ^^^^^^^^^^^^^^^^^^^^^
    ^^^^      ^^^^     ^^^    ^^
         ^^^^      ^^^
{RESET}"""

# Iceberg em ASCII
iceberg_ascii = f"""{CYAN}
               .    .
              / `.  `.     .
             /  `. `. `.  /|
            |    `. `. `.| |
            |      `. `.| |
            |        `..  |
            |          |  |
            |          |  |
           /           |  |
          /            |  |
         /_____________|__|
{RESET}"""

# Solzinho (sobreviveu)
sun_ascii = f"""{YELLOW}
       \\   |   /
         .-*-.
      ---( ☀ )---
         `-*-’
       /   |   \\
{RESET}"""

# Caveira com cruz (não sobreviveu)
skull_ascii = f"""{RED}
       †
      ☠ ☠
     (x__x)
      /||\\
       ||
      /  \\
{RESET}"""

# Início
print(f"{BOLD}=== Você sobreviveria ao Titanic? Descubra! ==={RESET}")
print(titanic_ascii)
print(iceberg_ascii)

# Coleta de dados
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

# Previsão
model = joblib.load('src/titanic_model.pkl')
prediction = model.predict(user_data)[0]

# Resultado
if prediction == 1:
    print(f"\n{GREEN}{BOLD}Você ganhou mais um dia de vida, aproveite! Sobreviveu ao Titanic!{RESET}")
    print(sun_ascii)
else:
    print(f"\n{RED}{BOLD}Você virou uma camiseta de saudades eternas. Não sobreviveu.{RESET}")
    print(skull_ascii)
