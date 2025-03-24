# Você teria sobrevivido ao Titanic?  

Este é um projeto baseado no desafio do Kaggle, com a base de dados dos passageiros do Titanic. Esse projeto é uma variação dele, interativo em Python que usa um modelo de machine learning para prever se você sobreviveria ao naufrágio do Titanic, com base em dados reais de 1912. A experiência inclui arte em ASCII e perguntas no terminal.

## Estrutura do Projeto

previsao-sobrevivencia-titanic/
├── data/
│   └── titanic_data.csv
├── src/
│   └── train_model.py
├── README.md
├── requirements.txt

## Como Rodar?

1. Clone este repositório:
```bash
git clone https://github.com/seu-usuario/previsao-sobrevivencia-titanic.git
cd previsao-sobrevivencia-titanic

2. Instale os pacotes necessários (recomenda-se usar um ambiente virtual):

pip install -r requirements.txt

## Treine o modelo:

python3 src/train_model.py

4. Execute a simulação:

python3 src/predict_user.py

## Modelo
O modelo é um RandomForestClassifier, treinado com os seguintes dados:

Classe (1ª, 2ª ou 3ª)

Idade

Nº de irmãos/cônjuges a bordo

Nº de pais/filhos a bordo

Valor pago pela passagem

Sexo

Porto de embarque

## Detalhes extras
Arte em ASCII ao iniciar a experiência.

Resposta final com desenho temático dependendo da previsão: ☀️ se sobreviveu, ☠️ se não.

## Requisitos
Python 3.8 ou superior

pandas

numpy

scikit-learn

joblib

(opcional) colorama – para colorir o terminal
