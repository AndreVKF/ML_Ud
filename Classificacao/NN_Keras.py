# Redes Neurais
# Conceitos
#   => Perceptron
#   => Treinamento/ajuste dos pesos
#   => Gradient Descent
#   => Cálculo do Delta
#   => Learning Rate (Taxa de Aprendizagem)
#   => Momentum
#   => Backpropagation

# Conceitos sobre deep learning
# Redes Neurais com pybrain
# Redes Neurais com scikit-learn
# Redes Neurais com keras (deep learning)

# Problemas que são resolvidos por algoritmos pré-determinados 
# (sistemas de recomendação, buscas, grafos, ordenações)

# Aplicações
#   => Descoberta de novos remédios
#   => Entendimento de linguagem natural
#   => Carros autônomos
#   => Reconhecimento facial
#   => Deep Learning

# Fornece um valor de entrada, a rede processa e retorna uma resposta
# Neurônio é ativado somente se o valor for maior que um liminar

# Perceptron: Entradas com pesos associada a uma função de ativação
# Neurônio Artificial
# Step Function (Função Degrau)
#   Peso positivo - Sinapse excitadora
#   Peso negativo - Sinapse inibidora
#   Pesos amplificam ou reduzem o sinal de entrada
# Sigmoide Function
# Hyperbolic Tangent (Função Tangente Hiperbólica)

# Erro
# Algoritmo mais simples
#   Erro = respostaCorreta - respostaCalculada
# Atualização dos Pesos
#   peso(n + 1) = peso(n) + (taxaAprendizagem * entrada * erro)

#Algoritmo
# i -> Inicializa pesos
# ii -> Calculo saídas
# iii -> Calculo erro
# iv -> Calculo pesos
# v -> Atualiza pesos

# Backpropagation -> Atualização da camada de saída para as camadas ocultas

# Parâmetros
# Learning Rate
# Momentum -> Busca escapar de mínimos locais

# Bias
# Unidade de viés, valores diferentes mesmo se todas as entradas forem zero
# Muda a saída com a unidade de bias

# Deep Learning
# São utilizadas outras técnicas
# "Problema do gradiente desaparecendo" - vanishing gradient problem - gradiente fica muito pequino mudanças nos pesos ficam pequenas
# Otras funções de ativação

########## Camadas Ocultas ##########
# Neurônios = (Entradas + Saídas)/2 => Ceiling
# Problemas linearmente separáveis não necessitam de camadas oculatas

import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, export
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import keras
from keras.models import Sequential
from keras.layers import Dense

'''
    Credit Data
'''
base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 3 - Pre-processamento com Pandas e scikit-learm\\credit_data.csv')
base.loc[base['age']<0, 'age'] = base[(base['age']>0)].mean()

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

# Adjust nan values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[: ,1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

# Standard Scaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Split Data
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

# Neural Network
classificador = Sequential()
classificador.add(Dense(units=2, activation='relu'))
classificador.add(Dense(units=2, activation='relu'))
classificador.add(Dense(units=1, activation='sigmoid'))
classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=100)

previsoes = classificador.predict(previsores_teste)
# Threshold
previsoes = (previsoes > 0.5)

precisao = accuracy_score(classe_teste, previsoes)

'''
    Base do Censo
'''
base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 3 - Pre-processamento com Pandas e scikit-learm\\census.csv')

previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

# Label Encoder

labelEncoder_previsores = LabelEncoder()
previsores[:, 1] = labelEncoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelEncoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelEncoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelEncoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelEncoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelEncoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelEncoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelEncoder_previsores.fit_transform(previsores[:, 13])

# One Hot Encoder
oneHotEncoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')
previsores = oneHotEncoder.fit_transform(previsores).toarray()

# Y
labelEncoder_classe = LabelEncoder()
classe = labelEncoder_classe.fit_transform(classe)

# Escalonamento dos dados
##### Escalonamento Parcial #####
# scalerCols = previsores[:, 102:]
# scaler = StandardScaler()
# previsores[:, 102:] = scaler.fit_transform(scalerCols)
##### Escalonamento Total #####
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Split dos dados
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

# Neural Network
classificador = Sequential()
classificador.add(Dense(units=55, activation='relu'))
classificador.add(Dense(units=55, activation='relu'))
classificador.add(Dense(units=1, activation='sigmoid'))
classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=100)

previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

precisao = accuracy_score(classe_teste, previsoes)
confusion_matrix(classe_teste, previsoes)