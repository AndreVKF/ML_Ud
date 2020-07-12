# Support Vector Machines
# Aprende hiperplanos de separação com margem máxima
# Busca da margem máxima - Criação de hiperplanos
#   => Convex Hulls

# Minimização de Erros e Custo
# 1/2|w|^2 + c*Sum(ai)
# c => punição por classificação incorreta
# c alto = tenta 100% de separação
# c baixo = permite mais erros

# SVMs não lineares (Kernel Trick)
# Transforma uma superficie linear em uma superficie nao linear
# Kernels
# (Linear, Gaussian, Polynomial, Tangent Hyperbolic)

# Vantagens:
# Não é muito influenciado por ruídos nos dados
# Utilizado para classificação e regressão
# Aprende conceitos não presentes nos dados originais
# Mais fácil de usar do que redes neurais
# Desvantanges:
# Testar várias combinações de parâmetros
# Lento
# Blackbox

import pandas as pd
import numpy as np

import collections

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, export
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

'''
    Base Crédito
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

# SVM
classificador = SVC(kernel='linear', random_state=1)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

# SVM Poly
classificador = SVC(kernel='poly', random_state=1)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

# SVM Poly
classificador = SVC(kernel='sigmoid', random_state=1)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

# SVM RBF
classificador = SVC(kernel='rbf', random_state=1, C=2)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

# Testes
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

'''
    Base de Dados Censo
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

# SVM
classificador = SVC(kernel='rbf', random_state=1, C=2)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)
# print(classificador.coef_)

# Verificar validez dos dados
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)