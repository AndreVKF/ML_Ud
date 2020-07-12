# K-Nearest Neighbour (KNN)

# Maioria dos métodos de aprendizagem constroem um modelo após o treinamento (dados são descartados após a criação do modelo)
# Métodos baseados em instâncias simplesmente armazenam os exemplos de treinamento
# Generalização/previsão é feita somente quando uma nova instância precisa ser classificada (lazy)
# Distância Euclidiana
# Encoding => Atributos categóricos para discretos 
# Escalonamento => Normalização/Padronização

# Algoritmo simples e poderoso
# Indicado qndo o relacionamento das características é complexo
# Valor de k pqno: dados com ruídos ou outliers podem prejudicar
# Valor de k grande: tendência a classificar a classe com mais elementos (overfitting) - valor default (3 ou 5)
# Pode ser lento para fazer previsões
# Pode utilizar distâncias:
#   => Distância Euclidiana
#   => Coeficiente de Pearson
#   => Índice de Tanimoto
#   => City Block

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

'''
    Base de Risco de Credito
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

# Arvore de decisao classificador
classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

# collections.Counter(classe_teste)

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
scalerCols = previsores[:, 102:]
scaler = StandardScaler()
previsores[:, 102:] = scaler.fit_transform(scalerCols)
##### Escalonamento Total #####
# scaler = StandardScaler()
# previsores = scaler.fit_transform(previsores)

# Split dos dados
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

# Decision Tree Classifier
classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classificador.fit(previsores_treinamento, classe_treinamento)

previsoes = classificador.predict(previsores_teste)

precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)