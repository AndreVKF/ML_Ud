# Abordagem probabilística (Teorema de Bayes)
# Tabela de probabilidades
# Vantagens
#   Rápido
#   Simplicidade de interpretação
#   Trabalha com muitas dimensões
#   Boas previsões em bases pequenas
# Desvantagens
#   Combinação de características (atributos independentes) - cada par é independente - nem sempre é verdade

import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer

'''
    Base de Credito Naive Bayes
'''
base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 4 - Aprendizagem bayesiana\\risco_credito.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

# Encoder
LabelEncoder = LabelEncoder()
previsores[:, 0] = LabelEncoder.fit_transform(previsores[:, 0])
previsores[:, 1] = LabelEncoder.fit_transform(previsores[:, 1])
previsores[:, 2] = LabelEncoder.fit_transform(previsores[:, 2])
previsores[:, 3] = LabelEncoder.fit_transform(previsores[:, 3])

classificador = GaussianNB()
classificador.fit(previsores, classe)

# Historico bom, divida alta, nenhuma garantia, renda > 35
resultado = classificador.predict([[0, 0, 1, 2], [3, 0, 0, 0]])
print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)

'''
    Base de Credito Maior
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

# Naive Bayes
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)

# Predict previsoes teste
previsoes = classificador.predict(previsores_teste)

# Percentual de acerto
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
scalerCols = previsores[:, 102:]
scaler = StandardScaler()
previsores[:, 102:] = scaler.fit_transform(scalerCols)

# Split dos dados
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

# Naive Bayes
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)

previsoes = classificador.predict(previsores_teste)

precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)