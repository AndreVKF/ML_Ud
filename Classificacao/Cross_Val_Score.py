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

from sklearn.model_selection import cross_val_score

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

# Classificador Gaussiano
classificador = GaussianNB()

resultados = cross_val_score(classificador, previsores, classe, cv=10)
resultados.mean()
resultados.std()