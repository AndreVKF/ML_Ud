from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

####### Credit Data #######
base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 3 - Pre-processamento com Pandas e scikit-learm\\credit_data.csv')
base.loc[base['age']<0]

# base[(base['age'] < 0)].index
# base.drop(base[(base['age'] < 0)].index, inplace=True)

# Preencher valores com as medias
base.loc[base['age']<0, 'age'] = base[(base['age']>0)]['age'].mean()

# pd.isnull(base)
# base.loc[pd.isnull(base['age'])]

# Quebra do dataframe
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

# SK filling data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:,1:4])

previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

# Escalonador
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Split Data
previsores_treinamento, previsores_teste, classe_treinamento, class_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

####### Census DataBase #######
census = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 3 - Pre-processamento com Pandas e scikit-learm\\census.csv')

previsores = census.iloc[:, 0:14].values
classe = census.iloc[:, 14].values

labelEncoderPrevisores = LabelEncoder()
# labels = labelEncoderPrevisores.fit_transform(previsores[:, 1])
previsores[:, 1] = labelEncoderPrevisores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelEncoderPrevisores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelEncoderPrevisores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelEncoderPrevisores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelEncoderPrevisores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelEncoderPrevisores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelEncoderPrevisores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelEncoderPrevisores.fit_transform(previsores[:, 13])

# Dummy variables
# etnia = census.iloc[:, 8].values
# etnia = labelEncoderPrevisores.fit_transform(etnia)

# Column Transformer
oneHotEncoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [1, 3, 5, 6, 7, 8, 9,13])], remainder='passthrough')
previsores = oneHotEncoder.fit_transform(previsores).toarray()

labelEncoder_Classe = LabelEncoder()
classe = labelEncoder_Classe.fit_transform(classe)

# Escalonamento
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)