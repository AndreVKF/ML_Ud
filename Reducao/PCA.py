import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
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

# Escalonamento dos dados
##### Escalonamento Parcial #####
# scalerCols = previsores[:, 102:]
# scaler = StandardScaler()
# previsores[:, 102:] = scaler.fit_transform(scalerCols)
##### Escalonamento Total #####
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Split dos dados
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)

# PCA
pca = PCA(n_components=6)

previsores_treinamento = pca.fit_transform(previsores_treinamento)
previsores_teste = pca.transform(previsores_teste)
componentes = pca.explained_variance_ratio_

# Random Forest
classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
classificador.fit(previsores_treinamento, classe_treinamento)

previsoes = classificador.predict(previsores_teste)

precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)