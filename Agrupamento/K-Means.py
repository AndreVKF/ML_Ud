# Agrupamento não supervisionado
# Classificação/Regressão
#   => Modelo que relaciona características com uma variável a ser prevista

# Agrupamento
#   => Cria novos dados
#   => Não tem um rótulo e o algoritmo aprende as relações entre os dados

# i -> Inicializar os centroids aleatoriamente (centros de um cluster)
# ii -> Para cada ponto na base de dados, calcular a distância para cada centroide e associar ao que estiver mais perto
# iii -> Calcular a média de todos os pontos ligados a cada centroide e definir um novo centroide (repetir passos ii e iii)

# Distância Euclidiana
# DE(x, y) = (SUM(xi - yi)^2)^1/2

# K-mean++
#   => Reduz a probabilidade de inicialização ruins
#   => Seleciona os centroides iniciais q estão longes uns dos outros
#   => O primeiro centroide é selecionado randomicamente, porém, os outros são selecionados baseados na distância do primeiro ponto

# Definição de clusters
#   => Genérico
#       clusters = (n/2)^1/2
#   => Elbow
#       vários valores de k
#       within-cluster sum of squares
#       WCSS = SUM distance(Pi, C1)^2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn.datasets.samples_generator import make_blobs

x, y = make_blobs(n_samples=200, centers=4)
plt.scatter(x[:,0], x[:,1])

kmeans = KMeans(n_clusters=4)
kmeans.fit(x)

previsoes = kmeans.predict(x)
plt.scatter(x[:,0], x[:,1], c=previsoes)

'''
    Base cartoes de credito
'''

base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 20 - Agrupamento com k-means\\credit_card_clients.csv', header=1)
base['BILL_TOTAL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3'] + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6']

x = base.iloc[:, [1,2,3,4,5,25]].values
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Elbow method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)

# K Mean
kmeans = KMeans(n_clusters=4, random_state=0)
previsoes = kmeans.fit_predict(x)

plt.scatter(x[previsoes==0, 0], x[previsoes==0, 1], s=100, c='red', label='C1')
plt.scatter(x[previsoes==1, 0], x[previsoes==1, 1], s=100, c='black', label='C2')
plt.scatter(x[previsoes==2, 0], x[previsoes==2, 1], s=100, c='orange', label='C3')
plt.scatter(x[previsoes==3, 0], x[previsoes==3, 1], s=100, c='blue', label='C4')

lista_clientes = np.column_stack((base, previsoes))
lista_clientes = lista_clientes[lista_clientes[:,26].argsort()]