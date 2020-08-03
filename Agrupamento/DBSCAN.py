# DBSCAN

# Density-Based Spatial Clustering of Application with Noise
# Baseado em densidade
# Não é necessário especificar o número de clusters
# Em geral apresenta melhores resultados que o K-Mean

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 21 - Agrupamento hierarquico\\credit_card_clients.csv', header=1)
base['BILL_TOTAL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3'] + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6'] + base['BILL_AMT1'] 

x = base.iloc[:, [1, 25]].values
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Agrupamento
dbscan = DBSCAN(eps=0.37, min_samples=4)
previsoes = dbscan.fit_predict(x)

unicos, quantidade = np.unique(previsoes, return_counts=True)

plt.scatter(x[previsoes==0, 0], x[previsoes==0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(x[previsoes==1, 0], x[previsoes==1, 1], s=100, c='orange', label='Cluster 1')
plt.scatter(x[previsoes==2, 0], x[previsoes==2, 1], s=100, c='green', label='Cluster 1')

lista_clientes = np.column_stack((base, previsoes))
lista_clientes = lista_clientes[lista_clientes[:, 26].argsort()]