# Agrupamento hierárquico

# Estrutura em formato de árvore q indica o número de clusters
# Abordagem aglomerativa: cada registro pertence ao seu próprio cluster e pares de cluster são unidos

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 21 - Agrupamento hierarquico\\credit_card_clients.csv', header=1)
base['BILL_TOTAL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3'] + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6'] + base['BILL_AMT1'] 

x = base.iloc[:, [1, 25]].values
scaler = StandardScaler()
x = scaler.fit_transform(x)

dendrograma = dendrogram(linkage(x, method='ward'))

hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
previsoes = hc.fit_predict(x)

plt.scatter(x[previsoes==0, 0], x[previsoes==0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(x[previsoes==1, 0], x[previsoes==1, 1], s=100, c='red', label='Cluster 2')
plt.scatter(x[previsoes==2, 0], x[previsoes==2, 1], s=100, c='red', label='Cluster 3')