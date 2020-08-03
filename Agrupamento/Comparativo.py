import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import  datasets


x, y = datasets.make_moons(n_samples=1500, noise=0.09)
plt.scatter(x[:, 0], x[:, 1], s=5)

cores = np.array(['red', 'blue'])

# KMeans
kmeans = KMeans(n_clusters=2)
previsoes = kmeans.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], color = cores[previsoes])

# Hierarquico
hc = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
previsoes = hc.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], color = cores[previsoes])

# DBSCAN
dbscan = DBSCAN(eps=0.1)
previsoes = dbscan.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], color = cores[previsoes])