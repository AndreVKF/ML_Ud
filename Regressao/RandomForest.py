# Random Forest
# Ensemble Learning (aprendizagem)
# Usa a média de várias regressão

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from yellowbrick.regressor import ResidualsPlot

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

'''
    Base Plano de Saude Pqna
'''
base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 15 - Outros tipos de regressao\\plano_saude2.csv')

x = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values

# Regressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x, y)
score = regressor.score(x, y)

# Teste
x_test = np.arange(min(x), max(x), 0.1)
x_test = x_test.reshape(-1, 1)

plt.scatter(x, y)
plt.plot(x_test, regressor.predict(x_test), color='r')

'''
    Base Housing Prices
'''
base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 14 - Regressao Linear\\house_prices.csv')

# Values
X = base.iloc[:, 3:19].values
y = base.iloc[:, 3].values

# Split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.3, random_state=0)

# Regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_treinamento, y_treinamento)
score = regressor.score(X_treinamento, y_treinamento)

previsoes = regressor.predict(X_teste)
score_teste = regressor.score(X_teste, y_teste)

# MAE
mae = mean_absolute_error(previsoes, y_teste)
