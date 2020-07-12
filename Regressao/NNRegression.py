# Regressao utilizando Redes Neurais
# 1 neurônio na camada de saída
# Função de ativação Linear

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
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
'''
    Base Plano de Saude Pqna
'''
base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 15 - Outros tipos de regressao\\plano_saude2.csv')

x = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values

# Escalonamento
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1))

# Regressor
regressor = MLPRegressor()
regressor.fit(x, y)
regressor.score(x, y)

# Plot
plt.scatter(x, y)
plt.plot(x, regressor.predict(x), color='r')

'''
    Housing Prices
'''
base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 14 - Regressao Linear\\house_prices.csv')

# Values
x = base.iloc[:, 3:19].values
y = base.iloc[:, 3].values

# Escalonamento
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1))

# Split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size=0.3, random_state=0)

# Regressor
regressor = MLPRegressor(hidden_layer_sizes=(9, 9, 9), verbose=True)
regressor.fit(x_treinamento, y_treinamento)

regressor.score(x_treinamento, y_treinamento)
regressor.score(x_teste, y_teste)

previsoes = scaler_y.inverse_transform(regressor._predict(x_teste))

