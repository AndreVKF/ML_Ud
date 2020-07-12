import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from yellowbrick.regressor import ResidualsPlot

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

'''
    Base de dados plano de saude 2
'''
base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 15 - Outros tipos de regressao\\plano_saude2.csv')

# Base dados
x = base.iloc[0:, 0:1].values
y = base.iloc[0:, 1].values

# Regressor
regressor = DecisionTreeRegressor()
regressor.fit(x, y)
score = regressor.score(x, y)

# Plot
plt.scatter(x, y)
plt.plot(x, regressor.predict(x), color='r')

x_teste = np.arange(min(x), max(x), 0.1).reshape(-1, 1)
plt.scatter(x, y)
plt.plot(x_teste, regressor.predict(x_teste), color='r')

regressor.predict(np.array(40).reshape(-1, 1))

'''
    Base de dados Housing Prices
'''
base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 14 - Regressao Linear\\house_prices.csv')

x = base.iloc[:, 3:19].values
y = base.iloc[:, 2].values

# Data Split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size=0.3, random_state=0)

# Regressao
regressor = DecisionTreeRegressor()
regressor.fit(x_treinamento, y_treinamento)
score = regressor.score(x_treinamento, y_treinamento)

# Prediction
previsoes = regressor.predict(x_teste)

# Metricas
mae = mean_absolute_error(y_teste, previsoes)

# Score na base de teste
regressor.score(x_teste, y_teste)
