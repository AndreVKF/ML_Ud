# Mean Square Error (MSE)
# MSE = 1 / N * SUM(fi - yi)^2

# Design Matrix (Algebra Linear)
# Base de dados com poucos atributos
# Inversão de matrizes q tem um custo computacional alto

# Gradient Descent
# Desempenho melhor com muitos atributos
# min C(B1, B2,..., Bn)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from yellowbrick.regressor import ResidualsPlot

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
'''
    Base Plano de Saúde
'''
base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 14 - Regressao Linear\\plano_saude.csv')
x = base.iloc[:,0].values
y = base.iloc[:,1].values

correlacao = np.corrcoef(x, y)

x = x.reshape(-1, 1)
regressor = LinearRegression()
regressor.fit(x, y)

#b0
regressor.intercept_

#b1
regressor.coef_

# Plot
plt.scatter(x, y)
plt.plot(x, regressor.predict(x), color='r')
plt.title('Regressão Linear Simples')
plt.xlabel('Idade')
plt.ylabel('Custo')

previsao = regressor.predict(np.array([1]).reshape(-1, 1))

score = regressor.score(x, y)

# YellowBrick
visualizador = ResidualsPlot(regressor)
visualizador.fit(x, y)
visualizador.poof()

'''
    Base de preço das casas regressao linear
'''
base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 14 - Regressao Linear\\house_prices.csv')

# Apenas 1 argumento
x = base.iloc[:, 5:6].values
y = base.iloc[:, 2].values

# Data split
x_treinamento, x_teste, y_treinamento, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Linear Regression
regressor = LinearRegression()
regressor.fit(x_treinamento, y_treinamento)

# Plot
plt.scatter(x_treinamento, y_treinamento)
plt.plot(x_treinamento, regressor.predict(x_treinamento), color='r')

previsoes = regressor.predict(x_teste)
resultado = abs(y_test - previsoes)
resultado.mean()

# Metricas
mae = mean_absolute_error(y_test, previsoes)
mse = mean_squared_error(y_test, previsoes)

# Plot base de testes
plt.scatter(x_teste, y_test)
plt.plot(x_teste, regressor.predict(x_teste), color='r')

regressor.score(x_teste, y_test)

'''
    Base de preço das casas regressao linear multipla
'''
base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 14 - Regressao Linear\\house_prices.csv')

# Varios argumentos
x = base.iloc[:, 3:19].values
y = base.iloc[:, 2].values

# Data split
x_treinamento, x_teste, y_treinamento, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Linear Regression
regressor = LinearRegression()
regressor.fit(x_treinamento, y_treinamento)

# Proxy de correlação entre os argumentos
regressor.score(x_treinamento, y_treinamento)

# Previsoes
previsoes = regressor.predict(x_teste)

# Metricas
mae = mean_absolute_error(y_test, previsoes)
regressor.score(x_teste, y_test)

regressor.intercept_
len(regressor.coef_)
