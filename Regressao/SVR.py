# SVR (Support Vector Regression)
# Mantém características das SVM para classificação
# Mais difícil de fazer as previsões por se tratar de números (muitas possibilidades)
# Parâmetro epsilon (Penalidade do treinamento, distância para o valor real)

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

'''
    Base Plano de Saude Pqna
'''
base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 15 - Outros tipos de regressao\\plano_saude2.csv')

x = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values

### Kernal Linear ###
regressor = SVR(kernel='linear')
regressor.fit(x, y)
score = regressor.score(x, y)

### Polynomial Kernel ###
regressor = SVR(kernel='poly', degree=3)
regressor.fit(x, y)
score = regressor.score(x, y)

### RBF Kernel ###
# Scaling parameter for regression #
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1))

regressor = SVR(kernel='rbf')
regressor.fit(x, y)
score = regressor.score(x, y)

# Plot
plt.scatter(x, y)
plt.plot(x, regressor.predict(x), color='r')

previsao1 = regressor.predict(scaler_x.transform(np.array(40).reshape(-1, 1)))
scaler_y.inverse_transform(previsao1)

'''
    Base Housing Prices
'''

base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 14 - Regressao Linear\\house_prices.csv')

# Values
X = base.iloc[:, 3:19].values
y = base.iloc[:, 3].values

# Escalonamento
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1))

# Split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.3, random_state=0)

# Regressor
regressor = SVR(kernel='rbf')
regressor.fit(X_treinamento, y_treinamento)
regressor.score(X_treinamento, y_treinamento)
regressor.score(X_teste, y_teste)
