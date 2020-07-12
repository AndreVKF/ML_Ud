import pandas as pd

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from yellowbrick.regressor import ResidualsPlot

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
'''
    Base Plano de Saúde
'''
base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 15 - Outros tipos de regressao\\plano_saude2.csv')

# Regressão Linear
X = base.iloc[:, 0:1].values
y = base.iloc[:, 1].values

regressor1 = LinearRegression()
regressor1.fit(X, y)
score1 = regressor1.score(X, y)

plt.scatter(X, y)
plt.plot(X, regressor1.predict(X), color='r')

# Regressão Polinomial
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

regressor2 = LinearRegression()
regressor2.fit(X_poly, y)

score2 = regressor2.score(X_poly, y)

plt.scatter(X, y)
plt.plot(X, regressor2.predict(X_poly), color='r')