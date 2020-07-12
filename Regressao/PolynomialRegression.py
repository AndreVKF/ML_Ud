import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from yellowbrick.regressor import ResidualsPlot

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import  PolynomialFeatures

'''
    Base de preço das casas regressao linear
'''
base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 14 - Regressao Linear\\house_prices.csv')

# Apenas 1 argumento
x = base.iloc[:, 3:19].values
y = base.iloc[:, 2].values

# Data split
x_treinamento, x_teste, y_treinamento, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Polynomial attributes
poly = PolynomialFeatures(degree=4)
x_treinamento_poly = poly.fit_transform(x_treinamento)
x_teste_poly = poly.transform(x_teste)

# Regressão linear
regressor = LinearRegression()
regressor.fit(x_treinamento_poly, y_treinamento)
previsoes = regressor.predict(x_teste_poly)

# Metrics
score = regressor.score(x_treinamento_poly, y_treinamento)
mae = mean_absolute_error(y_test, previsoes)


