import matplotlib.pyplot as plt
import pandas as pd

base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 25 - Deteccao de outliers\\credit_data.csv')
base = base.dropna()

base.loc[base.age<0, 'age'] = 40.92

# income x age
plt.scatter(base.iloc[:, 1], base.iloc[:, 2])

# income x loan
plt.scatter(base.iloc[:, 1], base.iloc[:, 3])

# age x loan
plt.scatter(base.iloc[:, 2], base.iloc[:, 3])

base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 25 - Deteccao de outliers\\census.csv')

# age x final weight
plt.scatter(base.iloc[:, 0], base.iloc[:, 2])