import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 25 - Deteccao de outliers\\credit_data.csv')
base = base.dropna()

# Outliers idade
plt.boxplot(base.iloc[:, 2], showfliers=True)
outliers_age = base[(base.age<-20)]

# Outliers Loan
plt.boxplot(base.iloc[:, 3])
outliers_loan = base[(base.loan>13400)]