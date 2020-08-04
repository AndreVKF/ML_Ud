import pandas as pd

from pyod.models.knn import KNN

base = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 25 - Deteccao de outliers\\credit_data.csv')
base = base.dropna()

detector = KNN()
detector.fit(base.iloc[:, 1:4])

confianca_previsoes = detector.decision_scores_

outliers = []

for i in range(len(previsoes)):
    print(previsoes[i])