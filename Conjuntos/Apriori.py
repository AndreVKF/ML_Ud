# Fase 1
# Descobrir todos os conjuntos de itens com suporte maior ou igual ao mínimo especificaod pelo usuário
# Fase 2
# A partir dos conjuntos de itens frequentes, descobrir as regras de associação com fator de confiança maior ou igual ao especificado pelo usuário

import pandas as pd

from apyori import apriori

'''
    Base dados pequena
'''
dados = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 17 - Algoritmo apriori\\mercado.csv', header=None)
transacoes = []

for i in range(0, 10):
    transacoes.append([str(dados.values[i, j]) for j in range(0, 4)])

regras = apriori(transacoes, min_support=0.003, min_confidence=0.2, min_lift=3, min_lenght=2)
resultados = list(regras)

resultados2 = [list(x) for x in resultados]

resultadoFormatado = []
for j in range(0, 3):
    resultadoFormatado.append([list(x) for x in resultados2[j][2]])

'''
    Base de dados maior
'''

dados = pd.read_csv('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 17 - Algoritmo apriori\\mercado2.csv', header=None)

transacoes = []

for i in range(0, 7501):
    transacoes.append([str(dados.values[i, j]) for j in range(0, 20)])

regras = apriori(transacoes, min_support=0.003, min_confidence=0.2, min_lift=3, min_lenght=2)

resultados = list(regras)

resultados2 = [list(x) for x in resultados]

resultadoFormatado = []
for j in range(0, 5):
    resultadoFormatado.append([list(x) for x in resultados2[j][2]])