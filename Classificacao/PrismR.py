# OneR
# Algoritmo gera 1 regra

# Testar coisas simples primeiro
# Um atributo faz todo o trabalho

# PrismR
# Regra gerada a partir de vários atributos

import Orange
from collections import Counter

'''
    Base de dados risco crédito
'''
base = Orange.data.Table('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 4 - Aprendizagem bayesiana\\risco_credito.csv')
base.domain

cn2_learner = Orange.classification.rules.CN2Learner()
classificador = cn2_learner(base)

for regras in classificador.rule_list:
    print(regras)

# Orange aceita atributos categóricos
resultado = classificador([['boa', 'alta', 'nenhuma', 'acima_35'], ['ruim', 'alta', 'adequada', '0_15']])

# Tag das classes
base.domain.class_var.values

for i in resultado:
    print(base.domain.class_var.values[i])

'''
    Base de dados risco de crédito completa
'''
base = Orange.data.Table('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 3 - Pre-processamento com Pandas e scikit-learm\\credit_data.csv')

base_dividida = Orange.evaluation.testing.sample(base, n=0.25)
base_treino = base_dividida[1]
base_teste = base_dividida[0]

# Algoritmo de regras
cn2_learner = Orange.classification.rules.CN2Learner()

# Classificador
classificador = cn2_learner(base_treino)

for regras in classificador.rule_list:
    print(regras)

resultado = Orange.evaluation.testing.TestOnTestData(base_treino, base_teste, [classificador])
Orange.evaluation.CA(resultado)

'''
    Majority Learner
'''
base = Orange.data.Table('C:\\Users\\André Viniciu\\Documents\\Python_ML\\Curso\\Secao 3 - Pre-processamento com Pandas e scikit-learm\\credit_data.csv')
base.domain

base_dividida = Orange.evaluation.testing.sample(base, n=0.25)
base_treino = base_dividida[1]
base_teste = base_dividida[0]

classificador = Orange.classification.MajorityLearner()
resultado = Orange.evaluation.testing.TestOnTestData(base_treino, base_teste, [classificador])
print(Orange.evaluation.CA(resultado))

# Base Line Classifier
print(Counter(str(d.get_class()) for d in base_teste))