# Analisis de datos y Manejo
import pandas as pd
import numpy as np
import random as rnd

# Visualizacion
import seaborn as sns
import matplotlib.pyplot as plt

#%matplotlib inline


train_df = pd.read_csv('~/.kaggle/competitions/titanic/train.csv')
test_df = pd.read_csv('~/.kaggle/competitions/titanic/test.csv')
combine = [train_df, test_df]

# Describiendo la data
print(train_df.columns.values)

# Previsualizacion
print(train_df.head())
print(train_df.tail())

# Mostrando el tipo de dato de las caracteristicas
train_df.info()
print('_'*40)
test_df.info()


# Descriniendo los datos
print(train_df.describe())
print(train_df.describe(include=['O']))
