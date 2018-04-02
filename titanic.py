# Analisis de datos y Manejo
import pandas as pd
import numpy as np
import random as rnd

# Visualizacion
import seaborn as sns
import matplotlib.pyplot as plt

# %matplotlib inline


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

# Analizando mediante Pivote de caracteristicas
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# Analisis mediante visualizacion de las caracteristicas
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
