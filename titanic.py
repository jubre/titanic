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


# Limpiando caracteristica Nombre y creando la caracteristica Title
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

print(pd.crosstab(train_df['Title'], train_df['Sex']))

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

print(train_df.head())

# Eliminamos la caracteristica Name
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape
