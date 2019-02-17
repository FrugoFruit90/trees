# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from drzewa.drzewa.constants import SUBS_TYPE
from sklearn.linear_model import LinearRegression


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Load the data
data = pd.read_csv('data/trees.csv', sep=';')

seed = 9001
y = data[SUBS_TYPE].fillna(0)
x = pd.concat([pd.get_dummies(data['GATUNEK_Scientific']), data[['OBWOD', 'SREDNICA_KORONY']]], axis=1)
# TODO: lepiej ciąć a potem pd.get_dummies(df, prefix=['col1', 'col2'])

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1) # use this for all data
x_train, x_test, y_train, y_test = train_test_split(data[['OBWOD', 'SREDNICA_KORONY']], y, test_size=.1)

lr = LinearRegression()

for metric in SUBS_TYPE:
    lr.fit(x_test, np.array(y_test[metric]))
    R_2 = int(round(lr.score(x_test, y_test[metric]), 2) * 100)
    mape = mean_absolute_percentage_error(lr.predict(x_test), y_test[metric])
    print(f'Przy estymacji {metric} R^2 wynosi {R_2}%, a średni procentowy błąd - {int(round(mape, 2))}')

    print(f'Parametry modelu: {lr.coef_, lr.intercept_} - odpowiednio obwód, średnica korony, i wyraz wolny')
