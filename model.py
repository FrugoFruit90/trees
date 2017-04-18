# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.externals import joblib
from drzewa.drzewa.constants import SUBS_TYPE
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv('data/trees.csv')

seed = 9001
y = data[SUBS_TYPE].fillna(0)
x = pd.concat([pd.get_dummies(data['gatunek']), data[['obwod', 'srednica_kor']]], axis=1)
# TODO: lepiej ciąć a potem pd.get_dummies(df, prefix=['col1', 'col2'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1)
x_circ_train, x_circ_test, y_circ_train, y_circ_test = train_test_split(data['obwod'], y, test_size=.1)

xgb_full = xgb.XGBRegressor()
lin_circ = xgb.XGBRegressor()

for metric in SUBS_TYPE:
    xgb_full.fit(x_train, np.array(y_train[metric]))
    print('R^2 dla XGB w pełnym modelu wynosi {}% przy estymacji {}'.format(
        int(round(xgb_full.score(x_test, y_test[metric]), 2) * 100), metric))
    joblib.dump(xgb_full, 'drzewa/drzewa/models/{}_full_xgb'.format(metric))

    lin_circ.fit(x_circ_train.values.reshape(-1, 1), np.array(y_circ_train[metric]))
    print('R^2 dla XGB w modelu obwodu wynosi {}% przy estymacji {}'.format(
        int(round(lin_circ.score(x_circ_test.values.reshape(-1, 1), y_circ_test[metric]), 2) * 100), metric))
    joblib.dump(lin_circ, 'drzewa/drzewa/models/{}_circ_xgb'.format(metric))
