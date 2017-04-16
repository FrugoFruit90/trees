import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.externals import joblib
from drzewa.drzewa.constants import SUBS_TYPE
# Load the data
data = pd.read_csv('data/trees.csv')
species_dummies = pd.get_dummies(data['gatunek'])

seed = 9001
y = data[SUBS_TYPE].fillna(0)
x = pd.concat([pd.get_dummies(data['gatunek']), data[['obwod', 'srednica_kor']]], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1)

xgb_r = xgb.XGBRegressor()
for metric in SUBS_TYPE:
    xgb_r.fit(x_train, np.array(y_train[metric]))
    print('R^2 dla lasu losowego wynosi {}% przy estymacji {}'.format(
        int(round(xgb_r.score(x_test, y_test[metric]), 2) * 100), metric))
    joblib.dump(xgb_r, 'drzewa/drzewa/models/{}_xgb'.format(metric))

