from sklearn.externals import joblib
from drzewa.drzewa.constants import SUBS_TYPE


def load_models():
    models = {}
    for mod_type in SUBS_TYPE:
        models[mod_type] = joblib.load('models/{}_xgb'.format(mod_type))
    return models

load_models()
print(load_models())
