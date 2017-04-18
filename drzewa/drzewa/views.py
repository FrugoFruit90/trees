from sklearn.externals import joblib
from drzewa.drzewa.constants import SUBS_TYPE
from django.http import JsonResponse


def load_models():
    models = {}
    for mod_type in SUBS_TYPE:
        models[mod_type] = joblib.load('models/{}_xgb'.format(mod_type))
    return models


def give_predictions(request):
    models = load_models()
    predictions = {}
    for item in request.data:
        predictions[item.pk] = [model.predict(item.data) for model in models]
    return JsonResponse(data=predictions)
