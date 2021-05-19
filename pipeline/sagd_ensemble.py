# @Author: Shounak Ray <Ray>
# @Date:   15-May-2021 01:05:99:993  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: sagd_ensemble.py
# @Last modified by:   Ray
# @Last modified time: 19-May-2021 15:05:54:544  GMT-0600
# @License: [Private IP]


import copy
import random
from math import sqrt

import numpy as np
import pandas as pd
# from xgboost import XGBRegressor
from mlxtend.regressor import StackingRegressor
from sklearn import ensemble
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor,
                              ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import (Lasso, LinearRegression, LogisticRegression,
                                  Ridge)
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import make_pipeline, make_union

random.seed("WW")

gb1 = GradientBoostingRegressor(n_estimators=50, random_state=1)
xt1 = ExtraTreesRegressor(bootstrap=True, n_estimators=50, random_state=1)
rf1 = RandomForestRegressor(bootstrap=True, n_estimators=50, random_state=1)
xt2 = ExtraTreesRegressor(bootstrap=False, n_estimators=50, random_state=1)
rf2 = RandomForestRegressor(bootstrap=False, n_estimators=50, random_state=1)
# xg1 = XGBRegressor(bootstrap=True, n_estimators=200,random_state=1)
ad1 = AdaBoostRegressor(n_estimators=50, random_state=1)
br1 = BaggingRegressor(n_estimators=50, random_state=1)

lm1 = LinearRegression()
lso = Lasso(random_state=1)
rg1 = Ridge(random_state=1)

kn1 = KNeighborsRegressor()


# all_models = [xt1, rf1, xg1]
all_models = [xt1, rf1, br1]
ww_ensemble = StackingRegressor(regressors=all_models, meta_regressor=LinearRegression())

# from ml_dataprep import well_pairs, prep_well_data
# w = 'A3_A4'
# train_x, train_y, test_x, test_y, test_dates = prep_well_data(w, d=92)


def evaluate_model(model, train_x, train_y, test_x, test_y):

    modname = str(model).split("egressor")[0]
    if modname == "Stacking":
        modname = "WWEnsemble"

    model.fit(train_x, train_y)
    results = model.predict(test_x)

    # print("RESULTS:", results)
    # print("TEST_Y", test_y)

    acc = [a for a in 1 - np.abs((results - test_y) / test_y) if a > 0 and a < 1]

    # print("ACCURACY:", np.mean(acc))
    # acc = [a for a in acc if np.isinf(a) == False]
    acc = np.mean(acc)

    rmse = sqrt(mean_squared_error(results, test_y))

    outcomes = pd.DataFrame({
        "predicted": results,
        "actual": test_y,
        "algorithm_name": modname[:modname.find("(")]})

    metrics = pd.DataFrame([{
        "algorithm_name": modname[:modname.find("(")],
        "accuracy":  acc,
        "RMSE":      rmse}])

    return(outcomes, metrics, model)


def genetic_ensemble(train_x, train_y, test_x, test_y):

    models_outputs = pd.DataFrame()
    metric_outputs = pd.DataFrame()

    # all_models = [gb1, rf1, rf2, ad1, br1, lm1, lso, rg1, xt1, xt2]
    all_models = [gb1, rf1, rf2, ad1, br1, xt1, xt2]

    for model in all_models:

        outcomes, metrics, mod = evaluate_model(model, train_x, train_y, test_x, test_y)
        models_outputs = models_outputs.append(outcomes)
        metric_outputs = metric_outputs.append(metrics)

    metric_outputs['algos'] = all_models
    finalists = metric_outputs.sort_values('RMSE')['algos']

    ensemble_outputs = pd.DataFrame()

    for n in range(1, 7):
        # print("N:", n)
        final_models = list(finalists)[:n]
        ensemble = StackingRegressor(regressors=final_models, meta_regressor=LinearRegression())
        outcomes, metrics, m = evaluate_model(ensemble, train_x, train_y, test_x, test_y)
        metrics['algos'] = copy.copy(m)
        metrics['algorithm_name'] = "WW_GeneticEnsemble_" + str(n) + str(id(ensemble))
        ensemble_outputs = ensemble_outputs.append(metrics)

    ensemble_outputs = ensemble_outputs.sort_values('RMSE')[:1]

    metric_outputs = metric_outputs.append(ensemble_outputs)
    metric_outputs = metric_outputs.sort_values('RMSE')

    final_model = copy.copy(list(metric_outputs['algos'])[0])
    metric_outputs = metric_outputs.drop('algos', axis=1)

    # print("MODEL METRICS")
    # print(metric_outputs)

    # print("FINAL MODEL")
    # print(final_model)

    return(models_outputs, metric_outputs, final_model)


def important_features(train_x, train_y, top_n=99):

    rf1.fit(train_x.values.astype(float), train_y.values)

    features_train = list(train_x.columns)
    imp = pd.DataFrame([features_train, list(rf1.feature_importances_)]).transpose()
    imp.columns = ['variable', 'importance']
    imp = imp.sort_values('importance', ascending=True)[0:top_n]

    return(imp)


#
# Class GeneticEnsembleRegressor():
#
#     def __init__(self):
#
#         self.models_outputs = pd.DataFrame()
#         self.metric_outputs = pd.DataFrame()
#         self.all_models = [gb1, rf1, rf2, ad1, br1, lm1, lso, rg1, xt1, xt2]
#
#
#     def train(self, train_x, train_y, test_x, test_y):
#
#         for model in self.all_models:
#
#             outcomes, metrics, mod = evaluate_model(model, train_x, train_y, test_x, test_y)
#             self.models_outputs = self.models_outputs.append(outcomes)
#             self.metric_outputs = self.metric_outputs.append(metrics)
#
#         self.metric_outputs['algos'] = self.all_models
#         self.finalists = self.metric_outputs.sort_values('RMSE')['algos']
#
#         self.ensemble_outputs = pd.DataFrame()
#
#         for n in range(1,7):
#             # print("N:", n)
#             self.final_models = self.finalists[:n]
#             self.ensemble = StackingRegressor(regressors=self.final_models, meta_regressor= LinearRegression())
#             outcomes, metrics, m = evaluate_model(self.ensemble, train_x, train_y, test_x, test_y)
#             metrics['algos'] = m.copy()
#             metrics['algorithm'] = "WW_GeneticEnsemble_" + str(n) + str(id(self.ensemble))
#             ensemble_outputs = self.ensemble_outputs.append(metrics)
#
#         self.ensemble_outputs = self.ensemble_outputs.sort_values('RMSE')[:1]
#
#         self.metric_outputs = self.metric_outputs.append(ensemble_outputs)
#         self.metric_outputs = self.metric_outputs.sort_values('RMSE')
#
#         self.final_model = list(self.metric_outputs['algos'])[0]
#         self.metric_outputs = self.metric_outputs.drop('algos', axis=1)
#
#     # print("MODEL METRICS")
#     # print(metric_outputs)
#
#     # print("FINAL MODEL")
#     # print(final_model)
#


#
