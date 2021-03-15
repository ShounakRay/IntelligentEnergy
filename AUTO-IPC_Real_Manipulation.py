# @Author: Shounak Ray <Ray>
# @Date:   13-Mar-2021 10:03:01:019  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: AUTO-IPC_Real_Manipulation.py
# @Last modified by:   Ray
# @Last modified time: 15-Mar-2021 17:03:28:286  GMT-0600
# @License: [Private IP]


# G0TO: CTRL + OPTION + G
# SELECT USAGES: CTRL + OPTION + U
# DOCBLOCK: CTRL + SHIFT + C
# GIT COMMIT: CMD + ENTER
# GIT PUSH: CMD + U P
# PASTE IMAGE: CTRL + OPTION + SHIFT + V
# Todo: Opt + K  Opt + T

import math
import os
import re
import sys
from collections import Counter
from functools import reduce
from pathlib import Path

import featuretools as ft
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from autofeat import AutoFeatRegressor, FeatureSelector
from matplotlib.patches import Rectangle
from sklearn.datasets import load_boston, load_diabetes
from sklearn.metrics import (explained_variance_score, max_error,
                             mean_absolute_error, mean_squared_error,
                             median_absolute_error, r2_score)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# from itertools import chain
# import timeit
# from functools import reduce
# import matplotlib.cm as cm
# from matplotlib.backends.backend_pdf import PdfPages
# import pickle
# import pandas_profiling
# !{sys.executable} -m pip install pandas-profiling

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

"""All Underlying Datasets
FIBER_DATA              --> Temperature/Distance data along production lines
DATA_INJECTION_STEAM    --> Metered Steam at Injection Sites
DATA_INJECTION_PRESS    --> Pressure at Injection Sites
DATA_PRODUCTION         --> Production Well Sensors
DATA_TEST               --> Oil, Water, Gas, and Fluid from Production Wells
FINALE                  --> Join of PRODUCTION_WELL_WSENSOR
                                                    and DATA_INJECTION_STEAM
"""

# HYPER-PARAMETERS
BINS = 5
FIG_SIZE = (220, 7)
DIR_EXISTS = Path('Data/Pickles').is_dir()


def correlation_matrix(df, FPATH, EXP_NAME, abs_arg=True, mask=True, annot=False,
                       type_corrs=['Pearson', 'Kendall', 'Spearman'],
                       cmap=sns.color_palette('flare', as_cmap=True), figsize=(24, 8), contrast_factor=1.0):
    """Outputs to console and saves to file correlation matrices given data with input features.
    Intended to represent the parameter space and different relationships within. Tool for modeling.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Input dataframe where each column is a feature that is to be correlated.
    FPATH : str
        Where the file should be saved. If directory is provided, it should already be created.
    abs_arg : bool
        Whether the absolute value of the correlations should be plotted (for strength magnitude).
        Impacts cmap, switches to diverging instead of sequential.
    mask : bool
        Whether only the bottom left triange of the matrix should be rendered.
    annot : bool
        Whether each cell should be annotated with its numerical value.
    type_corrs : list
        All the different correlations that should be provided. Default to all built-in pandas options for df.corr().
    cmap : child of matplotlib.colors
        The color map which should be used for the heatmap. Dependent on abs_arg.
    figsize : tuple
        The size of the outputted/saved heatmap (width, height).
    contrast_factor:
        The factor/exponent by which all the correlationed should be raised to the power of.
        Purpose is for high-contrast, better identification of highly correlated features.

    Returns
    -------
    pandas.core.frame.DataFrame
        The correlations given the input data, and other transformation arguments (abs_arg, contrast_factor)
        Furthermore, heatmap is saved in specifid format and printed to console.

    """
    input_data = {}

    # NOTE: Conditional assignment of sns.heatmap based on parameters
    # > Mask sns.heatmap parameter is conditionally controlled
    # > If absolute value is not chosen by the user, switch to divergent heatmap. Otherwise keep it as sequential.
    fig, ax = plt.subplots(ncols=len(type_corrs), sharey=True, figsize=figsize)
    for typec in type_corrs:
        input_data[typec] = (df.corr(typec.lower()).abs()**contrast_factor if abs_arg
                             else df.corr(typec.lower())**contrast_factor)
        sns_fig = sns.heatmap(input_data[typec],
                              mask=np.triu(df.corr().abs()) if mask else None,
                              ax=ax[type_corrs.index(typec)],
                              annot=annot,
                              cmap=cmap if abs_arg else sns.diverging_palette(240, 10, n=9, as_cmap=True)
                              ).set_title("{cortype}'s Correlation Matrix\n{EXP_NAME}".format(cortype=typec,
                                                                                              EXP_NAME=EXP_NAME))
    plt.tight_layout()
    sns_fig.get_figure().savefig(FPATH, bbox_inches='tight')

    plt.clf()

    return input_data


def performance(actual_Y: np.ndarray, predicted_Y: np.ndarray, slicedata='given'):
    metrics = {'Mean Squared Error (MSE) (units_y^squared)': mean_squared_error(actual_Y, predicted_Y, squared=True),
               'Root Mean Square Error (RMSE) (units_y)': mean_squared_error(actual_Y, predicted_Y, squared=False),
               'Coefficient of Determination (r^2) (dimensionless)': r2_score(actual_Y, predicted_Y),
               'Correlation Coefficient (r) (dimensionless)': (r2_score(actual_Y, predicted_Y)**0.5),
               'Explained Variance Score (EVS) (dimensionless)': explained_variance_score(actual_Y, predicted_Y),
               'Mean Absolute Error (Mean AE) (units_y)': mean_absolute_error(actual_Y, predicted_Y),
               'Median Absolute Error (Median AE) (units_y)': median_absolute_error(actual_Y, predicted_Y),
               'Maximum Residual Error (MAX RE) (units_y)': max_error(actual_Y, predicted_Y)}
    net = [re.findall(r'\((.*?)\)', metric) for metric in metrics]

    metrics = pd.DataFrame(net + [metrics.values()], index=metrics.keys(), columns=['Abbreviation, Units, Score'])

    print(metrics)

    return metrics


def rect_heatmap(rect_posx, hmap_data, title='Heatmap'):
    plt.subplots(figsize=(5, 20))
    hmap = sns.heatmap(hmap_data, annot=False)
    for p in rect_posx:
        hmap.add_patch(Rectangle((0, p), len(key_features), 1, edgecolor='blue', fill=False, lw=3))
    hmap.set_title(title)

    return hmap

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# > DATA INGESTION (Load of Pickle if available)
# Folder Specifications
if(DIR_EXISTS):
    DATA_INJECTION_ORIG = pd.read_pickle('Data/Pickles/DATA_INJECTION_ORIG.pkl')
    DATA_PRODUCTION_ORIG = pd.read_pickle('Data/Pickles/DATA_PRODUCTION_ORIG.pkl')
    DATA_TEST_ORIG = pd.read_pickle('Data/Pickles/DATA_TEST_ORIG.pkl')
    FIBER_DATA = pd.read_pickle('Data/Pickles/FIBER_DATA.pkl')
    DATA_INJECTION_STEAM = pd.read_pickle('Data/Pickles/DATA_INJECTION_STEAM.pkl')
    DATA_INJECTION_PRESS = pd.read_pickle('Data/Pickles/DATA_INJECTION_PRESS.pkl')
    DATA_PRODUCTION = pd.read_pickle('Data/Pickles/DATA_PRODUCTION.pkl')
    DATA_TEST = pd.read_pickle('Data/Pickles/DATA_TEST.pkl')
    PRODUCTION_WELL_INTER = pd.read_pickle('Data/Pickles/PRODUCTION_WELL_INTER.pkl')
    PRODUCTION_WELL_WSENSOR = pd.read_pickle('Data/Pickles/PRODUCTION_WELL_WSENSOR.pkl')
    FINALE = pd.read_pickle('Data/Pickles/FINALE.pkl')
else:
    __FOLDER__ = r'Data/Isolated/'
    PATH_INJECTION = __FOLDER__ + r'OLT injection data.xlsx'
    PATH_PRODUCTION = __FOLDER__ + r'OLT production data (rev 1).xlsx'
    PATH_TEST = __FOLDER__ + r'OLT well test data.xlsx'
    # Data Imports
    DATA_INJECTION_ORIG = pd.read_excel(PATH_INJECTION)
    DATA_PRODUCTION_ORIG = pd.read_excel(PATH_PRODUCTION)
    DATA_TEST_ORIG = pd.read_excel(PATH_TEST)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

model_feateng = AutoFeatRegressor(feateng_steps=2, verbose=3)
model_featsel = FeatureSelector(verbose=6)
train_pct = 0.8
TARGET = 'Heel_Pressure'
key_features = ['Bin_1', 'Bin_2', 'Bin_3', 'Bin_4', 'Bin_5']

FINALE = FINALE.fillna(0.000000001).replace(np.nan, 0.000000001)

for c in FINALE.columns:
    print(c, sum(FINALE[c].isna()))

# Ultimate Goal
# FINALE = FINALE[FINALE['test_flag'] == True].reset_index(drop=True)
# TARGET = 'Oil'

# FINALE_FILTERED = FINALE[[c for c in FINALE.columns if c not in ['Date', 'unique_id', 'Pad', 'test_flag',
# '24_Fluid',  '24_Oil', '24_Water', 'Oil', 'Water',
# 'Gas', 'Fluid', 'Toe_Pressure', 'Pump_Speed',
# 'Casing_Pressure', 'Tubing_Pressure', 'Heel_Temp',
# 'Toe_Temp', 'Bin_1', 'Bin_2', 'Bin_3', 'Bin_4',
# 'Bin_5']]].replace(0.0, 0.0000001)
FINALE_FILTERED = FINALE[[c for c in FINALE.columns if c not in ['Date', 'unique_id', 'Pad', 'test_flag',
                                                                 '24_Fluid',  '24_Oil', '24_Water', 'Oil', 'Water',
                                                                 'Gas', 'Fluid', 'Toe_Pressure', 'Pump_Speed',
                                                                 'Casing_Pressure', 'Tubing_Pressure', 'Heel_Temp',
                                                                 'Toe_Temp', 'Bin_1', 'Bin_2', 'Bin_3', 'Bin_4', 'Bin_5']]
                         ].replace(0.0, 0.0000001)
FINALE_FILTERED_wbins = FINALE[[c for c in FINALE.columns if c not in ['Date', 'unique_id', 'Pad', 'test_flag',
                                                                       '24_Fluid',  '24_Oil', '24_Water', 'Oil', 'Water',
                                                                       'Gas', 'Fluid', 'Toe_Pressure', 'Pump_Speed',
                                                                       'Casing_Pressure', 'Tubing_Pressure', 'Heel_Temp',
                                                                       'Toe_Temp', 'Heel_Pressure']]
                               ].replace(0.0, 0.0000001)


# list(FINALE_FILTERED.columns)
# FINALE_FILTERED = FINALE_FILTERED.astype(np.float64)

FINALE_FILTERED_ULTIMATE = FINALE[[c for c in FINALE.columns if c not in ['Date', 'unique_id', 'Pad', 'test_flag',
                                                                          '24_Fluid',  '24_Oil', '24_Water', 'Water',
                                                                          'Gas', 'Fluid', 'Toe_Pressure', 'Pump_Speed',
                                                                          'Casing_Pressure', 'Tubing_Pressure', 'Heel_Temp',
                                                                          'Toe_Temp', 'Bin_1', 'Bin_2', 'Heel_Pressure',
                                                                          'Bin_3', 'Bin_4', 'Bin_5']]
                                  ].replace(0.0, 0.0000001)

# aplt.hist(FINALE_FILTERED_ULTIMATE['Oil'], bins=100)

# FINALE_FILTERED = FINALE_FILTERED_ULTIMATE.copy()
# well = FINALE_FILTERED['Well'].unique()[0]
# Split into source datasets for each production well
for well in FINALE_FILTERED['Well'].unique():
    SOURCE = FINALE_FILTERED[FINALE_FILTERED['Well'] == well]
    SOURCE.drop(['Well'], axis=1, inplace=True)
    SOURCE.reset_index(drop=True, inplace=True)
    SOURCE = SOURCE.rolling(window=7).mean().fillna(method='bfill').fillna(method='ffill')

    SOURCE_wbins = FINALE_FILTERED_wbins[FINALE_FILTERED_wbins['Well'] == well]
    SOURCE_wbins.drop(['Well'], axis=1, inplace=True)
    SOURCE_wbins.reset_index(drop=True, inplace=True)
    SOURCE_wbins = SOURCE_wbins.rolling(window=7).mean().fillna(method='bfill').fillna(method='ffill')

    msk = np.random.rand(len(SOURCE)) < train_pct

    # Variable Selection (NOT Engineering)
    new_X = model_featsel.fit_transform(SOURCE[[c for c in SOURCE.columns if c != TARGET]],
                                        SOURCE[TARGET])
    filtered_features = new_X.columns

    correlated_df = SOURCE_wbins.corr()

    plt.subplots(figsize=(5, 20))
    final_data = correlated_df.abs()[key_features].dropna().drop(key_features[:6])
    pos = [list(final_data.index).index(val) for val in list(filtered_features)]

    hmap = sns.heatmap(final_data, annot=False)
    for p in pos:
        hmap.add_patch(Rectangle((0, p), len(key_features), 1, edgecolor='blue', fill=False, lw=3))
    hmap.set_title('Well ' + well)
    plt.tight_layout()
    hmap.get_figure().savefig('WELL-{WELL}_TARGET-{TARGET}.pdf'.format(WELL=well, TARGET=TARGET), bbox_inches='tight')

    # # >>> THIS IS THE REGRESSION PART
    # # Length filtering, no column filtering
    # # Filter training and testing sets to only include the selected features
    # TRAIN = SOURCE[msk][filtered_features.union([TARGET])].reset_index(drop=True)
    # TEST = SOURCE[~msk][filtered_features.union([TARGET])].reset_index(drop=True)
    #
    # X_TRAIN = TRAIN[[c for c in TRAIN.columns if c != TARGET]]
    # Y_TRAIN = pd.DataFrame(TRAIN[TARGET])
    #
    # X_TEST = TEST[[c for c in TEST.columns if c != TARGET]]
    # Y_TEST = pd.DataFrame(TEST[TARGET])
    #
    # feateng_X_TRAIN = model_feateng.fit_transform(X_TRAIN, Y_TRAIN)
    # feating_X_TEST = model_feateng.transform(X_TEST)
    #
    # # model_feateng.score(X_TRAIN, Y_TRAIN)
    # # model_feateng.new_feat_cols_
    # # plt.scatter(model_feateng.predict(X_TEST), Y_TEST[TARGET], s=2)
    # # model_feateng.score(X_TEST, Y_TEST)
    #
    # # performance(Y_TRAIN, model_feateng.predict(feateng_X_TRAIN))

# plt.plot(SOURCE['I08'])
#
# plt.plot(SOURCE['I08'] / SOURCE['I08'].quantile(0.92))
# plt.plot(SOURCE['I30'] / SOURCE['I30'].quantile(0.92))
# plt.plot(SOURCE['I10'] / SOURCE['I10'].quantile(0.92))
# plt.plot(SOURCE['I43'] / SOURCE['I43'].quantile(0.92))
# plt.plot(SOURCE['I29'] / SOURCE['I29'].quantile(0.92))
# plt.plot(SOURCE['I56'] / SOURCE['I56'].quantile(0.92))
#
# plt.plot(SOURCE['I32'] / SOURCE['I32'].quantile(0.92))
# SOURCE.corr()
# plt.plot(SOURCE['I29'] / SOURCE['I29'].quantile(0.92))
#
# plt.plot(SOURCE['I10'] / SOURCE['I10'].quantile(0.92))
# plt.plot(SOURCE[TARGET])

# _ = plt.hist(Y_TRAIN, bins=90)
# _ = plt.hist(model_feateng.predict(feateng_X_TRAIN), bins=90)

# Diverging: sns.diverging_palette(240, 10, n=9, as_cmap=True)
# https://seaborn.pydata.org/generated/seaborn.color_palette.html

# _ = correlation_matrix(SOURCE, 'Modeling.pdf', 'test', mask=False)


# dict(SOURCE.corr()['Heel_Pressure'].dropna().sort_values(ascending=False))

# model_metrics = performance(Y_TRAIN, model_feateng.predict(feateng_X_TRAIN))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


X, y = load_boston(True)
pd.DataFrame(X)

afreg = AutoFeatRegressor(verbose=1, feateng_steps=2)
# fit autofeat on less data, otherwise ridge reg model_feateng with xval will overfit on new features
X_train_tr = afreg.fit_transform(X[:480], y[:480])
X_test_tr = afreg.transform(X[480:])
print("autofeat new features:", len(afreg.new_feat_cols_))
print("autofeat MSE on training data:", mean_squared_error(pd.DataFrame(y[:480]), afreg.predict(X_train_tr)))
print("autofeat MSE on test data:", mean_squared_error(y[480:], afreg.predict(X_test_tr)))
print("autofeat R^2 on training data:", r2_score(y[:480], afreg.predict(X_train_tr)))
print("autofeat R^2 on test data:", r2_score(y[480:], afreg.predict(X_test_tr)))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

es = ft.EntitySet(id='ipc_entityset')

es = es.entity_from_dataframe(entity_id='FINALE', dataframe=FINALE,
                              index='unique_id', time_index='Date')

# Default primitives from featuretools
default_agg_primitives = ft.list_primitives()[(ft.list_primitives()['type'] == 'aggregation') &
                                              (ft.list_primitives()['valid_inputs'] == 'Numeric')
                                              ]['name'].to_list()
default_trans_primitives = [op for op in ft.list_primitives()[(ft.list_primitives()['type'] == 'transform') &
                                                              (ft.list_primitives()['valid_inputs'] == 'Numeric')
                                                              ]['name'].to_list()[:2]
                            if op not in ['scalar_subtract_numeric_feature']]

# DFS with specified primitives
feature_names = ft.dfs(entityset=es, target_entity='FINALE',
                       trans_primitives=default_trans_primitives,
                       agg_primitives=None,
                       max_depth=2,
                       # n_jobs=-1,
                       features_only=True)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Create Entity Set for this project
es = ft.EntitySet(id='ipc_entityset')

FIBER_DATA.reset_index(inplace=True)
DATA_INJECTION_STEAM.reset_index(inplace=True)
DATA_INJECTION_PRESS.reset_index(inplace=True)
DATA_PRODUCTION.reset_index(inplace=True)
DATA_TEST.reset_index(inplace=True)

# Create an entity from the client dataframe
# This dataframe already has an index and a time index
es = es.entity_from_dataframe(entity_id='DATA_INJECTION_STEAM', dataframe=DATA_INJECTION_STEAM.copy(),
                              make_index=True, index='id_injector', time_index='Date')
es = es.entity_from_dataframe(entity_id='DATA_INJECTION_PRESS', dataframe=DATA_INJECTION_PRESS.copy(),
                              make_index=True, index='id_injector', time_index='Date')

es = es.entity_from_dataframe(entity_id='DATA_PRODUCTION', dataframe=DATA_PRODUCTION.copy(),
                              make_index=True, index='id_production', time_index='Date')
es = es.entity_from_dataframe(entity_id='DATA_TEST', dataframe=DATA_TEST.copy(),
                              make_index=True, index='id_production', time_index='Date')
es = es.entity_from_dataframe(entity_id='FIBER_DATA', dataframe=FIBER_DATA.copy(),
                              make_index=True, index='id_production', time_index='Date')

rel_producer = ft.Relationship(es['DATA_PRODUCTION']['id_production'],
                               es['DATA_TEST']['id_production'])
es.add_relationship(rel_producer)

# Add relationships for all columns between injection steam and pressure data
injector_sensor_cols = [c for c in DATA_INJECTION_STEAM.columns.copy() if c not in ['Date']]
injector_sensor_cols = ['CI06']
for common_col in injector_sensor_cols:
    rel_injector = ft.Relationship(es['DATA_INJECTION_STEAM']['id_injector'],
                                   es['DATA_INJECTION_PRESS']['id_injector'])
    es = es.add_relationship(rel_injector)

es['DATA_INJECTION_STEAM']

# EOF
# EOF
