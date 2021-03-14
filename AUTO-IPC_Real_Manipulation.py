# @Author: Shounak Ray <Ray>
# @Date:   13-Mar-2021 10:03:01:019  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: AUTO-IPC_Real_Manipulation.py
# @Last modified by:   Ray
# @Last modified time: 14-Mar-2021 17:03:55:554  GMT-0600
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
import sys
from collections import Counter
from functools import reduce
from pathlib import Path

import featuretools as ft
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autofeat import AutoFeatRegressor
from sklearn.datasets import load_boston, load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

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


"""Major Notes:
> Ensure Python Version in root directory matches that of local directory
>> AKA ensure that there isn't a discrepancy between local and global setting of Python
> `pandas_profiling` will not work inside Atom, must user `jupyter-notebook` in terminal
"""

"""All Underlying Datasets
DATA_INJECTION_ORIG     --> The original injection data
DATA_PRODUCTION_ORIG    --> The original production data (rev 1)
DATA_TEST_ORIG          --> The originla testing data
FIBER_DATA              --> Temperature/Distance data along production lines
DATA_INJECTION_STEAM    --> Metered Steam at Injection Sites
DATA_INJECTION_PRESS    --> Pressure at Injection Sites
DATA_PRODUCTION         --> Production Well Sensors
DATA_TEST               --> Oil, Water, Gas, and Fluid from Production Wells
PRODUCTION_WELL_INTER   --> Join of DATA_TEST and DATA_PRODUCTION
PRODUCTION_WELL_WSENSOR --> Join of PRODUCTION_WELL_INTER and FIBER_DATA
FINALE                  --> Join of PRODUCTION_WELL_WSENSOR
                                                    and DATA_INJECTION_STEAM
"""

"""Filtering Notes:
> Excess Production Data is inside the dataset (more than just A and B pads/patterns)
>> These will be filtered out
> Excess (?) Fiber Data is inside the dataset (uncommon, otherwise unseen wells combos inside A  and B pads)
>> These will be filtered out
"""

# HYPER-PARAMETERS
BINS = 5
FIG_SIZE = (220, 7)
DIR_EXISTS = Path('Data/Pickles').is_dir()

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

model = AutoFeatRegressor(feateng_steps=2)
train_pct = 0.8
X_FROM_FINALE = FINALE[[c for c in FINALE.columns if c not in ['Date', 'unique_id', 'Pad', 'Well', 'test_flag',
                                                               '24_Fluid',  '24_Oil', '24_Water', 'Oil', 'Water',
                                                               'Gas', 'Fluid']]].replace(0.0, 0.1)[:100].to_numpy()
Y_FROM_FINALE = FINALE['Heel_Pressure'].replace(0.0, 0.1).to_numpy()

X_TRAIN = X_FROM_FINALE[:int(len(X_FROM_FINALE) * train_pct)]
Y_TRAIN = Y_FROM_FINALE[:int(len(Y_FROM_FINALE) * train_pct)]

pd.DataFrame(X_TRAIN)

X_TEST = X_FROM_FINALE[int(len(X_FROM_FINALE) * train_pct):]
Y_TEST = Y_FROM_FINALE[int(len(X_FROM_FINALE) * train_pct):]

df = model.fit_transform(X_TRAIN, Y_TRAIN)
model.score(X_FROM_FINALE, Y_FROM_FINALE)
plt.scatter(model.predict(X_FROM_FINALE), Y_FROM_FINALE, s=2)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

X, y = load_boston(True)
pd.DataFrame(y)

afreg = AutoFeatRegressor(verbose=1, feateng_steps=2)
# fit autofeat on less data, otherwise ridge reg model with xval will overfit on new features
X_train_tr = afreg.fit_transform(X[:480], y[:480])
X_test_tr = afreg.transform(X[480:])
print("autofeat new features:", len(afreg.new_feat_cols_))
print("autofeat MSE on training data:", mean_squared_error(y[:480], afreg.predict(X_train_tr)))
print("autofeat MSE on test data:", mean_squared_error(y[480:], afreg.predict(X_test_tr)))
print("autofeat R^2 on training data:", r2_score(y[:480], afreg.predict(X_train_tr)))
print("autofeat R^2 on test data:", r2_score(y[480:], afreg.predict(X_test_tr)))

for c in FINALE.columns:
    print(c, sum(FINALE[c].isna()))


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
