# @Author: Shounak Ray <Ray>
# @Date:   13-Mar-2021 10:03:01:019  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: AUTO-IPC_Real_Manipulation.py
# @Last modified by:   Ray
# @Last modified time: 16-Mar-2021 11:03:74:740  GMT-0600
# @License: [Private IP]


# G0TO: CTRL + OPTION + G
# SELECT USAGES: CTRL + OPTION + U
# DOCBLOCK: CTRL + SHIFT + C
# GIT COMMIT: CMD + ENTER
# GIT PUSH: CMD + U P
# PASTE IMAGE: CTRL + OPTION + SHIFT + V
# Todo: Opt + K  Opt + T

import json
import math
import os
import re
import sys
from collections import Counter
from functools import reduce
from itertools import chain
from pathlib import Path

import featuretools as ft
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from autofeat import AutoFeatRegressor, FeatureSelector
from matplotlib.patches import Rectangle
from networkx.readwrite import json_graph
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

_attrs = dict(id='id', source='source', target='target', key='key')

# This is stolen from networkx JSON serialization. It basically just changes what certain keys are.


def node_link_data(G, attrs=_attrs):
    """Return data in node-link format that is suitable for JSON serialization
    and use in Javascript documents.

    Parameters
    ----------
    G : NetworkX graph

    attrs : dict
        A dictionary that contains four keys 'id', 'source', 'target' and
        'key'. The corresponding values provide the attribute names for storing
        NetworkX-internal graph data. The values should be unique. Default
        value:
        :samp:`dict(id='id', source='source', target='target', key='key')`.

        If some user-defined graph data use these attribute names as data keys,
        they may be silently dropped.

    Returns
    -------
    data : dict
       A dictionary with node-link formatted data.

    Raises
    ------
    NetworkXError
        If values in attrs are not unique.

    Examples
    --------
    >>> from networkx.readwrite import json_graph
    >>> G = nx.Graph([(1,2)])
    >>> data = json_graph.node_link_data(G)

    To serialize with json

    >>> import json
    >>> s = json.dumps(data)

    Notes
    -----
    Graph, node, and link attributes are stored in this format. Note that
    attribute keys will be converted to strings in order to comply with
    JSON.

    The default value of attrs will be changed in a future release of NetworkX.

    See Also
    --------
    node_link_graph, adjacency_data, tree_data
    """
    multigraph = G.is_multigraph()
    id_ = attrs['id']
    source = attrs['source']
    target = attrs['target']
    # Allow 'key' to be omitted from attrs if the graph is not a multigraph.
    key = None if not multigraph else attrs['key']
    if len(set([source, target, key])) < 3:
        raise nx.NetworkXError('Attribute names are not unique.')
    data = {}
    data['directed'] = G.is_directed()
    data['multigraph'] = multigraph
    data['graph'] = G.graph
    data['nodes'] = [dict(chain(G.node[n].items(), [(id_, n), ('label', n)])) for n in G]
    if multigraph:
        data['links'] = [
            dict(chain(d.items(),
                       [('from', u), ('to', v), (key, k)]))
            for u, v, k, d in G.edges_iter(keys=True, data=True)]
    else:
        data['links'] = [
            dict(chain(d.items(),
                       [('from', u), ('to', v)]))
            for u, v, d in G.edges_iter(data=True)]
    return data


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
key_features = ['Bin_1', 'Bin_2', 'Bin_3', 'Bin_4', 'Bin_5'] + ['Heel_Temp', 'Toe_Temp',
                                                                'Toe_Pressure', 'Heel_Pressure',
                                                                'Casing_Pressure', 'Tubing_Pressure']

# Deletes any single-value columns
for col in FINALE.columns:
    if len(FINALE[col].unique()) == 1:
        FINALE.drop(col, inplace=True, axis=1)

# Fills all nan with small value (for feature selection)
# FINALE = FINALE.fillna(0.000000001).replace(np.nan, 0.000000001)

# for c in FINALE.columns:
#     print(c, sum(FINALE[c].isna()))

# Ultimate Goal
# FINALE = FINALE[FINALE['test_flag'] == True].reset_index(drop=True)
# TARGET = 'Oil'

# FINALE_FILTERED = FINALE[[c for c in FINALE.columns if c not in ['Date', 'unique_id', 'Pad', 'test_flag',
# '24_Fluid',  '24_Oil', '24_Water', 'Oil', 'Water',
# 'Gas', 'Fluid', 'Toe_Pressure', 'Pump_Speed',
# 'Casing_Pressure', 'Tubing_Pressure', 'Heel_Temp',
# 'Toe_Temp', 'Bin_1', 'Bin_2', 'Bin_3', 'Bin_4',
# 'Bin_5']]].replace(0.0, 0.0000001)
# FINALE_FILTERED = FINALE[[c for c in FINALE.columns if c not in ['Date', 'unique_id', 'Pad', 'test_flag',
#                                                                  '24_Fluid',  '24_Oil', '24_Water', 'Oil', 'Water',
#                                                                  'Gas', 'Fluid', 'Toe_Pressure', 'Pump_Speed',
#                                                                  'Casing_Pressure', 'Tubing_Pressure', 'Heel_Temp',
#                                                                  'Toe_Temp', 'Bin_1', 'Bin_2', 'Bin_3', 'Bin_4',
#                                                                  'Bin_5']]
#                          ].replace(0.0, 0.0000001)
FINALE_FILTERED_wbins = FINALE[[c for c in FINALE.columns if c not in ['Date', 'unique_id', 'Pad', 'test_flag',
                                                                       '24_Fluid',  '24_Oil', '24_Water', 'Oil', 'Water',
                                                                       'Gas', 'Fluid', 'Pump_Speed']]
                               ].replace(0.0, 0.0000001)


# list(FINALE_FILTERED.columns)
# FINALE_FILTERED = FINALE_FILTERED.astype(np.float64)

# FINALE_FILTERED_ULTIMATE = FINALE[[c for c in FINALE.columns if c not in ['Date', 'unique_id', 'Pad', 'test_flag',
#                                         '24_Fluid',  '24_Oil', '24_Water', 'Water',
#                                         'Gas', 'Fluid', 'Toe_Pressure', 'Pump_Speed',
#                                         'Casing_Pressure', 'Tubing_Pressure', 'Heel_Temp',
#                                         'Toe_Temp', 'Bin_1', 'Bin_2', 'Heel_Pressure',
#                                         'Bin_3', 'Bin_4', 'Bin_5']]
# ].replace(0.0, 0.0000001)

unique_well_list = FINALE_FILTERED_wbins['Well'].unique()
fig, ax = plt.subplots(ncols=len(unique_well_list), sharey=True, figsize=(35, 15))

# well = unique_well_list[0]
# Split into source datasets for each production well
correlation_tracker = {}
for well in unique_well_list:
    # SOURCE = FINALE_FILTERED[FINALE_FILTERED['Well'] == well]
    # SOURCE.drop(['Well'], axis=1, inplace=True)
    # SOURCE.reset_index(drop=True, inplace=True)
    # SOURCE = SOURCE.rolling(window=7).mean().fillna(method='bfill').fillna(method='ffill')

    SOURCE_wbins = FINALE_FILTERED_wbins[FINALE_FILTERED_wbins['Well'] == well]
    SOURCE_wbins.drop(['Well'], axis=1, inplace=True)
    SOURCE_wbins.reset_index(drop=True, inplace=True)

    SOURCE_wbins = SOURCE_wbins.rolling(window=7).mean().fillna(method='bfill').fillna(method='ffill')
    SOURCE_wbins = SOURCE_wbins.astype(np.float64)

    # msk = np.random.rand(len(SOURCE)) < train_pct
    #
    # # Variable Selection (NOT Engineering)
    # new_X = model_featsel.fit_transform(SOURCE[[c for c in SOURCE.columns if c != TARGET]],
    #                                     SOURCE[TARGET])
    # filtered_features = new_X.columns

    correlated_df = SOURCE_wbins.corr()
    final_data = correlated_df.abs()[key_features].drop(key_features)
    # pos = [list(final_data.index).index(val) for val in list(filtered_features)]

    hmap = sns.heatmap(final_data, annot=False, ax=ax[list(unique_well_list).index(well)],
                       cbar=False) if well != 'BP6' else sns.heatmap(final_data, annot=False,
                                                                     ax=ax[list(unique_well_list).index(well)])

    # for p in pos:
    #     hmap.add_patch(Rectangle((0, p), len(key_features), 1, edgecolor='blue', fill=False, lw=3))
    hmap.set_title('Well ' + well)
    plt.tight_layout()

    correlation_tracker[well] = final_data

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

# fig.colorbar(ax[12].collections[0], cax=ax[12])
hmap.get_figure().savefig('CrossCorrelation/WELL-{WELL}_TARGET-{TARGET}.pdf'.format(WELL='ALL_PRODUCTION',
                                                                                    TARGET='UNSPECIFIED'),
                          bbox_inches='tight')

# Calculate the means (macro-state)
MACRO_GROUPS = {'Bin Temps': ['Bin_1', 'Bin_2', 'Bin_3', 'Bin_4', 'Bin_5'],
                'End Temps': ['Heel_Temp', 'Toe_Temp'],
                'End Press': ['Toe_Pressure', 'Heel_Pressure'],
                'Lin Press': ['Casing_Pressure', 'Tubing_Pressure']}

# m_group = MACRO_GROUPS[0]
# well = unique_well_list[0]
G = nx.Graph()
macro_correlation_tracker = {}
macro_version = {}
pad_well_assoc = dict(FINALE[['Well', 'Pad']].values)
for well in unique_well_list:
    local_df = correlation_tracker[well]
    for m_group in MACRO_GROUPS.values():
        macro_i_avg = pd.DataFrame(data=final_data)[m_group].sum(axis=1)
        key_tag = list(MACRO_GROUPS.keys())[list(MACRO_GROUPS.values()).index(m_group)]
        macro_version[key_tag] = pd.DataFrame(macro_i_avg, columns=[key_tag])
        # Add edges to network graph
        G.add_weighted_edges_from(ebunch_to_add=list(zip(macro_version[key_tag].index,
                                                         [key_tag] * len(macro_version[key_tag]),
                                                         chain.from_iterable(macro_version[key_tag].values))),
                                  weight='correlation_weight',
                                  producer_well=well)
    macro_correlation_tracker[well] = macro_version.copy()
    macro_version.clear()

G.nodes

json_graph.node_link_data(G)
nx.write_gexf(G, "test.gexf")

print(json.dumps({'nodes': node_link_data(G)['nodes'], 'edges': node_link_data(G)['links']}, indent=4))

# pos = nx.spring_layout(G)
# nx.draw(G, pos)
# labels = nx.get_edge_attributes(G, 'correlation_weight')

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

#
# X, y = load_boston(True)
# pd.DataFrame(X)
#
# afreg = AutoFeatRegressor(verbose=1, feateng_steps=2)
# # fit autofeat on less data, otherwise ridge reg model_feateng with xval will overfit on new features
# X_train_tr = afreg.fit_transform(X[:480], y[:480])
# X_test_tr = afreg.transform(X[480:])
# print("autofeat new features:", len(afreg.new_feat_cols_))
# print("autofeat MSE on training data:", mean_squared_error(pd.DataFrame(y[:480]), afreg.predict(X_train_tr)))
# print("autofeat MSE on test data:", mean_squared_error(y[480:], afreg.predict(X_test_tr)))
# print("autofeat R^2 on training data:", r2_score(y[:480], afreg.predict(X_train_tr)))
# print("autofeat R^2 on test data:", r2_score(y[480:], afreg.predict(X_test_tr)))
#
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# es = ft.EntitySet(id='ipc_entityset')
#
# es = es.entity_from_dataframe(entity_id='FINALE', dataframe=FINALE,
#                               index='unique_id', time_index='Date')
#
# # Default primitives from featuretools
# default_agg_primitives = ft.list_primitives()[(ft.list_primitives()['type'] == 'aggregation') &
#                                               (ft.list_primitives()['valid_inputs'] == 'Numeric')
#                                               ]['name'].to_list()
# default_trans_primitives = [op for op in ft.list_primitives()[(ft.list_primitives()['type'] == 'transform') &
#                                                               (ft.list_primitives()['valid_inputs'] == 'Numeric')
#                                                               ]['name'].to_list()[:2]
#                             if op not in ['scalar_subtract_numeric_feature']]
#
# # DFS with specified primitives
# feature_names = ft.dfs(entityset=es, target_entity='FINALE',
#                        trans_primitives=default_trans_primitives,
#                        agg_primitives=None,
#                        max_depth=2,
#                        # n_jobs=-1,
#                        features_only=True)
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# # Create Entity Set for this project
# es = ft.EntitySet(id='ipc_entityset')
#
# FIBER_DATA.reset_index(inplace=True)
# DATA_INJECTION_STEAM.reset_index(inplace=True)
# DATA_INJECTION_PRESS.reset_index(inplace=True)
# DATA_PRODUCTION.reset_index(inplace=True)
# DATA_TEST.reset_index(inplace=True)
#
# # Create an entity from the client dataframe
# # This dataframe already has an index and a time index
# es = es.entity_from_dataframe(entity_id='DATA_INJECTION_STEAM', dataframe=DATA_INJECTION_STEAM.copy(),
#                               make_index=True, index='id_injector', time_index='Date')
# es = es.entity_from_dataframe(entity_id='DATA_INJECTION_PRESS', dataframe=DATA_INJECTION_PRESS.copy(),
#                               make_index=True, index='id_injector', time_index='Date')
#
# es = es.entity_from_dataframe(entity_id='DATA_PRODUCTION', dataframe=DATA_PRODUCTION.copy(),
#                               make_index=True, index='id_production', time_index='Date')
# es = es.entity_from_dataframe(entity_id='DATA_TEST', dataframe=DATA_TEST.copy(),
#                               make_index=True, index='id_production', time_index='Date')
# es = es.entity_from_dataframe(entity_id='FIBER_DATA', dataframe=FIBER_DATA.copy(),
#                               make_index=True, index='id_production', time_index='Date')
#
# rel_producer = ft.Relationship(es['DATA_PRODUCTION']['id_production'],
#                                es['DATA_TEST']['id_production'])
# es.add_relationship(rel_producer)
#
# # Add relationships for all columns between injection steam and pressure data
# injector_sensor_cols = [c for c in DATA_INJECTION_STEAM.columns.copy() if c not in ['Date']]
# injector_sensor_cols = ['CI06']
# for common_col in injector_sensor_cols:
#     rel_injector = ft.Relationship(es['DATA_INJECTION_STEAM']['id_injector'],
#                                    es['DATA_INJECTION_PRESS']['id_injector'])
#     es = es.add_relationship(rel_injector)
#
# es['DATA_INJECTION_STEAM']
#
# # EOF
# # EOF
