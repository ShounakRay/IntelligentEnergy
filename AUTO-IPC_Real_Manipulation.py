# @Author: Shounak Ray <Ray>
# @Date:   13-Mar-2021 10:03:01:019  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: AUTO-IPC_Real_Manipulation.py
# @Last modified by:   Ray
# @Last modified time: 18-Mar-2021 01:03:97:971  GMT-0600
# @License: [Private IP]


# G0TO: CTRL + OPTION + G
# SELECT USAGES: CTRL + OPTION + U
# DOCBLOCK: CTRL + SHIFT + C
# GIT COMMIT: CMD + ENTER
# GIT PUSH: CMD + U P
# PASTE IMAGE: CTRL + OPTION + SHIFT + V
# Todo: Opt + K  Opt + T

import colorsys
import datetime
import json
import math
import os
import re
import shutil
import sys
from collections import Counter
from functools import reduce
from itertools import chain
from pathlib import Path
from random import shuffle

import featuretools as ft
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from autofeat import AutoFeatRegressor, FeatureSelector
from colorutils import Color
from matplotlib.patches import Rectangle
from networkx.readwrite import json_graph
from sklearn import manifold
from sklearn.datasets import load_boston, load_diabetes
from sklearn.decomposition import PCA
from sklearn.metrics import (euclidean_distances, explained_variance_score,
                             max_error, mean_absolute_error,
                             mean_squared_error, median_absolute_error,
                             r2_score)

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
distinct_colors = ["#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059", "#FFDBE5",
                   "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87", "#5A0007", "#809693",
                   "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80", "#61615A", "#BA0900", "#6B7900",
                   "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100", "#DDEFFF", "#7B4F4B", "#A1C299",
                   "#0AA6D8", "#00846F", "#FFB500", "#C2FFED", "#A079BF", "#CC0744",
                   "#C0B9B2", "#C2FF99", "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68",
                   "#7A87A1", "#788D66", "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED",
                   "#886F4C", "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
                   "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00", "#7900D7",
                   "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700", "#549E79", "#FFF69F",
                   "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329", "#5B4534", "#FDE8DC", "#404E55",
                   "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C", "#83AB58", "#D1F7CE", "#004B28",
                   "#C8D0F6", "#A3A489", "#806C66", "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59",
                   "#8ADBB4", "#5B4E51", "#C895C5", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94",
                   "#7ED379", "#7A7BFF", "#D68E01", "#353339", "#78AFA1", "#FEB2C6", "#75797C", "#837393",
                   "#943A4D", "#B5F4FF", "#D2DCD5", "#9556BD", "#6A714A", "#02525F", "#0AA3F7", "#E98176",
                   "#DBD5DD", "#5EBCD1", "#3D4F44", "#7E6405", "#02684E", "#962B75", "#8D8546", "#9695C5", "#E773CE",
                   "#D86A78", "#3E89BE", "#CA834E", "#518A87", "#5B113C", "#55813B", "#E704C4", "#A97399",
                   "#4B8160", "#59738A", "#FF5DA7", "#F7C9BF", "#643127", "#513A01", "#6B94AA", "#51A058", "#A45B02",
                   "#E20027", "#E7AB63", "#4C6001", "#9C6966", "#64547B", "#97979E", "#006A66",
                   "#F4D749", "#0045D2", "#006C31", "#DDB6D0", "#7C6571", "#9FB2A4", "#00D891", "#15A08A", "#BC65E9",
                   "#FFFFFE", "#C6DC99", "#671190", "#6B3A64", "#F5E1FF", "#FFA0F2", "#CCAA35", "#374527",
                   "#8BB400", "#797868", "#C6005A", "#3B000A", "#C86240", "#29607C", "#402334", "#7D5A44", "#CCB87C",
                   "#B88183", "#AA5199", "#B5D6C3", "#A38469", "#9F94F0", "#A74571", "#B894A6", "#71BB8C", "#00B433",
                   "#789EC9", "#6D80BA", "#953F00", "#5EFF03", "#E4FFFC", "#1BE177", "#BCB1E5", "#76912F", "#003109",
                   "#0060CD", "#D20096", "#895563", "#A76F42", "#89412E", "#1A3A2A", "#494B5A",
                   "#A88C85", "#F4ABAA", "#A3F3AB", "#00C6C8", "#EA8B66", "#958A9F", "#BDC9D2", "#9FA064", "#BE4700",
                   "#658188", "#83A485", "#453C23", "#47675D", "#3A3F00", "#DFFB71", "#868E7E", "#98D058",
                   "#6C8F7D", "#D7BFC2", "#3C3E6E", "#D83D66", "#2F5D9B", "#6C5E46", "#D25B88", "#5B656C", "#00B57F",
                   "#545C46", "#866097", "#365D25", "#252F99", "#00CCFF", "#674E60", "#FC009C", "#92896B"]
diverging_palette_Global = sns.diverging_palette(240, 10, n=9, as_cmap=True)

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
    data['nodes'] = [dict(chain(G.nodes[n].items(), [(id_, n), ('label', n)])) for n in G]
    if multigraph:
        data['links'] = [
            dict(chain(d.items(),
                       [('from', u), ('to', v), (key, k)]))
            for u, v, k, d in G.edges(keys=True, data=True)]
    else:
        data['links'] = [
            dict(chain(d.items(),
                       [('from', u), ('to', v)]))
            for u, v, d in G.edges(data=True)]
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
                              mask=np.triu(input_data[typec]) if mask else None,
                              ax=ax[type_corrs.index(typec)],
                              annot=annot,
                              cmap=cmap if abs_arg else diverging_palette_Global
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


def remove_dupl_edges(G_func):
    # REMOVE DUPLICATE EDGES
    stripped_list = list(set([tuple(set(edge)) for edge in G_func.edges()]))
    stripped_list = [(u, v, d) for u, v, d in G_func.edges(data=True) if (u, v) in stripped_list]
    G_func.remove_edges_from([e for e in G.edges()])
    G_func.add_edges_from(stripped_list)

    return G_func


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def generate_depn_animation(df, groupby, time_feature, fig_size=(12.5, 9), resolution='high',
                            period=7, moving=False, dpi=150):
    # For every well...
    for w in df[groupby].unique():
        filtered_df = df[df[groupby] == w]
        images = []
        traversed_dates = []

        print('ANIMATION >> ' + w + ': Determining Frames...')
        all_dates = filtered_df['Date'].values
        if not moving:
            focus_dates = np.array(filtered_df[time_feature].index)
            good_indices = np.array(filtered_df[time_feature].index)[::period]
            focus_dates = [all_dates[i] for i in good_indices]
            for d_i in range(len(focus_dates) - 1):
                frame_start_ind = focus_dates[d_i]
                frame_end_ind = focus_dates[d_i + 1]
                traversed_dates.append((frame_start_ind, frame_end_ind))
                data_frame = filtered_df[(filtered_df[time_feature] > frame_start_ind) &
                                         (filtered_df[time_feature] < frame_end_ind)
                                         ].select_dtypes(float).corr()
                images.append(data_frame)
        elif moving:
            for d_i in range(len(all_dates) - period):
                frame_start_ind = all_dates[d_i]
                frame_end_ind = all_dates[d_i + period]
                traversed_dates.append((frame_start_ind, frame_end_ind))
                data_frame = filtered_df[(filtered_df[time_feature] > frame_start_ind) &
                                         (filtered_df[time_feature] < frame_end_ind)
                                         ].select_dtypes(float).corr()
                images.append(data_frame)
        else:
            raise ValueError('FAILED: moving paramaters is FATAL.')

        print('ANIMATION >> Image Vector Dimension: ' + str(len(images)))

        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(images[0], annot=False, center=0.0, cbar=False, square=True,
                    cmap=diverging_palette_Global, mask=np.triu(images[0])
                    ).set_title(str(traversed_dates[0][0]) + ' to ' + str(traversed_dates[0][1]))
        plt.tight_layout()

        def init():
            sns.heatmap(images[0], annot=False, center=0.0, cbar=False, square=True,
                        cmap=diverging_palette_Global, mask=np.triu(images[0])
                        ).set_title(str(traversed_dates[0][0]) + ' to ' + str(traversed_dates[0][1]))
            plt.tight_layout()

        def animate(i):
            data = images[i]
            sns.heatmap(data, vmax=.8, annot=False, center=0.0, cbar=False, square=True,
                        cmap=diverging_palette_Global, mask=np.triu(images[0])
                        ).set_title(str(traversed_dates[i][0]) + ' to ' + str(traversed_dates[i][1]))
            plt.tight_layout()

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(images), repeat=False)

        print('ANIMATION >> Writing Animation...')

        try:
            writer = animation.writers['ffmpeg']
        except KeyError:
            writer = animation.writers['avconv']
        writer = writer(fps=60)
        anim.save('CrossCorrelation/test.mp4', writer=writer, dpi=dpi)

        print('ANIMATION >> Animation created.')

        return images


def util_delete_alone(df):
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col, inplace=True, axis=1)
    return df


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
    FINALE = pd.read_csv('Data/FINALE_INTERP.csv')
    FINALE.drop(FINALE.columns[:2], axis=1, inplace=True)
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

TARGET = 'Heel_Pressure'
key_features = ['Bin_1', 'Bin_2', 'Bin_3', 'Bin_4', 'Bin_5']
DATE_BINS = 5

FINALE = util_delete_alone(FINALE)

FINALE_FILTERED_wbins = FINALE[[c for c in FINALE.columns if c not in ['unique_id', 'Pad', 'test_flag',
                                                                       '24_Fluid',  '24_Oil', '24_Water', 'Oil',
                                                                       'Water', 'Gas', 'Fluid', 'Pump_Speed']]
                               ].replace(0.0, 0.0000001)

plt.subplots(figsize=(20, 20))
hmapc = sns.heatmap(FINALE_FILTERED_wbins.corr(), cmap=diverging_palette_Global)
hmapc.get_figure().savefig('CrossCorrelation/Injector Cross Correlation.png', bbox_inches='tight')

unique_well_list = FINALE_FILTERED_wbins['Well'].unique()
fig, ax = plt.subplots(ncols=len(unique_well_list), nrows=DATE_BINS, sharey=True, figsize=(35, 35))

# well = unique_well_list[0]
# Split into source datasets for each production well
correlation_tracker = {}
for well in unique_well_list:
    well_date_range = sorted(
        set(FINALE_FILTERED_wbins[FINALE_FILTERED_wbins['Well'] == well
                                  ].drop(['Well'], 1).reset_index(drop=True)['Date']))
    dbins = list(chunks(well_date_range, int(len(well_date_range) / DATE_BINS)))
    if(len(dbins) > DATE_BINS):
        del dbins[-1]
    for dbin in dbins:
        SOURCE_wbins = FINALE_FILTERED_wbins[FINALE_FILTERED_wbins['Well'] == well]
        SOURCE_wbins = SOURCE_wbins.drop(['Well'], axis=1, inplace=False).reset_index(drop=True)

        first_date = dbin[0]
        last_date = dbin[-1]

        SOURCE_wbins = SOURCE_wbins[(SOURCE_wbins['Date'] >= first_date) &
                                    (SOURCE_wbins['Date'] <= last_date)].drop('Date', 1).reset_index(drop=True)
        SOURCE_wbins = SOURCE_wbins.rolling(window=7).mean().fillna(method='bfill').fillna(method='ffill')
        SOURCE_wbins = SOURCE_wbins.astype(np.float64)

        correlated_df = SOURCE_wbins.corr()
        final_data = correlated_df[key_features].drop(key_features)
        # pos = [list(final_data.index).index(val) for val in list(filtered_features)]

        hmap_row = list(unique_well_list).index(well)
        hmap_col = dbins.index(dbin)
        hmap = sns.heatmap(final_data, annot=False, ax=ax[hmap_col][hmap_row], center=0.0,
                           cbar=False, cmap=diverging_palette_Global
                           ) if well != unique_well_list[-1] else sns.heatmap(final_data,
                                                                              annot=False,
                                                                              ax=ax[hmap_col][hmap_row],
                                                                              cmap=diverging_palette_Global,
                                                                              center=0.0)

        hmap.set_title(well + ': ' + str(first_date) + ' > ' + str(last_date))
        plt.tight_layout()

        correlation_tracker[well] = {dbins.index(dbin): final_data}

hmap.get_figure().savefig('CrossCorrelation/WELL-{WELL}_TARGET-{TARGET}_TIMED-{TIME}.png'.format(
    WELL='ALL_PRODUCTION',
    TARGET='UNSPECIFIED',
    TIME='True'), bbox_inches='tight')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Calculate the means (macro-state)
# MACRO_GROUPS = {'Bin Temps': ['Bin_1', 'Bin_2', 'Bin_3', 'Bin_4', 'Bin_5'],
#                 'End Temps': ['Heel_Temp', 'Toe_Temp'],
#                 'End Press': ['Toe_Pressure', 'Heel_Pressure'],
#                 'Lin Press': ['Casing_Pressure', 'Tubing_Pressure']}
MACRO_GROUPS = {'Bin Temps': ['Bin_1', 'Bin_2', 'Bin_3', 'Bin_4', 'Bin_5']}

# m_group = MACRO_GROUPS[0]
# well = unique_well_list[0]
G = nx.Graph()
macro_correlation_tracker = {}
macro_version = {}
pad_well_assoc = dict(FINALE[['Well', 'Pad']].values)
for well in unique_well_list:
    local_df = correlation_tracker[well]
    for m_group in MACRO_GROUPS.values():
        macro_i_avg = pd.DataFrame(data=local_df)[m_group].mean(axis=1)
        key_tag = list(MACRO_GROUPS.keys())[list(MACRO_GROUPS.values()).index(m_group)] + ', ' + well
        macro_version[key_tag] = pd.DataFrame(macro_i_avg, columns=[key_tag])
        # Add edges to network graph
        G.add_weighted_edges_from(ebunch_to_add=list(zip(macro_version[key_tag].index,
                                                         [key_tag] * len(macro_version[key_tag]),
                                                         chain.from_iterable(macro_version[key_tag].values))),
                                  weight='value',
                                  producer_well=well)
    macro_correlation_tracker[well] = macro_version.copy()
    macro_version.clear()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Filtering/Truncation (p << 80 ISH)
percentile = 50
weights_all = list(set([d['value'] for u, v, d in G.edges(data=True)]))
plt.hist(weights_all, bins=50)
# sorted(weights_all, reverse=True)
cutoff = np.percentile([c for c in weights_all if c > 0], percentile)
# cutoff = min(weights_all)
print('STATUS: Cuttoff is: ' + str(cutoff) + ' @ {} percentile'.format(percentile))
G_dupl = nx.Graph([(u, v, d) for u, v, d in G.edges(data=True) if d['value'] > cutoff])

# stripped_list = list(set([tuple(set(edge)) for edge in G_dupl.edges()]))
# stripped_list = [(u, v, d) for u, v, d in G_dupl.edges(data=True) if (u, v) in stripped_list]
# G_func.remove_edges_from([e for e in G.edges()])
# G_func.add_edges_from(stripped_list)

G_dupl = remove_dupl_edges(G_dupl)
# nx.node_link_data(remove_dupl_edges(G_dupl))
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Remove anything that isn't temperature bins or injectors
keep_columns = [c for c in FINALE_FILTERED_wbins.columns if 'Bin' in c or 'I' in c]
forbidden_columns = [c for c in FINALE_FILTERED_wbins.columns if c not in keep_columns]
remove = [node for node in G_dupl.nodes if node in forbidden_columns]
G_dupl.remove_nodes_from(remove)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Set note attributes (whether well is injector or producer)
well_attr = [e[-1].strip() for e in [n.split(',') if len(n.split(',')) != 1 else ['Injector'] for n in G_dupl.nodes]]
well_attr = dict(zip(G_dupl.nodes, well_attr))
nx.set_node_attributes(G_dupl, well_attr, 'group')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Add injector cross-correlations
G_node_inj = [n for n in G_dupl.nodes if 'I' in n]
adj_matrix = FINALE_FILTERED_wbins[G_node_inj].corr()
_ = correlation_matrix(FINALE_FILTERED_wbins[G_node_inj], 'CrossCorrelation/selective_matrix.png',
                       '', abs_arg=False)
G_internal = nx.from_pandas_adjacency(adj_matrix, create_using=nx.Graph)
inj_internal_edges = [(to, fr, val['weight']) for to, fr, val in list(G_internal.edges(data=True))]
G_dupl.add_weighted_edges_from(ebunch_to_add=inj_internal_edges, weight='value')

# Remove duplicate edges
G_dupl = remove_dupl_edges(G_dupl)
# Remove any self-loops
G_dupl.remove_edges_from(nx.selfloop_edges(G_dupl))
# Remove and childless nodes
loners = list(nx.isolates(G_dupl))
G_dupl.remove_nodes_from(loners)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Check JSON data (for reference)
json_graph.node_link_data(G_dupl)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# VIS.JS
json_dict = {'nodes': node_link_data(G_dupl)['nodes'], 'edges': node_link_data(G_dupl)['links']}

# Add edge color inheritance
for ed_i in range(len(json_dict['edges'])):
    if(json_dict['edges'][ed_i]['to'] in G_node_inj and json_dict['edges'][ed_i]['from'] in G_node_inj):
        json_dict['edges'][ed_i]['color'] = {'color': "red"}
    else:
        json_dict['edges'][ed_i]['color'] = {'inherit': "from"}

# Restructure nodes and edges data
node_data = str(json_dict['nodes']).replace('[', '').replace(']', '')
edges_data = str(json_dict['edges']).replace('[', '').replace(']', '')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Assign colors (of producers)
group_color = list(set(["""'{}': {{'color':{{'background':'grey'}}, 'borderWidth':3}}""".format(val)
                        for val in [a['group'] for n, a in G_dupl.nodes(data=True)] if val != 'Injector']))
N = len(group_color)
# sns.color_palette("tab10", len([c for c in macro_i_avg.keys()])).as_hex()
shuffle(distinct_colors)
group_color = [item.replace('grey', str(distinct_colors[group_color.index(item)])) for item in group_color]
# group_color += ["""'{}': {{'color':{{'background':'#FF0000'}}, 'borderWidth':3}}""".format(val) for val in G_node_inj]
group_color = str(group_color)[2:-2].replace('"', '')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Duplicate the base HTML file
shutil.copy2('nx_visjs.html', 'nx_visjs_TAILOR.html')

# Key for inserting options, etc.
# insertions = {74: node_data,
#               78: edges_data}

insertions = {73: node_data,
              77: edges_data,
              86: group_color}

# Insert options, etc.
for line_num, ins_text in list(insertions.items()):
    # # Insert node and edge data in html file
    with open('nx_visjs_TAILOR.html', 'r') as b:
        lines = b.readlines()

    with open('nx_visjs_TAILOR.html', 'w') as b:
        for i, line in enumerate(lines):
            if i == line_num:
                b.write(ins_text)
            b.write(line)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ANIMATIONS

data_temp = FINALE_FILTERED_wbins[FINALE_FILTERED_wbins['Well'] == well]
# data_temp = data_temp[(data_temp['Date'] >= first_date) & (data_temp['Date'] <= last_date)]

all_data = generate_depn_animation(data_temp[data_temp['Well'] == 'AP3'].reset_index(drop=True),
                                   'Well', 'Date', period=100, moving=False, dpi=350)
