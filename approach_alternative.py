# @Author: Shounak Ray <Ray>
# @Date:   19-Mar-2021 08:03:81:813  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: approach_alternative.py
# @Last modified by:   Ray
# @Last modified time: 30-Mar-2021 14:03:81:818  GMT-0600
# @License: [Private IP]

import math
import os
import sys
from collections import Counter
from itertools import chain
from multiprocessing import Pool

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import Anomaly_Detection_PKG
except Exception:
    sys.path.append('/Users/Ray/Documents/Github/AnomalyDetection')
    from Anomaly_Detection_PKG import *

import warnings

import defs

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

plot_eda = False
plot_geo = True

# Display hyperparams
relative_radius = 50
dimless_radius = relative_radius * 950
focal_period = 20

# Scaling, window constants
POS_TL = (478, 71)
POS_TR = (1439, 71)
POS_BL = (478, 1008)
POS_BR = (1439, 1008)
x_delta = POS_TL[0] | POS_BL[0]
y_delta = POS_TR[0] | POS_BR[0]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# INGESTION
FINALE = pd.read_csv('Data/combined_ipc.csv').infer_objects()
DATA_INJECTION_ORIG = pd.read_pickle('Data/Pickles/DATA_INJECTION_ORIG.pkl')
DATA_PRODUCTION_ORIG = pd.read_pickle('Data/Pickles/DATA_PRODUCTION_ORIG.pkl')

PRO_PAD_KEYS = dict(zip(DATA_PRODUCTION_ORIG['Well'], DATA_PRODUCTION_ORIG['Pad']))
INJ_PAD_KEYS = dict(zip(DATA_INJECTION_ORIG['Well'], DATA_INJECTION_ORIG['Pad']))

available_pads_transformed = ['A', 'B']
available_pwells_transformed = [k for k, v in PRO_PAD_KEYS.items() if v in available_pads_transformed]

UPSCALAR = 100
BASE_UPSCALED = 0

# DEFINITIONS


def filter_negatives(df, columns):
    """Filter out all the rows in the DataFrame whose rows contain negatives.

    Parameters
    ----------
    df : DataFrame
        The intitial, unfiltered DataFrame.
    columns : pandas Index, or any iterable
        Which columns to consider.

    Returns
    -------
    DataFrame
        The filtered DataFrame, may contain NaNs (intentionally).

    """
    for num_col in columns:
        df = df[~(df[num_col] < 0)].reset_index(drop=True)
    return df


def convert_to_date(df, date_col):
    """Given a DataFrame and it's time column, assign dtype to datetime.date().

    Parameters
    ----------
    df : DataFrameFrame
        The original DataFrame with [ambiguous] dtypes.
    date_col : object (string)
        Name of column with time date.

    Returns
    -------
    DataFrame
        A corrected/re-assigned version of the DataFrame.

    """
    df[date_col] = pd.to_datetime(df[date_col])
    df[date_col] = [d.date() for d in df[date_col]]
    return df


def pressure_lambda(row):
    """Picks the available pressure out of casing and tubing pressures.

    Parameters
    ----------
    row : Series
        The row in the respective DataFrame.

    Returns
    -------
    int or nan (float)
        The chosen pressure.

    """
    if not math.isnan(row['Casing_Pressure']):
        # if Casing_Pressure exists
        return row['Casing_Pressure']
    elif not math.isnan(row['Tubing_Pressure']):
        # if Casing_Pressure doesn't exist but Tubing_Pressure does
        return row['Tubing_Pressure']
    else:
        # If neither Casing_Pressure or Tubing_Pressure exist
        return math.nan


def drop_singles(df):
    dropped = []
    for col in df.columns:
        if len(df[col].unique()) <= 1:
            dropped.append(col)
            df.drop(col, axis=1, inplace=True)
    return df, dropped


def intermediates(p1, p2, nb_points=8):
    """"Return a list of nb_points equally spaced points
    between p1 and p2"""
    # If we have 8 intermediate points, we have 8+1=9 spaces
    # between p1 and p2
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    return [[p1[0] + i * x_spacing, p1[1] + i * y_spacing]
            for i in range(0, nb_points + 2)]


def euclidean_2d_distance(c1, c2):
    x1 = c1[0]
    x2 = c2[0]
    y1 = c1[1]
    y2 = c2[1]
    return math.hypot(x2 - x1, y2 - y1)


def injector_candidates(production_pad, production_well,
                        pro_well_pad_relationship, injector_coordinates, production_coordinates,
                        relative_radius=relative_radius):

    if(production_pad is not None or production_well is not None):
        inclusion = pd.DataFrame([], columns=injector_coordinates.keys(), index=production_coordinates.keys())
        for iwell, iwell_point in injector_coordinates.items():
            x_test = iwell_point[0]
            y_test = iwell_point[1]
            inj_allpro_tracker = []
            for pwell, pwell_points in production_coordinates.items():
                inj_pro_level_tracker = []
                for pwell_point in pwell_points:
                    center_x = pwell_point[0]
                    center_y = pwell_point[1]
                    inclusion_condition = (x_test - center_x)**2 + (y_test - center_y)**2 < relative_radius**2
                    inj_pro_level_tracker.append(inclusion_condition)
                state = any(inj_pro_level_tracker)
                inj_allpro_tracker.append(state)
            inclusion[iwell] = inj_allpro_tracker

        inclusion = inclusion.rename_axis('PRO_Well').reset_index()
        if(production_pad is not None):
            inclusion['PRO_Pad'] = [pro_well_pad_relationship.get(pwell) for pwell in inclusion['PRO_Well']]

        if(production_pad is not None):
            type_filtered_inclusion = inclusion[inclusion['PRO_Pad'] == production_pad].reset_index(drop=True)
        elif(production_well is not None):
            type_filtered_inclusion = inclusion[inclusion['PRO_Well'] == production_well].reset_index(drop=True)

        all_possible_candidates = type_filtered_inclusion.select_dtypes(bool).columns
        candidate_injectors_full = dict([(candidate, any(type_filtered_inclusion[candidate]))
                                         for candidate in all_possible_candidates])
        candidate_injectors = [k for k, v in candidate_injectors_full.items() if v is True]

        return candidate_injectors, candidate_injectors_full
    else:
        raise ValueError('ERROR: Fatal parameter inputs for `injector_candidates`')


def plot_geo_candidates(candidate_dict, group_type, PRO_finalcoords, INJ_relcoords,
                        POS_TL=POS_TL, POS_BR=POS_BR, POS_TR=POS_TR):
    # Producer connections
    aratio = (POS_TR[0] - POS_TL[0]) / (POS_BR[1] - POS_TR[1])
    fig, ax = plt.subplots(figsize=(20 * aratio, 20))
    colors = cm.rainbow(np.linspace(0, 1, len(PRO_finalcoords.keys())))
    for k, v in PRO_finalcoords.items():
        all_x = [c[0] for c in v]
        all_y = [c[1] for c in v]
        plt.scatter(all_x, all_y, linestyle='solid', color=colors[list(PRO_finalcoords.keys()).index(k)])
        plt.plot(all_x, all_y, color=colors[list(PRO_finalcoords.keys()).index(k)])
        plt.scatter(all_x, all_y, color=list(
            (*colors[list(PRO_finalcoords.keys()).index(k)][:3], *[0.1])), s=dimless_radius)
    # Injector connections
    all_x = [t[0] for t in INJ_relcoords.values()]
    all_y = [t[1] for t in INJ_relcoords.values()]
    ax.scatter(all_x, all_y)
    for i, txt in enumerate(INJ_relcoords.keys()):
        if(txt in candidate_dict.get(group_type)):
            ax.scatter(all_x[i], all_y[i], color='green', s=200)
        else:
            ax.scatter(all_x[i], all_y[i], color='red', s=200)
        ax.annotate(txt, (all_x[i] + 2, all_y[i] + 2))
    plt.title('Producer Well and Injector Space, Overlaps')
    plt.tight_layout()
    plt.savefig('Modeling Reference Files/Producer-Injector Overlap for {}.png'.format(group_type))


def produce_injector_aggregates(candidates_by_prodtype, injector_data, group_type):
    INJECTOR_AGGREGATES = {}
    for category, cat_candidates in candidates_by_prodtype.items():
        # Select candidates (not all the wells)
        local_candidates = cat_candidates.copy()
        absence = []
        for cand in local_candidates:
            if cand not in all_injs:
                print('> STATUS: Candidate {} removed assessing {}, unavailable in initial data'.format(cand,
                                                                                                        category))
                absence.append(cand)
        local_candidates = [el for el in local_candidates if el not in absence]

        melted_inj = pd.melt(injector_data, id_vars=['Date'], value_vars=local_candidates,
                             var_name='Injector', value_name='Steam')
        melted_inj['INJ_Pad'] = melted_inj['Injector'].apply(lambda x: INJ_PAD_KEYS.get(x))
        melted_inj = melted_inj[~melted_inj['INJ_Pad'].isna()].reset_index(drop=True)
        # To groupby injector pads, by=['Date', 'INJ_Pad']
        agg_inj = melted_inj.groupby(by=['Date'], axis=0, sort=False, as_index=False).sum()
        agg_inj[group_type] = category
        INJECTOR_AGGREGATES[category] = agg_inj
    INJECTOR_AGGREGATES = pd.concat(INJECTOR_AGGREGATES.values()).reset_index(drop=True)

    return INJECTOR_AGGREGATES


def visualize_anomalies(ft):
    fig, ax = plt.subplots(figsize=(12, 7))
    ft[ft['anomaly'] == 'Yes'][['date', 'selection']].plot(
        x='date', y='selection', kind='scatter', c='red', s=3, ax=ax)
    ft[ft['anomaly'] == 'No'][['date', 'selection']].plot(x='date', y='selection', kind='line', ax=ax)
    plt.show()


# with Pool(os.cpu_count() - 1) as pool:
#     pool.map(process_local_anomalydetection, arguments)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
FINALE['PRO_Pad'] = FINALE['PRO_Well'].apply(lambda x: PRO_PAD_KEYS.get(x))
FINALE = FINALE.dropna(subset=['PRO_Well']).reset_index(drop=True)

FINALE = filter_negatives(FINALE, FINALE.select_dtypes(float))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# plt.figure(figsize=(25, 25))
# sns.heatmap(FINALE.isna())

TEMP = FINALE.copy()
TEMP.columns = list(TEMP.columns.str.lower())
col_reference = dict(zip(TEMP.columns, FINALE.columns))

all_continuous_columns = TEMP.select_dtypes(float).columns
all_prod_wells = list(FINALE['PRO_Well'].unique())
TEMP['weights'] = None
lc_injectors = [k for k, v in col_reference.items() if 'I' in v][1:]

cumulative_tagged = []
for cont_col in all_continuous_columns:
    rep_multiplier = len(all_prod_wells) if cont_col not in lc_injectors else 1
    arguments = list(zip([TEMP] * rep_multiplier, all_prod_wells, [cont_col] * rep_multiplier))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if __name__ == '__main__':
            with Pool(os.cpu_count() - 1) as pool:
                data_outputs = pool.starmap(defs.process_local_anomalydetection, arguments)
    cumulative_tagged.extend(data_outputs)

    print('> STATUS: {} % progress with `{}`'.format(
        round(100.0 * list(all_continuous_columns).index(cont_col) / len(all_continuous_columns), 3), cont_col))

anomaly_tag_tracker = pd.concat(cumulative_tagged).reset_index(drop=True)

anomaly_tag_tracker[list(anomaly_tag_tracker.columns)[:-1]].drop_duplicates()

# plt.plot(linear_time_range)

# plt.figure(figsize=(12, 8))
# # plt.plot(ft['scores'])
# plt.plot(ft['selection'] / max(ft['selection'].dropna()))
# plt.plot(adjusted_anomscores)
#
# plt.figure(figsize=(12, 8))
# plt.plot(ft['date'].apply(lambda x: (x - min_date).days / range)**2)
# plt.plot(abs(ft['scores'] / max(ft['scores'].dropna())))
# plt.plot(ft['final_weight'] / sorted(ft['final_weight'].dropna())[-1:])
# min(ft['scores'])

_ = plt.hist(ft['scores'])

# ft.to_html('anomaly_decision.html')
visualize_anomalies(ft)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # INJECTOR/PRODUCER TRANSLATIONS  # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # Injector wells to producer well overlaps (only those spanning different producer pads)
# INJ_well_to_pad_overlaps = {'I16': ['AP3', 'BP1'],
#                             'I21': ['AP3', 'BP1'],
#                             'I27': ['AP3', 'BP1'],
#                             'I37': ['FP1', 'BP6'],
#                             'I72': ['FP1', 'BP6'],
#                             'I47': ['CP2', 'EP2'],
#                             'I44': ['CP3', 'EP2'],
#                             'I42': ['CP6', 'EP2'],
#                             'CI7': ['CP7', 'EP2'],
#                             'CI8': ['CP8', 'EP2']}
# PRO_injpad_to_well = {'E3': ['EP2', 'EP3', 'EP4', 'EP5', 'EP6', 'EP7'],
#                       'E2': ['EP2', 'EP3', 'EP4', 'EP5', 'EP6', 'EP7'],
#                       'E1': ['EP2', 'EP3', 'EP4', 'EP5', 'EP6', 'EP7'],
#                       'A': ['AP4', 'AP5', 'AP6', 'AP7', 'A8'],
#                       '15-05': ['AP4', 'AP6', 'AP7'],
#                       '16-05': ['AP2', 'AP3', 'AP4', 'AP5', 'AP6', 'AP7']}

# # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # #
# INJECTOR COORDINATE TRANSFORMATIONS
INJ_relcoords = {}
INJ_relcoords = {'I02': '(757, 534)',
                 'I03': '(709, 519)',
                 'I04': '(760, 488)',
                 'I05': '(708, 443)',
                 'I06': '(825, 537)',
                 'I07': '(823, 461)',
                 'I08': '(997, 571)',
                 'I09': '(940, 516)',
                 'I10': '(872, 489)',
                 'I11': '(981, 477)',
                 'I12': '(1026, 495)',
                 'I13': '(1034, 444)',
                 'I14': '(935, 440)',
                 'I15': '(709, 686)',
                 'I16': '(694, 611)',
                 'I17': '(758, 649)',
                 'I18': '(760, 571)',
                 'I19': '(818, 684)',
                 'I20': '(880, 645)',
                 'I21': '(817, 606)',
                 'I22': '(881, 565)',
                 'I23': '(946, 682)',
                 'I24': '(1066, 679)',
                 'I25': '(1063, 604)',
                 'I26': '(995, 643)',
                 'I27': '(940, 604)',
                 'I28': '(758, 801)',
                 'I29': '(701, 766)',
                 'I30': '(825, 763)',
                 'I31': '(759, 736)',
                 'I32': '(871, 716)',
                 'I33': '(939, 739)',
                 'I34': '(873, 801)',
                 'I35': '(1023, 727)',
                 'I36': '(996, 789)',
                 'I37': '(1061, 782)',
                 'I38': '(982, 529)',
                 'I86': '(792, 385)',
                 'I80': '(880, 416)',
                 'I61': '(928, 370)',
                 'I59': '(928, 334)',
                 'I60': '(1036, 374)',
                 'I47': '(1085, 411)',
                 'I44': '(1144, 409)'}
# for inj in [k for k in INJ_PAD_KEYS.keys() if INJ_PAD_KEYS[k] in ['A', '15-05', '16-05', '11-05', '10-05',
#                                                                   '09-05', '06-05', '08-05']]:
#     INJ_relcoords[inj] = input(prompt='Please enter coordinates for injector {}'.format(inj))
for k, v in INJ_relcoords.items():
    # String to tuple
    INJ_relcoords[k] = eval(v)
    v = INJ_relcoords[k]
    # Re-scaling
    INJ_relcoords[k] = (v[0] - x_delta, y_delta - v[1])


# # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # #
# PRODUCER COORDINATE TRANSFORMATIONS

PRO_relcoords = {}
PRO_relcoords = {'AP2': '(616, 512) <> (683, 557) <> (995, 551)',
                 'AP3': '(601, 522) <> (690, 582) <> (995, 582)',
                 'AP4': '(621, 504) <> (691, 526) <> (1052, 523)',
                 'AP5': '(616, 483) <> (688, 505) <> (759, 507) <> (1058, 501)',
                 'AP6': '(606, 470) <> (685, 478) <> (827, 472) <> (910, 472)',
                 'AP7': '(602, 461) <> (674, 456) <> (846, 452) <> (992, 450)',
                 'AP8': '(593, 456) <> (674, 429) <> (1009, 427)',
                 'BP1': '(541, 733) <> (609, 654) <> (674, 633) <> (916, 636) <> (992, 635) <> (1015, 629)',
                 'BP2': '(541, 747) <> (630, 670) <> (1014, 668)',
                 'BP3': '(541, 760) <> (647, 704) <> (1016, 697)',
                 'BP4': '(555, 772) <> (691, 752) <> (908, 750)',
                 'BP5': '(555, 784) <> (838, 786) <> (1010, 748)',
                 'BP6': '(555, 803) <> (690, 821) <> (1026, 817)'}
# Get relative position inputs
# for well in [pw for pw in PRO_PAD_KEYS.keys() if 'A' in pw or 'B' in pw]:
#     PRO_relcoords[well] = input(prompt='Please enter coordinates for producer {}'.format(well))
# Re-format relative positions
for k, v in PRO_relcoords.items():
    # Parsing
    PRO_relcoords[k] = [chunk.strip() for chunk in v.split('<>')]
    # String to tuple
    PRO_relcoords[k] = [eval(chunk) for chunk in PRO_relcoords[k]]
# Re-scale relative positions (cartesian, not jS, system)
for k, v in PRO_relcoords.items():
    transformed = []
    for coordinate in PRO_relcoords[k]:
        transformed.append((coordinate[0] - x_delta, y_delta - coordinate[1]))
    PRO_relcoords[k] = transformed
# Find n points connecting the points
PRO_finalcoords = {}
for k, v in PRO_relcoords.items():
    discrete_links = []
    for coordinate_i in range(len(PRO_relcoords[k]) - 1):
        c1 = PRO_relcoords[k][coordinate_i]
        c2 = PRO_relcoords[k][coordinate_i + 1]
        num_points = int(euclidean_2d_distance(c1, c2) / focal_period)
        discrete_ind = intermediates(c1, c2, nb_points=num_points)
        discrete_links.append(discrete_ind)
    PRO_finalcoords[k] = list(chain.from_iterable(discrete_links))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # NAIVE INJECTOR SELECTION ALGO # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

candidates_by_prodpad = {}
reports_by_prodpad = {}
for pad in available_pads_transformed:
    candidates, report = injector_candidates(production_pad=pad,
                                             production_well=None,
                                             pro_well_pad_relationship=PRO_PAD_KEYS,
                                             injector_coordinates=INJ_relcoords,
                                             production_coordinates=PRO_finalcoords,
                                             relative_radius=50)
    candidates_by_prodpad[pad] = candidates
    reports_by_prodpad[pad] = report

candidates_by_prodwell = {}
reports_by_prodwell = {}
for pwell in available_pwells_transformed:
    candidates, report = injector_candidates(production_pad=None,
                                             production_well=pwell,
                                             pro_well_pad_relationship=None,
                                             injector_coordinates=INJ_relcoords,
                                             production_coordinates=PRO_finalcoords,
                                             relative_radius=50)
    candidates_by_prodwell[pwell] = candidates
    reports_by_prodwell[pwell] = report


if(plot_geo):
    plot_geo_candidates(candidates_by_prodwell, 'BP6', PRO_finalcoords, INJ_relcoords)


#
#
#
#
#
#


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # PRODUCER-INJECTOR DIST. MATRIX # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
pro_inj_distance = pd.DataFrame([], columns=INJ_relcoords.keys(), index=PRO_finalcoords.keys())
pro_inj_distance = pro_inj_distance.rename_axis('PRO_Well').reset_index()
operat = 'mean'
for injector in INJ_relcoords.keys():
    iwell_coord = INJ_relcoords.get(injector)
    PRO_Well_uniques = pro_inj_distance['PRO_Well']
    inj_specific_distances = []
    for pwell in PRO_Well_uniques:
        pwell_coords = PRO_finalcoords.get(pwell)
        point_distances = [euclidean_2d_distance(pwell_coord, iwell_coord) for pwell_coord in pwell_coords]
        dist_store = np.mean(point_distances) if operat == 'mean' else min(point_distances)
        inj_specific_distances.append(dist_store)
    pro_inj_distance[injector] = inj_specific_distances

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # PAD-LEVEL SENSOR DATA # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # #
# PRODUCER PAD-LEVEL AGGREGATOIN
unique_pro_pads = list(FINALE['PRO_Pad'].unique())
all_pro_data = ['PRO_Well',
                'PRO_Pump_Speed',
                'PRO_Time_On',
                'PRO_Casing_Pressure',
                'PRO_Heel_Pressure',
                'PRO_Toe_Pressure',
                'PRO_Heel_Temp',
                'PRO_Toe_Temp',
                'PRO_Pad',
                'PRO_Duration',
                'PRO_Alloc_Oil',
                'PRO_Alloc_Water',
                'Bin_1',
                'Bin_2',
                'Bin_3',
                'Bin_4',
                'Bin_5']

FINALE_pro = FINALE[all_pro_data + ['Date']]
FINALE_pro, dropped_cols = drop_singles(FINALE_pro)

# Smoothen out the Oil Data
unique_pro_wells = list(FINALE_pro['PRO_Well'].unique())

aggregation_dict = {'PRO_Pump_Speed': 'sum',
                    'PRO_Time_On': 'mean',
                    'PRO_Casing_Pressure': 'mean',
                    'PRO_Heel_Pressure': 'mean',
                    'PRO_Toe_Pressure': 'mean',
                    'PRO_Heel_Temp': 'mean',
                    'PRO_Toe_Temp': 'mean',
                    # 'PRO_Duration': 'mean',
                    'PRO_Alloc_Oil': 'sum',
                    'PRO_Alloc_Water': 'sum',
                    'Bin_1': 'mean',
                    'Bin_2': 'mean',
                    'Bin_3': 'mean',
                    'Bin_4': 'mean',
                    'Bin_5': 'mean'}

FINALE_agg_pro = FINALE_pro.groupby(by=['Date', 'PRO_Pad'], axis=0,
                                    sort=False, as_index=False).agg(aggregation_dict)

if(plot_eda):
    # FIGURE PLOTTING (PRODUCTION PAD-LEVEL STATISTICS)
    master_rows = len(unique_pro_pads)
    master_cols = len(FINALE_agg_pro.select_dtypes(float).columns)
    fig, ax = plt.subplots(nrows=master_rows, ncols=master_cols, figsize=(200, 50))
    for pad in unique_pro_pads:
        temp_pad = FINALE_agg_pro[FINALE_agg_pro['PRO_Pad'] == pad].sort_values('Date').reset_index(drop=True)
        d_1 = list(temp_pad['Date'])[0]
        d_n = list(temp_pad['Date'])[-1]
        numcols = FINALE_agg_pro.select_dtypes(float).columns
        for col in numcols:
            temp = temp_pad[[col]]
            temp = temp.interpolate('linear')
            # if all(temp.isna()):
            #     temp = temp.fillna(0)
            subp = ax[unique_pro_pads.index(pad)][list(numcols).index(col)]
            subp.plot(temp[col], label='Producer ' + pad + ', Metric ' +
                      col + '\n{} > {}'.format(d_1, d_n))
            subp.legend()
            plt.tight_layout()
        plt.tight_layout()

    plt.savefig('pro_pads_cols_ts.png')

# # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # #
# PRODUCER WELL-LEVEL AGGREGATION
FINALE_agg_pro_pwell = FINALE_pro.groupby(by=['Date', 'PRO_Well'], axis=0,
                                          sort=False, as_index=False).agg(aggregation_dict)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # INJECTOR PAD-LEVEL STEAM ALLOCATION # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # #
# INJECTOR BY PRODUCER PAD-LEVEL AGGREGATION
# This is just to delete the implicit duplicates found in the data (each production well has the same injection data)
FINALE_inj = FINALE[FINALE['PRO_Well'] == 'AP3'].reset_index(drop=True).drop('PRO_Well', 1)
all_injs = [c for c in FINALE_inj.columns if 'I' in c and '_' not in c]

# INJECTOR_AGGREGATES = {}
# for propad, pad_candidates in candidates_by_prodpad.items():
#     # Select candidates (not all the wells)
#     local_candidates = pad_candidates.copy()
#     absence = []
#     for cand in local_candidates:
#         if cand not in all_injs:
#             print('> STATUS: Candidate {} removed, unavailable in initial data'.format(cand))
#             absence.append(cand)
#     local_candidates = [el for el in local_candidates if el not in absence]
#
#     FINALE_melted_inj = pd.melt(FINALE_inj, id_vars=['Date'], value_vars=local_candidates,
#                                 var_name='Injector', value_name='Steam')
#     FINALE_melted_inj['INJ_Pad'] = FINALE_melted_inj['Injector'].apply(lambda x: INJ_PAD_KEYS.get(x))
#     FINALE_melted_inj = FINALE_melted_inj[~FINALE_melted_inj['INJ_Pad'].isna()].reset_index(drop=True)
#     # To groupby injector pads, by=['Date', 'INJ_Pad']
#     FINALE_agg_inj = FINALE_melted_inj.groupby(by=['Date'], axis=0, sort=False, as_index=False).sum()
#     FINALE_agg_inj['PRO_Pad'] = propad
#     INJECTOR_AGGREGATES[propad] = FINALE_agg_inj
# INJECTOR_AGGREGATES = pd.concat(INJECTOR_AGGREGATES.values()).reset_index(drop=True)


INJECTOR_AGGREGATES = produce_injector_aggregates(candidates_by_prodpad, FINALE_inj, 'PRO_Pad')
INJECTOR_AGGREGATES_PWELL = produce_injector_aggregates(candidates_by_prodwell, FINALE_inj, 'PRO_Well')


if(plot_eda):
    # FIGURE PLOTTING (INJECTION PAD-LEVEL STATISTICS)
    unique_inj_pads = list(FINALE_agg_inj['INJ_Pad'].unique())
    fig, ax = plt.subplots(nrows=len(unique_inj_pads), figsize=(15, 80))
    for pad in unique_inj_pads:
        subp = ax[unique_inj_pads.index(pad)]
        temp = FINALE_agg_inj[FINALE_agg_inj['INJ_Pad'] == pad].sort_values('Date').reset_index(drop=True)
        d_1 = list(temp['Date'])[0]
        d_n = list(temp['Date'])[-1]
        subp.plot(temp['Steam'], label='Injector ' + pad + '\n{} > {}'.format(d_1, d_n))
        subp.legend()
        plt.tight_layout()

    plt.savefig('inj_pads_ts.png')

# # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # #
# INJECTOR BY PRODUCER PAD-LEVEL AGGREGATION


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # #  COMBINE INJECTOR AND PRODUCER AGGREGATIONS # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
PRODUCER_AGGREGATES = FINALE_agg_pro[FINALE_agg_pro['PRO_Pad'].isin(available_pads_transformed)]
PRODUCER_AGGREGATES_PWELL = FINALE_agg_pro_pwell[FINALE_agg_pro_pwell['PRO_Well'].isin(available_pwells_transformed)]

COMBINED_AGGREGATES = pd.merge(PRODUCER_AGGREGATES, INJECTOR_AGGREGATES,
                               how='inner', on=['Date', 'PRO_Pad']).drop(['PRO_Alloc_Water'], axis=1)

COMBINED_AGGREGATES_PWELL = pd.merge(PRODUCER_AGGREGATES_PWELL, INJECTOR_AGGREGATES_PWELL,
                                     how='inner', on=['Date', 'PRO_Well']).drop(['PRO_Alloc_Water'], axis=1)

COMBINED_AGGREGATES, dropped = drop_singles(COMBINED_AGGREGATES)
COMBINED_AGGREGATES_PWELL, dropped_pwell = drop_singles(COMBINED_AGGREGATES_PWELL)

# COMBINED_AGGREGATES_temp = COMBINED_AGGREGATES.rolling(window=7).mean()
# COMBINED_AGGREGATES_temp['PRO_Pad'] = COMBINED_AGGREGATES['PRO_Pad']
#
# plt.figure(figsize=(24, 16))
# # COMBINED_AGGREGATES['PRO_Oil'].plot()
# COMBINED_AGGREGATES[(COMBINED_AGGREGATES['PRO_Pad'] == 'A') & (COMBINED_AGGREGATES['PRO_Oil'] > 0)]['PRO_Oil'].plot()
# COMBINED_AGGREGATES[(COMBINED_AGGREGATES['PRO_Pad'] == 'B') & (COMBINED_AGGREGATES['PRO_Oil'] > 0)]['PRO_Oil'].plot()

# plt.figure(figsize=(24, 16))
# # COMBINED_AGGREGATES_temp['PRO_Oil'].plot()
# COMBINED_AGGREGATES_temp[(COMBINED_AGGREGATES_temp['PRO_Pad'] == 'A') &
#                          (COMBINED_AGGREGATES_temp['PRO_Oil'] > 0)]['PRO_Oil'].plot()
# COMBINED_AGGREGATES_temp[(COMBINED_AGGREGATES_temp['PRO_Pad'] == 'B') &
#                          (COMBINED_AGGREGATES_temp['PRO_Oil'] > 0)]['PRO_Oil'].plot()

COMBINED_AGGREGATES_PWELL.to_csv('Data/combined_ipc_aggregates_PWELL.csv')
COMBINED_AGGREGATES.to_csv('Data/combined_ipc_aggregates.csv')

# plt.figure(figsize=(10, 100))
# sns.heatmap(COMBINED_AGGREGATES.sort_values(['PRO_Pad']).select_dtypes(float))
# COMBINED_AGGREGATES.isna().sum()
# EOF

# EOF
