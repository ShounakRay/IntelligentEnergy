# @Author: Shounak Ray <Ray>
# @Date:   19-Mar-2021 08:03:81:813  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: approach_alternative.py
# @Last modified by:   Ray
# @Last modified time: 20-Apr-2021 15:04:63:631  GMT-0600
# @License: [Private IP]

import ast
import math
import os
import pickle
import sys
from datetime import datetime
from itertools import chain
from multiprocessing import Pool

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import Anomaly_Detection_PKG
except Exception:
    sys.path.append('/Users/Ray/Documents/Github/AnomalyDetection')
    from Anomaly_Detection_PKG import *

import warnings


def ensure_cwd(expected_parent):
    init_cwd = os.getcwd()
    sub_dir = init_cwd.split('/')[-1]

    if(sub_dir != expected_parent):
        new_cwd = init_cwd
        print(f'\x1b[91mWARNING: "{expected_parent}" folder was expected to be one level ' +
              f'lower than parent directory! Project CWD: "{sub_dir}" (may already be properly configured).\x1b[0m')
    else:
        new_cwd = init_cwd.replace('/' + sub_dir, '')
        print(f'\x1b[91mWARNING: Project CWD will be set to "{new_cwd}".')
        os.chdir(new_cwd)


if __name__ == '__main__':
    try:
        _EXPECTED_PARENT_NAME = os.path.abspath(__file__ + "/..").split('/')[-1]
    except Exception:
        _EXPECTED_PARENT_NAME = 'pipeline'
        print('\x1b[91mWARNING: Seems like you\'re running this in a Python interactive shell. ' +
              f'Expected parent is manually set to: "{_EXPECTED_PARENT_NAME}".\x1b[0m')
    ensure_cwd(_EXPECTED_PARENT_NAME)
    sys.path.insert(1, os.getcwd() + '/_references')
    sys.path.insert(1, os.getcwd() + '/' + _EXPECTED_PARENT_NAME)
    import _accessories
    import _context_managers
    import _multiprocessed.defs as defs
    import _traversal


# _traversal.print_tree_to_txt(PATH='_configs/FILE_STRUCTURE.txt')

_ = """
#######################################################################################################################
#############################################   HYPERPARAMETER SETTINGS   #############################################
#######################################################################################################################
"""
plot_eda = False
plot_geo = False

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

MAX = 150.0

OLD_DUPLICATES = ['PRO_Alloc_Oil', 'PRO_Pump_Speed']

_ = """
#######################################################################################################################
#################################################   DATA INGESTIONS   #################################################
#######################################################################################################################
"""


def _DETECTION():
    return


def _AGGREGATION():
    return


FINALE = _accessories.retrieve_local_data_file('Data/combined_ipc_engineered_phys.csv')
PRO_PAD_KEYS = _accessories.retrieve_local_data_file('Data/INJECTION_[Well, Pad].pkl')
INJ_PAD_KEYS = _accessories.retrieve_local_data_file('Data/PRODUCTION_[Well, Pad].pkl')

available_pads_transformed = ['A', 'B']
available_pwells_transformed = [k for k, v in PRO_PAD_KEYS.items() if v in available_pads_transformed]

_ = """
#######################################################################################################################
###############################################   FUNCTION DEFINITIONS   ##############################################
#######################################################################################################################
"""


def filter_negatives(df, columns, out=True, placeholder=0):
    """Filter out all the rows in the DataFrame whose rows contain negatives.
    TODO: PENDING!!!! Groupby when finding means

    Parameters
    ----------
    df : DataFrame
        The intitial, unfiltered DataFrame.
    columns : pandas Index, or any iterable
        Which columns to consider.
    out : boolean
        Whether the "negative rows" should be deleted from the DataFrame.
    placeholder : int OR str
        The value or strategy used in "treating" the "negative rows."

    Returns
    -------
    DataFrame
        The filtered DataFrame, may contain NaNs (intentionally).

    """
    for num_col in columns:
        if(out):
            df = df[~(df[num_col] < 0)].reset_index(drop=True)
        else:
            if(placeholder == 'mean'):
                placeholder = df[num_col].mean()
            elif(placeholder == 'median'):
                placeholder = df[num_col].median()
            elif(type(placeholder) == int or type(placeholder) == float):
                pass
            else:
                raise ValueError(f'Incompatible argument entered for `placeholder`: {placeholder}')
            df.loc[df[num_col] < 0, num_col] = placeholder
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


def plot_aggregation_eda(df, resp_feature_1, resp_feature_2, wells_iterator, pad_val):
    fig, ax = plt.subplots(figsize=(50, 25), nrows=2, ncols=2)
    ax[0][0].set_title(f'Aggregation breakdown of {resp_feature_1}')
    for pwell in wells_iterator:
        _temp = df[df['PRO_Well'] == pwell][resp_feature_1].reset_index(drop=True)
        _temp.plot(ax=ax[0][0], linewidth=0.9)
    _temp = COMBINED_AGGREGATES[COMBINED_AGGREGATES['PRO_Pad'] == pad_val][resp_feature_1]
    (_temp / 1).plot(ax=ax[0][0], c='black')

    ax[0][1].set_title(f'Histogram of {resp_feature_1}')
    ax[0][1].hist(_temp, bins=200)

    ax[1][0].set_title(f'Aggregation breakdown of {resp_feature_2}')
    for pwell in wells_iterator:
        _temp = df[df['PRO_Well'] == pwell][resp_feature_2].reset_index(drop=True)
        _temp.plot(ax=ax[1][0], linewidth=0.9)
    _temp = COMBINED_AGGREGATES[COMBINED_AGGREGATES['PRO_Pad'] == pad_val][resp_feature_2]
    (_temp / 1).plot(ax=ax[1][0], c='black')

    ax[1][1].set_title(f'Histogram of {resp_feature_2}')
    ax[1][1].hist(_temp, bins=200)


def plot_weights_eda(df, groupby_val, groupby_col, time_col='Date', weight_col='weight', col_thresh=None):
    plt.figure(figsize=(30, 20))
    _temp = df[df[groupby_col] == groupby_val].sort_values(time_col).reset_index(drop=True)
    if(col_thresh is None):
        iter_cols = _temp.columns
    else:
        iter_cols = _temp.columns[:col_thresh]
    # Plot all features (normalized to 100 max)
    for col in [c for c in iter_cols if c not in ['Date', 'PRO_Pad', 'PRO_Well', 'weight', 'PRO_Alloc_Oil']]:
        __temp = _temp[['Date', col]].copy().fillna(_temp[col].mean())
        __temp[col] = MAX * __temp[col] / (MAX if max(__temp[col]) is np.nan else max(__temp[col]))
        # plt.hist(__temp[col], bins=100)
        if(col in ['PRO_Adj_Alloc_Oil', 'Steam', 'PRO_Adj_Pump_Speed']):
            lw = 0.75
        else:
            lw = 0.3
        plt.plot(__temp[time_col], __temp[col], linewidth=lw, label=col)
    plt.legend(loc='upper left', ncol=2)

    # # Plot weight
    plt.plot(_temp[time_col], _temp[weight_col])
    plt.title(f'Weight Time Series for {groupby_col} = {groupby_val}')
    plt.savefig(f'Manipulation Reference Files/Weight TS {groupby_col} = {groupby_val}.png')


def specialized_anomaly_detection(FINALE):
    def perform_detection(TEMP, injector_names, all_prod_wells, benchmark_plot=False):
        all_continuous_columns = TEMP.select_dtypes(np.number).columns

        # Conduct Anomaly Detection
        cumulative_tagged = []          # Stores the outputs from the customized, anomaly detection function
        temporal_benchmarks = []        # Used to track multi-process benchmarking for utlimate vizualization
        process_init = datetime.now()
        for cont_col in all_continuous_columns:
            rep_multiplier = len(all_prod_wells) if cont_col not in injector_names else 1
            arguments = list(zip([TEMP] * rep_multiplier, all_prod_wells, [cont_col] * rep_multiplier))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if __name__ == '__main__':
                    with Pool(os.cpu_count() - 1) as pool:
                        data_outputs = pool.starmap(defs.process_local_anomalydetection, arguments)
            cumulative_tagged.extend(data_outputs)

            process_final = datetime.now()
            temporal_benchmarks.append((cont_col, (process_final - process_init).total_seconds()))
            _accessories._print('> STATUS: {} % progress with `{}`'.format(
                round(100.0 * (list(all_continuous_columns).index(cont_col) + 1) / len(all_continuous_columns), 3),
                cont_col), color='CYAN')

        if(benchmark_plot):
            fig_path = 'Manipulation Reference Files/Benchmarking Anomaly Detection.png'
            # Save benchmarking materials
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot([t[1] for t in temporal_benchmarks])
            ax.set_xlabel('Feature (the middle are injectors)')
            ax.set_ylabel('Cumulative Time (s)')
            ax.set_title('Multiprocessing – Benchmarking Anomaly Detection')
            _accessories.auto_make_path(fig_path)
            fig.savefig(fig_path)

        # Reformat the tagged anomalies to a format more accessible DataFrame
        anomaly_tag_tracker = pd.concat(cumulative_tagged).reset_index(drop=True)

        return anomaly_tag_tracker

    def coalesce_datasets(anomaly_tag_tracker, FINALE, injector_names):
        sole_injector_data = anomaly_tag_tracker[anomaly_tag_tracker['feature'].isin(   # Only the injector data
            injector_names)].reset_index(drop=True).drop('group', axis=1)

        # Duplicate the injector data for all available producer wells
        rep_tracker = []
        for pwell in all_prod_wells:
            sole_injector_data['group'] = pwell
            rep_tracker.append(sole_injector_data.copy())
        sole_injector_data_rep = pd.concat(rep_tracker).reset_index(drop=True)
        sole_injector_data_rep['feature'] = sole_injector_data_rep['feature'].apply(lambda x: col_reference.get(x))
        sole_injector_data_rep.drop(['selection', 'detection_iter', 'anomaly'], axis=1, inplace=True)

        # NOTE: Do not consider old pump speed and allocated oil value in the weighting transformations
        _temp_OLD_DUPLICATES = [c.lower() for c in OLD_DUPLICATES]
        anomaly_tag_tracker = anomaly_tag_tracker[
            ~anomaly_tag_tracker['feature'].isin(_temp_OLD_DUPLICATES)].reset_index(drop=True)
        anomaly_tag_tracker.drop(['selection', 'detection_iter', 'anomaly'], axis=1, inplace=True)

        # Concatenate the non-injector and the duplicated-injector tables together
        cumulative_anomalies = pd.concat([sole_injector_data_rep, anomaly_tag_tracker], axis=0).reset_index(drop=True)

        # Find the daily, unweighted average of the column-specific weights
        reformatted_anomalies = cumulative_anomalies.groupby(['group', 'date'])[['updated_score']].mean().reset_index()
        reformatted_anomalies.columns = ['PRO_Well', 'Date', 'weight']
        reformatted_anomalies = reformatted_anomalies.infer_objects()

        # Date Formatting (just to be sure)
        FINALE['Date'] = pd.to_datetime(FINALE['Date'])
        reformatted_anomalies['Date'] = pd.to_datetime(reformatted_anomalies['Date'])

        # Merge this anomaly data into the original, highest-resolution base table
        FINALE = pd.merge(FINALE.infer_objects(), reformatted_anomalies, 'inner', on=['Date', 'PRO_Well'])

        return FINALE

    # Anomaly Detection Preparation
    TEMP = FINALE.copy()
    TEMP.columns = list(TEMP.columns.str.lower())
    col_reference = dict(zip(TEMP.columns, FINALE.columns))
    lc_injectors = [k for k, v in col_reference.items() if 'I' in v]
    all_prod_wells = list(FINALE['PRO_Well'].unique())

    anomaly_tag_tracker = perform_detection(TEMP, injector_names=lc_injectors,
                                            all_prod_wells=all_prod_wells, benchmark_plot=True)

    anom_original_merged = coalesce_datasets(anomaly_tag_tracker, FINALE, injector_names=lc_injectors)

    return anom_original_merged


_ = """
#######################################################################################################################
#############################################   MINOR DATA PRE-PROCESSING  ############################################
#######################################################################################################################
"""

FINALE['PRO_Pad'] = FINALE['PRO_Well'].apply(lambda x: PRO_PAD_KEYS.get(x))
FINALE = FINALE.dropna(subset=['PRO_Well']).reset_index(drop=True)

FINALE = filter_negatives(FINALE, FINALE.select_dtypes(float), out=True)
# FINALE.drop(FINALE.columns[0], axis=1, inplace=True)

_ = """
#######################################################################################################################
################################################   ANOMALY DETECTION   ################################################
#######################################################################################################################
"""
FINALE = specialized_anomaly_detection(FINALE)

os.system('say finished anomaly detection')

_ = """
#######################################################################################################################
########################################   INJECTOR > RELATIVE REPRESENTATION   #######################################
#######################################################################################################################
"""


def get_injector_coordinates():
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
    return INJ_relcoords


_ = """
#######################################################################################################################
########################################   PRODUCER > RELATIVE REPRESENTATION   #######################################
#######################################################################################################################
"""

# NORTHING AND EASTING INPUTS
# all_file_paths = [f for f in sorted(
#     ['Data/Coordinates/' + c for c in list(os.walk('Data/Coordinates'))[0][2]]) if '.txt' in f]
# all_wells = list(FINALE['PRO_Well'].unique())
# all_pads = list(FINALE['PRO_Pad'].unique())
#
# liner_bounds = pd.read_excel('Data/Coordinates/Liner Depths (measured depth).xlsx').infer_objects()
#
# all_files = {}
# all_positions = {}
# for file_path in all_file_paths:
#     well_group = str([group for group in all_wells + ['I2'] if group in file_path][0])
#     lines = open(file_path, 'r', errors='ignore').readlines()
#     all_files[file_path] = lines
#
#     try:
#         data_line = [line.split('\n') for line in ''.join(
#             map(str, lines)).split('\n\n') if 'Local Coordinates' in line][0]
#         data_line = [line.split('\t') for line in data_line if line != '']
#     except Exception:
#         data_line = [line[0].split('\t') for line in data_line if line != '']
#     data_start_index = sorted([data_line.index(line) for line in data_line if '0.0' in line[0]])[0]
#     data_string = data_line[data_start_index:]
#     data_string = [re.sub(' +', ' ', line[0].strip()) + '\n' for line in data_string]
#     dummy_columns = ''.join(map(str, ['col_' + str(i) + ' ' for i in range(len(data_string[0].split(' ')))])) + '\n'
#     str_obj_input = StringIO(dummy_columns + ''.join(map(str, data_string)))
#     df = pd.read_csv(str_obj_input, sep=' ', error_bad_lines=False).dropna(1).infer_objects()
#     df = df.select_dtypes(np.number)
#     df.columns = ['Depth', 'Incl', 'Azim', 'SubSea_Depth', 'Vertical_Depth', 'Local_Northing', 'Local_Easting',
#                   'UTM_Northing', 'UTM_Easting', 'Vertical_Section', 'Dogleg']
#     # df = df[['UTM_Easting', 'UTM_Northing']]
#
#     start_bound = float(liner_bounds[liner_bounds['Well'] == well_group]['Liner Start (mD)'])
#     end_bound = float(liner_bounds[liner_bounds['Well'] == well_group]['Liner End (mD)'])
#     final_df = df[(df['Depth'] > start_bound) & (df['Depth'] < end_bound)]
#     all_positions[well_group] = final_df.sort_values('Depth').reset_index(drop=True)
#
# fig, ax = plt.subplots(nrows=len(all_positions.keys()), ncols=2, figsize=(15, 40))
# for well in all_positions.keys():
#     group_df = all_positions[well]  # /all_positions[well].max()
#     axes_1 = ax[list(all_positions.keys()).index(well)][0]
#     axes_1.plot(group_df['UTM_Easting'], group_df['UTM_Northing'])
#     axes_1.set_xlabel('UTM Easting')
#     axes_1.set_ylabel('UTM Northing')
#     axes_1.set_title(well + ' UTM Coordinates')
#     # axes_1.set_ylim(1000000 + 3.963 * 10**6, 1000000 + 5.963 * 10**6)
#
#     axes_2 = ax[list(all_positions.keys()).index(well)][1]
#     axes_2.plot(group_df['Local_Easting'], group_df['Local_Northing'])
#     axes_1.set_xlabel('Local Easting')
#     axes_1.set_xlabel('Local Northing')
#     axes_2.set_title(well + ' Local Coordinates')
#     axes_2.set_ylim(0, 225)
# plt.tight_layout()
# fig.suptitle('Coordinates Bounded by Provided Liner Depths XLSX File')
# plt.savefig('Modeling Reference Files/Candidate Selection Images/provided_coordinate_plots.png',
#             bbox_inches='tight')
#


def get_injector_coordinates():
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


_ = """
#######################################################################################################################
########################################   NAIVE INJECTOR SELECTION ALGORITHM   #######################################
#######################################################################################################################
"""

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

with open('Data/candidates_by_prodpad.pkl', 'wb') as fp:
    pickle.dump(candidates_by_prodpad, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open('Data/candidates_by_prodwell.pkl', 'wb') as fp:
    pickle.dump(candidates_by_prodwell, fp, protocol=pickle.HIGHEST_PROTOCOL)

os.system('say finished candidate selection')

_ = """
#######################################################################################################################
########################################   PRODUCER/INJECTOR DISTANCE MATRIX    #######################################
#######################################################################################################################
"""

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

pro_inj_distance.infer_objects().to_pickle('Data/injector_producer_dist_matrix.pkl')

os.system('say finished distance matrix')

_ = """
#######################################################################################################################
###########################################   PRODUCTION-DATA AGGREGATION   ###########################################
#######################################################################################################################
"""
_ = """
####################################
######  PAD-LEVEL AGGREGATION ######
####################################
"""
unique_pro_pads = list(FINALE['PRO_Pad'].unique())
all_pro_data = ['PRO_Well',
                'PRO_Adj_Pump_Speed',
                # 'PRO_Pump_Speed',
                # 'PRO_Time_On',
                'PRO_Casing_Pressure',
                'PRO_Heel_Pressure',
                'PRO_Toe_Pressure',
                'PRO_Heel_Temp',
                'PRO_Toe_Temp',
                'PRO_Pad',
                # 'PRO_Duration',
                'PRO_Adj_Alloc_Oil',
                # 'PRO_Alloc_Oil',
                'PRO_Total_Fluid',
                # 'PRO_Alloc_Water',
                'Bin_1',
                'Bin_2',
                'Bin_3',
                'Bin_4',
                'Bin_5',
                'Bin_6',
                'Bin_7',
                'Bin_8']

FINALE_pro = FINALE[all_pro_data + ['Date', 'weight']]
FINALE_pro, dropped_cols = drop_singles(FINALE_pro)

aggregation_dict = {'PRO_Adj_Pump_Speed': 'mean',
                    # 'PRO_Pump_Speed': 'sum',
                    # 'PRO_Time_On': 'mean',
                    'PRO_Casing_Pressure': 'mean',
                    'PRO_Heel_Pressure': 'mean',
                    'PRO_Toe_Pressure': 'mean',
                    'PRO_Heel_Temp': 'mean',
                    'PRO_Toe_Temp': 'mean',
                    'PRO_Adj_Alloc_Oil': 'sum',
                    # 'PRO_Duration': 'mean',
                    # 'PRO_Alloc_Oil': 'sum',
                    'PRO_Total_Fluid': 'sum',
                    # 'PRO_Alloc_Water': 'sum',
                    'Bin_1': 'mean',
                    'Bin_2': 'mean',
                    'Bin_3': 'mean',
                    'Bin_4': 'mean',
                    'Bin_5': 'mean',
                    'Bin_6': 'mean',
                    'Bin_7': 'mean',
                    'Bin_8': 'mean',
                    'weight': 'mean'}

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

os.system('say finished producer pad aggregation')

# _ = """
# ####################################
# #####  WELL-LEVEL AGGREGATION ######
# ####################################
# """
# FINALE_agg_pro_pwell = FINALE_pro.groupby(by=['Date', 'PRO_Well'], axis=0,
#                                           sort=False, as_index=False).agg(aggregation_dict)

_ = """
#######################################################################################################################
############################################   INJECTOR-DATA AGGREGATION   ############################################
#######################################################################################################################
"""
# This is just to delete the implicit duplicates found in the data (each production well has the same injection data)
FINALE_inj = FINALE[FINALE['PRO_Well'] == 'AP3'].reset_index(drop=True).drop('PRO_Well', 1)
all_injs = [c for c in FINALE_inj.columns if 'I' in c and '_' not in c]

_ = """
####################################
#####  PAD-LEVEL AGGREGATION #######
####################################
"""
INJECTOR_AGGREGATES = produce_injector_aggregates(candidates_by_prodpad, FINALE_inj, 'PRO_Pad')

# _ = """
# ####################################
# #####  WELL-LEVEL AGGREGATION ######
# ####################################
# """
# INJECTOR_AGGREGATES_PWELL = produce_injector_aggregates(candidates_by_prodwell, FINALE_inj, 'PRO_Well')

os.system('say finished injector aggregation')

_ = """
#######################################################################################################################
#####################################   INJECTOR/PRODUCER AGGREGATION – MERGING   #####################################
#######################################################################################################################
"""
_ = """
####################################
########  PAD-LEVEL MERGING ########
####################################
"""
PRODUCER_AGGREGATES = FINALE_agg_pro[FINALE_agg_pro['PRO_Pad'].isin(available_pads_transformed)]
COMBINED_AGGREGATES = pd.merge(PRODUCER_AGGREGATES, INJECTOR_AGGREGATES,
                               how='inner', on=['Date', 'PRO_Pad'])
COMBINED_AGGREGATES, dropped = drop_singles(COMBINED_AGGREGATES)

os.system('say finished producer aggregation')

COMBINED_AGGREGATES.infer_objects().to_csv('Data/combined_ipc_aggregates.csv')

os.system('say saved files to csv')

# _ = """
# ####################################
# ########  WELL-LEVEL MERGING #######
# ####################################
# """
# PRODUCER_AGGREGATES_PWELL = FINALE_agg_pro_pwell[FINALE_agg_pro_pwell['PRO_Well'].isin(available_pwells_transformed)]
# COMBINED_AGGREGATES_PWELL = pd.merge(PRODUCER_AGGREGATES_PWELL, INJECTOR_AGGREGATES_PWELL,
#                                      how='inner', on=['Date', 'PRO_Well'])
# COMBINED_AGGREGATES_PWELL, dropped_pwell = drop_singles(COMBINED_AGGREGATES_PWELL)
# COMBINED_AGGREGATES_PWELL.infer_objects().to_csv('Data/combined_ipc_aggregates_PWELL.csv')

# _ = """
# ####################################
# ##  AGGREGATION EDA – WELL LEVEL ###
# ####################################
# """
# plot_aggregation_eda(COMBINED_AGGREGATES, 'PRO_Adj_Alloc_Oil', 'weight',
#                      available_pwells_transformed[:7], 'A')
#
# os.system('say finished plotting aggregation')

_ = """
####################################
###########  WEIGHT EDA ############
####################################
"""
for pad in available_pads_transformed:
    plot_weights_eda(COMBINED_AGGREGATES, groupby_val=pad, groupby_col='PRO_Pad', time_col='Date', weight_col='weight')

# for pwell in available_pwells_transformed:
#     plot_weights_eda(COMBINED_AGGREGATES_PWELL, groupby_val=pwell,
#                      groupby_col='PRO_Well', time_col='Date', weight_col='weight')

os.system('say finished weight exploratory analysis')


# plt.figure(figsize=(10, 100))
# sns.heatmap(COMBINED_AGGREGATES.sort_values(['PRO_Pad']).select_dtypes(float))
# COMBINED_AGGREGATES.isna().sum()
# EOF

# EOF
