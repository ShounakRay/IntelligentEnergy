# @Author: Shounak Ray <Ray>
# @Date:   19-Mar-2021 08:03:81:813  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: approach_alternative.py
# @Last modified by:   Ray
# @Last modified time: 22-Mar-2021 13:03:34:341  GMT-0600
# @License: [Private IP]

import math
from itertools import chain

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
            for i in range(1, nb_points + 1)]


# INGESTION
FINALE = pd.read_csv('Data/combined_ipc.csv').infer_objects()
DATA_INJECTION_ORIG = pd.read_pickle('Data/Pickles/DATA_INJECTION_ORIG.pkl')
DATA_PRODUCTION_ORIG = pd.read_pickle('Data/Pickles/DATA_PRODUCTION_ORIG.pkl')

PRO_PAD_KEYS = dict(zip(DATA_PRODUCTION_ORIG['Well'], DATA_PRODUCTION_ORIG['Pad']))
INJ_PAD_KEYS = dict(zip(DATA_INJECTION_ORIG['Well'], DATA_INJECTION_ORIG['Pad']))

# list(FINALE.columns)

# [OPTIONAL] FINALE PRE-PROCESSING
# finale_replace = {'date': 'Date',
#                   'producer_well': 'Well',
#                   'prod_casing_pressure': 'Casing_pressure',
#                   'pad': 'Pad',
#                   'prod_bhp_heel': 'Heel_Pressure',
#                   'prod_bhp_toe': 'Toe_Pressure',
#                   'hours_on_prod': 'Time_On'}
# _ = [finale_replace.update({'bin_{}'.format(i): 'Bin_{}'.format(i)}) for i in range(1, 6)]
# _ = [finale_replace.update({col: col}) for col in FINALE.columns if col not in finale_replace.keys()]
# FINALE.drop(['op_approved', 'eng_approved', 'uwi'], 1, inplace=True)
# FINALE.columns = FINALE.columns.to_series().map(finale_replace)

FINALE['PRO_Pad'] = FINALE['PRO_Well'].apply(lambda x: PRO_PAD_KEYS.get(x))
FINALE = FINALE.dropna(subset=['PRO_Well']).reset_index(drop=True)

# # # # # # # # # # # # # # # # # # # # PAD-LEVEL SENSOR DATA # # # # # # # # # # # # # # # # # # # # # # #

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
                'PRO_Oil',
                'PRO_Water',
                'PRO_Gas',
                'PRO_Fluid',
                'PRO_Chlorides']
FINALE_pro = FINALE[all_pro_data + ['Date']]

FINALE_pro, dropped_cols = drop_singles(FINALE_pro)

# FINALE_agg = FINALE_pro.groupby(by=['Date', 'Pad'], axis=0, sort=False, as_index=False).sum()
FINALE_melted_pro = pd.melt(FINALE, id_vars=['Date'], value_vars=all_pro_data, var_name='metric', value_name='PRO_Pad')
FINALE_melted_pro['PRO_Pad'] = FINALE_melted_pro['PRO_Pad'].apply(lambda x: PRO_PAD_KEYS.get(x))

FINALE_agg_pro = FINALE_pro.groupby(by=['Date', 'PRO_Pad'], axis=0,
                                    sort=False, as_index=False).agg({'PRO_Pump_Speed': 'sum',
                                                                     'PRO_Time_On': 'mean',
                                                                     'PRO_Casing_Pressure': 'mean',
                                                                     'PRO_Heel_Pressure': 'mean',
                                                                     'PRO_Duration': 'mean',
                                                                     'PRO_Toe_Pressure': 'mean',
                                                                     'PRO_Oil': 'sum',
                                                                     'PRO_Water': 'sum',
                                                                     'PRO_Gas': 'sum',
                                                                     'PRO_Fluid': 'sum',
                                                                     'PRO_Chlorides': 'sum'})

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


# # # # # # # # # # # # # # # # INJECTOR PAD-LEVEL STEAM ALLOCATION # # # # # # # # # # # # # # # # # # # #

FINALE_inj = FINALE[FINALE['PRO_Well'] == 'AP3'].reset_index(drop=True).drop('PRO_Well', 1)
all_injs = [c for c in FINALE_inj.columns if 'I' in c and '_' not in c]
FINALE_melted_inj = pd.melt(FINALE_inj, id_vars=['Date'], value_vars=all_injs,
                            var_name='Injector', value_name='Steam')
FINALE_melted_inj['INJ_Pad'] = FINALE_melted_inj['Injector'].apply(lambda x: INJ_PAD_KEYS.get(x))
FINALE_melted_inj = FINALE_melted_inj[~FINALE_melted_inj['INJ_Pad'].isna()].reset_index(drop=True)
FINALE_agg_inj = FINALE_melted_inj.groupby(by=['Date', 'INJ_Pad'], axis=0, sort=False, as_index=False).sum()

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


# # # # # # # # # # # # # # # # INJECTOR ASSOCIATIONS # # # # # # # # # # # # # # # # # # # #

# Injector wells and injector pad relationships
INJ_well_to_pad = dict(zip(FINALE_melted_inj['Injector'], FINALE_melted_inj['INJ_Pad']))
# Injector wells to producer well overlaps (only those spanning different producer pads)
INJ_well_to_pad_overlaps = {'I16': ['AP3', 'BP1'],
                            'I21': ['AP3', 'BP1'],
                            'I27': ['AP3', 'BP1'],
                            'I37': ['FP1', 'BP6'],
                            'I72': ['FP1', 'BP6'],
                            'I47': ['CP2', 'EP2'],
                            'I44': ['CP3', 'EP2'],
                            'I42': ['CP6', 'EP2'],
                            'CI7': ['CP7', 'EP2'],
                            'CI8': ['CP8', 'EP2']}
PRO_injpad_to_well = {'E3': ['EP2', 'EP3', 'EP4', 'EP5', 'EP6', 'EP7'],
                      'E2': ['EP2', 'EP3', 'EP4', 'EP5', 'EP6', 'EP7'],
                      'E1': ['EP2', 'EP3', 'EP4', 'EP5', 'EP6', 'EP7'],
                      'A': ['AP4', 'AP5', 'AP6', 'AP7', 'A8'],
                      '15-05': ['AP4', 'AP6', 'AP7'],
                      '16-05': ['AP2', 'AP3', 'AP4', 'AP5', 'AP6', 'AP7']}

PRO_relcoords = {'AP2': '(696, 554) <> (995, 551)',
                 'AP3': '(700, 582) <> (995, 582)',
                 'AP4': '(691, 526) <> (1052, 523)',
                 'AP5': '(684, 498) <> (759, 507) <> (1058, 501)',
                 'AP6': '(695, 478) <> (827, 472) <> (910, 472)',
                 'AP7': '(693, 454) <> (846, 452) <> (992, 450)',
                 'AP8': '(690, 429) <> (910, 427)',
                 'BP1': '(684, 633) <> (992, 635) <> (1015, 620)',
                 'BP2': '(708, 668) <> (1014, 668)',
                 'BP3': '(703, 702) <> (1016, 697)',
                 'BP4': '(663, 752) <> (908, 750)',
                 'BP5': '(676, 783) <> (838, 786) <> (1010, 745)',
                 'BP6': '(690, 821) <> (1026, 817)'}
POS_TL = (478, 71)
POS_TR = (1439, 71)
POS_BL = (478, 1008)
POS_BR = (1439, 1008)
PRO_relcoords = {}
# Get relative position inputs
for well in [pw for pw in PRO_PAD_KEYS.keys() if 'A' in pw or 'B' in pw]:
    PRO_relcoords[well] = input(prompt='Please enter coordinates for {}'.format(well))
# Re-format relative positions
for k, v in PRO_relcoords.items():
    PRO_relcoords[k] = [chunk.strip() for chunk in v.split('<>')]
    PRO_relcoords[k] = [eval(chunk) for chunk in PRO_relcoords[k]]
# Re-scale relative positions (cartesian, not jS, system)
x_delta = POS_TL[0] | POS_BL[0]
y_delta = POS_TR[0] | POS_BR[0]
for k, v in PRO_relcoords.items():
    transformed = []
    for coordinate in PRO_relcoords[k]:
        transformed.append((coordinate[0] - x_delta, y_delta - coordinate[1]))
    PRO_relcoords[k] = transformed
# Find n points connecting the points
connections = {}
arbitrary_length = 40
for k, v in PRO_relcoords.items():
    discrete_links = []
    for coordinate_i in range(len(PRO_relcoords[k]) - 1):
        c1 = PRO_relcoords[k][coordinate_i]
        c2 = PRO_relcoords[k][coordinate_i + 1]
        x1 = c1[0]
        x2 = c2[0]
        y1 = c1[1]
        y2 = c2[1]
        num_points = int(math.hypot(x2 - x1, y2 - y1) / arbitrary_length)
        discrete_ind = intermediates(c1, c2, nb_points=num_points)
        discrete_links.append(discrete_ind)
    connections[k] = list(chain.from_iterable(discrete_links))
# Plot connections to confirm
colors = cm.rainbow(np.linspace(0, 1, len(connections.keys())))
for k, v in connections.items():
    all_x = [c[0] for c in v]
    all_y = [c[1] for c in v]
    plt.scatter(all_x, all_y, linestyle='solid', color=colors[list(connections.keys()).index(k)])
    plt.plot(all_x, all_y, color=colors[list(connections.keys()).index(k)])

# color=colors[list(PRO_relcoords.keys()).index(k)]
list(chain.from_iterable(discrete_links))
# EOF

# EOF
