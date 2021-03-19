# @Author: Shounak Ray <Ray>
# @Date:   19-Mar-2021 08:03:81:813  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: approach_alternative.py
# @Last modified by:   Ray
# @Last modified time: 19-Mar-2021 16:03:34:348  GMT-0600
# @License: [Private IP]

import math
import sys

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


# INGESTION
FINALE = pd.read_csv('Data/combined_ipc.csv').infer_objects()
DATA_INJECTION_ORIG = pd.read_pickle('Data/Pickles/DATA_INJECTION_ORIG.pkl')
DATA_PRODUCTION_ORIG = pd.read_pickle('Data/Pickles/DATA_PRODUCTION_ORIG.pkl')

PRO_PAD_KEYS = dict(zip(DATA_PRODUCTION_ORIG['Well'], DATA_PRODUCTION_ORIG['Pad']))
INJ_PAD_KEYS = dict(zip(DATA_INJECTION_ORIG['Well'], DATA_INJECTION_ORIG['Pad']))

# list(FINALE_pro.columns)

# FINALE PROCESSING
finale_replace = {'date': 'Date',
                  'producer_well': 'Well',
                  'prod_casing_pressure': 'Casing_pressure',
                  'pad': 'Pad',
                  'prod_bhp_heel': 'Heel_Pressure',
                  'prod_bhp_toe': 'Toe_Pressure',
                  'hours_on_prod': 'Time_On'}
_ = [finale_replace.update({'bin_{}'.format(i): 'Bin_{}'.format(i)}) for i in range(1, 6)]
_ = [finale_replace.update({col: col}) for col in FINALE.columns if col not in finale_replace.keys()]

FINALE.drop(['op_approved', 'eng_approved', 'uwi'], 1, inplace=True)
FINALE.columns = FINALE.columns.to_series().map(finale_replace)
FINALE['Pad'] = FINALE['Well'].apply(lambda x: PRO_PAD_KEYS.get(x))
FINALE = FINALE.dropna(subset=['Well']).reset_index(drop=True)

# # # # # # # # # # # # # # # # # # # # PAD-LEVEL SENSOR DATA # # # # # # # # # # # # # # # # # # # # # # #

unique_pro_pads = list(FINALE['Pad'].unique())
all_pro_data = [*FINALE.columns[1:7], *FINALE.columns[-6:]]
FINALE_pro = FINALE[all_pro_data + ['Date']]

# FINALE_agg = FINALE_pro.groupby(by=['Date', 'Pad'], axis=0, sort=False, as_index=False).sum()
FINALE_melted_pro = pd.melt(FINALE, id_vars=['Date'], value_vars=all_pro_data, var_name='metric', value_name='Well')
FINALE_melted_pro['Pad'] = FINALE_melted_pro['Well'].apply(lambda x: PRO_PAD_KEYS.get(x))

FINALE_agg_pro = FINALE_pro.groupby(by=['Date', 'Pad'], axis=0,
                                    sort=False, as_index=False).agg({'spm': 'sum',
                                                                     'Time_On': 'mean',
                                                                     'Casing_pressure': 'mean',
                                                                     'Heel_Pressure': 'mean',
                                                                     'Toe_Pressure': 'mean',
                                                                     'test_oil': 'sum',
                                                                     'test_water': 'sum',
                                                                     'pump_size': 'mean',
                                                                     'pump_efficiency': 'mean'})
master_rows = len(unique_pro_pads)
master_cols = len(FINALE_agg_pro.select_dtypes(float).columns)
fig, ax = plt.subplots(nrows=master_rows, ncols=master_cols, figsize=(200, 50))
for pad in unique_pro_pads:
    temp_pad = FINALE_agg_pro[FINALE_agg_pro['Pad'] == pad].sort_values('Date').reset_index(drop=True)
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

FINALE_inj = FINALE[FINALE['Well'] == 'AP3'].reset_index(drop=True).drop('Well', 1)
all_injs = [c for c in FINALE_inj.columns if 'I' in c]
FINALE_melted_inj = pd.melt(FINALE_inj, id_vars=['Date'], value_vars=all_injs,
                            var_name='Injector', value_name='Steam')
FINALE_melted_inj['Pad'] = FINALE_melted_inj['Injector'].apply(lambda x: INJ_PAD_KEYS.get(x))
FINALE_agg_inj = FINALE_melted_inj.groupby(by=['Date', 'Pad'], axis=0, sort=False, as_index=False).sum()

unique_inj_pads = list(FINALE_agg_inj['Pad'].unique())
fig, ax = plt.subplots(nrows=len(unique_inj_pads), figsize=(15, 80))
for pad in unique_inj_pads:
    subp = ax[unique_inj_pads.index(pad)]
    temp = FINALE_agg_inj[FINALE_agg_inj['Pad'] == pad].sort_values('Date').reset_index(drop=True)
    d_1 = list(temp['Date'])[0]
    d_n = list(temp['Date'])[-1]
    subp.plot(temp['Steam'], label='Injector ' + pad + '\n{} > {}'.format(d_1, d_n))
    subp.legend()
    plt.tight_layout()

plt.savefig('inj_pads_ts.png')


# # # # # # # # # # # # # # # # INJECTOR ASSOCIATIONS # # # # # # # # # # # # # # # # # # # #

dict(zip(FINALE_melted_inj['Injector'], FINALE_melted_inj['Pad']))
