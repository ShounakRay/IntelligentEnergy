# @Author: Shounak Ray <Ray>
# @Date:   19-Mar-2021 08:03:81:813  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: approach_alternative.py
# @Last modified by:   Ray
# @Last modified time: 19-Mar-2021 10:03:13:132  GMT-0600
# @License: [Private IP]

import math

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
DATA_INJECTION_ORIG = DATA_PRODUCTION_ORIG = pd.read_pickle('Data/Pickles/DATA_INJECTION_ORIG.pkl')

# INJECTION PAD PROCESSING
DATA_INJECTION = DATA_INJECTION_ORIG.reset_index(drop=True)
DATA_INJECTION.columns = ['Date', 'Pad', 'Well', 'UWI_Identifier', 'Time_On',
                          'Alloc_Steam', 'Meter_Steam', 'Casing_Pressure',
                          'Tubing_Pressure', 'Reason', 'Comment']
DATA_INJECTION_KEYS = ['Date', 'Pad', 'Well', 'Meter_Steam',
                       'Casing_Pressure', 'Tubing_Pressure']
DATA_INJECTION = DATA_INJECTION[DATA_INJECTION_KEYS]
DATA_INJECTION['Pressure'] = DATA_INJECTION.apply(pressure_lambda, axis=1)
DATA_INJECTION.drop(['Casing_Pressure', 'Tubing_Pressure'],
                    axis=1, inplace=True)
DATA_INJECTION = filter_negatives(DATA_INJECTION,
                                  DATA_INJECTION.select_dtypes(
                                      include=['float64']).columns)
DATA_INJECTION = convert_to_date(DATA_INJECTION, 'Date')
DATA_INJECTION = DATA_INJECTION.infer_objects()

# FINALE PROCESSING
FINALE.columns = [elem.replace('date',
                               'Date').replace('bin_',
                                               'Bin_').replace('producer_well',
                                                               'Well') for elem in list(FINALE.columns)]
FINALE = FINALE.dropna(subset=['Well']).reset_index(drop=True)
FINALE.drop(FINALE.columns[-6:-1], 1, inplace=True)
FINALE = FINALE[FINALE['Well'] == 'AP3'].reset_index(drop=True).drop('Well', 1)
FINALE_melted = pd.melt(FINALE, id_vars=['Date'],
                        value_vars=FINALE.columns[1:-1], var_name='Injector', value_name='Steam')

INJ_PAD_KEYS = dict(zip(DATA_INJECTION['Well'], DATA_INJECTION['Pad']))

FINALE_melted['Pad'] = FINALE_melted['Injector'].apply(lambda x: INJ_PAD_KEYS.get(x))
FINALE_agg = FINALE_melted.groupby(by=['Date', 'Pad'], axis=0, sort=False, as_index=False).sum()

unique_inj_pads = list(FINALE_agg['Pad'].unique())
fig, ax = plt.subplots(nrows=len(unique_inj_pads), figsize=(15, 80))
for pad in unique_inj_pads:
    subp = ax[unique_inj_pads.index(pad)]
    temp = FINALE_agg[FINALE_agg['Pad'] == pad].sort_values('Date').reset_index(drop=True)
    subp.plot(temp['Steam'] / max(temp['Steam']), label=pad)
    subp.legend()
