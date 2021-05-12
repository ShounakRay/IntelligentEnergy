# @Author: Shounak Ray <Ray>
# @Date:   27-Apr-2021 09:04:39:394  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: test.py
# @Last modified by:   Ray
# @Last modified time: 12-May-2021 11:05:36:365  GMT-0600
# @License: [Private IP]


import ast
import math

import _references._accessories as _accessories
import matplotlib.pyplot as plt
import pandas as pd

_ = """
#######################################################################################################################
##################################################   STEAM DATA PREP   ################################################
#######################################################################################################################
"""


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


__FOLDER__ = r'Data/Isolated/'
PATH_INJECTION = __FOLDER__ + r'OLT injection data.xlsx'
DATA_INJECTION_ORIG = pd.read_excel(PATH_INJECTION)

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
# > Pivot so columns feature all injection wells and cells are steam values
DATA_INJECTION = DATA_INJECTION.pivot_table(['Meter_Steam', 'Pressure'],
                                            'Date', ['Well'], dropna=False)
DATA_INJECTION.columns.names = (None, None)
DATA_INJECTION_PRESS = DATA_INJECTION['Pressure']
list(DATA_INJECTION_PRESS)
_ = """
#######################################################################################################################
##################################################   CANDIDATE CONSIDERATION   ########################################
#######################################################################################################################
"""
# {'FP1': ['I37', 'I72', 'I70'],
#  'FP2': ['I64', 'I73', 'I69', 'I37', 'I72', 'I70'],
#  'FP3': ['I64', 'I74', 'I73', 'I69', 'I71'],
#  'FP4': ['I74', 'I71', 'I75', 'I76'],
#  'FP5': ['I67', 'I75', 'I77', 'I76', 'I66'],
#  'FP6': ['I67', 'I65', 'I78', 'I77', 'I79', 'I68'],
#  'FP7': ['I65', 'I68', 'I79'],
#  'CP1': ['I25', 'I24', 'I26', 'I08'],
#  'CP2': ['I24', 'I49', 'I45', 'I46', 'I39', 'I47'],
#  'CP3': ['I47', 'I39', 'I46', 'I45', 'I49'],
#  'CP4': ['I44', 'I43', 'I45', 'I51', 'I48'],
#  'CP5': ['I40', 'I43', 'I51', 'I50'],
#  'CP6': ['I40', 'I41', 'I50', 'CI06'],
#  'CP7': ['I42', 'I41', 'CI06'],
#  'CP8': ['I41', 'I42', 'CI06'],
#  'EP2': ['I61', 'I60', 'I53'],
#  'EP3': ['I59', 'I52', 'I61', 'I60', 'I53'],
#  'EP4': ['I59', 'I52', 'I57', 'I54'],
#  'EP5': ['I62', 'I57', 'I56', 'I54'],
#  'EP6': ['I62', 'I56', 'I58', 'I55'],
#  'EP7': ['I63', 'I56', 'I55']}
CANDIDATES = _accessories.retrieve_local_data_file('Data/Pickles/WELL_Candidates.pkl', mode=2)
# Manually add other injector data
EXTRA = ['FP1', 'FP2', 'FP3', 'FP4', 'FP5', 'FP6', 'FP7'] + \
        ['CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CP7', 'CP8'] + \
        ['EP2', 'EP3', 'EP4', 'EP5', 'EP6', 'EP7']
# for extra_prod in EXTRA:
#     CANDIDATES[extra_prod] = ast.literal_eval(input(f'Candidates for {extra_prod}:'))
#     print('Stored...')
all_dfs = []
for pwell, cands in CANDIDATES.items():
    df = pd.DataFrame(DATA_INJECTION_PRESS[cands].mean(axis=1), columns=['Pressure_Average'])
    df['PRO_Well'] = pwell
    all_dfs.append(df)
concatenated = pd.concat(all_dfs).reset_index()

_accessories.save_local_data_file(concatenated, 'Data/candidate_selected_pressures_dTime.csv')

concatenated = concatenated[concatenated['PRO_Well'].isin(EXTRA)].reset_index(drop=True)
available = list(concatenated['PRO_Well'].unique())
fig, ax = plt.subplots(nrows=len(available), ncols=2, figsize=(10, 30), constrained_layout=True)
for pwell in available:
    # Get aggregated pressures
    axis = ax[available.index(pwell)][0]
    _temp = concatenated[concatenated['PRO_Well'] == pwell].sort_values('Date')
    axis.plot(_temp['Date'], _temp['Pressure_Average'])
    axis.set_title(f'Producer: {pwell}')

    axis = ax[available.index(pwell)][1]
    axis.text(0.5, 0.5, str(CANDIDATES.get(pwell)),
              horizontalalignment='center',
              verticalalignment='center')
    axis.axis('off')
fig.suptitle('Selective, Average Producer Pressures Over Time')
fig.savefig('Manipulation Reference Files/Final Schematics/Selective, Average Producer Pressures Over Time.png')
# plt.tight_layout()
