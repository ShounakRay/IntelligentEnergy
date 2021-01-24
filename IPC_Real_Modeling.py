# @Author: Shounak Ray <Ray>
# @Date:   2021-01-21T13:48:32-07:00
# @Email:  rijshouray@gmail.com
# @Filename: IPC_Real_Modeling.py
# @Last modified by:   Ray
# @Last modified time: 24-Jan-2021 01:01:41:412  GMT-0700
# @License: No License for Distribution

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

import numpy as np
import pandas as pd
import pandas_profiling

"""Major Notes:
> Ensure Python Version in root directory matches that of local directory
"""

# import pickle
# from itertools import chain
# import timeit
# from functools import reduce
# import matplotlib.cm as cm
# import matplotlib.pyplot as plt

# Folder Specifications
__FOLDER__ = r'Data/'
PATH_INJECTION = __FOLDER__ + r'OLT injection data.xlsx'
PATH_PRODUCTION = __FOLDER__ + r'OLT production data.xlsx'
PATH_TEST = __FOLDER__ + r'OLT well test data.xlsx'

# Data Imports
DATA_INJECTION_ORIG = pd.read_excel(PATH_INJECTION)
DATA_PRODUCTION_ORIG = pd.read_excel(PATH_PRODUCTION)
DATA_TEST_ORIG = pd.read_excel(PATH_TEST)

# Hyper-parameters
BINS = 5

# TODO: Integrate `reshape_well_data` with `condense_fiber` and optimize


def reshape_well_data(original):
    """Restructures the original, raw fiber data to desired format.

    Parameters
    ----------
    original : DataFrame
        The raw data provided by IPC for fiber tracking.

    Returns
    -------
    DataFrame
        The melted/re-structured output, optimized for base-table integration.

    """
    df = original.T.reset_index()
    df.drop([i for i in range(0, 9)], inplace=True, axis=1)
    df.columns = df.iloc[0]
    df.drop(0, inplace=True)
    df['Date/Time :'] = [complete_date.split('@')[0].strip()
                         for complete_date in list(df['Date/Time :'])]
    df['Date/Time :'] = pd.to_datetime(df['Date/Time :'])

    df = pd.melt(df, id_vars=['Date/Time :'], value_vars=list(df.columns[1:])
                 ).sort_values('Date/Time :').reset_index(drop=True)
    df.columns = ['Date', 'Distance', 'Temperature']
    df['Distance'] = pd.to_numeric(df['Distance'])
    df['Temperature'] = pd.to_numeric(df['Temperature'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = [d.date() for d in df['Date']]
    df['Date'] = pd.to_datetime(df['Date'])

    return df

# TODO: (?) Multiple Statistical Metrics for Fiber Bins
# TODO: (?) Optimize pd.cut binning


def condense_fiber(well_df, BINS):
    """Condenses Fiber Data Each Injection Well Liner into specified bins
    for all recorded dates.

    Parameters
    ----------
    well_df : DataFrame
        The specific injection well liner to be processed.
    BINS : list
        The number of segments/intervals to split  `well_df` in to.

    Returns
    -------
    DataFrame
        The binned temperature values for all available days in `well_df`.

    """
    test_case = well_df
    all_dates = list(set(test_case['Date']))
    all_dates.sort()
    cume_dict = {}

    for d_i in range(len(all_dates)):
        selected_date = all_dates[d_i]
        filtered_test_case = test_case[test_case['Date'] == selected_date]
        filtered_test_case = filtered_test_case.drop('Date', 1)
        bins = np.linspace(filtered_test_case['Distance'].min(),
                           filtered_test_case['Distance'].max(), BINS + 1)
        distances = filtered_test_case['Distance'].copy()
        filtered_test_case.drop('Distance', 1, inplace=True)

        # filtered_test_case = (filtered_test_case.groupby(pd.cut(
        #     distances, bins, include_lowest=True))).agg({'Temperature':
        #                                                  ['mean']}
        #                                                 )['Temperature']
        filtered_test_case = (filtered_test_case.groupby(pd.cut(
            distances, bins, include_lowest=True))).mean()['Temperature']

        filtered_test_case.index = filtered_test_case.index.to_list()
        filtered_test_case = filtered_test_case.reset_index(
            drop=True).reset_index()
        filtered_test_case['Distance_Bin'] = filtered_test_case['index'] + 1
        filtered_test_case.drop('index', 1, inplace=True)
        filtered_test_case['Date'] = selected_date
        filtered_test_case = filtered_test_case.T.reset_index(drop=True)
        filtered_test_case.columns = ['Bin_' + str(col_i + 1)
                                      for col_i in filtered_test_case.columns]
        filtered_test_case['Date'] = filtered_test_case.loc[2][
            0].to_pydatetime().date()
        filtered_test_case.drop([1, 2], axis=0, inplace=True)

        cume_dict[d_i] = dict(filtered_test_case.loc[0])

    # Converting from dict to DataFrame and Reordering
    final = pd.DataFrame.from_dict(cume_dict).T

    cols = list(final.columns)
    cols.insert(0, cols.pop())
    final = final[cols]
    return final


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


def diagnostic_nan(df):
    """Returns the percentage of NaN values in each column in DataFrame.

    Parameters
    ----------
    df : DataFrame
        Inputted data to undergo analysis.

    Returns
    -------
    VOID
        Nothing. Statements Printed.

    """
    print('Percentage of NaN in each column.\nOut of '
          + str(df.shape[0]) + ' rows:')
    print(((df.isnull().astype(int).sum()) * 100 / df.shape[0]).sort_values(
        ascending=False))
    print('')


# Confirm Concatenation and Pickling
# TODO: Resolve and Optimize File IO
# Reformat all individual well data
well_set = {}
ind_FIBER_DATA = []
well_docs = [x[0] for x in os.walk(
    r'Data/DTS')][1:]
for well in well_docs:
    files = os.listdir(well)
    well_var_names = []
    for file in files:
        # error_bad_lines=False TO AVOID READ ERRORS
        var_name = file.replace('.xlsx', '').replace('.csv', '')
        if ".xlsx" in file:
            try:
                exec(var_name + ' = pd.read_excel(\"' + well
                     + '/' + file + '")')
            except Exception as e:
                print('Unable to read: ' + file + ', ' + str(e))
                continue
        elif ".csv" in file:
            try:
                exec(var_name + ' = pd.read_csv(\"' + well + '/' + file + '")')
            except Exception as e:
                print('Unable to read: ' + file + ', ' + str(e))
                continue
        well_var_names.append(var_name)
        # DataFrame Pre-Processing
        try:
            exec(var_name + ' = reshape_well_data(' + var_name + ')')
            exec('temp = condense_fiber(' + var_name + ', BINS)')
            temp["Well"] = var_name.replace('DTS', '')
            exec('ind_FIBER_DATA.append(temp)')
        except Exception as e:
            print("Error manipulating " + var_name + ": " + str(e))

    well_set[well.split('/')[-1]] = well_var_names.copy()
    well_var_names.clear()

# Process took 11 min 53 sec (File IO)

# with open('fiber_well_list_of_df.pkl', 'wb') as f:
#     pickle.dump(ind_FIBER_DATA, f)

FIBER_DATA = pd.concat(ind_FIBER_DATA, axis=0,
                       ignore_index=True).sort_values('Date').reset_index(
                           drop=True)

# with open('FIBER_DATA_DataFrame.pkl', 'wb') as f:
#     pickle.dump(FIBER_DATA, f)

# Data Processing - DATA_INJECTION
# Column Filtering, Pressure Reassignment, DateTime Setting
# > Delete Rows with Negative Numerical Cells
DATA_INJECTION = DATA_INJECTION_ORIG.reset_index(drop=True)
DATA_INJECTION.columns = ['Date', 'Pad', 'Well', 'UWI_Identifier', 'Time_On',
                          'Alloc_Steam', 'Meter_Steam', 'Casing_Pressure',
                          'Tubing_Pressure', 'Reason', 'Comment']
DATA_INJECTION_KEYS = ['Date', 'Well', 'Meter_Steam',
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
# Pivot so columns feature all injection wells and cells are steam values
DATA_INJECTION = DATA_INJECTION.pivot_table(['Meter_Steam', 'Pressure'],
                                            'Date', ['Well'])
# DATA_INJECTION['Date'] = DATA_INJECTION.index
# DATA_INJECTION.reset_index(inplace=True, drop=True)
DATA_INJECTION.columns.names = (None, None)
DATA_INJECTION_STEAM = DATA_INJECTION['Meter_Steam'].reset_index()
DATA_INJECTION_PRESS = DATA_INJECTION['Pressure'].reset_index()

# Data Processing - DATA_PRODUCTION
# Column Filtering, DateTime Setting, Delete Rows with Negative Numerical Cells
DATA_PRODUCTION = DATA_PRODUCTION_ORIG.reset_index(drop=True)
DATA_PRODUCTION.columns = ['Date', 'Pad', 'Well', 'UWI_Identifier', 'Time_On',
                           'Downtime_Code', 'Alloc_Oil', 'Alloc_Water',
                           'Alloc_Gas', 'Alloc_Steam', 'Steam_To_Producer',
                           'Hourly_Meter_Steam', 'Daily_Meter_Steam',
                           'Pump_Speed', 'Tubing_Pressure', 'Casing_Pressure',
                           'Heel_Pressure', 'Toe_Pressure', 'Heel_Temp',
                           'Toe_Temp', 'Last_Test_Date', 'Reason', 'Comment']
DATA_PRODUCTION_KEYS = ['Date', 'Well', 'Time_On', 'Hourly_Meter_Steam',
                        'Daily_Meter_Steam', 'Pump_Speed',
                        'Tubing_Pressure', 'Casing_Pressure', 'Heel_Pressure',
                        'Toe_Pressure', 'Heel_Temp', 'Toe_Temp']
DATA_PRODUCTION = DATA_PRODUCTION[DATA_PRODUCTION_KEYS]
DATA_PRODUCTION = filter_negatives(DATA_PRODUCTION,
                                   DATA_PRODUCTION.select_dtypes(
                                       include=['float64']).columns[1:])
DATA_PRODUCTION = convert_to_date(DATA_PRODUCTION, 'Date')
DATA_PRODUCTION = DATA_PRODUCTION.infer_objects()

# Data Processing - DATA_TEST
# Column Filtering, DateTime Setting, Delete Rows with Negative Numerical Cells
# TODO: (?) Do we care about BSW, Chlorides, Pump_Speed, Pump_Efficiency, Pump_Size
DATA_TEST = DATA_TEST_ORIG.reset_index(drop=True)
DATA_TEST.columns = ['Pad', 'Well', 'Start_Time', 'End_Time', 'Duration',
                     'Effective_Date', '24_Fluid', '24_Oil', '24_Hour', 'Oil',
                     'Water', 'Gas', 'Fluid', 'BSW', 'Chlorides',
                     'Pump_Speed', 'Pump_Efficiency', 'Pump_Size',
                     'Operator_Approved', 'Operator_Rejected',
                     'Operator_Comment', 'Engineering_Approved',
                     'Engineering_Rejected', 'Engineering_Comment']
DATA_TEST_KEYS = ['Well', 'Duration', 'Effective_Date',
                  '24_Fluid', '24_Oil', '24_Hour',
                  'Oil', 'Water', 'Gas', 'Fluid']
DATA_TEST = DATA_TEST[DATA_TEST_KEYS]
DATA_TEST = filter_negatives(DATA_TEST,
                             DATA_TEST.select_dtypes(
                                 include=['float64']).columns)
DATA_TEST = convert_to_date(DATA_TEST, 'Effective_Date')
DATA_TEST.rename(columns={'Effective_Date': 'Date'}, inplace=True)
DATA_TEST = DATA_TEST.infer_objects()

"""All Underlying Datasets
FIBER_DATA              --> Temperature/Distance data along production lines
DATA_INJECTION_STEAM    --> Metered Steam at Injection Sites
DATA_INJECTION_PRESSURE --> Pressure at Injection Sites
DATA_PRODUCTION         --> Production Well Sensors
DATA_TEST               --> Oil, Water, Gas, and Fluid from Production Wells

"""

# FINALIZED DATASETS, DIAGNOSTICS
PRODUCTION_WELL_OVERLAP = set.intersection(*map(set, [FIBER_DATA['Well'],
                                                      DATA_PRODUCTION['Well'],
                                                      DATA_TEST['Well']]))

# TODO: !! Update Data Schematic
# Base Off DATA_PRODUCTION
PRODUCTION_WELL_INTER = pd.merge(DATA_PRODUCTION, DATA_TEST,
                                 how='outer', on=['Date', 'Well'])
PRODUCTION_WELL_WSENSOR = pd.merge(PRODUCTION_WELL_INTER, FIBER_DATA,
                                   how='outer', on=['Date', 'Well'])
FINALE = pd.merge(PRODUCTION_WELL_WSENSOR, DATA_INJECTION_STEAM,
                  how='outer', on='Date')

# > These following lines MUST be run in JUPYTER
report = pandas_profiling.ProfileReport(FINALE,
                                        explorative=True,
                                        progress_bar=True)
# display(report)
# report.to_file('FINALE.html')


# TODO: !! Verify Fully-Merged Data Table Diagnostically

# # DIAGNOSTICS
# # Verify Well Counts and Expected Overlaps
# DATA_PRODUCTION['Well'].value_counts()
# DATA_INJECTION['Well'].value_counts()
# DATA_TEST['Well'].value_counts()

# # Observe Pressures Over Time in INJECTION_DATA
# df = DATA_INJECTION[['Date', 'Well', 'Casing_Pressure', 'Tubing_Pressure']
#                     ][DATA_INJECTION['Well'] == 'CI06'].reset_index(
#                         drop=True)
# plt.figure(figsize=(24, 19))
# plt.scatter(df['Date'], df['Casing_Pressure'], s=10)
# plt.scatter(df['Date'], df['Tubing_Pressure'], s=10)
#
# DATA_INJECTION['Pad'].unique()
# DATA_PRODUCTION['Pad'].unique()
# DATA_TEST['Pad'].unique()
#
# DATA_INJECTION['Well'].unique()
# DATA_PRODUCTION['Well'].unique()
# DATA_TEST['Well'].unique()

# # Merging
# dfs = [DATA_INJECTION.copy(), DATA_PRODUCTION.copy(), DATA_TEST.copy()]
# # > Very Long DataFrame
# df_final = reduce(lambda left, right: pd.merge(
#     left, right, on=['Date', 'Well', 'Pad'], how='outer'), dfs)
# # > Better Option to retain most data
# pd.merge(DATA_PRODUCTION, DATA_TEST, on=['Date', 'Well', 'Pad'], how='outer')

# BP1DTS.describe()
# DATA_TEST.describe()
# DATA_PRODUCTION.describe()
# DATA_INJECTION.describe()

#################
#################
# vis_date_range = list(set(BP1DTS['Date']))[:60]
# colors = cm.bwr(np.linspace(0, 1, len(vis_date_range)))
# plt.figure(figsize=(16, 8))
# for date in vis_date_range:
#     plt.scatter(BP1DTS[BP1DTS['Date'] == date]['Distance'],
#                 BP1DTS[BP1DTS['Date'] == date]
#                 ['Temperature'], s=1,
#                 color=colors[list(set(BP1DTS['Date'])).index(date)])
# plt.title('Temperature VS. Distance')
# plt.xlabel('Distance')
# plt.ylabel('Temperature')
# plt.show()
# plt.close()
#################
#################


#
