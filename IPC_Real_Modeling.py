# @Author: Shounak Ray <Ray>
# @Date:   2021-01-21T13:48:32-07:00
# @Email:  rijshouray@gmail.com
# @Filename: IPC_Real_Modeling.py
# @Last modified by:   Ray
# @Last modified time: 22-Jan-2021 01:01:99:997  GMT-0700
# @License: No License

# G0TO: CTRL + OPTION + G
# SELECT USAGES: CTRL + OPTION + U
# DOCBLOCK: CTRL + SHIFT + C

import os
import timeit
from functools import reduce

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# TODO: Optimize Initial Pandas Reformatting Function


def reshape_well_data(original):
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


# Reformat all individual well data
well_set = {}
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
        except Exception as e:
            print("Error manipulating " + var_name)

    well_set[well.split('/')[-1]] = well_var_names.copy()
    well_var_names.clear()

# Data Processing - DATA_INJECTION
DATA_INJECTION = DATA_INJECTION_ORIG.reset_index(drop=True)
DATA_INJECTION.columns = ['Date', 'Pad', 'Well', 'UWI_Identifier', 'Time_On',
                          'Alloc_Steam', 'Meter_Steam', 'Casing_Pressure',
                          'Tubing_Pressure', 'Reason', 'Comment']
DATA_INJECTION_KEYS = ['Date', 'Pad', 'Well', 'Time_On',
                       'Meter_Steam', 'Casing_Pressure', 'Tubing_Pressure']
DATA_INJECTION = DATA_INJECTION[DATA_INJECTION_KEYS]
DATA_INJECTION['Date'] = pd.to_datetime(DATA_INJECTION['Date'])

# Data Processing - DATA_PRODUCTION
DATA_PRODUCTION = DATA_PRODUCTION_ORIG.reset_index(drop=True)
DATA_PRODUCTION.columns = ['Date', 'Pad', 'Well', 'UWI_Identifier', 'Time_On',
                           'Downtime_Code', 'Alloc_Oil', 'Alloc_Water',
                           'Alloc_Gas', 'Alloc_Steam', 'Steam_To_Producer',
                           'Hourly_Meter_Steam', 'Daily_Meter_Steam',
                           'Pump_Speed', 'Tubing_Pressure', 'Casing_Pressure',
                           'Heel_Pressure', 'Toe_Pressure', 'Heel_Temp',
                           'Toe_Temp', 'Last_Test_Date', 'Reason', 'Comment']
DATA_PRODUCTION_KEYS = ['Date', 'Pad', 'Well', 'Time_On', 'Hourly_Meter_Steam',
                        'Daily_Meter_Steam', 'Pump_Speed',
                        'Tubing_Pressure', 'Casing_Pressure', 'Heel_Pressure',
                        'Toe_Pressure', 'Heel_Temp', 'Toe_Temp',
                        'Last_Test_Date']
DATA_PRODUCTION = DATA_PRODUCTION[DATA_PRODUCTION_KEYS]
DATA_PRODUCTION['Date'] = pd.to_datetime(DATA_PRODUCTION['Date'])

# Data Processing - DATA_TEST
DATA_TEST = DATA_TEST_ORIG.reset_index(drop=True)
DATA_TEST.columns = ['Pad', 'Well', 'Start_Time', 'End_Time', 'Duration',
                     'Effective_Date', '24_Fluid', '24_Oil', '24_Hour', 'Oil',
                     'Water', 'Gas', 'Fluid', 'BSW', 'Chlorides',
                     'Pump_Speed', 'Pump_Efficiency', 'Pump_Size',
                     'Operator_Approved', 'Operator_Rejected',
                     'Operator_Comment', 'Engineering_Approved',
                     'Engineering_Rejected', 'Engineering_Comment']
DATA_TEST_KEYS = ['Pad', 'Well', 'Duration', 'Effective_Date',
                  '24_Fluid', '24_Oil', '24_Hour',
                  'Oil', 'Water', 'Gas', 'Fluid', 'BSW', 'Chlorides',
                  'Pump_Speed', 'Pump_Efficiency', 'Pump_Size']
DATA_TEST = DATA_TEST[DATA_TEST_KEYS]
DATA_TEST['Effective_Date'] = pd.to_datetime(DATA_TEST['Effective_Date'])
DATA_TEST['Effective_Date'] = [d.date() for d in DATA_TEST['Effective_Date']]
DATA_TEST['Effective_Date'] = pd.to_datetime(DATA_TEST['Effective_Date'])
DATA_TEST.rename(columns={'Effective_Date': 'Date'}, inplace=True)

# TODO: Bin Fiber Data
test_case = BP16DTS

# %timeit condense_fiber(test_case, BINS)

# TODO: Extend `condense_fiber` for all dates, efficiently, and %%timeit
# TODO: After extending `condense_fiber`, update DOCBLOCK


def condense_fiber(well_df, BINS):
    """Condenses Fiber Data Each Injection Well Liner into specified bins.

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
    ex_date = test_case['Date'].min()
    filtered_test_case = test_case[test_case['Date'] == ex_date]
    filtered_test_case.drop('Date', 1, inplace=True)
    bins = np.linspace(filtered_test_case['Distance'].min(),
                       filtered_test_case['Distance'].max(), BINS + 1)
    distances = filtered_test_case['Distance'].copy()
    filtered_test_case.drop('Distance', 1, inplace=True)

    filtered_test_case = (filtered_test_case.groupby(pd.cut(
        distances, bins, include_lowest=True))).agg({'Temperature':
                                                     ['mean', 'std']}
                                                    )['Temperature']
    filtered_test_case.index = filtered_test_case.index.to_list()
    filtered_test_case = filtered_test_case.reset_index(
        drop=True).reset_index()
    filtered_test_case['Distance_Bin'] = filtered_test_case['index'] + 1
    filtered_test_case.drop('index', 1, inplace=True)

    return filtered_test_case


# TODO: Create Analytics Base Table (Refer to Data Schematics)
# Base Off DATA_PRODUCTION


# TODO: Verify Data Table Diagnostically

# # DIAGNOSTICS
# # Verify Well Counts and Expected Overlaps
# DATA_PRODUCTION['Well'].value_counts()
# DATA_INJECTION['Well'].value_counts()
# DATA_TEST['Well'].value_counts()

# # Observe Pressures Over Time in INJECTION_DATA
# df = DATA_INJECTION[['Date', 'Well', 'Casing_Pressure', 'Tubing_Pressure']
#                     ][DATA_INJECTION['Well'] == 'CI06'].reset_index(drop=True)
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
#     plt.scatter(BP1DTS[BP1DTS['Date'] == date]['Distance'], BP1DTS[BP1DTS['Date'] == date]
#                 ['Temperature'], s=1, color=colors[list(set(BP1DTS['Date'])).index(date)])
# plt.title('Temperature VS. Distance')
# plt.xlabel('Distance')
# plt.ylabel('Temperature')
# plt.show()
# plt.close()
#################
#################


#
