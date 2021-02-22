# @Author: Shounak Ray <Ray>
# @Date:   2021-01-21T13:48:32-07:00
# @Email:  rijshouray@gmail.com
# @Filename: IPC_Real_Modeling.py
# @Last modified by:   Ray
# @Last modified time: 21-Feb-2021 22:02:53:537  GMT-0700
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
from collections import Counter
from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from Anomaly_Detection_PKG import *
except Exception:
    sys.path.append('/Users/Ray/Documents/GitHub/AnomalyDetection')
    from Anomaly_Detection_PKG import *

# from itertools import chain
# import timeit
# from functools import reduce
# import matplotlib.cm as cm

# from matplotlib.backends.backend_pdf import PdfPages

# import pickle
# import pandas_profiling
# !{sys.executable} -m pip install pandas-profiling


"""Major Notes:
> Ensure Python Version in root directory matches that of local directory
>> AKA ensure that there isn't a discrepancy between local and global setting of Python
> `pandas_profiling` will not work inside Atom, must user `jupyter-notebook` in terminal
"""

"""All Underlying Datasets
DATA_INJECTION_ORIG     --> The original injection data
DATA_PRODUCTION_ORIG    --> The original production data (rev 1)
DATA_TEST_ORIG          --> The originla testing data
FIBER_DATA              --> Temperature/Distance data along production lines
DATA_INJECTION_STEAM    --> Metered Steam at Injection Sites
DATA_INJECTION_PRESS    --> Pressure at Injection Sites
DATA_PRODUCTION         --> Production Well Sensors
DATA_TEST               --> Oil, Water, Gas, and Fluid from Production Wells
PRODUCTION_WELL_INTER   --> Join of DATA_TEST and DATA_PRODUCTION
PRODUCTION_WELL_WSENSOR --> Join of PRODUCTION_WELL_INTER and FIBER_DATA
FINALE                  --> Join of PRODUCTION_WELL_WSENSOR
                                                    and DATA_INJECTION_STEAM
"""

"""Filtering Notes:
> Excess Production Data is inside the dataset (more than just A and B pads/patterns)
>> These will be filtered out
> Excess (?) Fiber Data is inside the dataset (uncommon, otherwise unseen wells combos inside A  and B pads)
>> These will be filtered out
"""

# HYPER-PARAMETERS
BINS = 5
FIG_SIZE = (220, 7)
DIR_EXISTS = Path('Data/Pickles').is_dir()

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

# DATA_INJECTION_ORIG.to_pickle('Pickles/DATA_INJECTION_ORIG.pkl')
# DATA_PRODUCTION_ORIG.to_pickle('Pickles/DATA_PRODUCTION_ORIG.pkl')
# DATA_TEST_ORIG.to_pickle('Pickles/DATA_TEST_ORIG.pkl')
# # ind_FIBER_DATA.to_pickle('Pickles/ind_FIBER_DATA.pkl')
# FIBER_DATA.to_pickle('Pickles/FIBER_DATA.pkl')
# DATA_INJECTION_STEAM.to_pickle('Pickles/DATA_INJECTION_STEAM.pkl')
# DATA_INJECTION_PRESS.to_pickle('Pickles/DATA_INJECTION_PRESS.pkl')
# DATA_PRODUCTION.to_pickle('Pickles/DATA_PRODUCTION.pkl')
# DATA_TEST.to_pickle('Pickles/DATA_TEST.pkl')
# PRODUCTION_WELL_INTER.to_pickle('Pickles/PRODUCTION_WELL_INTER.pkl')
# PRODUCTION_WELL_WSENSOR.to_pickle('Pickles/PRODUCTION_WELL_WSENSOR.pkl')
# FINALE.to_pickle('Pickles/FINALE.pkl')


# FUNCTION DEFINITIONS
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
    df['Date/Time :'] = [complete_date.split('@')[0].strip() for complete_date in list(df['Date/Time :'])]
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
        filtered_test_case.columns = ['Bin_' + str(col_i + 1) for col_i in filtered_test_case.columns]
        filtered_test_case['Date'] = filtered_test_case.loc[2][0].to_pydatetime().date()
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
    print('Percentage of NaN in each column.\nOut of ' + str(df.shape[0]) + ' rows:')
    print(((df.isnull().astype(int).sum()) * 100 / df.shape[0]).sort_values(ascending=False))
    print('')


def write_ts_matrix(df, groupby, time_feature, mpl_PDF, features_filter):
    """For each unique pair name/well, produce time-dependent plots
    of given selected feature.

    Parameters
    ----------
    df : DataFrame
        The original dataset to be investigated.
    groupby : string
        Feature name to group information using.
    time_feature : string
        Feature name representing time data.
    mpl_PDF : PdfPages Object
        What's used to write the pdf.
    features_filter : list
        List of features to be analyzed on PDF.
        I believe this list should have a minimum length of ~10.

    Returns
    -------
    VOID
        Nothing is returned.

    """
    print('STATUS: T.S. MATRIX >> Processing Matrix...')
    for w in df[groupby].unique():
        t = [w + " – " + feat for feat in features_filter[2:]]
        tsplot = df[df[groupby] == w].plot(title=t, x=time_feature,
                                           subplots=True,
                                           layout=(1, len(features_filter)),
                                           figsize=FIG_SIZE)
        plt.suptitle(w, fontsize=41)
        mpl_PDF.savefig(tsplot[0][0].get_figure())
    print('STATUS: T.S. MATRIX >> Confirming Matrix Process...')

# TODO: ?!!! Number of days a well is shut in at a time (dynamically change)


# Handle NANs at tail and head differently
# TODO: ! Columns skipped because there was nothing to interpolate?
def interpol(df, cols, time_index='Date', method='time', limit=15, limit_area=None):
    missing = []
    for well in cols:
        current = pd.to_numeric(df.set_index(time_index)[well])
        # Check if there is anything to interpolate
        if(any(current.isnull())):
            # percentage of NAN in columns
            prop_init = Counter(current.isnull()).get(True) / len(current)
            current.index = pd.DatetimeIndex(current.index)
            current.interpolate(method=method, limit=limit,
                                limit_area=limit_area, inplace=True)
            try:  # Determine
                prop_fin = Counter(current.isnull()).get(True) / len(current)
                if(prop_fin == prop_init):
                    outcome = 'no change'
                elif(prop_fin < prop_init):
                    outcome = 'lowered'
                else:
                    outcome = 'increased'
            except TypeError:  # there is no missing value, no True in Counter
                prop_fin = 0.0
                outcome = 'lowered'
            missing.append((well, prop_init, prop_fin, outcome))
            df[well] = current.values

    out = pd.DataFrame(missing, columns=['COLUMN', 'INITIAL_NAN', 'FINAL_NAN', 'OUTCOME'])
    df.fillna(0.0, inplace=True)
    return df, out


# Assumes that DataFrame is numerical and has 'Date' and 'Well' columns
# Interpolate Missing Values (surrounded by non-NA values)
def complete_interpol(df, cols, PIVOT=True):
    # col = DATA_PRODUCTION.columns[4:][0]
    all_dfs = []

    for col in cols:
        filtered = df[['Date', col, 'Well']]
        if(PIVOT):
            sensor_pivoted = filtered.pivot_table(col, 'Date', 'Well',
                                                  dropna=False).reset_index()
            sensor_pivoted.columns.names = [None]
        sensor_pivoted, sensor_pivoted_NANBENCH = interpol(sensor_pivoted,
                                                           sensor_pivoted.columns[1:],
                                                           method='time')
        sensor_pivoted = pd.melt(sensor_pivoted, id_vars='Date', value_vars=sensor_pivoted.columns[1:],
                                 var_name='Well', value_name=col)
        if(PIVOT):
            temp = sensor_pivoted.merge(filtered, on=list(sensor_pivoted.columns), how='left')
            all_dfs.append(temp)
        else:
            all_dfs.append(sensor_pivoted)

    return reduce(lambda left, right: pd.merge(left, right, on=['Date', 'Well']), all_dfs)


def viz_to_confirm(df, well, feature):
    dumbo = df.set_index('Date')
    fig = dumbo[dumbo['Well'] == well][feature].plot(figsize=(12, 4))
    return fig


PAD_KEYS = dict(zip(DATA_PRODUCTION_ORIG['Well'], DATA_PRODUCTION_ORIG['Pad']))


# FIBER DATA INGESTION AND REFORMATTING (~12 mins)


# TODO: !! Resolve and Optimize File IO
# TODO: Modularize File IO
if not DIR_EXISTS:
    well_set = {}
    ind_FIBER_DATA = []
    well_docs = [x[0] for x in os.walk(r'Data/DTS')][1:]
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
                    exec(var_name + ' = pd.read_csv(\"' + well +
                         '/' + file + '")')
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

    FIBER_DATA = pd.concat(ind_FIBER_DATA, axis=0,
                           ignore_index=True).sort_values('Date').reset_index(
                               drop=True)
FIBER_DATA = FIBER_DATA.infer_objects()
FIBER_DATA = complete_interpol(FIBER_DATA, FIBER_DATA.columns[1:6])
# dict(FIBER_DATA.isnull().mean() * 100)
FIBER_DATA['Pad'] = [PAD_KEYS.get(well) for well in FIBER_DATA['Well']]


# # Confirm Concatenation and Pickling
# with open('Pickles/ind_FIBER_DATA.pkl', 'wb') as f:
#     pickle.dump(ind_FIBER_DATA, f)
# with open('Pickles/FIBER_DATA_DataFrame.pkl', 'wb') as f:
#     pickle.dump(FIBER_DATA, f)

# DATA PROCESSING - DATA_INJECTION
# Column Filtering, Pressure Reassignment, DateTime Setting
# > Delete Rows with Negative Numerical Cells
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
# > Split into Steam and Pressure DataFrames
DATA_INJECTION_STEAM = DATA_INJECTION['Meter_Steam'].reset_index()
DATA_INJECTION_PRESS = DATA_INJECTION['Pressure'].reset_index()
# DATA_INJECTION_STEAM = complete_interpol(DATA_INJECTION_STEAM, DATA_INJECTION_STEAM.columns[1:], PIVOT=False)
# DATA_INJECTION_PRESS = complete_interpol(DATA_INJECTION_PRESS, DATA_INJECTION_PRESS.columns[1:], PIVOT=False)
DATA_INJECTION_STEAM, DATA_INJECTION_STEAM_NANBENCH = interpol(DATA_INJECTION_STEAM,
                                                               DATA_INJECTION_STEAM.columns[1:])
DATA_INJECTION_PRESS, DATA_INJECTION_PRESS_NANBENCH = interpol(DATA_INJECTION_PRESS,
                                                               DATA_INJECTION_PRESS.columns[1:])
# dict(DATA_INJECTION_STEAM.isnull().mean() * 100)
# dict(DATA_INJECTION_PRESS.isnull().mean() * 100)


# DATA PROCESSING - DATA_PRODUCTION
# Column Filtering, DateTime Setting, Delete Rows with Negative Numerical Cells
DATA_PRODUCTION = DATA_PRODUCTION_ORIG.reset_index(drop=True)
DATA_PRODUCTION.columns = ['Date', 'Pad', 'Well', 'UWI_Identifier', 'Time_On',
                           'Downtime_Code', 'Alloc_Oil', 'Alloc_Water',
                           'Alloc_Gas', 'Alloc_Steam', 'Steam_To_Producer',
                           'Hourly_Meter_Steam', 'Daily_Meter_Steam',
                           'Pump_Speed', 'Tubing_Pressure', 'Casing_Pressure',
                           'Heel_Pressure', 'Toe_Pressure', 'Heel_Temp',
                           'Toe_Temp', 'Last_Test_Date', 'Reason', 'Comment']
DATA_PRODUCTION_KEYS = ['Date', 'Pad', 'Well', 'Pump_Speed',
                        'Tubing_Pressure', 'Casing_Pressure', 'Heel_Pressure',
                        'Toe_Pressure', 'Heel_Temp', 'Toe_Temp']
DATA_PRODUCTION = DATA_PRODUCTION[DATA_PRODUCTION_KEYS]
DATA_PRODUCTION = filter_negatives(DATA_PRODUCTION,
                                   DATA_PRODUCTION.select_dtypes(include=['float64']).columns[1:])
DATA_PRODUCTION = convert_to_date(DATA_PRODUCTION, 'Date')
DATA_PRODUCTION = DATA_PRODUCTION.infer_objects()
DATA_PRODUCTION = complete_interpol(DATA_PRODUCTION, DATA_PRODUCTION.columns[3:])
DATA_PRODUCTION['Pad'] = [PAD_KEYS.get(well) for well in DATA_PRODUCTION['Well']]

# dict(DATA_PRODUCTION.isnull().mean() * 100)

# viz_to_confirm(DATA_PRODUCTION, 'AP2', 'Pump_Speed').plot()

# TODO: !! Filter anomalies in [BHP] Pressure Data
# > Kris and I found some data issues in our SQL server and we just had it
# >> corrected. Some of the producer bottom hole pressure data I sent is false.
# > Some wells with significant changes are AP5, AP6, BP3, BP5 and BP6.
# >> I haven’t looked through C, E and F pads but I assume there would be some
# >> bad data there as well.
# col_of_int = ['Heel_Pressure', 'Toe_Pressure']
# all_wells = set(DATA_PRODUCTION['Well'])
#
# # Very rudimentary, manual anomaly detection
# pp = PdfPages('IPC_Validation_initial.pdf')
# write_ts_matrix(DATA_PRODUCTION, 'Well', 'Date', pp, DATA_PRODUCTION.columns)
# pp.close()


# DATA PROCESSING - DATA_TEST
# Column Filtering, DateTime Setting, Delete Rows with Negative Numerical Cells
# TODO: (?) BSW, Chlorides, Pump_Speed, Pump_Efficiency, Pump_Size
DATA_TEST = DATA_TEST_ORIG.reset_index(drop=True)
DATA_TEST.columns = ['Pad', 'Well', 'Start_Time', 'End_Time', 'Duration',
                     'Effective_Date', '24_Fluid', '24_Oil', '24_Hour', 'Oil',
                     'Water', 'Gas', 'Fluid', 'BSW', 'Chlorides',
                     'Pump_Speed', 'Pump_Efficiency', 'Pump_Size',
                     'Operator_Approved', 'Operator_Rejected',
                     'Operator_Comment', 'Engineering_Approved',
                     'Engineering_Rejected', 'Engineering_Comment']
DATA_TEST_KEYS = ['Well', 'Pad', 'Duration', 'Effective_Date',
                  '24_Fluid', '24_Oil', '24_Hour',
                  'Oil', 'Water', 'Gas', 'Fluid']
DATA_TEST = DATA_TEST[DATA_TEST_KEYS]
DATA_TEST = filter_negatives(DATA_TEST, DATA_TEST.select_dtypes(include=['float64']).columns)
DATA_TEST = convert_to_date(DATA_TEST, 'Effective_Date')
DATA_TEST.rename(columns={'Effective_Date': 'Date'}, inplace=True)
DATA_TEST = DATA_TEST.infer_objects()
DATA_TEST = complete_interpol(DATA_TEST, DATA_TEST.columns[4:])
DATA_TEST['Pad'] = [PAD_KEYS.get(well) for well in DATA_TEST['Well']]
# dict(DATA_TEST.isnull().mean() * 100)

# TODO: !! Update Data Schematic
# TODO: !! Verify Columns of Underlying Datasets

# CREATE ANALYTIC BASE TABLED, MERGED
# Base Off DATA_PRODUCTION

# The wells and pads are identical across the L/R sources, however extraneous since !(a and b)
PRODUCTION_WELL_INTER = pd.merge(DATA_PRODUCTION, DATA_TEST,
                                 how='outer', on=['Date', 'Pad', 'Well'], indicator=True)
# Add test flag based on NAN occurence after outer join
PRODUCTION_WELL_INTER['test_flag'] = PRODUCTION_WELL_INTER['_merge'].replace(['both', 'left_only'],
                                                                             [True, False])
PRODUCTION_WELL_INTER.drop('_merge', axis=1, inplace=True)

# dict(PRODUCTION_WELL_INTER.isnull().mean() * 100)
PRODUCTION_WELL_WSENSOR = pd.merge(PRODUCTION_WELL_INTER, FIBER_DATA,
                                   how='inner', on=['Date', 'Pad', 'Well'])
# dict(PRODUCTION_WELL_WSENSOR.isnull().mean() * 100)
FINALE = pd.merge(PRODUCTION_WELL_WSENSOR, DATA_INJECTION_STEAM,
                  how='inner', on='Date')
# ADD PAD KEYS based on production data, along with unique id (for DSP)
# FINALE['Pad'] = [PAD_KEYS.get(well) for well in FINALE['Well']]
FINALE['unique_id'] = FINALE.index + 1
# Reorder columns
FINALE = FINALE[['unique_id', 'Date', 'Pad', 'Well'] +
                list(FINALE.columns[2:FINALE.shape[1] - 1])]
# drop columns with duplicate NAMES (doesn't assess values)
FINALE = FINALE.loc[:, ~FINALE.columns.duplicated()]

# dict(FINALE.isnull().mean() * 100)
# _ = plt.hist(FINALE['Pump_Speed'], bins=200)

FINALE.to_csv('Data/FINALE_INTERP.csv')
DATA_PRODUCTION['Well'].unique()

# production, injection, test, fiber

_ = """
# ANOMALY DETECTION AND FILTERING
data = FINALE.copy()
well = 'AP2'  # Production Well
feat = 'Daily_Meter_Steam'
mtds = ['Offline Outlier']
mds = ['overall', 'overall']
cnts = ['0.2', '0.2']
ALL_FEATURES = ['Hourly_Meter_Steam',
                'Daily_Meter_Steam',
                'Pump_Speed',
                'Tubing_Pressure',
                'Casing_Pressure',
                'Heel_Pressure',
                'Toe_Pressure',
                'Heel_Temp',
                'Toe_Temp',
                '24_Fluid',
                '24_Oil',
                '24_Hour',
                'Oil',
                'Water',
                'Gas']

# TODO: Snake-case all ALL_FEATURES directly in package
# ft, total, info, windows = anomaly_detection(data, well, feat, ALL_FEATURES=ALL_FEATURES, method=mtds, mode=mds,
#                                              gamma='scale', nu=0.3, model_name='rbf', N_EST=100,
#                                              diff_thresh=100, contamination=cnts, plot=True, n_jobs=-1,
#                                              iteration=1, TIME_COL='Date', GROUPBY_COL='Well')

# Pickle Absolutely Everything, minimize data injestion time for local testing
DATA_INJECTION_ORIG.to_pickle('Data/Pickles/DATA_INJECTION_ORIG.pkl')
DATA_PRODUCTION_ORIG.to_pickle('Data/Pickles/DATA_PRODUCTION_ORIG.pkl')
DATA_TEST_ORIG.to_pickle('Data/Pickles/DATA_TEST_ORIG.pkl')
FIBER_DATA.to_pickle('Data/Pickles/FIBER_DATA.pkl')
DATA_INJECTION_STEAM.to_pickle('Data/Pickles/DATA_INJECTION_STEAM.pkl')
DATA_INJECTION_PRESS.to_pickle('Data/Pickles/DATA_INJECTION_PRESS.pkl')
DATA_PRODUCTION.to_pickle('Data/Pickles/DATA_PRODUCTION.pkl')
DATA_TEST.to_pickle('Data/Pickles/DATA_TEST.pkl')
PRODUCTION_WELL_INTER.to_pickle('Data/Pickles/PRODUCTION_WELL_INTER.pkl')
PRODUCTION_WELL_WSENSOR.to_pickle('Data/Pickles/PRODUCTION_WELL_WSENSOR.pkl')
FINALE.to_pickle('Data/Pickles/FINALE.pkl')


# Base Table Done
# Verifying Anomalies

#
# TODO: !! Verify Fully-Merged Data Table Diagnostically

# PRELIMINARY ANALYSIS/DIAGNOSIS OF ANALYTIC BASE TABLE
# > These following lines MUST be run in JUPYTER
# report = pandas_profiling.ProfileReport(FINALE,
#                                         explorative=True,
#                                         progress_bar=True)
# display(report)
# report.to_file('FINALE.html')

"""

_ = """DIAGNOSTICS
#################
# DATE OVERLAP #
#################
# FINALIZED DATASETS, DIAGNOSTICS
PRODUCTION_WELL_OVERLAP = set.intersection(*map(set, [FIBER_DATA['Well'],
                                                      DATA_PRODUCTION['Well'],
                                                      DATA_TEST['Well']]))

#################
# VISUALIZATION #
#################
# # Observe Pressures Over Time in INJECTION_DATA
# df = DATA_INJECTION[['Date', 'Well', 'Casing_Pressure', 'Tubing_Pressure']
#                     ][DATA_INJECTION['Well'] == 'CI06'].reset_index(
#                         drop=True)
# plt.figure(figsize=(24, 19))
# plt.scatter(df['Date'], df['Casing_Pressure'], s=10)
# plt.scatter(df['Date'], df['Tubing_Pressure'], s=10)
#### #### ####
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

"""
