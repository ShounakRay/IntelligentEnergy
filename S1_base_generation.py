# @Author: Shounak Ray <Ray>
# @Date:   19-Mar-2021 10:03:97:973  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: base_generation.py
# @Last modified by:   Ray
# @Last modified time: 19-Apr-2021 15:04:05:059  GMT-0600
# @License: [Private IP]


import os
from multiprocessing import Pool
from typing import Final

import _acessories
import pandas as pd

_ = """
#######################################################################################################################
#############################################   HYPERPARAMETER SETTINGS   #############################################
#######################################################################################################################
"""
# # # # # # # # # # # # # # # # # # # # # # # #
# SOURCE INGESTION
data_dir = 'Data/Isolated/'
fiber_dir = 'Data/DTS/'
ap2_path = fiber_dir + 'AP2/AP2THERM.xlsx'

FORMAT_COLUMNS: Final = {'INJECTION': ['Date', 'INJ_Pad', 'INJ_Well', 'INJ_UWI', 'INJ_Time_On', 'INJ_Alloc_Steam',
                                       'INJ_Meter_Steam', 'INJ_Casing_BHP', 'INJ_Tubing_Pressure', 'INJ_Reason',
                                       'INJ_Comment'],
                         'PRODUCTION': ['Date', 'PRO_Pad', 'PRO_Well', 'PRO_UWI', 'PRO_Time_On',
                                        'PRO_Downtime_Code', 'PRO_Alloc_Oil', 'PRO_Alloc_Water', 'PRO_Alloc_Gas',
                                        'PRO_Alloc_Steam', 'PRO_Alloc_Steam_To_Producer', 'PRO_Hourly_Meter_Steam',
                                        'PRO_Daily_Meter_Steam', 'PRO_Pump_Speed', 'PRO_Tubing_Pressure',
                                        'PRO_Casing_Pressure', 'PRO_Heel_Pressure',  'PRO_Toe_Pressure',
                                        'PRO_Heel_Temp', 'PRO_Toe_Temp', 'PRO_Last_Test_Date', 'PRO_Reason',
                                        'PRO_Comment'],
                         'PRODUCTION_TEST': ['PRO_Pad', 'PRO_Well', 'Date', 'PRO_End_Time',
                                             'PRO_Duration', 'PRO_Effective_Date', 'PRO_24_Fluid', 'PRO_24_Oil',
                                             'PRO_24_Water', 'PRO_Oil', 'PRO_Water', 'PRO_Gas', 'PRO_Fluid',
                                             'PRO_BSW', 'PRO_Chlorides', 'PRO_Pump_Speed', 'PRO_Pump_Efficiency',
                                             'PRO_Pump_Size', 'PRO_Operator_Approved', 'PRO_Operator_Rejected',
                                             'PRO_Operator_Comment', 'PRO_Engineering_Approved',
                                             'PRO_Engineering_Rejected', 'PRO_Engineering_Comment']}

CHOICE_COLUMNS: Final = {'INJECTION': ['Date', 'INJ_Pad', 'INJ_Well', 'INJ_Meter_Steam', 'INJ_Casing_BHP',
                                       'INJ_Tubing_Pressure'],
                         'PRODUCTION': ['Date', 'PRO_UWI', 'PRO_Well', 'PRO_Pump_Speed', 'PRO_Time_On',
                                        'PRO_Casing_Pressure', 'PRO_Heel_Pressure', 'PRO_Toe_Pressure',
                                        'PRO_Heel_Temp', 'PRO_Toe_Temp', 'PRO_Alloc_Oil', 'PRO_Alloc_Water'],
                         'PRODUCTION_TEST': ['PRO_Pad', 'PRO_Well', 'Date', 'PRO_Duration', 'PRO_Oil',
                                             'PRO_Water', 'PRO_Gas', 'PRO_Fluid', 'PRO_Chlorides',
                                             'PRO_Pump_Efficiency', 'PRO_Engineering_Approved']}

_ = """
#######################################################################################################################
###############################################   FUNCTION DEFINITIONS   ##############################################
#######################################################################################################################
"""


def filter_out(datasets, FORMAT=FORMAT_COLUMNS, CHOICE=CHOICE_COLUMNS):
    for name, df in datasets.items():
        _temp = df.copy()
        _temp.columns = FORMAT.get(name)
        _temp = _temp[CHOICE.get(name)]
        df['name'] = _temp.infer_objects()


def aggregate_fiber(producer_wells, **kwargs):
    def get_fiber_data(producer, bins=8):
        def combine_data(producer):
            filedir = fiber_dir + producer + "/"
            files = os.listdir(filedir)
            combined = []
            for f in files:
                print(f)
                try:
                    fiber_data = pd.read_csv(filedir + f, error_bad_lines=False).T.iloc[1:]
                except Exception:
                    fiber_data = pd.read_excel(filedir + f).T.iloc[1:]
                combined.append(fiber_data)
            return pd.concat(combined)
        combined = combine_data(producer)
        combined = combined.iloc[:, 9:]
        combined = combined.apply(pd.to_numeric)
        combined = combined.reset_index()
        combined = combined.sort_values('index')
        combined['index'] = combined['index'].str.split(" ").str[0]

        max_length = int(max(combined.columns[1:]))
        segment_length = int(max_length / bins)

        condensed = []
        for i, s in enumerate(range(0, max_length + 1, segment_length)):
            condensed.append(pd.DataFrame(
                {'Bin_' + str(i + 1): list(combined.iloc[:, s:s + segment_length].mean(axis=1))}))

        condensed = pd.concat(condensed, axis=1)
        condensed['Date'] = combined['index']
        condensed = condensed.sort_values('Date').set_index('Date')

        return condensed
    aggregated_fiber = []
    for producer in producer_wells[1:]:
        condensed = get_fiber_data(producer)
        condensed['PRO_Well'] = producer
        aggregated_fiber.append(condensed)

    aggregated_fiber = pd.concat(aggregated_fiber)
    aggregated_fiber = aggregated_fiber.dropna(how='all', axis=1)

    # Processing for AP2 which is thermocouple and in different format
    AP2_df = pd.read_excel(ap2_path)
    AP2_df.columns = ['Date', 'Bin_1', 'Bin_2', 'Bin_3', 'Bin_4', 'Bin_5', 'Bin_6', 'Bin_7', 'Bin_8']
    AP2_df.drop([0, 1], axis=0, inplace=True)
    AP2_df.index = AP2_df['Date']
    AP2_df.drop('Date', axis=1, inplace=True)
    AP2_df['PRO_Well'] = 'AP2'
    AP2_df.index = pd.to_datetime(kwargs.get(ap2_path).index)

    # Concatenating aggregated fiber and AP2 data
    aggregated_fiber = pd.concat([aggregated_fiber, AP2_df.infer_objects()]).reset_index(drop=True)


def finalize_all(datasets, skip=['FIBER'], coerce_date=True):
    for name, df in datasets.items():
        _temp = df.infer_objects()
        if(coerce_date):
            _temp['Date'] = pd.to_datetime(_temp['Date'])
        datasets['name'] = _temp


def merge(datasets):
    df = pd.merge(datasets['PRODUCTION'], datasets['FIBER'], how='outer', on=['Date', 'PRO_Well'])
    df = pd.merge(df, datasets['INJECTION_TABLE'], how='outer', on=['Date'])
    df = pd.merge(df, datasets['PRODUCTION_TEST'], how='left', on=['Date', 'PRO_Well'])


_ = """
#######################################################################################################################
##################################################   CORE EXECUTION  ##################################################
#######################################################################################################################
"""


def _INGESTION():
    filepaths = [data_dir + "OLT injection data.xlsx",
                 data_dir + "OLT production data (rev 1).xlsx",
                 data_dir + "OLT well test data.xlsx"]
    with Pool(os.cpu_count() - 1) as pool:
        inj, pro, protest = pool.starmap(_acessories.retrieve_local_data_file, list(zip(filepaths,
                                                                                        [False] * len(filepaths))))
    DATASETS = {'INJECTION': inj, 'PRODUCTION': pro, 'PRODUCTION_TEST': protest}

    producer_wells = [p for p in os.listdir(fiber_dir) if p[0] != '.']

    DATASETS['FIBER'] = aggregate_fiber([i for i in producer_wells if i != 'AP2'])

    _temp = DATASETS['PRODUCTION']
    DATASETS['PRODUCTION'] = _temp[_temp['PRO_Well'].isin(producer_wells)]

    DATASETS['INJECTION_TABLE'] = pd.pivot_table(DATASETS['INJECTION'], values='INJ_Meter_Steam',
                                                 index='Date', columns='INJ_Well').reset_index()

    merged_df = merge(DATASETS)

    finalize_all(DATASETS, skip=['FIBER'])

    merged_df = merged_df.dropna(subset=['PRO_Well', 'PRO_UWI'], how='any').reset_index(drop=True)

    _acessories.save_local_data_file

    df.infer_objects().to_csv('Data/combined_ipc.csv', index=False)

# EOF
