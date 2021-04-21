# @Author: Shounak Ray <Ray>
# @Date:   19-Mar-2021 10:03:97:973  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: base_generation.py
# @Last modified by:   Ray
# @Last modified time: 21-Apr-2021 00:04:81:817  GMT-0600
# @License: [Private IP]


import os
import sys
from multiprocessing import Pool
from typing import Final

import pandas as pd


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

    # import _context_managers
    # import _traversal

# _traversal.print_tree_to_txt(PATH='_configs/FILE_STRUCTURE.txt')

_ = """
#######################################################################################################################
#############################################   HYPERPARAMETER SETTINGS   #############################################
#######################################################################################################################
"""
data_dir = 'Data/Isolated/'
fiber_dir = 'Data/DTS/'
ap2_path = fiber_dir + 'AP2/AP2THERM.xlsx'
filepaths = [data_dir + "OLT injection data.xlsx",
             data_dir + "OLT production data (rev 1).xlsx",
             data_dir + "OLT well test data.xlsx"]

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
        datasets[name] = _temp.infer_objects()


def merge(datasets):
    df = pd.merge(datasets['PRODUCTION'], datasets['FIBER'], how='outer', on=['Date', 'PRO_Well'])
    df = pd.merge(df, datasets['INJECTION_TABLE'], how='outer', on=['Date'])
    df = pd.merge(df, datasets['PRODUCTION_TEST'], how='left', on=['Date', 'PRO_Well'])

    df = df.dropna(subset=['PRO_Well', 'PRO_UWI'], how='any').reset_index(drop=True)

    return df


def get_fiber_pwells(fiber_dir):
    return [p for p in os.listdir(fiber_dir) if p[0] != '.']


def ingest_sources(filepaths):
    with Pool(os.cpu_count() - 1) as pool:
        inj, pro, protest = pool.map(_accessories.retrieve_local_data_file, filepaths)
    DATASETS = {'INJECTION': inj, 'PRODUCTION': pro, 'PRODUCTION_TEST': protest}

    return DATASETS


def ingest_fiber(producer_wells, **kwargs):
    def get_fiber_data(producer, bins=8):
        def combine_data(producer):
            filedir = fiber_dir + producer + "/"
            files = os.listdir(filedir)
            with Pool(os.cpu_count() - 1) as pool:
                args = [filedir + f for f in files]
                combined = pool.map(_accessories.retrieve_local_data_file, args)
                combined = [df.T.iloc[1:] for df in combined]
            return pd.concat(combined).infer_objects()
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
    AP2_df = _accessories.retrieve_local_data_file(kwargs.get('ap2_path'))
    AP2_df.columns = ['Date', 'Bin_1', 'Bin_2', 'Bin_3', 'Bin_4', 'Bin_5', 'Bin_6', 'Bin_7', 'Bin_8']
    AP2_df.drop([0, 1], axis=0, inplace=True)
    AP2_df.index = AP2_df['Date']
    AP2_df['PRO_Well'] = 'AP2'
    AP2_df['Date'] = pd.to_datetime(AP2_df.index)
    AP2_df.reset_index(drop=True)

    # Concatenating aggregated fiber and AP2 data
    aggregated_fiber = pd.concat([aggregated_fiber, AP2_df.infer_objects()]).reset_index(drop=True)

    return aggregated_fiber


def save_well_pad_relations(datasets, direc='Data/', only=['INJECTION', 'PRODUCTION']):
    for name, df in datasets.items():
        if(name not in only):
            continue
        structure = dict(zip(df['Well'], df['Pad']))
        _accessories.save_local_data_file(structure, f'{direc}{name}_[Well, Pad].pkl')


_ = """
#######################################################################################################################
##################################################   CORE EXECUTION  ##################################################
#######################################################################################################################
"""


def _INGESTION():
    _accessories._print('Ingesting INJECTION, PRODUCTION, and PRODUCTION_TEST data...', color='LIGHTYELLOW_EX')
    DATASETS = ingest_sources(filepaths)
    save_well_pad_relations(DATASETS)
    filter_out(DATASETS)

    _accessories._print('Ingesting and transforming FIBER data...', color='LIGHTYELLOW_EX')
    producer_wells = get_fiber_pwells(fiber_dir)
    DATASETS['FIBER'] = ingest_fiber([i for i in producer_wells if i != 'AP2'], ap2_path=ap2_path)

    _accessories._print('Transforming and filtering...', color='LIGHTYELLOW_EX')
    _temp = DATASETS['PRODUCTION']
    DATASETS['PRODUCTION'] = _temp[_temp['PRO_Well'].isin(producer_wells)]
    DATASETS['INJECTION_TABLE'] = pd.pivot_table(DATASETS['INJECTION'], values='INJ_Meter_Steam',
                                                 index='Date', columns='INJ_Well').reset_index()

    _accessories._print('Merging and saving...', color='LIGHTYELLOW_EX')
    _accessories.finalize_all(DATASETS, skip=['FIBER'])
    merged_df = merge(DATASETS)
    _accessories.save_local_data_file(merged_df, 'Data/combined_ipc.csv')


if __name__ == '__main__':
    _INGESTION()

# EOF
