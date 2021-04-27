# @Author: Shounak Ray <Ray>
# @Date:   30-Mar-2021 11:03:53:534  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: feature_engineering.py
# @Last modified by:   Ray
# @Last modified time: 26-Apr-2021 17:04:58:589  GMT-0600
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
    import _multiprocessed.defs as defs

    # import _traversal


# _traversal.print_tree_to_txt(PATH='_configs/FILE_STRUCTURE.txt')

_ = """
#######################################################################################################################
#############################################   HYPERPARAMETER SETTINGS   #############################################
#######################################################################################################################
"""
NOT_REQUIRED: Final = ['PRO_Pump_Efficiency', 'PRO_Engineering_Approved', 'PRO_Alloc_Water_Cut',
                       'PRO_Theo_Fluid', 'PRO_Alloc_Factor', 'adj_PRO_Theo_Fluid',
                       'PRO_Adj_Alloc_Water', 'PRO_Adj_Pump_Efficiency', 'Field_Steam', 'PRO_Pump_Speed', 'PRO_UWI',
                       'PRO_Chlorides', 'PRO_Gas', 'PRO_Oil', 'PRO_Fluid', 'PRO_Time_On', 'PRO_Duration', 'PRO_Water',
                       'PRO_Pad', 'PRO_Alloc_Water']

_ = """
#######################################################################################################################
###############################################   FUNCTION DEFINITIONS   ##############################################
#######################################################################################################################
"""


def engineer_initial_features(df):
    df['PRO_Total_Fluid'] = df['PRO_Alloc_Oil'] + df['PRO_Alloc_Water']
    df['PRO_Fluid'] = df['PRO_Oil'] + df['PRO_Water']
    df['PRO_Alloc_Water_Cut'] = df['PRO_Alloc_Water'] / df['PRO_Total_Fluid']
    df['PRO_Water_cut'] = df['PRO_Water'] / df['PRO_Fluid']
    df['PRO_Adj_Pump_Speed'] = (df['PRO_Pump_Speed'] * df['PRO_Time_On']) / 24.0


def engineer_adjusted_features(df, injectors):
    df['adj_PRO_Theo_Fluid'] = df['PRO_Theo_Fluid'] / df['PRO_Alloc_Factor']
    df['PRO_Adj_Alloc_Oil'] = df['adj_PRO_Theo_Fluid'] * (1 - df['PRO_Water_cut'])
    df['PRO_Adj_Alloc_Water'] = df['adj_PRO_Theo_Fluid'] * df['PRO_Water_cut']
    df['PRO_Adj_Pump_Efficiency'] = df['adj_PRO_Theo_Fluid'] / \
        df['PRO_Adj_Pump_Speed'] * 10
    df['Field_Steam'] = df[injectors].sum(axis=1)


def get_injector_wells(df_joined, forbidden=['PRO_UWI']):
    return [c for c in df_joined.columns if 'I' in c and c not in forbidden]


def theoretical_fluid(df):
    with Pool(os.cpu_count() - 1) as pool:
        pwells = df['PRO_Well'].dropna().unique()
        args = zip([df] * len(pwells), pwells)
        theoretical_df = pool.starmap(defs._theo_fluid, args)
        theoretical_df = pd.concat(theoretical_df).reset_index(drop=True)
    theoretical_df.loc[theoretical_df['PRO_Total_Fluid'] < 0, 'PRO_Total_Fluid'] = 0

    return theoretical_df


def allocation_factor(df):
    combined = df.groupby(['Date']).sum().reset_index()
    combined['PRO_Alloc_Factor'] = combined['PRO_Theo_Fluid'] / combined['PRO_Total_Fluid']
    df = pd.merge(df, combined[['Date', 'PRO_Alloc_Factor']], how='left', on=['Date'])

    return df


_ = """
#######################################################################################################################
##################################################   CORE EXECUTION  ##################################################
#######################################################################################################################
"""


def _FEATENG_PHYS():
    _accessories._print('Ingesting JOINED DATA data...', color='LIGHTYELLOW_EX')
    DATASETS = {'JOINED_SOURCE': _accessories.retrieve_local_data_file('Data/combined_ipc_ALL.csv')}

    _accessories._print('Engineering initial physics features...', color='LIGHTYELLOW_EX')
    engineer_initial_features(DATASETS['JOINED_SOURCE'])

    _accessories._print('Engineering theoretical fluid...', color='LIGHTYELLOW_EX')
    DATASETS['THEORETICAL'] = theoretical_fluid(DATASETS['JOINED_SOURCE'])
    _accessories._print('Engineering allocation factor...', color='LIGHTYELLOW_EX')
    DATASETS['THEORETICAL'] = allocation_factor(DATASETS['THEORETICAL'])

    _accessories._print('Engineering adjusted features...', color='LIGHTYELLOW_EX')
    injectors = get_injector_wells(DATASETS['THEORETICAL'])
    engineer_adjusted_features(DATASETS['THEORETICAL'], injectors)

    _accessories._print('Health checks and saving...', color='LIGHTYELLOW_EX')
    DATASETS['THEORETICAL'].drop(NOT_REQUIRED, axis=1, inplace=True)
    _accessories.finalize_all(DATASETS, skip=[])

    _accessories.save_local_data_file(DATASETS['THEORETICAL'], 'Data/combined_ipc_engineered_phys_ALL.csv')


if __name__ == '__main__':
    _FEATENG_PHYS()

# EOF
