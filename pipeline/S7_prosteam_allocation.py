# @Author: Shounak Ray <Ray>
# @Date:   14-Apr-2021 09:04:63:632  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: S6_steam_allocation.py
# @Last modified by:   Ray
# @Last modified time: 17-May-2021 11:05:66:660  GMT-0600
# @License: [Private IP]

import os
import sys
from typing import Final

import numpy as np
import pandas as pd

# TODO: Heel constraints
# TODO: Fiber Consideration


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


if True:
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


_ = """
#######################################################################################################################
#############################################   HYPERPARAMETER SETTINGS   #############################################
#######################################################################################################################
"""

# TEMP: Local file testing
field_df = pd.read_csv('Data/field_data_pressures.csv').drop('Unnamed: 0', axis=1)


_ = """
#######################################################################################################################
##############################################   FUNCTION DEFINITIONS   ###############################################
#######################################################################################################################
"""


def get_field_solution(field_df, date, macro_solution, chloride_solution, Op_Params):
    # pad = 'E'
    field_res = []
    field_kpi = []

    for pad in sorted(list(field_df['pad'].dropna().unique())):
        res = well_level_steam_solution(field_df, date, macro_solution, pad,
                                        Op_Params.steam_available, chloride_solution, Op_Params.well_pump_constraint,
                                        Op_Params.well_steam_constraint, Op_Params.pres_steam_percent,
                                        Op_Params.recent_days, Op_Params.group, Op_Params.watercut_source)
        kpi = {"pad": pad,
               "alloc_steam": res['alloc_steam'].sum(),
               "recomm_steam": res['recomm_steam'].sum(),
               "chl_steam": res['chl_steam'].sum(),
               "spm": res['spm'].sum(),
               "target_spm": res['target_spm'].sum(),
               "oil": res['oil'].sum(),
               "target_oil": res['target_oil'].sum(),
               "oil_per": res['target_oil'].sum() / res['oil'].sum() - 1,
               "fluid": res['total_fluid'].sum(),
               "target_fluid": res['target_fluid'].sum(),
               "fluid_per": res['target_fluid'].sum() / res['total_fluid'].sum() - 1,
               "water_cut": res['water'].sum() / res['total_fluid'].sum(),
               "target_water_cut": 1 - res['target_oil'].sum() / res['target_fluid'].sum(),
               "calc_pump_efficiency": res['calc_pump_efficiency'].dropna().mean(),
               "test_pump_efficiency": res['pump_efficiency'].dropna().mean(),
               "sor": res['alloc_steam'].sum() / res['oil'].sum(),
               "target_sor": res['recomm_steam'].sum() / res['target_oil'].sum()}
        field_res.append(res)
        field_kpi.append(kpi)

    return pd.concat(field_res), pd.DataFrame(field_kpi)


def well_level_steam_solution(field_df, date, macro_solution, pad, field_steam, chloride_solution, pump_constraint,
                              steam_constraint, pres_steam_percent, recent_days, group, watercut_source):
    group_steam = dict((a['pad'], a['alloc_steam']) for a in macro_solution[[group, 'alloc_steam']].to_dict('records'))
    group_target = dict((a['pad'], a['pred']) for a in macro_solution[[group, 'pred']].to_dict('records'))

    selected_dates = field_df[field_df['date'] <= date]['date'].unique()[-recent_days:]
    wells = field_df[field_df['pad'] == pad]['producer_well'].unique()

    latest_df = field_df[(field_df['date'] == selected_dates[-1]) & (field_df['producer_well'].isin(wells))]
    recent_df = field_df[(field_df['date'].isin(selected_dates)) & (field_df['producer_well'].isin(wells))]

    recent_stf = recent_df['alloc_steam'].sum() / recent_df['total_fluid'].sum()
    latest_stf = latest_df['alloc_steam'].sum() / latest_df['total_fluid'].sum()

    stf_ratio = np.mean([group_steam[pad] / group_target[pad], recent_stf, latest_stf])

    steam_avail = group_steam[pad]
    fluid_target = steam_avail / stf_ratio

    #####
    benchmark_df = recent_df.groupby(['producer_well', 'pad'])[['alloc_steam', 'oil', 'water', 'total_fluid',
                                                                'spm', 'test_water_cut', 'pressure_average'
                                                                ]].mean().reset_index()
    benchmark_df = pd.merge(benchmark_df.rename(columns={'test_water_cut': 'recent_water_cut'}),
                            latest_df[['producer_well', 'volume_per_stroke',
                                       'pump_efficiency', 'test_water_cut']]
                            .rename(columns={'test_water_cut': 'latest_water_cut'}))
    benchmark_df['recent_water_cut'] = benchmark_df[['recent_water_cut', 'latest_water_cut']].mean(axis=1)

    benchmark_df['recent_sor'] = benchmark_df['alloc_steam'] / benchmark_df['oil'].replace(0, 1)

    benchmark_df = benchmark_df.drop(['alloc_steam', 'oil', 'water', 'total_fluid', 'spm'], axis=1)
    benchmark_df = pd.merge(benchmark_df.rename(columns={'test_water_cut': 'recent_water_cut'}),
                            latest_df[['producer_well', 'alloc_steam', 'oil',
                                       'water', 'total_fluid', 'spm']], how='left', on='producer_well')

    benchmark_df.loc[benchmark_df['alloc_steam'] < 1, watercut_source] = 1
    benchmark_df.loc[benchmark_df['oil'] < 1, watercut_source] = 1

    steam_solution = benchmark_df[benchmark_df['producer_well'].isin(wells)]
    steam_solution = pd.merge(steam_solution, chloride_solution[['producer_well', 'chl_steam']],
                              how='left', on='producer_well')
    steam_solution['chl_steam'] = steam_solution['chl_steam'].fillna(0)
    steam_solution['meas_water_cut'] = steam_solution['water'] / steam_solution['total_fluid']

    ###
    steam_solution['target_fluid'] = fluid_target * \
        (1 - steam_solution[watercut_source]) / (1 - steam_solution[watercut_source]).sum()
    steam_solution['pump_efficiency'] = steam_solution['pump_efficiency'].fillna(
        int(steam_solution['pump_efficiency'].dropna().mean()))
    steam_solution['calc_pump_efficiency'] = steam_solution['total_fluid'] / \
        steam_solution['spm'] / steam_solution['volume_per_stroke'] * 100

    steam_solution['volume_per_stroke'] = steam_solution['volume_per_stroke'].replace(np.inf, np.nan)
    steam_solution['volume_per_stroke'] = steam_solution['volume_per_stroke'].fillna(
        int(np.mean(steam_solution['volume_per_stroke'].dropna())))

    steam_solution['target_spm'] = steam_solution['target_fluid'] / \
        steam_solution['pump_efficiency'] * 100 / steam_solution['volume_per_stroke']

    # ADD MIN/MAX PUMP SPEED
    steam_solution['pump_min'] = [pump_constraint[w]['min'] for w in steam_solution['producer_well']]
    steam_solution['pump_max'] = [pump_constraint[w]['max'] for w in steam_solution['producer_well']]

    for i in range(0, 10):
        steam_solution.loc[steam_solution['target_spm'] > steam_solution['pump_max'],
                           'target_spm'] = steam_solution['pump_max']
        steam_solution.loc[steam_solution['target_spm'] < steam_solution['pump_min'],
                           'target_spm'] = steam_solution['pump_min']

        steam_solution['target_fluid'] = steam_solution['target_spm'] * \
            steam_solution['pump_efficiency'] / 100 * steam_solution['volume_per_stroke']
        fluid_adj = fluid_target - steam_solution['target_fluid'].sum()

        steam_solution.loc[steam_solution['target_spm'] < steam_solution['pump_max'], 'target_fluid'] = steam_solution['target_fluid'] + fluid_adj * (
            1 - steam_solution[steam_solution['target_spm'] < steam_solution['pump_max']]['recent_water_cut']) / (1 - steam_solution[steam_solution['target_spm'] < steam_solution['pump_max']]['recent_water_cut']).sum()

    steam_solution['pressure_average'] = steam_solution['pressure_average'].fillna(2000)

    steam_solution['base_steam'] = list(1 * steam_avail * (steam_solution['target_fluid'
                                                                          ] / steam_solution['target_fluid'].sum()))
    steam_solution['pres_steam'] = (steam_solution['pressure_average'].mean(
    ) - steam_solution['pressure_average']) / steam_solution['pressure_average'].mean()
    steam_solution['pres_steam'] = pres_steam_percent * steam_avail * steam_solution['pres_steam']
    # steam_solution['pres_steam'] = steam_solution['base_steam'] * steam_solution['pres_steam']

    steam_solution['recomm_steam'] = steam_solution['base_steam'] + steam_solution['chl_steam']
    steam_solution['recomm_steam'] = steam_solution['recomm_steam'] + steam_solution['pres_steam']

    steam_solution['stf_ratio'] = (steam_solution['recomm_steam'] / steam_solution['target_fluid'] + stf_ratio) / 2
    # steam_solution['stf_ratio'] = steam_solution['alloc_steam'] / steam_solution['total_fluid']

    # ADD MIN/MAX STEAM VALUES
    steam_solution['steam_min'] = [steam_constraint[w]['min'] for w in steam_solution['producer_well']]
    steam_solution['steam_max'] = [steam_constraint[w]['max'] for w in steam_solution['producer_well']]

    ###
    steam_avail = steam_solution['recomm_steam'].sum()

    for i in range(0, 10):
        steam_solution.loc[steam_solution['recomm_steam'] >
                           steam_solution['steam_max'], 'recomm_steam'] = steam_solution['steam_max']
        steam_solution.loc[steam_solution['recomm_steam'] <
                           steam_solution['steam_min'], 'recomm_steam'] = steam_solution['steam_min']

        steam_adj = steam_avail - steam_solution['recomm_steam'].sum()
        steam_solution.loc[steam_solution['recomm_steam'] < steam_solution['steam_max'], 'recomm_steam'] = steam_solution['recomm_steam'] + steam_adj * (
            1 - steam_solution[steam_solution['recomm_steam'] < steam_solution['steam_max']]['target_fluid']) / (1 - steam_solution[steam_solution['recomm_steam'] < steam_solution['steam_max']]['target_fluid']).sum()

    # steam_solution['target_fluid'] = steam_solution['base_steam']/steam_solution['stf_ratio']
    steam_solution['target_oil'] = steam_solution['target_fluid'] * (1 - steam_solution[watercut_source])
    steam_solution['target_spm'] = steam_solution['target_fluid'] / \
        steam_solution['pump_efficiency'] * 100 / steam_solution['volume_per_stroke']
    steam_solution['target_sor'] = (steam_solution['recomm_steam'] / steam_solution['target_oil'])

    # steam_solution[['producer_well', 'alloc_steam', 'recomm_steam', 'spm', 'target_spm', 'oil',
    #                 'target_oil', 'total_fluid', 'target_fluid', 'recent_water_cut', 'meas_water_cut',
    #                 'recent_sor', 'target_sor', 'chl_steam']]

    return steam_solution


_ = """
#######################################################################################################################
#####################################################   WRAPPERS  #####################################################
#######################################################################################################################
"""


_ = """
#######################################################################################################################
##################################################   CORE EXECUTION   #################################################
#######################################################################################################################
"""


def _PRODUCER_ALLOCATION():
    _accessories._print('Ingesting...')

    DATASETS = {}

    _accessories._print('Finalizing and saving producer steam allocation data...')
    _accessories.finalize_all(DATASETS)


if __name__ == '__main__':
    _PRODUCER_ALLOCATION()


# EOF
