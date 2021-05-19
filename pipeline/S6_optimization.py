# @Author: Shounak Ray <Ray>
# @Date:   22-Apr-2021 10:04:63:638  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: S6_optimization.py
# @Last modified by:   Ray
# @Last modified time: 19-May-2021 12:05:81:811  GMT-0600
# @License: [Private IP]


import ast
import datetime
import os
import subprocess
import sys
from typing import Final

import h2o
import pandas as pd
from h2o_prediction import h2o_model_prediction


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


def check_java_dependency():
    OUT_BLOCK = '»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»\n'
    # Get the major java version in current environment
    java_major_version = int(subprocess.check_output(['java', '-version'],
                                                     stderr=subprocess.STDOUT).decode().split('"')[1].split('.')[0])

    # Check if environment's java version complies with H2O requirements
    # http://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html#requirements
    if not (java_major_version >= 8 and java_major_version <= 14):
        raise ValueError('STATUS: Java Version is not between 8 and 14 (inclusive).\n' +
                         'H2O cluster will not be initialized.')

    print("\x1b[32m" + 'STATUS: Java dependency versions checked and confirmed.')
    print(OUT_BLOCK)


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
    import _context_managers
    import S8_injsteam_allocation as S8_SALL

    # Check java dependency
    check_java_dependency()


_ = """
#######################################################################################################################
#############################################   HYPERPARAMETER SETTINGS   #############################################
#######################################################################################################################
"""

# H2O Server Constants
IP_LINK: Final = 'localhost'                                  # Initializing the server on the local host (temporary)
SECURED: Final = True if(IP_LINK != 'localhost') else False   # https protocol doesn't work locally, should be True
PORT: Final = 54321                                           # Always specify the port that the server should use
SERVER_FORCE: Final = True                                    # Tries to init new server if existing connection fails

steam_range = {
    "A": {
        "min": 0,
        "max": 2000,
    },

    "B": {
        "min": 0,
        "max": 2000,
    },

    "C": {
        "min": 0,
        "max": 2000,
    },

    "E": {
        "min": 0,
        "max": 2000,
    },

    "F": {
        "min": 0,
        "max": 2000,
    },
}

# TODO: EXTERNAL: Model generation should happen in a specific file
BEST_MODEL_PATHS = _accessories.retrieve_local_data_file("Data/Model Candidates/best_models.pkl",  mode=2)

MAPPING = ast.literal_eval(open('mapping.txt', 'r').read().split('[Private IP]\n\n')[1])
INV_MAPPING = {v: k for k, v in MAPPING.items()}

_ = """
#######################################################################################################################
##############################################   FUNCTION DEFINITIONS   ###############################################
#######################################################################################################################
"""

# def create_scenarios(pad_df, date, features, steam_range):
#     # GET LATEST OPERATING SCENARIOS
#     op_condition = pad_df[pad_df['date'] == date]
#     scenario_df = pd.DataFrame([{"alloc_steam": a} for a in range(steam_range['min'], steam_range['max'] + 5, 5)])
#
#     # GET CURRENT CONDITIONS
#     for f in features:
#         if f not in list(scenario_df.columns):
#             scenario_df[f] = op_condition.iloc[-1][f]
#
#     return scenario_df
#
#
# def generate_optimization_table(field_df, pad_df, date, steam_range=steam_range,
#                                 grouper='pad', target='total_fluid', time_col='date',
#                                 forward_days=30):
#     optimization_table = []
#
#     def get_subset(df, date, group, grouper_name=grouper, time_col=time_col, tail_days=365):
#         subset_df = df[df[grouper_name] == group].reset_index(drop=True)
#         subset_df = subset_df[subset_df[time_col] <= date].tail(tail_days)
#         features = list(subset_df.select_dtypes(float).columns)
#
#         return subset_df.reset_index(drop=True), features
#
#     # df = field_df
#     def get_testdfs(model, df, group, features, grouper_name=grouper, time_col=time_col, forward_days=forward_days):
#         # group = 'A'
#         test_df = df[(df[grouper] == group) & (df[time_col] > date)].head(forward_days).dropna(axis=1, how='all')
#         # grouper='PRO_Pad'
#         # time_col='Date'
#         test_df.columns = [MAPPING.get(c) if MAPPING.get(c) != '' else MAPPING.get(c) for c in test_df.columns]
#         orig_features = [c for c in model._model_json['output']['names'] if c in list(test_df)]
#         # print('TEST_DF_1: ' + str(list(test_df)))
#         if 'PRO_Alloc_Steam' in list(test_df):
#             orig_features.append('Steam')
#             test_df.rename(columns={'PRO_Alloc_Steam': 'Steam'}, inplace=True)
#         elif 'alloc_steam' in list(test_df):
#             orig_features.append('Steam')
#             test_df.rename(columns={'alloc_steam': 'Steam'}, inplace=True)
#         # print('TEST_DF_2: ' + str(list(test_df)))
#         # print(orig_features)
#         compatible_df = test_df[orig_features].infer_objects()
#         wanted_types = {k: 'real' if v == float or v == int else 'enum'
#                         for k, v in dict(compatible_df.dtypes).items()}
#         test_pred = model.predict(h2o.H2OFrame(compatible_df,
#                                                column_types=wanted_types)).as_data_frame()['predict']
#         test_actual = test_df[MAPPING.get(target)]
#
#         return test_pred.reset_index(drop=True), test_actual.reset_index(drop=True)
#
#     def configure_scenario_locally(subset_df, date, model, g, features, steam_range=steam_range):
#         scenario_df = create_scenarios(subset_df, date, features, steam_range[g])
#         scenario_df.columns = [MAPPING.get(c) if MAPPING.get(c) != '' else MAPPING.get(c) for c in scenario_df.columns]
#         orig_features = [c for c in model._model_json['output']['names'] if c in list(scenario_df)]
#         # print('SCENARIO_DF_1: ' + str(list(scenario_df)))
#         if 'PRO_Alloc_Steam' in list(scenario_df):
#             orig_features.append('Steam')
#             scenario_df.rename(columns={'PRO_Alloc_Steam': 'Steam'}, inplace=True)
#         elif 'alloc_steam' in list(scenario_df):
#             orig_features.append('Steam')
#             scenario_df.rename(columns={'alloc_steam': 'Steam'}, inplace=True)
#         # print('SCENARIO_DF_2: ' + str(list(scenario_df)))
#         # print(orig_features)
#         compatible_df = scenario_df[orig_features].infer_objects()
#         wanted_types = {k: 'real' if v == float or v == int else 'enum'
#                         for k, v in dict(compatible_df.dtypes).items()}
#         scenario_df['total_fluid'] = list(model.predict(h2o.H2OFrame(compatible_df, column_types=wanted_types)
#                                                         ).as_data_frame()['predict'])
#         scenario_df[grouper] = g
#
#         scenario_df['rmse'] = mean_squared_error(test_actual, test_pred, squared=False)
#         scenario_df['algorithm'] = model.model_id
#         scenario_df['accuracy'] = 1 - abs(test_pred.values - test_actual.values) / test_actual
#
#         return scenario_df
#
#     for d_col in ['PRO_Adj_Pump_Speed', 'Bin_1', 'Bin_8', 'PRO_Adj_Alloc_Oil']:
#         if d_col in field_df.columns:
#             field_df.drop(d_col, axis=1, inplace=True)
#
#     for g in pad_df[grouper].dropna(axis=0).unique():
#         _accessories._print(f"CREATING SCENARIO TABLE FOR: {grouper} {g}.")
#
#         subset_df, features = get_subset(pad_df.copy(), date, g)
#
#         # Running on INDEX 1
#         model = h2o.load_model(BEST_MODEL_PATHS.get(g)[0])
#
#         with _accessories.suppress_stdout():
#             test_pred, test_actual = get_testdfs(model, field_df, g, features)
#             scenario_df = configure_scenario_locally(subset_df.copy(), date, model, g, features)
#
#         optimization_table.append(scenario_df)
#
#     _accessories._print(f"Finished optimization table for all groups on {date}")
#     optimization_table = pd.concat(optimization_table).infer_objects()
#     optimization_table = optimization_table.sort_values([grouper, 'Steam'], ascending=[True, False])
#     optimization_table['exchange_rate'] = optimization_table['Steam'] / optimization_table['PRO_Total_Fluid']
#     optimization_table = optimization_table[[grouper, 'Steam', 'PRO_Total_Fluid',
#                                              'exchange_rate', 'rmse', 'accuracy', 'algorithm']].reset_index(drop=True)
#
#     return optimization_table
#
#
# def optimize(optimization_table, group, steam_avail):
#
#     # OPTIMIZE BASED ON CONSTRAINTS
#     solution = optimization_table.groupby([group]).first().reset_index()
#     steam_usage = solution['Steam'].sum()
#     total_output = solution['PRO_Total_Fluid'].sum()
#
#     print("Initiating optimization... seed:", steam_usage, "target:", steam_avail, "total output:", total_output)
#
#     while steam_usage > steam_avail:
#
#         lowest_delta = solution['exchange_rate'].astype(float).idxmax()
#
#         steam_cutoff = solution.loc[lowest_delta]['Steam']
#         asset_cut = [solution.loc[lowest_delta][group]]
#         to_drop = optimization_table[(optimization_table[group].isin(asset_cut)) &
#                                      (optimization_table['Steam'] >= steam_cutoff)].index
#
#         optimization_table = optimization_table.drop(to_drop)
#         solution = optimization_table.groupby([group]).first().reset_index()  # OPTIMAL SOLUTION
#         steam_usage = solution['Steam'].sum()
#
#     return solution
#
#
# def chloride_control(date, field_df, steam_avail):
#     chl_df = field_df[field_df['date'] == str(date)].sort_values('chloride_contrib', ascending=False)
#     chl_df = chl_df[chl_df['alloc_steam'] > 0].sort_values('chloride_contrib', ascending=False)
#
#     chl_df['cumulative_chl_contrib'] = chl_df['chloride_contrib'].cumsum()
#     chl_df['chl_adj'] = 1
#     chl_df.loc[chl_df['cumulative_chl_contrib'] > 0.6, 'chl_adj'] = -1
#
#     plus_stm = chl_df[chl_df['chl_adj'] == 1]['chloride_contrib'].sum()
#     minus_stm = chl_df[chl_df['chl_adj'] == -1]['chloride_contrib'].sum()
#
#     ratio = plus_stm / minus_stm
#
#     chl_df.loc[chl_df['chl_adj'] == 1, 'chl_steam'] = list(chl_df[chl_df['chl_adj'] == 1]
#                                                            ['chloride_contrib'].sort_values(ascending=False))
#     chl_df.loc[chl_df['chl_adj'] == -1, 'chl_steam'] = list(ratio * chl_df[chl_df['chl_adj'] == -1]
#                                                             ['chloride_contrib'].sort_values())
#
#     chl_df['chl_steam'] = chl_df['chl_steam'] * chl_df['chl_adj']
#     chl_df['date'] = date
#
#     return chl_df[['date', 'pad', 'producer_well', 'alloc_steam', 'chl_steam',
#                    'chlorides', 'cumulative_chl_contrib']].fillna(0)
#
#
# def create_group_data(field_df, group='pad'):
#     # field_df = pd.read_csv('Data/S3 Files/combined_ipc_aggregates.csv')
#     # field_df = pd.read_csv('Data/field_data_pressures.csv').drop('Unnamed: 0', axis=1)
#     field_df = field_df.groupby(['date', group]).agg({'prod_casing_pressure': 'mean',
#                                                       'prod_bhp_heel': 'mean',
#                                                       'prod_bhp_toe': 'mean',
#                                                       'oil': 'sum',
#                                                       # 'water': 'sum',
#                                                       'total_fluid': 'sum',
#                                                       'S': 'sum',
#                                                       'spm': 'sum',
#                                                       }).reset_index().dropna().sort_values(['date', 'pad'])
#
#     field_df['sor'] = field_df['S'] / field_df['oil']
#     return field_df
#
# # date = '2020-05-17'
#
#
# def parallel_optimize(field_df, date, grouper='pad', target='total_fluid', steam_col='Steam', time_col='date'):
#     field_df.columns = [INV_MAPPING.get(c) for c in list(field_df) if INV_MAPPING.get(c) != '']
#     field_df = field_df[[c for c in list(field_df) if c != None]]
#
#     pad_df = create_group_data(field_df.copy(), grouper)
#     chloride_solution = chloride_control(date, field_df, Op_Params['steam_available'])
#     chloride_solution['chl_steam'] = 0.1 * \
#         chloride_solution['chl_steam'] * 6000
#
#     chl_delta = chloride_solution.groupby('pad')['chl_steam'].sum().to_dict()
#
#     _accessories._print('DATE: ' + date)
#     day_df = field_df[field_df[time_col] == str(date)]
#     if day_df.empty:
#         raise ValueError('There\'s no data for this particular day.')
#     steam_avail = int(day_df[steam_col].sum())
#     try:
#         optimization_table = generate_optimization_table(field_df, pad_df, date, steam_range)
#         solution = optimize(optimization_table, grouper, steam_avail)
#         solution[time_col] = date
#         return solution, chloride_solution
#     except Exception as e:
#         _accessories._print('HIT AN EXCEPTION: ' + str(e))
#         return
#
#
# # def run(data, dates):
# #     # data = DATASETS['AGGREGATED'].copy()
# #     print("ACTIVE CPU COUNT:", mp.cpu_count() - 1)
# #
# #     with mp.Pool(processes=mp.cpu_count() - 1) as pool:
# #         # try:
# #         args = list(zip([data.copy()] * len(dates), dates))
# #         agg = pool.starmap(parallel_optimize, args)
# #         # except Exception as e:
# #         #     raise Exception(e)
# #         pool.close()
# #         pool.terminate()
# #
# #     agg_final = pd.concat(agg)
# #     return agg_final
#
#
# # field_df = pd.read_csv('Data/S2 Files/combined_ipc_engineered_phys_ALL.csv')
# # field_df['chloride_contrib'] = 0.5
# # def well_setpoints(solution, well_constraints):
# #     # WELL LEVEL SOLUTION
# #     # 1. NAIVE SOLUTION (divide by number of wells)
# #     # 2. SOR-ECONOMICS BASED (STEAM SHOULD NOT BE MORE THAN MAX SOR x OIL PRODUCTION)
# #     # 3. CHLORIDE CONTROL (MIN STEAM SHOULD RESIST CHLORIDE BREAKTHROUGHS)
# #     # 4. CORRELATION BASED SOLUTION
# #     return
# # def max_sor_steam():
# #     return
# # def min_chloride_control_steam():
# #     return

_ = """
#######################################################################################################################
##########################################   PETE'S OPTIMIZATION FUNCTIONS  ###########################################
#######################################################################################################################
"""
# NOTE: Pasting Pete's upgraded optimization functions here represents an attempt to coerce them in my pipeline


class Optimization_Params:
    def __init__(self, date, steam_available, steam_variance, pad_steam_constraint, well_steam_constraint,
                 well_pump_constraint, watercut_source, chl_steam_percent, pres_steam_percent, recent_days,
                 hist_days, target, features, group):
        self.date = date
        self.steam_available = steam_available
        self.steam_variance = steam_variance
        self.pad_steam_constraint = pad_steam_constraint
        self.well_steam_constraint = well_steam_constraint
        self.well_pump_constraint = well_pump_constraint
        self.watercut_source = watercut_source
        self.chl_steam_percent = chl_steam_percent
        self.pres_steam_percent = pres_steam_percent
        self.recent_days = recent_days
        self.hist_days = hist_days
        self.target = target
        self.features = features
        self.group = group


def calc_average_pressure(field_df, well_interactions):
    """This is used in the master function"""
    # TEMP: calc_average_pressure(field_df, well_interactions)
    for prod_well in field_df['producer_well'].unique():
        well_df = field_df[field_df['producer_well'] == prod_well]

        # GET THE PRESSURE COLUMN IN THE DATASET
        cols = [w.lower() + '_pressure' for w in well_interactions[prod_well]]

        # IF THE PRESSURE DATA FOR THAT INJECTOR IS NOT AVAILABLE, LOCALLY REMOVE IT FROM `well_interactions`
        # NOTE: There is no pressure data for some injectors (i80 and onwards) since there is no steam data for
        #       these injectors (i80 and onwards). So, these injectors are excluded from the average calculation.
        _ = [cols.remove(c) for c in cols.copy() if c not in well_df.columns]

        pressure_df = well_df[cols]
        pressure_series = pressure_df.mean(axis=1)

        field_df.loc[field_df['producer_well'] == prod_well, 'pressure_average'] = pressure_series

    return field_df


def set_defaults(field_df, pad_steam_constraint, well_steam_constraint, well_pump_constraint):

    default_pad_steam_constraint = {"A": {"min": 0, "max": 2000},
                                    "B": {"min": 0, "max": 2000},
                                    "C": {"min": 0, "max": 2000},
                                    "E": {"min": 0, "max": 2000},
                                    "F": {"min": 1200, "max": 2000}}

    default_well_steam_constraint = {a: {"min": 0, "max": 400} for a in field_df['producer_well'].unique()}
    default_well_pump_constraint = {a: {"min": 1, "max": 4.5} for a in field_df['producer_well'].unique()}

    pad_steam_constraint = default_pad_steam_constraint if pad_steam_constraint == {} else pad_steam_constraint
    well_steam_constraint = default_well_steam_constraint if well_steam_constraint == {} else well_steam_constraint
    well_pump_constraint = default_well_pump_constraint if well_pump_constraint == {} else well_pump_constraint

    return pad_steam_constraint, well_steam_constraint, well_pump_constraint


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

    pad_sol = pd.DataFrame(field_kpi)
    well_sol = pd.concat(field_res)

    trimmed_kpi = pd.DataFrame([{"date": date,
                                 "alloc_steam": well_sol['alloc_steam'].sum(),
                                 "recomm_steam": well_sol['recomm_steam'].sum(),
                                 "spm":well_sol['spm'].sum(),
                                 "target_spm": well_sol['target_spm'].sum(),
                                 "oil":well_sol['oil'].sum(),
                                 "target_oil":well_sol['target_oil'].sum(),
                                 "oil_per":well_sol['target_oil'].sum() / well_sol['oil'].sum() - 1,
                                 "fluid":well_sol['total_fluid'].sum(),
                                 "target_fluid":well_sol['target_fluid'].sum(),
                                 "meas_water_cut": well_sol['water'].sum() / well_sol['total_fluid'].sum(),
                                 "target_water_cut": 1 - well_sol['target_oil'].sum() / well_sol['target_fluid'].sum(),
                                 "calc_pump_efficiency": well_sol['calc_pump_efficiency'].dropna().mean(),
                                 "test_pump_efficiency":well_sol['pump_efficiency'].dropna().mean(),
                                 "sor": well_sol['alloc_steam'].sum() / well_sol['oil'].sum(),
                                 "target_sor": well_sol['recomm_steam'].sum() / well_sol['target_oil'].sum()}])

    return well_sol, pad_sol, trimmed_kpi


def create_scenarios(pad_df, date, features, pad_steam_range, pad_chl_delta, steam_variance):
    # GET LATEST OPERATING SCENARIOS
    op_condition = pad_df[pad_df['date'] == date]
    current_steam = op_condition['alloc_steam'].sum()

    scenario_df = pd.DataFrame([{"alloc_steam": a}
                                for a in range(int(pad_steam_range['min'] - pad_chl_delta),
                                               int(pad_steam_range['max'] - pad_chl_delta) + 5, 5)])

    scenario_df['steam_op_state'] = 'normal'
    scenario_df.loc[scenario_df['alloc_steam'] >= current_steam * (1 + steam_variance), 'steam_op_state'] = 'excess'
    scenario_df.loc[scenario_df['alloc_steam'] <= current_steam * (1 - steam_variance), 'steam_op_state'] = 'reduced'
    scenario_df.loc[scenario_df['alloc_steam'] == scenario_df['alloc_steam'].min(), 'steam_op_state'] = 'minimum'

    # GET CURRENT CONDITIONS
    for f in features:
        if f not in scenario_df.columns:
            scenario_df[f] = op_condition.iloc[-1][f]

    return scenario_df


def generate_optimization_table(field_df, pad_df, date, features, target,
                                group, steam_range, chl_delta, steam_variance):
    rell = {'A': 159.394495, 'B': 132.758275, 'C': 154.587740, 'E': 151.573186, 'F': 103.389248}
    optimization_table = []

    # g='A'
    for g in pad_df[group].unique():
        print("CREATING SCENARIO TABLE FOR:", group, g)
        subset_df = pad_df[pad_df[group] == g].fillna(0)
        subset_df = subset_df[(subset_df['date'] <= date) &
                              (subset_df['date'] >= (str(pd.to_datetime(date) -
                                                         pd.Timedelta(365, unit="d")).split(" ")[0]))]

        test_df = field_df[(field_df[group] == g) & (field_df['date'] > date)].head(30).fillna(0)
        test_df['date'] = pd.to_datetime(test_df['date'])

        # #####################################################
        # ### NOTE: FEATURE ENGINEERING AND NEW MODEL SWAP –> This is done directly in `h2o_prediction.py` ####
        # NOTE: Only do feature engineering if needed (as in, the best model requires engineered features)
        # feature_engineering()

        # models_outputs, metric_outputs, model = sagd_ensemble(subset_df[features],
        #                                                       subset_df[target],
        #                                                       test_df[features],
        #                                                       test_df[target])

        model_path = BEST_MODEL_PATHS.get(g)[0]

        models_outputs, metric_outputs, model = h2o_model_prediction(model_path,
                                                                     test_df[['date', target] + features].copy(),
                                                                     tolerable_rmse=rell.get(g))

        ######################################################

        scenario_df = create_scenarios(subset_df, date, features +
                                       [target], steam_range[g], chl_delta[g], steam_variance)
        scenario_df['date'] = pd.to_datetime(date)

        scenario_df['pred'] = sorted(h2o_model_prediction(model_path,
                                                          scenario_df[['date', target] + features],
                                                          tolerable_rmse=rell.get(g),
                                                          just_predictions=True)['predicted'])
        scenario_df[group] = g

        scenario_df['rmse'] = metric_outputs.iloc[0].to_dict()['RMSE']
        scenario_df['algorithm'] = metric_outputs.iloc[0].to_dict()['algorithm_name']
        scenario_df['accuracy'] = metric_outputs.iloc[0].to_dict()['accuracy']

        optimization_table.append(scenario_df)

    optimization_table = pd.concat(optimization_table)
    optimization_table = optimization_table.sort_values([group, 'alloc_steam'], ascending=[True, False])
    optimization_table['exchange_rate'] = optimization_table['alloc_steam'] / optimization_table['pred']
    optimization_table = optimization_table[[group, 'alloc_steam', 'pred', 'exchange_rate',
                                             'steam_op_state', 'rmse', 'accuracy', 'algorithm']].reset_index(drop=True)

    return optimization_table


def chloride_control(date, field_df, steam_avail):

    chl_df = field_df[field_df['date'] == str(date)].sort_values('chloride_contrib', ascending=False)
    chl_df = chl_df[chl_df['alloc_steam'] > 0].sort_values('chloride_contrib', ascending=False)

    chl_df['cumulative_chl_contrib'] = chl_df['chloride_contrib'].cumsum()
    chl_df['chl_adj'] = 1
    chl_df.loc[chl_df['cumulative_chl_contrib'] > 0.6, 'chl_adj'] = -1

    plus_stm = chl_df[chl_df['chl_adj'] == 1]['chloride_contrib'].sum()
    minus_stm = chl_df[chl_df['chl_adj'] == -1]['chloride_contrib'].sum()

    ratio = plus_stm / minus_stm

    chl_df.loc[chl_df['chl_adj'] == 1, 'chl_steam'] = list(chl_df[chl_df['chl_adj'] == 1]
                                                           ['chloride_contrib'].sort_values(ascending=False))
    chl_df.loc[chl_df['chl_adj'] == -1, 'chl_steam'] = list(ratio * chl_df[chl_df['chl_adj'] == -1]
                                                            ['chloride_contrib'].sort_values())

    chl_df['chl_steam'] = chl_df['chl_steam'] * chl_df['chl_adj']
    chl_df['date'] = date

    return chl_df[['date', 'pad', 'producer_well', 'alloc_steam', 'chl_steam',
                   'chlorides', 'cumulative_chl_contrib']].fillna(0)


def create_group_data(field_df, group):
    field_df = field_df.groupby(['date', group]).agg({'prod_casing_pressure': 'mean',
                                                      'prod_bhp_heel': 'mean',
                                                      'prod_bhp_toe': 'mean',
                                                      'oil': 'sum',
                                                      'water': 'sum',
                                                      'total_fluid': 'sum',
                                                      'alloc_steam': 'sum',
                                                      'spm': 'sum',
                                                      }).reset_index().dropna().sort_values(['date', 'pad'])

    field_df['sor'] = field_df['alloc_steam'] / field_df['oil']
    return field_df


def parallel_optimize(field_df, date, Op_Params):
    pad_df = create_group_data(field_df, Op_Params.group)
    # day_df = pad_df[pad_df['date'] == str(date)]

    chloride_solution = chloride_control(date, field_df, Op_Params.steam_available)
    chloride_solution['chl_steam'] = Op_Params.chl_steam_percent * \
        chloride_solution['chl_steam'] * Op_Params.steam_available

    chl_delta = chloride_solution.groupby('pad')['chl_steam'].sum().to_dict()

    optimization_table = generate_optimization_table(field_df, pad_df, date,
                                                     Op_Params.features, Op_Params.target, Op_Params.group,
                                                     steam_range=Op_Params.pad_steam_constraint, chl_delta=chl_delta,
                                                     steam_variance=Op_Params.steam_variance)
    solution = optimize(optimization_table, Op_Params.group, Op_Params.steam_available)
    solution['date'] = date

    return solution, chloride_solution


def optimize(optimization_table, group, steam_avail):
    # OPTIMIZE BASED ON CONSTRAINTS
    solution = optimization_table.groupby([group]).first().reset_index()
    steam_usage = solution['alloc_steam'].sum()
    total_output = solution['pred'].sum()

    print("Initiating optimization... seed:", steam_usage, "target:", steam_avail, "total output:", total_output)
    optimization_table[optimization_table['pad'] == 'F']

    while steam_usage > steam_avail:
        # REDUCE EXCESS STEAM FIRST!
        lowest_delta = solution[(solution['steam_op_state'] == 'excess')]
        if lowest_delta.shape[0] == 0:
            lowest_delta = solution[(solution['steam_op_state'] == 'normal')]
        if lowest_delta.shape[0] == 0:
            lowest_delta = solution[(solution['steam_op_state'] == 'reduced')]
        if lowest_delta.shape[0] == 0:
            # IF NO SOLUTION AVAILABLE
            break

        lowest_delta = lowest_delta['exchange_rate'].astype(float).idxmax()

        steam_cutoff = solution.loc[lowest_delta]['alloc_steam']
        asset_cut = [solution.loc[lowest_delta][group]]
        to_drop = optimization_table[(optimization_table[group].isin(asset_cut)) &
                                     (optimization_table['alloc_steam'] >= steam_cutoff)].index

        optimization_table = optimization_table.drop(to_drop)
        solution = optimization_table.groupby([group]).first().reset_index()  # OPTIMAL SOLUTION
        steam_usage = solution['alloc_steam'].sum()

    return solution


_ = """
#######################################################################################################################
#####################################################   WRAPPERS  #####################################################
#######################################################################################################################
"""


def setup_and_server(SECURED=SECURED, IP_LINK=IP_LINK, PORT=PORT, SERVER_FORCE=SERVER_FORCE):
    # Initialize the cluster
    h2o.init(https=SECURED,
             ip=IP_LINK,
             port=PORT,
             start_h2o=SERVER_FORCE)


def shutdown_confirm(h2o_instance: type(h2o)) -> None:
    """Terminates the provided H2O cluster.

    Parameters
    ----------
    cluster : type(h2o)
        The H2O instance where the server was initialized.

    Returns
    -------
    None
        Nothing. ValueError may be raised during processing and cluster metrics may be printed.

    """
    # """DATA SANITATION"""
    # _provided_args = locals()
    # name = sys._getframe(0).f_code.co_name
    # _expected_type_args = {'h2o_instance': [type(h2o)]}
    # _expected_value_args = {'h2o_instance': None}
    # util_data_type_sanitation(_provided_args, _expected_type_args, name)
    # util_data_range_sanitation(_provided_args, _expected_value_args, name)
    # """END OF DATA SANITATION"""

    # SHUT DOWN the cluster after you're done working with it
    h2o_instance.remove_all()
    h2o_instance.cluster().shutdown()


def configure_aggregates(aggregate_results, aggregate_reference, rell, grouper='pad'):
    aggregate_results['allowable_rmse'] = aggregate_results[grouper].apply(lambda x: rell.get(x))
    aggregate_results['rel_rmse'] = (aggregate_results['rmse']) - aggregate_results['allowable_rmse']
    aggregate_results.columns = ['PRO_Pad',
                                 'Reccomended_Steam',
                                 'Predicted_Total_Fluid',
                                 'exchange_rate',
                                 'RMSE',
                                 'Accuracy',
                                 'Algorithm',
                                 'Date',
                                 'Tolerable_RMSE',
                                 'Relative_RMSE']
    aggregate_results = pd.merge(aggregate_results, aggregate_reference, on=['Date', 'PRO_Pad'])

    return aggregate_results


_ = """
#######################################################################################################################
##################################################   CORE EXECUTION  ##################################################
#######################################################################################################################
"""


def _OPTIMIZATION(field_df, date, well_interactions,
                  steam_available=6000, steam_variance=0.25, pad_steam_constraint={},
                  well_steam_constraint={}, well_pump_constraint={}, watercut_source='recent_water_cut',
                  chl_steam_percent=0.1, pres_steam_percent=0.15, recent_days=45, hist_days=365,
                  target='total_fluid', group='pad'):
    """This seems to be the "master" function"""

    # NOTE: Map everything to Pete's output standard (based on `MAPPING` and `INV_MAPPING`)
    field_df.columns = [INV_MAPPING.get(c) if INV_MAPPING.get(c) != '' else c for c in field_df.columns]

    features = [  # 'pressure_average',
        'prod_casing_pressure',
        'prod_bhp_heel',
        'prod_bhp_toe',
        'alloc_steam',
        # 'spm'
    ]

    # NOTE: Get pressures
    field_df = calc_average_pressure(field_df, well_interactions)

    # NOTE: Set hyperparams
    pad_steam_constraint, well_steam_constraint, well_pump_constraint = set_defaults(field_df, pad_steam_constraint,
                                                                                     well_steam_constraint,
                                                                                     well_pump_constraint)
    Op_Params = Optimization_Params(date, steam_available, steam_variance, pad_steam_constraint, well_steam_constraint,
                                    well_pump_constraint, watercut_source, chl_steam_percent, pres_steam_percent,
                                    recent_days, hist_days, target, features, group)

    field_df.columns = [f.lower() for f in list(field_df)]

    # NOTE: Pad-level Allocation – Optimization
    macro_solution, chloride_solution = parallel_optimize(field_df, date, Op_Params)

    # NOTE: Pad-level Allocation – Solution, Well-Level Solution
    well_sol, pad_sol, field_kpi = get_field_solution(field_df, date, macro_solution, chloride_solution, Op_Params)

    # NOTE: Injector pre-allocation data
    well_allocations = well_sol.set_index('producer_well')['recomm_steam'].to_dict()

    return well_allocations, well_sol, pad_sol, field_kpi


def _OPTIMIZATION(data=None, aggregate_reference=None, producer_taxonomy=None, _return=True, flow_ingest=True,
                  start_date='2015-04-01', end_date='2020-12-20', engineered=True,
                  today=False, singular_date=''):
    """PURPOSE: TO GENERATE AN OPTIMIZED *PAD* RECCOMMENDATION FOR A PARTICULAR DATE ~OR~ RANGE OF DATES
       INPUTS:
       1 – ONE MERGED DATASET, IDEALLY AGGREGATED ON THE PAD LEVEL
       PROCESSING:
       OUTPUT: 1 – A DATASET W/ STEAM ALLOCATIONS FOR EACH PAD W/ PERFORMANCE METRICS AND ORIGINAL DATA
                   RESOLUTION: For a particular date (or range of dates), per pad
               2 – A DICTIONARY W/ STEAM ALLOCATIONS FOR EACH PAD
                   *Note: This is same as the above dataset, but trimmed and restructured for accessibility*
    """
    """The `engineered` parameter is a high-level preference, meaning that this setting coerces which type of model
       to use for optimization – even if that type is not the best model."""

    rell = {'A': 159.394495, 'B': 132.758275, 'C': 154.587740, 'E': 151.573186, 'F': 103.389248}
    if singular_date != '':
        start_date, end_date = singular_date
    elif today:
        # WARNING: This may result in a crash if there is no data available for this day
        # TODO: To fix this issue, always get the most recent day from the inputted dataset
        start_date, end_date = str(datetime.datetime.now()).split(' ')[0]

    _accessories._print('Initializing H2O server to access model files...')
    setup_and_server()

    if flow_ingest:
        _accessories._print('Loading the most basic, aggregated + mathematically engineered datasets from LAST...')
        DATASETS = {'AGGREGATED': data}
    else:
        _accessories._print('Loading the most basic, aggregated + mathematically engineered datasets from SAVED...')
        DATASETS = {'AGGREGATED':
                    _accessories.retrieve_local_data_file('Data/S3 Files/combined_ipc_aggregates_ALL.csv')}

    # results = []
    # for date in dates:
    #     try:
    #         solution = parallel_optimize(date)
    #     except Exception:
    #         continue
    #     results.append(solution)
    # aggregate_results = pd.concat(results)

    # TODO: Model selection
    # TODO: Load acceptable ranges from S5_Modeling

    _accessories._print('Performing backtesting...')
    dates = pd.date_range(*(start_date, end_date)).strftime('%Y-%m-%d')

    field_df = DATASETS['AGGREGATED_ENG'].copy()

    # TEMP: Assigns pad relationships to data if it's not already av
    if 'PRO_Pad' not in list(field_df):
        # producer_taxonomy = _accessories.retrieve_local_data_file('Data/Pickles/PRODUCTION_[Well, Pad].pkl', mode=2)
        field_df['PRO_Pad'] = field_df['PRO_Well'].apply(lambda x: producer_taxonomy.get(x))

    macro_solution, chloride_solution = parallel_optimize(field_df.copy(), dates[0])
    aggregate_results = configure_aggregates(macro_solution, aggregate_reference, rell)

    _accessories._print('Shutting down H2O server...')
    shutdown_confirm(h2o)

    if _return:
        return aggregate_results, chloride_solution
    else:
        _accessories.save_local_data_file(aggregate_results,
                                          f'Data/S6 Files/Right_Aggregates_{start_date}_{end_date}.csv')


if __name__ == '__main__':
    _OPTIMIZATION()


# ###
# aggregate_results = aggregate_results.reset_index()
# aggregate_results[aggregate_results['pad'] == 'A']['accuracy'].plot()
#
# aggregate_results = aggregate_results.rename(columns={"alloc_steam": "recomm_steam"})
#
# merged_df = pd.merge(aggregate_results, field_df, how='left', on=['date', 'pad'])
# merged_df['date'] = pd.to_datetime(merged_df['date'])
# merged_df = merged_df.set_index('date')
# merged_df['pred_oil'] = merged_df['pred'] * (1 - merged_df['water'] / merged_df['total_fluid'])
#
# merged_df[merged_df['pad'] == 'E']['alloc_steam'].plot(color='black', figsize=(11, 8))
# merged_df[merged_df['pad'] == 'E']['recomm_steam'].plot(color='red', figsize=(11, 8))
#
# merged_df[merged_df['pad'] == 'A']['total_fluid'].plot(color='black', figsize=(11, 8))
# merged_df[merged_df['pad'] == 'A']['pred'].plot(color='brown', figsize=(11, 8))


#
