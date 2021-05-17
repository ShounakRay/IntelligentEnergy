# @Author: Shounak Ray <Ray>
# @Date:   22-Apr-2021 10:04:63:638  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: S6_optimization.py
# @Last modified by:   Ray
# @Last modified time: 17-May-2021 16:05:22:221  GMT-0600
# @License: [Private IP]


import datetime
import os
import subprocess
import sys
from typing import Final

import billiard as mp
import h2o
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


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

BEST_MODEL_PATHS = _accessories.retrieve_local_data_file("Data/Model Candidates/best_models.pkl",
                                                         mode=2)

Op_Params = {
    "date": "2020-12-01",
    "steam_available": 6000,
    "steam_variance": 0.25,
    "pad_steam_constraint": {
        "A": {"min": 0, "max": 2000},
        "B": {"min": 0, "max": 2000},
        "C": {"min": 0, "max": 2000},
        "E": {"min": 0, "max": 2000},
        "F": {"min": 1200, "max": 2000},
    },
    "well_steam_constraint": {
        "AP2": {"min": 0, "max": 400},
        "CP7": {"min": 0, "max": 400},
        "CP8": {"min": 0, "max": 400},
        "EP2": {"min": 0, "max": 400},
        "EP3": {"min": 0, "max": 400},
        "EP4": {"min": 0, "max": 400},
        "EP5": {"min": 0, "max": 400},
        "CP6": {"min": 0, "max": 400},
        "EP7": {"min": 0, "max": 400},
        "FP2": {"min": 0, "max": 400},
        "FP3": {"min": 0, "max": 400},
        "FP4": {"min": 0, "max": 400},
        "FP5": {"min": 0, "max": 400},
        "FP6": {"min": 0, "max": 400},
        "FP7": {"min": 0, "max": 400},
        "FP1": {"min": 0, "max": 400},
        "CP5": {"min": 0, "max": 400},
        "EP6": {"min": 0, "max": 400},
        "CP3": {"min": 0, "max": 400},
        "CP4": {"min": 0, "max": 400},
        "AP3": {"min": 0, "max": 400},
        "AP4": {"min": 0, "max": 400},
        "AP5": {"min": 0, "max": 400},
        "AP6": {"min": 0, "max": 400},
        "AP8": {"min": 0, "max": 400},
        "BP1": {"min": 0, "max": 400},
        "AP7": {"min": 0, "max": 400},
        "BP3": {"min": 0, "max": 400},
        "BP4": {"min": 0, "max": 400},
        "BP5": {"min": 0, "max": 400},
        "BP6": {"min": 0, "max": 400},
        "CP1": {"min": 0, "max": 400},
        "CP2": {"min": 0, "max": 400},
        "BP2": {"min": 0, "max": 400},
    },
    "well_pump_constraint": {
        "AP2": {"min": 1, "max": 4.5},
        "CP7": {"min": 1, "max": 4.5},
        "CP8": {"min": 1, "max": 4.5},
        "EP2": {"min": 1, "max": 4.5},
        "EP3": {"min": 1, "max": 4.5},
        "EP4": {"min": 1, "max": 4.5},
        "EP5": {"min": 1, "max": 4.5},
        "CP6": {"min": 1, "max": 4.5},
        "EP7": {"min": 1, "max": 4.5},
        "FP2": {"min": 1, "max": 4.5},
        "FP3": {"min": 1, "max": 4.5},
        "FP4": {"min": 1, "max": 4.5},
        "FP5": {"min": 1, "max": 4.5},
        "FP6": {"min": 1, "max": 4.5},
        "FP7": {"min": 1, "max": 4.5},
        "FP1": {"min": 1, "max": 4.5},
        "CP5": {"min": 1, "max": 4.5},
        "EP6": {"min": 1, "max": 4.5},
        "CP3": {"min": 1, "max": 4.5},
        "CP4": {"min": 1, "max": 4.5},
        "AP3": {"min": 1, "max": 4.5},
        "AP4": {"min": 1, "max": 4.5},
        "AP5": {"min": 1, "max": 4.5},
        "AP6": {"min": 1, "max": 4.5},
        "AP8": {"min": 1, "max": 4.5},
        "BP1": {"min": 1, "max": 4.5},
        "AP7": {"min": 1, "max": 4.5},
        "BP3": {"min": 1, "max": 4.5},
        "BP4": {"min": 1, "max": 4.5},
        "BP5": {"min": 1, "max": 4.5},
        "BP6": {"min": 1, "max": 4.5},
        "CP1": {"min": 1, "max": 4.5},
        "CP2": {"min": 1, "max": 4.5},
        "BP2": {"min": 1, "max": 4.5},
    },
    "watercut_source": "meas_water_cut",
    "chl_steam_percent": 0.1,
    "pres_steam_percent": 0.15,
    "recent_days": 45,
    "hist_days": 365,
    "target": "total_fluid",
    "features": [
        "prod_casing_pressure",
        "prod_bhp_heel",
        "prod_bhp_toe",
        "alloc_steam",
    ],
    "group": "pad",
}

MAPPING = {'date': 'Date',
           'uwi': 'PRO_UWI',
           'producer_well': 'PRO_Well',
           'spm': 'PRO_Adj_Pump_Speed',
           'hours_on_prod': 'PRO_Time_On',
           'prod_casing_pressure': 'PRO_Casing_Pressure',
           'prod_bhp_heel': 'PRO_Heel_Pressure',
           'prod_bhp_toe': 'PRO_Toe_Pressure',
           'prod_bht_heel': 'PRO_Heel_Temp',
           'prod_bht_toe': 'PRO_Toe_Temp',
           'oil': 'PRO_Adj_Alloc_Oil',
           'water': 'PRO_Alloc_Water',
           'bin_1': 'Bin_1',
           'bin_2': 'Bin_2',
           'bin_3': 'Bin_3',
           'bin_4': 'Bin_4',
           'bin_5': 'Bin_5',
           'bin_6': 'Bin_6',
           'bin_7': 'Bin_7',
           'bin_8': 'Bin_8',
           'ci06_steam': '',
           'ci07_steam': '',
           'ci08_steam': '',
           'i02_steam': '',
           'i03_steam': '',
           'i04_steam': '',
           'i05_steam': '',
           'i06_steam': '',
           'i07_steam': '',
           'i08_steam': '',
           'i09_steam': '',
           'i10_steam': '',
           'i11_steam': '',
           'i12_steam': '',
           'i13_steam': '',
           'i14_steam': '',
           'i15_steam': '',
           'i16_steam': '',
           'i17_steam': '',
           'i18_steam': '',
           'i19_steam': '',
           'i20_steam': '',
           'i21_steam': '',
           'i22_steam': '',
           'i23_steam': '',
           'i24_steam': '',
           'i25_steam': '',
           'i26_steam': '',
           'i27_steam': '',
           'i28_steam': '',
           'i29_steam': '',
           'i30_steam': '',
           'i31_steam': '',
           'i32_steam': '',
           'i33_steam': '',
           'i34_steam': '',
           'i35_steam': '',
           'i36_steam': '',
           'i37_steam': '',
           'i38_steam': '',
           'i39_steam': '',
           'i40_steam': '',
           'i41_steam': '',
           'i42_steam': '',
           'i43_steam': '',
           'i44_steam': '',
           'i45_steam': '',
           'i46_steam': '',
           'i47_steam': '',
           'i48_steam': '',
           'i49_steam': '',
           'i50_steam': '',
           'i51_steam': '',
           'i52_steam': '',
           'i53_steam': '',
           'i54_steam': '',
           'i55_steam': '',
           'i56_steam': '',
           'i57_steam': '',
           'i58_steam': '',
           'i59_steam': '',
           'i60_steam': '',
           'i61_steam': '',
           'i62_steam': '',
           'i63_steam': '',
           'i64_steam': '',
           'i65_steam': '',
           'i66_steam': '',
           'i67_steam': '',
           'i68_steam': '',
           'i69_steam': '',
           'i70_steam': '',
           'i71_steam': '',
           'i72_steam': '',
           'i73_steam': '',
           'i74_steam': '',
           'i75_steam': '',
           'i76_steam': '',
           'i77_steam': '',
           'i78_steam': '',
           'i79_steam': '',
           'pad': 'PRO_Pad',
           'test_oil': 'PRO_Oil',
           'test_water': 'PRO_Water',
           'test_chlorides': 'PRO_Chlorides',
           'test_spm': '',
           'pump_size': '',
           'pump_efficiency': 'PRO_Pump_Efficiency',
           'op_approved': 'op_approved',
           'op_comment': 'op_comment',
           'eng_approved': 'PRO_Engineering_Approved',
           'eng_comment': 'eng_comment',
           'test_total_fluid': 'test_total_fluid',
           'total_fluid': 'PRO_Total_Fluid',
           'volume_per_stroke': 'volume_per_stroke',
           'theoretical_fluid': 'theoretical_fluid',
           'test_water_cut': 'test_water_cut',
           'theoretical_water': 'theoretical_water',
           'theoretical_oil': 'theoretical_water',
           'alloc_steam': 'PRO_Alloc_Steam',
           'sor': 'sor',
           'chlorides': 'PRO_Chlorides',
           'meas_water_cut': 'meas_water_cut',
           'field_chloride': 'field_chloride',
           'chloride_contrib': 'chloride_contrib',
           'field': 'field',
           'pressure_average': 'pressure_average',
           'ci06_pressure': '',
           'ci07_pressure': '',
           'ci08_pressure': '',
           'i02_pressure': '',
           'i03_pressure': '',
           'i04_pressure': '',
           'i05_pressure': '',
           'i06_pressure': '',
           'i07_pressure': '',
           'i08_pressure': '',
           'i09_pressure': '',
           'i10_pressure': '',
           'i11_pressure': '',
           'i12_pressure': '',
           'i13_pressure': '',
           'i14_pressure': '',
           'i15_pressure': '',
           'i16_pressure': '',
           'i17_pressure': '',
           'i18_pressure': '',
           'i19_pressure': '',
           'i20_pressure': '',
           'i21_pressure': '',
           'i22_pressure': '',
           'i23_pressure': '',
           'i24_pressure': '',
           'i25_pressure': '',
           'i26_pressure': '',
           'i27_pressure': '',
           'i28_pressure': '',
           'i29_pressure': '',
           'i30_pressure': '',
           'i31_pressure': '',
           'i32_pressure': '',
           'i33_pressure': '',
           'i34_pressure': '',
           'i35_pressure': '',
           'i36_pressure': '',
           'i37_pressure': '',
           'i38_pressure': '',
           'i39_pressure': '',
           'i40_pressure': '',
           'i41_pressure': '',
           'i42_pressure': '',
           'i43_pressure': '',
           'i44_pressure': '',
           'i45_pressure': '',
           'i46_pressure': '',
           'i47_pressure': '',
           'i48_pressure': '',
           'i49_pressure': '',
           'i50_pressure': '',
           'i51_pressure': '',
           'i52_pressure': '',
           'i53_pressure': '',
           'i54_pressure': '',
           'i55_pressure': '',
           'i56_pressure': '',
           'i57_pressure': '',
           'i58_pressure': '',
           'i59_pressure': '',
           'i60_pressure': '',
           'i61_pressure': '',
           'i62_pressure': '',
           'i63_pressure': '',
           'i64_pressure': '',
           'i65_pressure': '',
           'i66_pressure': '',
           'i67_pressure': '',
           'i68_pressure': '',
           'i69_pressure': '',
           'i70_pressure': '',
           'i71_pressure': '',
           'i72_pressure': '',
           'i73_pressure': '',
           'i74_pressure': '',
           'i75_pressure': '',
           'i76_pressure': '',
           'i77_pressure': '',
           'i78_pressure': '',
           'i79_pressure': '',
           'i80': '',
           'i82': '',
           'i83': '',
           'i84': '',
           'i85': '',
           'i86': '',
           'i87': '',
           'i88': '',
           'i89': '',
           'i90': '',
           'i91': '',
           'i92': '',
           'i93': ''}
INV_MAPPING = {v: k for k, v in MAPPING.items()}

_ = """
#######################################################################################################################
##############################################   FUNCTION DEFINITIONS   ###############################################
#######################################################################################################################
"""


def create_scenarios(pad_df, date, features, steam_range):
    # GET LATEST OPERATING SCENARIOS
    op_condition = pad_df[pad_df['date'] == date]
    scenario_df = pd.DataFrame([{"alloc_steam": a} for a in range(steam_range['min'], steam_range['max'] + 5, 5)])

    # GET CURRENT CONDITIONS
    for f in features:
        if f not in list(scenario_df.columns):
            scenario_df[f] = op_condition.iloc[-1][f]

    return scenario_df


def generate_optimization_table(field_df, date, steam_range=steam_range,
                                grouper='pad', target='total_fluid', time_col='date',
                                forward_days=30):
    optimization_table = []

    def get_subset(df, date, group, grouper_name=grouper, time_col=time_col, tail_days=365):
        subset_df = df[df[grouper_name] == group].reset_index(drop=True)
        subset_df = subset_df[subset_df[time_col] <= date].tail(tail_days)
        features = list(subset_df.select_dtypes(float).columns)

        return subset_df.reset_index(drop=True), features

    # df = field_df
    def get_testdfs(model, df, group, features, grouper_name=grouper, time_col=time_col, forward_days=forward_days):
        # group = 'A'
        test_df = df[(df[grouper] == group) & (df[time_col] > date)].head(forward_days).dropna(axis=1, how='all')
        # grouper='PRO_Pad'
        # time_col='Date'
        test_df.columns = [MAPPING.get(c) if MAPPING.get(c) != '' else MAPPING.get(c) for c in test_df.columns]
        orig_features = [c for c in model._model_json['output']['names'] if c in list(test_df)]
        compatible_df = test_df[orig_features].infer_objects()
        wanted_types = {k: 'real' if v == float or v == int else 'enum'
                        for k, v in dict(compatible_df.dtypes).items()}
        test_pred = model.predict(h2o.H2OFrame(compatible_df,
                                               column_types=wanted_types)).as_data_frame()['predict']
        test_actual = test_df[MAPPING.get(target)]

        return test_pred.reset_index(drop=True), test_actual.reset_index(drop=True)

    def configure_scenario_locally(subset_df, date, model, g, features, steam_range=steam_range):
        scenario_df = create_scenarios(subset_df, date, features, steam_range[g])
        scenario_df.columns = [MAPPING.get(c) if MAPPING.get(c) != '' else MAPPING.get(c) for c in scenario_df.columns]
        orig_features = [c for c in model._model_json['output']['names'] if c in list(scenario_df)]
        compatible_df = scenario_df[orig_features].infer_objects()
        wanted_types = {k: 'real' if v == float or v == int else 'enum'
                        for k, v in dict(compatible_df.dtypes).items()}
        scenario_df['total_fluid'] = list(model.predict(h2o.H2OFrame(compatible_df, column_types=wanted_types)
                                                        ).as_data_frame()['predict'])
        scenario_df[grouper] = g

        scenario_df['rmse'] = mean_squared_error(test_actual, test_pred, squared=False)
        scenario_df['algorithm'] = model.model_id
        scenario_df['accuracy'] = 1 - abs(test_pred.values - test_actual.values) / test_actual

        return scenario_df

    for d_col in ['PRO_Adj_Pump_Speed', 'Bin_1', 'Bin_8', 'PRO_Adj_Alloc_Oil']:
        if d_col in field_df.columns:
            field_df.drop(d_col, axis=1, inplace=True)

    for g in field_df[grouper].dropna(axis=0).unique():
        _accessories._print(f"CREATING SCENARIO TABLE FOR: {grouper} {g}.")

        subset_df, features = get_subset(field_df, date, g)
        print('GOT SUBSET')

        # file = open('Modeling Reference Files/6086 – ENG: True, WEIGHT: False, TIME: 20/MODELS_6086.pkl', 'rb')

        # Running on INDEX 1
        model = h2o.load_model(BEST_MODEL_PATHS.get(g)[0])
        print('LOADED MODEL')

        print('GOT TEST DATAFRAMES AND CONFIGURING LOCALLY')
        with _accessories.suppress_stdout():
            test_pred, test_actual = get_testdfs(model, field_df, g, features)
            scenario_df = configure_scenario_locally(subset_df, date, model, g, features)

        optimization_table.append(scenario_df)

    _accessories._print(f"Finished optimization table for all groups on {date}")
    optimization_table = pd.concat(optimization_table).infer_objects()
    optimization_table = optimization_table.sort_values([grouper, 'PRO_Alloc_Steam'], ascending=[True, False])
    optimization_table['exchange_rate'] = optimization_table['PRO_Alloc_Steam'] / optimization_table['PRO_Total_Fluid']
    optimization_table = optimization_table[[grouper, 'PRO_Alloc_Steam', 'PRO_Total_Fluid',
                                             'exchange_rate', 'rmse', 'accuracy', 'algorithm']].reset_index(drop=True)

    return optimization_table


def optimize(optimization_table, group, steam_avail):

    # OPTIMIZE BASED ON CONSTRAINTS
    solution = optimization_table.groupby([group]).first().reset_index()
    steam_usage = solution['PRO_Alloc_Steam'].sum()
    total_output = solution['PRO_Total_Fluid'].sum()

    print("Initiating optimization... seed:", steam_usage, "target:", steam_avail, "total output:", total_output)

    while steam_usage > steam_avail:

        lowest_delta = solution['exchange_rate'].astype(float).idxmax()

        steam_cutoff = solution.loc[lowest_delta]['PRO_Alloc_Steam']
        asset_cut = [solution.loc[lowest_delta][group]]
        to_drop = optimization_table[(optimization_table[group].isin(asset_cut)) &
                                     (optimization_table['PRO_Alloc_Steam'] >= steam_cutoff)].index

        optimization_table = optimization_table.drop(to_drop)
        solution = optimization_table.groupby([group]).first().reset_index()  # OPTIMAL SOLUTION
        steam_usage = solution['PRO_Alloc_Steam'].sum()

    return solution


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


def create_group_data(field_df, group='pad'):
    # field_df = pd.read_csv('Data/S3 Files/combined_ipc_aggregates.csv')
    # field_df = pd.read_csv('Data/field_data_pressures.csv').drop('Unnamed: 0', axis=1)
    field_df = field_df.groupby(['date', group]).agg({'prod_casing_pressure': 'mean',
                                                      'prod_bhp_heel': 'mean',
                                                      'prod_bhp_toe': 'mean',
                                                      'oil': 'sum',
                                                      # 'water': 'sum',
                                                      'total_fluid': 'sum',
                                                      'alloc_steam': 'sum',
                                                      'spm': 'sum',
                                                      }).reset_index().dropna().sort_values(['date', 'pad'])

    field_df['sor'] = field_df['alloc_steam'] / field_df['oil']
    return field_df

# dates = ['2020-05-17']


def parallel_optimize(field_df, date, grouper='pad', target='total_fluid', steam_col='alloc_steam', time_col='date'):
    field_df.columns = [INV_MAPPING.get(c) for c in list(field_df) if INV_MAPPING.get(c) != '']
    field_df = field_df[[c for c in list(field_df) if c != None]]

    pad_df = create_group_data(field_df.copy(), grouper)
    chloride_solution = chloride_control(date, field_df, Op_Params['steam_available'])
    chloride_solution['chl_steam'] = 0.1 * \
        chloride_solution['chl_steam'] * 6000

    chl_delta = chloride_solution.groupby('pad')['chl_steam'].sum().to_dict()

    _accessories._print('DATE: ' + date)
    day_df = field_df[field_df[time_col] == str(date)]
    if day_df.empty:
        raise ValueError('There\'s no data for this particular day.')
    steam_avail = int(day_df[steam_col].sum())
    try:
        print('HEdRE')
        optimization_table = generate_optimization_table(field_df, date, steam_range)
        solution = optimize(optimization_table, grouper, steam_avail)
        solution[time_col] = date
        return solution, chloride_solution
    except Exception as e:
        _accessories._print('HIT AN EXCEPTION: ' + str(e))
        return


# def run(data, dates):
#     # data = DATASETS['AGGREGATED_NOENG'].copy()
#     print("ACTIVE CPU COUNT:", mp.cpu_count() - 1)
#
#     with mp.Pool(processes=mp.cpu_count() - 1) as pool:
#         # try:
#         args = list(zip([data.copy()] * len(dates), dates))
#         agg = pool.starmap(parallel_optimize, args)
#         # except Exception as e:
#         #     raise Exception(e)
#         pool.close()
#         pool.terminate()
#
#     agg_final = pd.concat(agg)
#     return agg_final


# field_df = pd.read_csv('Data/S2 Files/combined_ipc_engineered_phys_ALL.csv')
# field_df['chloride_contrib'] = 0.5
# def well_setpoints(solution, well_constraints):
#     # WELL LEVEL SOLUTION
#     # 1. NAIVE SOLUTION (divide by number of wells)
#     # 2. SOR-ECONOMICS BASED (STEAM SHOULD NOT BE MORE THAN MAX SOR x OIL PRODUCTION)
#     # 3. CHLORIDE CONTROL (MIN STEAM SHOULD RESIST CHLORIDE BREAKTHROUGHS)
#     # 4. CORRELATION BASED SOLUTION
#     return
# def max_sor_steam():
#     return
# def min_chloride_control_steam():
#     return


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


def configure_aggregates(aggregate_results, rell, grouper='pad'):
    aggregate_results['allowable_rmse'] = aggregate_results[grouper].apply(lambda x: rell.get(x))
    aggregate_results['rel_rmse'] = (aggregate_results['rmse']) - aggregate_results['allowable_rmse']
    old_raw = _accessories.retrieve_local_data_file('Data/S3 Files/combined_ipc_aggregates_ALL.csv')
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
    aggregate_results = pd.merge(aggregate_results, old_raw, on=['Date', 'PRO_Pad'])

    return aggregate_results


_ = """
#######################################################################################################################
##################################################   CORE EXECUTION  ##################################################
#######################################################################################################################
"""


def _OPTIMIZATION(data=None, _return=True, flow_ingest=True,
                  start_date='2015-04-01', end_date='2020-12-20', engineered=True,
                  today=False, singular_date=''):
    rell = {'A': 159.394495, 'B': 132.758275, 'C': 154.587740, 'E': 151.573186, 'F': 103.389248}
    if singular_date != '':
        start_date = singular_date
        end_date = singular_date
    elif today:
        start_date = str(datetime.datetime.now()).split(' ')[0]
        end_date = str(datetime.datetime.now()).split(' ')[0]

    _accessories._print('Initializing H2O server to access model files...')
    setup_and_server()

    if flow_ingest:
        _accessories._print(
            'Loading the most basic, aggregated + mathematically engineered datasets from LAST STEP...')
        DATASETS = {'AGGREGATED_NOENG': data,
                    'AGGREGATED_ENG':
                    _accessories.retrieve_local_data_file('Data/S4 Files/combined_ipc_engineered_math.csv')}
    else:
        _accessories._print(
            'Loading the most basic, aggregated + mathematically engineered datasets from SAVED DATA...')
        DATASETS = {'AGGREGATED_NOENG':
                    _accessories.retrieve_local_data_file('Data/S3 Files/combined_ipc_aggregates_ALL.csv'),
                    'AGGREGATED_ENG':
                    _accessories.retrieve_local_data_file('Data/S4 Files/combined_ipc_engineered_math.csv')}

    # results = []
    # for date in dates:
    #     try:
    #         solution = parallel_optimize(date)
    #     except Exception:
    #         continue
    #     results.append(solution)
    # aggregate_results = pd.concat(results)

    # TODO: Model selection
    # TODO: Load acceptable ranges

    _accessories._print('Performing backtesting...')
    dates = pd.date_range(*(start_date, end_date)).strftime('%Y-%m-%d')
    if engineered:
        field_df = DATASETS['AGGREGATED_ENG'].copy()
    else:
        # field_df['chloride_contrib'] = 0.5
        field_df = DATASETS['AGGREGATED_NOENG'].copy()

    if 'PRO_Pad' not in list(field_df):
        relations = _accessories.retrieve_local_data_file('Data/Pickles/PRODUCTION_[Well, Pad].pkl', mode=2)
        field_df['PRO_Pad'] = field_df['PRO_Well'].apply(lambda x: relations.get(x))

    macro_solution, chloride_solution = parallel_optimize(field_df.copy(), dates[0])
    aggregate_results = configure_aggregates(macro_solution, rell)

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
