# @Author: Shounak Ray <Ray>
# @Date:   22-Apr-2021 10:04:63:638  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: S6_optimization.py
# @Last modified by:   Ray
# @Last modified time: 10-May-2021 09:05:79:794  GMT-0600
# @License: [Private IP]


import os
import pickle
import subprocess
import sys
import traceback
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
injector_wells = {
    "A": [],
    "B": [],
    "C": [],
    "D": [],
    "E": [],
    "F": []
}

BEST_MODEL_PATHS = _accessories.retrieve_local_data_file('Data/Model Candidates/best_models_nomatheng.pkl')

_ = """
#######################################################################################################################
##############################################   FUNCTION DEFINITIONS   ###############################################
#######################################################################################################################
"""


def data_prep(field_df, *args):
    # PREPARE DATA FRAME, CLEANING, FEATURE ENGINEERING, OTHERS...
    return field_df


def build_pad_model(pad_df, injector_wells, features, target):
    # PREDICT TARGET ON THE PAD LEVEL, GIVEN INJECTOR WELLS,
    return model, metrics


def create_scenarios(pad_df, date, features, steam_range):
    # GET LATEST OPERATING SCENARIOS
    op_condition = pad_df[pad_df['Date'] == date]
    scenario_df = pd.DataFrame([{"Steam": a} for a in range(steam_range['min'], steam_range['max'] + 5, 5)])

    # GET CURRENT CONDITIONS
    for f in features:
        if f not in list(scenario_df.columns):
            scenario_df[f] = op_condition.iloc[-1][f]

    return scenario_df


def generate_optimization_table(field_df, date, steam_range=steam_range,
                                grouper='PRO_Pad', target='PRO_Total_Fluid', time_col='Date',
                                forward_days=30):
    optimization_table = []

    def get_subset(df, date, group, grouper_name=grouper, tail_days=365):
        subset_df = df[df[grouper_name] == group].reset_index(drop=True)
        subset_df = subset_df[subset_df[time_col] <= date].tail(tail_days)
        features = list(subset_df.select_dtypes(float).columns)

        return subset_df.reset_index(drop=True), features

    def get_testdfs(df, group, features, grouper_name=grouper, time_col=time_col, forward_days=forward_days):
        test_df = df[(df[grouper] == group) & (df[time_col] > date)].head(forward_days)
        wanted_types = {k: 'real' if v == float or v == int else 'enum'
                        for k, v in dict(test_df[features].infer_objects().dtypes).items()}
        test_pred = model.predict(h2o.H2OFrame(test_df[features],
                                               column_types=wanted_types)).as_data_frame()['predict']
        test_actual = test_df[target]

        return test_pred.reset_index(drop=True), test_actual.reset_index(drop=True)

    def configure_scenario_locally(subset_df, date, model, g, features, steam_range=steam_range):
        scenario_df = create_scenarios(subset_df, date, features, steam_range[g])
        wanted_types = {k: 'real' if v == float or v == int else 'enum'
                        for k, v in dict(scenario_df[features].infer_objects().dtypes).items()}
        scenario_df['Total_Fluid'] = list(model.predict(h2o.H2OFrame(scenario_df[features],
                                                                     column_types=wanted_types)
                                                        ).as_data_frame()['predict'])
        scenario_df[grouper] = g

        scenario_df['rmse'] = mean_squared_error(test_actual, test_pred, squared=False)
        scenario_df['algorithm'] = model.model_id
        scenario_df['accuracy'] = 1 - abs(test_pred.values - test_actual.values) / test_actual

        return scenario_df

    for d_col in ['PRO_Adj_Pump_Speed', 'Bin_1', 'Bin_8', 'PRO_Adj_Alloc_Oil']:
        if d_col in field_df.columns:
            field_df.drop(d_col, axis=1, inplace=True)

    for g in field_df[grouper].unique():
        _accessories._print(f"CREATING SCENARIO TABLE FOR: {grouper} and {g}.")

        subset_df, features = get_subset(field_df, date, g)

        # file = open('Modeling Reference Files/6086 – ENG: True, WEIGHT: False, TIME: 20/MODELS_6086.pkl', 'rb')

        model = h2o.load_model(BEST_MODEL_PATHS.get(g)[0])

        with _accessories.suppress_stdout():
            test_pred, test_actual = get_testdfs(field_df, g, features)
            scenario_df = configure_scenario_locally(subset_df, date, model, g, features)

        optimization_table.append(scenario_df)

    _accessories._print(f"Finished optimization table for all groups on {date}")
    optimization_table = pd.concat(optimization_table).infer_objects()
    optimization_table = optimization_table.sort_values([grouper, 'Steam'], ascending=[True, False])
    optimization_table['exchange_rate'] = optimization_table['Steam'] / optimization_table['Total_Fluid']
    optimization_table = optimization_table[[grouper, 'Steam', 'Total_Fluid',
                                             'exchange_rate', 'rmse', 'accuracy', 'algorithm']].reset_index(drop=True)

    return optimization_table


def optimize(optimization_table, group, steam_avail):

    # OPTIMIZE BASED ON CONSTRAINTS
    solution = optimization_table.groupby([group]).first().reset_index()
    steam_usage = solution['Steam'].sum()
    total_output = solution['Total_Fluid'].sum()

    print("Initiating optimization... seed:", steam_usage, "target:", steam_avail, "total output:", total_output)

    while steam_usage > steam_avail:

        lowest_delta = solution['exchange_rate'].astype(float).idxmax()

        steam_cutoff = solution.loc[lowest_delta]['Steam']
        asset_cut = [solution.loc[lowest_delta][group]]
        to_drop = optimization_table[(optimization_table[group].isin(asset_cut)) &
                                     (optimization_table['Steam'] >= steam_cutoff)].index

        optimization_table = optimization_table.drop(to_drop)
        solution = optimization_table.groupby([group]).first().reset_index()  # OPTIMAL SOLUTION
        steam_usage = solution['Steam'].sum()

    return solution


def parallel_optimize(field_df, date, grouper='PRO_Pad', target='PRO_Total_Fluid', steam_col='Steam', time_col='Date'):
    # field_df = data.copy()
    # data = dates[0]
    _accessories._print('DATE: ' + date)
    day_df = field_df[field_df[time_col] == str(date)]
    steam_avail = int(day_df[steam_col].sum())
    try:
        optimization_table = generate_optimization_table(field_df, date, steam_range)
        solution = optimize(optimization_table, grouper, steam_avail)
        solution[time_col] = date
        return solution
    except Exception as e:
        _accessories._print(str(e))
        return
        # print(traceback.print_exc())
        # pass


def run(data, dates):
    # data = DATASETS['AGGREGATED_NOENG'].copy()
    print("ACTIVE CPU COUNT:", mp.cpu_count() - 1)

    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        # try:
        args = list(zip([data.copy()] * len(dates), dates))
        agg = pool.starmap(parallel_optimize, args)
        # except Exception as e:
        #     raise Exception(e)
        pool.close()
        pool.terminate()

    agg_final = pd.concat(agg)
    return agg_final


def well_setpoints(solution, well_constraints):

    # WELL LEVEL SOLUTION
    # 1. NAIVE SOLUTION (divide by number of wells)
    # 2. SOR-ECONOMICS BASED (STEAM SHOULD NOT BE MORE THAN MAX SOR x OIL PRODUCTION)
    # 3. CHLORIDE CONTROL (MIN STEAM SHOULD RESIST CHLORIDE BREAKTHROUGHS)
    # 4. CORRELATION BASED SOLUTION

    return


def max_sor_steam():
    return


def min_chloride_control_steam():
    return


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
    # Double checking...
    try:
        snapshot(h2o_instance.cluster)
        raise ValueError('ERROR: H2O cluster improperly closed!')
    except Exception:
        pass


def configure_aggregates(aggregate_results, rell):
    aggregate_results['rmse'] = aggregate_results['rmse'] / 2
    aggregate_results['rel_rmse'] = (aggregate_results['rmse']) - \
        aggregate_results['PRO_Pad'].apply(lambda x: rell.get(x))

    return aggregate_results


_ = """
#######################################################################################################################
##################################################   CORE EXECUTION  ##################################################
#######################################################################################################################
"""


def _OPTIMIZATION(start_date='2015-04-01', end_date='2020-12-20', engineered=True):
    rell = {'A': 159.394495, 'B': 132.758275, 'C': 154.587740, 'E': 151.573186, 'F': 103.389248}

    _accessories._print('Initializing H2O server to access model files...')
    setup_and_server()

    _accessories._print('Loading the most basic, aggregated + mathematically engineered datasets...')
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
    aggregate_results = run(DATASETS['AGGREGATED_NOENG'].copy(), dates)
    aggregate_results = configure_aggregates(aggregate_results, rell)

    _accessories.save_local_data_file(aggregate_results,
                                      f'Data/S6 Files/Aggregates_{start_date}_{end_date}.csv')

    _accessories._print('Shutting down H2O server...')
    shutdown_confirm(h2o)


if __name__ == '__main__':
    _OPTIMIZATION()


# merged_df.to_csv('btest.csv')


# # Plot
# _temp = aggregate_results.sort_values('Date')
# _temp = _temp[_temp['PRO_Pad'] == 'A']
# _temp = _temp[['Date', 'Steam', 'Pred_Steam']]
# plt.figure(figsize=(20, 10))
# plt.plot(_temp['Date'], _temp['Steam'])
# plt.plot(_temp['Date'], _temp['Pred_Steam'])


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
