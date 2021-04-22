# @Author: Shounak Ray <Ray>
# @Date:   22-Apr-2021 10:04:63:638  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: S6_optimization.py
# @Last modified by:   Ray
# @Last modified time: 22-Apr-2021 17:04:51:513  GMT-0600
# @License: [Private IP]


import pickle
import traceback
from typing import Final

import _references._accessories as _accessories
import billiard as mp
import h2o
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

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
        if f not in scenario_df.columns:
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

        return subset_df, features

    def get_testdf(df, group, grouper_name=grouper, time_col=time_col, forward_days=forward_days):
        test_df = field_df[(field_df[grouper] == g) & (field_df[time_col] > date)].head(forward_days)

    for g in field_df[grouper].unique():
        _accessories._print("CREATING SCENARIO TABLE FOR:", grouper, g)

        subset_df, features = get_subset(field_df, date, g)

        model_path = 'Modeling Reference Files/Round 5060/GBM_3_AutoML_20210422_120411'
        model = h2o.load_model(model_path)

        # # # #

        def

        scenario_df = create_scenarios(subset_df, date, features, steam_range[g])
        scenario_df['pred'] = list(model.predict(h2o.H2OFrame(scenario_df[features])).as_data_frame()['predict'])
        scenario_df[grouper] = g
        test_pred = model.predict(h2o.H2OFrame(test_df[features])).as_data_frame()['predict'].iloc[1:]

        test_actual = test_df[target].iloc[1:]
        scenario_df['rmse'] = mean_squared_error(test_actual, test_pred, squared=False)
        scenario_df['algorithm'] = 'BGM H2O-Model'
        scenario_df['accuracy'] = abs(test_pred.values - test_actual.values) / test_actual

        # # # #

        # scenario_df.plot(x='alloc_steam', y='pred')
        optimization_table.append(scenario_df)

    optimization_table = pd.concat(optimization_table)
    optimization_table = optimization_table.sort_values([grouper, 'Steam'], ascending=[True, False])
    optimization_table['exchange_rate'] = optimization_table['Steam'] / optimization_table['pred']
    optimization_table = optimization_table[[grouper, 'Steam', 'pred',
                                             'exchange_rate', 'rmse', 'accuracy', 'algorithm']].reset_index(drop=True)

    return optimization_table


def parallel_optimize(field_df, date, grouper='PRO_Pad', target='PRO_Total_Fluid', steam_col='Steam', time_col='Date'):
    _accessories._print('DATE: ' + date)
    day_df = field_df[field_df[time_col] == str(date)]
    steam_avail = int(day_df[steam_col].sum())
    try:
        optimization_table = generate_optimization_table(field_df, date, target, steam_range)
        solution = optimize(optimization_table, grouper, steam_avail)
        solution[time_col] = date
        return solution
    except Exception:
        pass


def run(dates):
    print("ACTIVE CPU COUNT:", mp.cpu_count() - 1)
    pool = mp.Pool(processes=mp.cpu_count() - 1)

    try:
        agg = pool.map(parallel_optimize, dates)
    except Exception:
        print(traceback.print_exc())
        pool.close()
        # raise Exception(e)
    pool.terminate()

    agg = pd.concat(agg)
    return agg


def optimize(optimization_table, group, steam_avail):

    # OPTIMIZE BASED ON CONSTRAINTS
    solution = optimization_table.groupby([group]).first().reset_index()
    steam_usage = solution['Steam'].sum()
    total_output = solution['pred'].sum()

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


_ = """
#######################################################################################################################
##################################################   CORE EXECUTION  ##################################################
#######################################################################################################################
"""


def _OPTIMIZATION():
    setup_and_server()

    DATASETS = {'AGGREGATED_NOENG': _accessories.retrieve_local_data_file('Data/combined_ipc_aggregates.csv')}

    dates = pd.date_range('2020-06-01', '2020-12-20').strftime('%Y-%m-%d')

    # results = []
    # for date in dates:
    #     try:
    #         solution = parallel_optimize(date)
    #     except Exception:
    #         continue
    #     results.append(solution)
    # aggregate_results = pd.concat(results)

    _accessories.auto_make_path('Optimization Reference Files/Backtests/')

    aggregate_results = run(dates)

    rell = {'A': 170, 'B': 131}
    aggregate_results['rmse'].describe()
    aggregate_results['rmse'] = aggregate_results['rmse'] / 2
    aggregate_results['rel_rmse'] = (aggregate_results['rmse']) - \
        aggregate_results['PRO_Pad'].apply(lambda x: rell.get(x))
    aggregate_results.head(50)
    aggregate_results.to_csv('AGGREGATES_PARTIAL_2019.csv')
# merged_df.to_csv('btest.csv')


# Plot
_temp = aggregate_results.sort_values('date')
_temp = _temp[_temp['PRO_Pad'] == 'A']
_temp = _temp[['date', 'Steam', 'pred']]
plt.figure(figsize=(20, 10))
plt.plot(_temp['date'], _temp['Steam'])
plt.plot(_temp['date'], _temp['pred'])


###
aggregate_results = aggregate_results.reset_index()
aggregate_results[aggregate_results['pad'] == 'A']['accuracy'].plot()

aggregate_results = aggregate_results.rename(columns={"alloc_steam": "recomm_steam"})

merged_df = pd.merge(aggregate_results, field_df, how='left', on=['date', 'pad'])
merged_df['date'] = pd.to_datetime(merged_df['date'])
merged_df = merged_df.set_index('date')
merged_df['pred_oil'] = merged_df['pred'] * (1 - merged_df['water'] / merged_df['total_fluid'])

merged_df[merged_df['pad'] == 'E']['alloc_steam'].plot(color='black', figsize=(11, 8))
merged_df[merged_df['pad'] == 'E']['recomm_steam'].plot(color='red', figsize=(11, 8))

merged_df[merged_df['pad'] == 'A']['total_fluid'].plot(color='black', figsize=(11, 8))
merged_df[merged_df['pad'] == 'A']['pred'].plot(color='brown', figsize=(11, 8))


#
