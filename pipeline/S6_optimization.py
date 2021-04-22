# @Author: Shounak Ray <Ray>
# @Date:   22-Apr-2021 10:04:63:638  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: S6_optimization.py
# @Last modified by:   Ray
# @Last modified time: 22-Apr-2021 12:04:72:721  GMT-0600
# @License: [Private IP]


import pickle
import traceback

import _references._accessories as _accessories
import billiard as mp
import h2o
import pandas as pd
from matplotlib import pyplot as plt

# from sagd_ensemble import genetic_ensemble as sagd_ensemble


field_df = pd.read_csv("Data/ipc_data.csv")

list(field_df)
field_df = field_df.groupby(['date', 'pad']).agg({
    'prod_casing_pressure': 'mean',
    'prod_bhp_heel': 'mean',
    'prod_bhp_toe': 'mean',
    'oil': 'sum',
    'water': 'sum',
    'total_fluid': 'sum',
    'alloc_steam': 'sum',
    'spm': 'sum',
}).reset_index().dropna().sort_values(['date', 'pad'])

field_df['sor'] = field_df['alloc_steam'] / field_df['oil']

# for g in field_df['pad'].unique():
#     field_df[field_df['pad']==g].rolling(10).mean()['alloc_steam'].plot()
#     plt.show()
#     # field_df[field_df['pad']==g]['oil'].plot()
#     # field_df[field_df['pad']==g]['spm'].plot()
#     field_df[(field_df['pad']==g)&(field_df['sor']<=10)].rolling(10).mean()['sor'].plot()
#     plt.show()

_ = """
#######################################################################################################################
##################################################   MODELING  ##################################################
######################################################################################################################
"""

# STEAM MORE CORRELATED WITH TOTAL FLUID THAN PUMP
field_df[field_df['pad'] == 'A'][['alloc_steam', 'spm', 'total_fluid']].corr()


features = [
    'prod_casing_pressure',
    'prod_bhp_heel',
    'prod_bhp_toe',
    'alloc_steam',
    # 'spm'
]

target = 'total_fluid'


s = 0
pad_df = field_df[field_df['pad'] == 'A'].iloc[s:s + 500]

test_x = pad_df.tail(45)[features]
test_y = pad_df.tail(45)[target]

train_df = pad_df[pad_df.index.isin(test_x.index) == False].sample(frac=0.8)
train_x = train_df[features]
train_y = train_df[target]

models_outputs, metric_outputs, model = sagd_ensemble(train_x, train_y, test_x, test_y)

pad_df['pred'] = model.predict(pad_df[features])

pad_df[target].plot()
pad_df['pred'].plot()


_ = """
#######################################################################################################################
##################################################   OPTIMIZATION  ##################################################
######################################################################################################################
"""


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


# features = [
#              'prod_casing_pressure',
#              'prod_bhp_heel',
#              'prod_bhp_toe',
#              'alloc_steam',
#              # 'spm'
#         ]
#
# target = 'total_fluid'
# date = '2020-12-01'


h2o.init(https=False,
         port=54321,
         start_h2o=True)


def generate_optimization_table(field_df, date, features, target, group, steam_range):

    optimization_table = []

    for g in field_df[group].unique():
        print("CREATING SCENARIO TABLE FOR:", group, g)
        subset_df = field_df[field_df[group] == g]
        subset_df = subset_df[subset_df['Date'] <= date].tail(365)

        date = '2017-12-20'

        test_df = field_df[(field_df[group] == g) & (field_df['Date'] > date)].head(30)

        model_path = 'Modeling Reference Files/Round 5060/GBM_3_AutoML_20210422_120411'
        model = h2o.load_model(model_path)

        # # # #

        subset_df = subset_df[subset_df['PRO_Pad'] == g].reset_index(drop=True)

        features = list(subset_df.columns)

        scenario_df = create_scenarios(subset_df, date, features, steam_range[g])
        scenario_df['pred'] = list(model.predict(h2o.H2OFrame(scenario_df[features])).as_data_frame()['predict'])
        scenario_df[group] = g
        test_pred = model.predict(h2o.H2OFrame(test_df[features]))
        target = 'PRO_Total_Fluid'
        test_actual = test_df[target]
        print(test_pred)
        print(test_actual)
        scenario_df['rmse'] = (test_pred - test_actual)**2
        scenario_df['algorithm'] = 'your_algo'
        scenario_df['accuracy'] = 1 - abs(test_pred - test_actual) / test_actual

        # # # #

        # scenario_df.plot(x='alloc_steam', y='pred')
        optimization_table.append(scenario_df)

    optimization_table = pd.concat(optimization_table)
    optimization_table = optimization_table.sort_values([group, 'alloc_steam'], ascending=[True, False])
    optimization_table['exchange_rate'] = optimization_table['alloc_steam'] / optimization_table['pred']
    optimization_table = optimization_table[[group, 'alloc_steam', 'pred',
                                             'exchange_rate', 'rmse', 'accuracy', 'algorithm']].reset_index(drop=True)

    return optimization_table


group = 'pad'
dates = pd.date_range('2020-01-01', '2020-12-20').strftime('%Y-%m-%d')


parallel_optimize(dates[100])


field_df = pd.read_csv('Data/combined_ipc_aggregates.csv')


def parallel_optimize(date):
    day_df = field_df[field_df['Date'] == str(date)]
    steam_avail = int(day_df['Steam'].sum())
    optimization_table = generate_optimization_table(field_df, date, features, target, 'PRO_Pad', steam_range)
    solution = optimize(optimization_table, group, steam_avail)
    solution['date'] = date
    # outputs.append(solution)
    return solution


def run():
    print("ACTIVE CPU COUNT:", mp.cpu_count() - 1)
    pool = mp.Pool(processes=mp.cpu_count() - 1)

    try:
        aggregate_results = pool.map(parallel_optimize, dates)
    except Exception as e:
        print(traceback.print_exc())
        pool.close()
        # raise Exception(e)
    pool.terminate()

    aggregate_results = pd.concat(aggregate_results)
    return aggregate_results


aggregate_results = run()
# merged_df.to_csv('btest.csv')

aggregate_results = pd.read_csv('btest.csv').drop(['Unnamed: 0', 'index', 'Unnamed: 0.1'], axis=1)
# aggregate_results.to_csv('btest.csv')


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


def optimize(optimization_table, group, steam_avail):

    # OPTIMIZE BASED ON CONSTRAINTS
    solution = optimization_table.groupby([group]).first().reset_index()
    steam_usage = solution['alloc_steam'].sum()
    total_output = solution['pred'].sum()

    print("Initiating optimization... seed:", steam_usage, "target:", steam_avail, "total output:", total_output)

    while steam_usage > steam_avail:

        lowest_delta = solution['exchange_rate'].astype(float).idxmax()

        steam_cutoff = solution.loc[lowest_delta]['alloc_steam']
        asset_cut = [solution.loc[lowest_delta][group]]
        to_drop = optimization_table[(optimization_table[group].isin(asset_cut)) &
                                     (optimization_table['alloc_steam'] >= steam_cutoff)].index

        optimization_table = optimization_table.drop(to_drop)
        solution = optimization_table.groupby([group]).first().reset_index()  # OPTIMAL SOLUTION
        steam_usage = solution['alloc_steam'].sum()

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


#
