# @Author: Shounak Ray <Ray>
# @Date:   26-Apr-2021 10:04:89:899  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: test.py
# @Last modified by:   Ray
# @Last modified time: 27-Apr-2021 14:04:45:458  GMT-0600
# @License: [Private IP]


import glob
import os
import pickle
import subprocess
import sys
from io import StringIO

import _pickle as cPickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

_ = """
#######################################################################################################################
##############################################   FUNCTION DEFINITIONS   ###############################################
#######################################################################################################################
"""


def see_performance(benchmarks_combined, groupby_option, kind='kde'):
    fig, ax = plt.subplots(figsize=(12, 9))
    for group_conditions, group in benchmarks_combined.groupby(['Math_Eng', 'Weighted', groupby_option]):
        math_eng, weighted, grouper = group_conditions
        label = f'Eng: {math_eng}, Wgt: {weighted}, Other: {grouper}'
        sub_df = group
        sub_df.plot(x='Run_Time', y='Rel_Val_RMSE', ax=ax, kind=kind, label=label, stacked=True)
    _ = plt.title(f'Relative RMSE vs. Run Time per Modeling Configuration (for all {groupby_option}s)')
    _ = plt.legend(loc='upper left')


def get_best_models(benchmarks_combined, grouper='Group', top=3):
    for name, group_df in benchmarks_combined.groupby(grouper):
        best = group_df.sort_values(['Rel_Val_RMSE', 'RMSE'], ascending=True)[:top]
        best = list(best['path'].values)
        yield (name, best)


_ = """
#######################################################################################################################
##########################################   VISUALIZING BACKTEST RESULTS   ###########################################
#######################################################################################################################
"""

# INTERPRET AGGREGATES
data = pd.read_csv('Optimization Reference Files/Backtests/Aggregates_2015-04-01_2020-12-20.csv')
data = data[data['PRO_Pad'] == 'B'].reset_index(drop=True).sort_values('Date')
data['accuracy'] = 1 - data['accuracy']
data = data[data['Date'] > '2015-12-30'].reset_index(drop=True)

_ = plt.figure(figsize=(20, 12))
_ = plt.title('PAD B Optimization Reccomendation')
_ = plt.xlabel('Days since 2019-12-30')
_ = plt.ylabel('Volume and Accuracy (Dual)')
_ = data['Steam'].plot(label="Reccomended Steam", legend=True)
_ = data['Total_Fluid'].plot(label="Predicted Total Fluid", legend=True)
# data['accuracy'].plot(secondary_y=True, label="Accuracy", legend=True)
_ = data['rel_rmse'].plot(secondary_y=True, label="Relative RMSE", legend=True)


_ = """
#######################################################################################################################
#########################################   TEMPORAL + CONFIG BENCHMARKING   ##########################################
#######################################################################################################################
"""

with open('_configs/modeling_benchmarks.txt') as file:
    lines = file.readlines()[8:]
lines_obj = StringIO(''.join(lines))
temporal_benchmarks = pd.read_csv(lines_obj, sep=",").infer_objects().dropna()
temporal_benchmarks.columns = ['Math_Eng', 'Weighted', 'Run_Time', 'Duration', 'Run_Tag', 'Save_Time']

# # #

# all_pickles = glob.glob("Modeling Reference Files/*/*/*pkl")
all_csvs = glob.glob("Modeling Reference Files/*/*csv")
all_perf_files = []
for path in all_csvs:
    all_perf_files.append([path.split('/MODELS_')[-1].split('.pkl')[0], path])

data_storage = []
for run_tag, path in all_perf_files:
    if(path.endswith('.pkl')):
        with open(path, 'r') as file:
            lines = file.readlines()
            # lines = StringIO(''.join()
        data = pd.read_csv(StringIO(''.join(lines)), delim_whitespace=True).dropna(axis=1).reset_index(drop=True)
    elif(path.endswith('.csv')):
        data = pd.read_csv(path, sep=',').reset_index(drop=True)
    data.columns = ['Name', 'RMSE', 'Rel_RMSE', 'Val_RMSE', 'Rel_Val_RMSE', 'Group', 'Tolerated RMSE', 'Run_Tag']
    data['path'] = path
    data_storage.append(data.infer_objects())

aggregated_metrics = pd.concat(data_storage).reset_index(drop=True)
aggregated_metrics = aggregated_metrics[aggregated_metrics['Rel_Val_RMSE'] <= 100]
aggregated_metrics = aggregated_metrics.groupby(['Group', 'Run_Tag'],
                                                group_keys=False).apply(lambda x:
                                                                        x.sort_values(['Rel_Val_RMSE', 'RMSE'],
                                                                                      ascending=True))
aggregated_metrics['Model_Type'] = aggregated_metrics['Name'].str.split('_').str[0]

aggregated_metrics[aggregated_metrics['Group'] == 'A'].sort_values(['Rel_Val_RMSE', 'RMSE'], ascending=True)

benchmarks_combined = pd.merge(aggregated_metrics, temporal_benchmarks, 'inner', on='Run_Tag').infer_objects()
benchmarks_combined['Save_Time'] = pd.to_datetime(benchmarks_combined['Save_Time'])
# # benchmarks_combined['Run_Time'] = benchmarks_combined['Run_Time'].astype(float)
# benchmarks_combined[(benchmarks_combined['Math_Eng'] == True) &
#                     (benchmarks_combined['Weighted'] == True)].plot(x='Run_Time', y='Rel_Val_RMSE',
#                                                                     ax=ax,
#                                                                     kind='scatter', label='Eng + Weight',
#                                                                     c='blue')
# benchmarks_combined[(benchmarks_combined['Math_Eng'] == True) &
#                     (benchmarks_combined['Weighted'] == False)].plot(x='Run_Time', y='Rel_Val_RMSE',
#                                                                      ax=ax,
#                                                                      kind='scatter', label='Eng',
#                                                                      c='purple')
# benchmarks_combined[(benchmarks_combined['Math_Eng'] == False) &
#                     (benchmarks_combined['Weighted'] == True)].plot(x='Run_Time', y='Rel_Val_RMSE',
#                                                                     ax=ax,
#                                                                     kind='scatter', label='Weight',
#                                                                     c='red')
# benchmarks_combined[(benchmarks_combined['Math_Eng'] == False) &
#                     (benchmarks_combined['Weighted'] == False)].plot(x='Run_Time', y='Rel_Val_RMSE',
#                                                                      ax=ax,
#                                                                      kind='scatter', label='Naive',
#                                                                      c='green')
# _ = plt.legend(loc='upper left')

see_performance(benchmarks_combined, 'Group', kind='scatter')
_accessories._distinct_colors

get_best_models(benchmarks_combined)


# Associate run tag to model RMSE


model_info = pd.read_csv('Modeling Reference Files/5433 – ENG: False, WEIGHT: True, TIME: 60/MODELS_5433.csv')
model_info = model_info[model_info['group'] == 'B']
model_info = model_info.sort_values('Rel. Val. RMSE').reset_index(drop=True)
top_candidates = model_info[model_info['Rel. Val. RMSE'] <= 0].sort_values('Rel. RMSE')


fig, ax = plt.subplots(figsize=(8, 6))
temporal_benchmarks.groupby(['math_eng', 'weighted']).plot(x='run_time', y='duration', ax=ax)


# plt.plot(data[data['accuracy'] >= 0]['accuracy'])
