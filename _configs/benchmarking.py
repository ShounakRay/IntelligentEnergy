# @Author: Shounak Ray <Ray>
# @Date:   26-Apr-2021 10:04:89:899  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: test.py
# @Last modified by:   Ray
# @Last modified time: 28-Apr-2021 12:04:35:354  GMT-0600
# @License: [Private IP]


import glob
import os
import pickle
import subprocess
import sys
from io import StringIO
from random import shuffle

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


def get_benchmarks(time_path='_configs/modeling_benchmarks.txt', perf_path="Modeling Reference Files/*/*csv"):
    def get_time_benchmarks(path=time_path):
        # GET TIME BENCHMARKING FILES
        with open(path) as file:
            lines = file.readlines()[8:]
        lines_obj = StringIO(''.join(lines))
        temporal_benchmarks = pd.read_csv(lines_obj, sep=",").infer_objects().dropna()
        temporal_benchmarks.columns = ['Math_Eng', 'Weighted', 'Run_Time', 'Duration', 'Run_Tag', 'Save_Time']
        temporal_benchmarks['Save_Time'] = pd.to_datetime(temporal_benchmarks['Save_Time'])

        return temporal_benchmarks.infer_objects()

    def get_perf_benchmarks(path=perf_path):
        # GET PERFORMANCE BENCHMARKING FILES
        # all_pickles = glob.glob("Modeling Reference Files/*/*/*pkl")
        all_csvs = glob.glob(path)
        all_perf_files = []
        for path in all_csvs:
            all_perf_files.append([path.split('/MODELS_')[-1].split('.pkl')[0], path])
        data_storage = []
        for run_tag, path in all_perf_files:
            if(path.endswith('.pkl')):
                with open(path, 'r') as file:
                    lines = file.readlines()
                    # lines = StringIO(''.join()
                data = pd.read_csv(StringIO(''.join(lines)), delim_whitespace=True).dropna(
                    axis=1).reset_index(drop=True)
            elif(path.endswith('.csv')):
                data = pd.read_csv(path, sep=',').reset_index(drop=True)
            data.columns = ['Name', 'RMSE', 'Rel_RMSE', 'Val_RMSE',
                            'Rel_Val_RMSE', 'Group', 'Tolerated RMSE', 'Run_Tag']
            data['path'] = ''.join([t + '/' for t in path.split('/MODELS_')[:-1]]) + data['Name']
            data_storage.append(data.infer_objects())
        aggregated_metrics = pd.concat(data_storage).reset_index(drop=True).infer_objects()
        # EXCLUDE ANY ANOMALOUS, SUPER-HIGH RMSE VALUES
        # aggregated_metrics = aggregated_metrics[aggregated_metrics['Rel_Val_RMSE'] <= 100]
        aggregated_metrics['Model_Type'] = aggregated_metrics['Name'].str.split('_').str[0]

        return aggregated_metrics

    temporal = get_time_benchmarks()
    performance = get_perf_benchmarks()

    # MERGE TEMPORAL AND PERFORMANCE BENCHMARKS
    return temporal, performance, pd.merge(performance, temporal, 'inner', on='Run_Tag').infer_objects()


def see_performance(benchmarks_combined, groupby_option,
                    first_two=['Math_Eng', 'Weighted'],
                    x='Run_Time', y='Rel_Val_RMSE',
                    kind='kde', colors=_accessories._distinct_colors(), FIGSIZE=(12, 9)):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_facecolor('lightgray')
    shuffle(colors)
    groupby_obj = benchmarks_combined.groupby(first_two + [groupby_option])
    i = 0
    for group_conditions, group in groupby_obj:
        math_eng, weighted, grouper = group_conditions
        label = f'Eng: {math_eng}, Wgt: {weighted}, Other: {grouper}'
        sub_df = group
        color = colors[i]
        sub_df.plot(x=x, y=y, ax=ax, kind=kind, label=label, stacked=False, c=color, lw=3)
        i += 1
    _ = plt.title(f'Relative RMSE vs. Run Time per Modeling Configuration (for all {groupby_option}s)')
    plt.legend(title='Configs', bbox_to_anchor=(0.5, -0.16),
               loc='lower center', fontsize='small', ncol=5,
               fancybox=True, shadow=True, facecolor='white')


def get_best_models(benchmarks_combined, grouper='Group', sort_by=['Rel_Val_RMSE', 'Rel_RMSE'], top=3):
    top_df = {}
    for name, group_df in benchmarks_combined.groupby(grouper):
        best = group_df.sort_values(sort_by, ascending=True)[:top]
        best = list(best['path'].values)
        top_df[name] = best

    return top_df


def macro_performance(benchmarks, consideration=10):
    best_possibles = benchmarks.groupby(['Group',
                                         'Tolerated RMSE']
                                        )['Rel_Val_RMSE'].nsmallest(consideration).reset_index().drop('level_2', 1)
    best_possibles = best_possibles.groupby(['Group', 'Tolerated RMSE'])['Rel_Val_RMSE'].mean().reset_index()
    best_possibles['Best RMSE Proportion'] = (best_possibles['Tolerated RMSE'] + best_possibles['Rel_Val_RMSE']
                                              ) / (best_possibles['Tolerated RMSE'] * 0.1)

    return best_possibles


_ = """
#######################################################################################################################
#########################################   TEMPORAL + CONFIG BENCHMARKING   ##########################################
#######################################################################################################################
"""

temporal, performance, benchmarks = get_benchmarks(time_path='_configs/modeling_benchmarks.txt',
                                                   perf_path="Modeling Reference Files/*/*csv")


macro_best = macro_performance(benchmarks, consideration=5)


sparse_df = benchmarks[benchmarks['Rel_Val_RMSE'] <= 80].groupby(['Group',
                                                                  'Math_Eng',
                                                                  'Weighted',
                                                                  'Duration']
                                                                 )['Rel_Val_RMSE'].nsmallest(3).reset_index()
# sparse_df['Group'].unique()
see_performance(sparse_df, first_two=['Math_Eng', 'Weighted'],
                x='Duration', y='Rel_Val_RMSE',
                groupby_option='Group', kind='kde', FIGSIZE=(21, 11))

best = get_best_models(performance, sort_by=['Rel_Val_RMSE', 'Rel_RMSE'])
_accessories.save_local_data_file(best, 'Data/Model Candidates/best_models.pkl')


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

# EOF

# EOF
