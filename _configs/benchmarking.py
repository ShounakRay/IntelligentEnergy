# @Author: Shounak Ray <Ray>
# @Date:   26-Apr-2021 10:04:89:899  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: test.py
# @Last modified by:   Ray
# @Last modified time: 15-May-2021 13:05:81:815  GMT-0600
# @License: [Private IP]


import glob
import os
import subprocess
import sys
from io import StringIO
from random import shuffle
from typing import Final

import h2o
import matplotlib.pyplot as plt
import numpy as np
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


_ = """
#######################################################################################################################
##############################################   FUNCTION DEFINITIONS   ###############################################
#######################################################################################################################
"""


def setup_and_server(SECURED=SECURED, IP_LINK=IP_LINK, PORT=PORT, SERVER_FORCE=SERVER_FORCE):
    # Initialize the cluster
    h2o.init(https=SECURED,
             ip=IP_LINK,
             port=PORT,
             start_h2o=SERVER_FORCE)


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
            data['path'] = ''.join([t + '/Models/' for t in path.split('/MODELS_')[:-1]]) + data['Name']
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
    plt.legend(title='Configs', bbox_to_anchor=(0.5, -0.19),
               loc='lower center', fontsize='medium', ncol=5,
               fancybox=True, shadow=True, facecolor='white')


def get_best_models(benchmarks_combined, grouper='Group', sort_by=['Rel_Val_RMSE', 'Rel_RMSE'], top=3):
    top_file_paths = {}
    top_dfs = {}
    for name, group_df in benchmarks_combined.groupby(grouper):
        best = group_df.sort_values(sort_by, ascending=True)[:top]
        top_dfs[name] = best.reset_index(drop=True).drop(['Group', 'Run_Tag', 'Run_Time'], axis=1)
        best = list(best['path'].values)
        top_file_paths[name] = best

    return top_file_paths, top_dfs


def macro_performance(benchmarks, consideration=10):
    best_possibles = benchmarks.groupby(['Group',
                                         'Tolerated RMSE']
                                        )['Rel_Val_RMSE'].nsmallest(consideration).reset_index().drop('level_2', 1)
    best_possibles = best_possibles.groupby(['Group', 'Tolerated RMSE'])['Rel_Val_RMSE'].mean().reset_index()
    best_possibles['Best RMSE Proportion'] = (best_possibles['Tolerated RMSE'] + best_possibles['Rel_Val_RMSE']
                                              ) / (best_possibles['Tolerated RMSE'] * 0.1)

    return best_possibles


# def create_validation_splits(DATA_PATH_PAD, pd_data_pad, group_colname='PRO_Pad', TRAIN_VAL_SPLIT=0.95):
#     # NOTE: Global Dependencies:
#
#     # Split into train/test (CV) and holdout set (per each class of grouping)
#     # pd_data_pad = pd.read_csv(DATA_PATH_PAD_vanilla).drop('Unnamed: 0', axis=1)
#     unique_pads = list(pd_data_pad[group_colname].unique())
#     grouped_data_split = {}
#     for u_pad in unique_pads:
#         filtered_by_group = pd_data_pad[pd_data_pad[group_colname] == u_pad].sort_values('Date').reset_index(drop=True)
#         data_pad_loop, data_pad_validation_loop = [dat.reset_index(drop=True).infer_objects()
#                                                    for dat in np.split(filtered_by_group,
#                                                                        [int(TRAIN_VAL_SPLIT *
#                                                                             len(filtered_by_group))])]
#         grouped_data_split[u_pad] = (data_pad_loop, data_pad_validation_loop)
#
#     # Holdout and validation reformatting
#     with _accessories.suppress_stdout():
#         data_pad = pd.concat([v[0] for k, v in grouped_data_split.items()]).reset_index(drop=True).infer_objects()
#         wanted_types = {k: 'real' if v == float or v == int else 'enum' for k, v in dict(data_pad.dtypes).items()}
#         data_pad = h2o.H2OFrame(data_pad, column_types=wanted_types)
#         data_pad_validation = pd.concat([v[1] for k, v in grouped_data_split.items()]
#                                         ).reset_index(drop=True).infer_objects()
#         data_pad_validation = h2o.H2OFrame(data_pad_validation, column_types=wanted_types)
#
#         # Create pad validation relationships
#         pad_relationship_validation = {}
#         for u_pad in unique_pads:
#             df_loop = data_pad_validation.as_data_frame()
#             df_loop = df_loop[df_loop[group_colname] == u_pad].drop(
#                 ['Date'], axis=1).infer_objects().reset_index(drop=True)
#             local_wanted_types = {k: v for k, v in wanted_types.items() if k in df_loop.columns}
#             df_loop = h2o.H2OFrame(df_loop, column_types=local_wanted_types)
#             pad_relationship_validation[u_pad] = df_loop
#         # Create pad training relationships
#         pad_relationship_training = {}
#         for u_pad in unique_pads:
#             df_loop = data_pad.as_data_frame()
#             df_loop = df_loop[df_loop[group_colname] == u_pad].drop(
#                 [group_colname], axis=1).infer_objects().reset_index(drop=True)
#             local_wanted_types = {k: v for k, v in wanted_types.items() if k in df_loop.columns}
#             df_loop = h2o.H2OFrame(df_loop, column_types=local_wanted_types)
#             pad_relationship_training[u_pad] = df_loop
#
#     _accessories._print('STATUS: Server initialized and data imported.')
#
#     return data_pad, pad_relationship_validation, pad_relationship_training


_ = """
#######################################################################################################################
#########################################   TEMPORAL + CONFIG BENCHMARKING   ##########################################
#######################################################################################################################
"""
temporal, performance, benchmarks = get_benchmarks(time_path='Data/S5 Files/modeling_benchmarks.txt',
                                                   perf_path="Modeling Reference Files/*/*csv")

macro_best = macro_performance(benchmarks, consideration=5)
# macro_best.at[0, 'Rel_Val_RMSE'] = 5.1251363
# macro_best.at[0, 'Best RMSE Proportion'] = (macro_best.at[0, 'Tolerated RMSE'] + macro_best.at[0, 'Rel_Val_RMSE'])/(159.394495 * 0.1)
#
# macro_best.at[3, 'Rel_Val_RMSE'] = -10.692769
# macro_best.at[3, 'Best RMSE Proportion'] = (macro_best.at[3, 'Tolerated RMSE'] + macro_best.at[3, 'Rel_Val_RMSE'])/(151.573186 * 0.1)

sparse_df = benchmarks[benchmarks['Rel_Val_RMSE'] <= 80].groupby(['Group',
                                                                  'Math_Eng',
                                                                  'Weighted',
                                                                  'Duration']
                                                                 )['Rel_Val_RMSE'].nsmallest(3).reset_index()
see_performance(sparse_df, first_two=['Math_Eng', 'Weighted'],
                x='Duration', y='Rel_Val_RMSE',
                groupby_option='Group', kind='line', FIGSIZE=(21, 11))

best, details = get_best_models(benchmarks, sort_by=['Rel_Val_RMSE', 'Rel_RMSE'])

_accessories.save_local_data_file(best, 'Data/Model Candidates/best_models.pkl')
_accessories.save_local_data_file(details, 'Data/Model Candidates/best_models_details.pkl')

_ = """
#######################################################################################################################
##########################################   GET RE-CALCULATED VALIDATION   ##########################################
#######################################################################################################################
"""
source_data = _accessories.retrieve_local_data_file('Data/S3 Files/combined_ipc_aggregates_ALL.csv').infer_objects()
# original, validation, training = create_validation_splits('', source_data)
# source_data = source_data[source_data['PRO_Pad'] == 'A'].sort_values('Date').reset_index(drop=True)

validation_set = source_data[(source_data['Date'] > '2020-09-07') &
                             (source_data['Date'] < '2021-01-20')].reset_index(drop=True)

setup_and_server()
for category, model_paths in best.items():
    local_data = source_data[source_data['PRO_Pad'] == category].select_dtypes(float)
    wanted_types = {k: 'real' if v == float or v == int else 'enum' for k, v in dict(local_data.dtypes).items()}
    tolerable_rmse = details[category]['Tolerated RMSE'].drop_duplicates()[0]
    with _accessories.suppress_stdout():
        local_data = h2o.H2OFrame(local_data, column_types=wanted_types)
    for model_path in model_paths:
        with _accessories.suppress_stdout():
            model = h2o.load_model(model_path)
            predictions = model.predict(local_data).as_data_frame().infer_objects()
        val_rmse = np.sqrt(np.mean((local_data.as_data_frame()['PRO_Total_Fluid'] - predictions['predict'])**2))
        rel_val_rmse = val_rmse - tolerable_rmse

        # TRAINING

        print(f'For category {category}, path {model_path}, REL_RMSE: {model.rmse(rel_val_rmse)}')


_ = """
#######################################################################################################################
##########################################   VISUALIZING BACKTEST RESULTS   ###########################################
#######################################################################################################################
"""

# INTERPRET AGGREGATES
data = pd.read_csv('Optimization Reference Files/Backtests/Aggregates_2015-04-01_2020-12-20.csv')
data['accuracy'] = 1 - data['accuracy']

orig_data = _accessories.retrieve_local_data_file('Data/combined_ipc_aggregates_ALL.csv')
orig_data['Steam_Original'] = orig_data['Steam']
orig_data.drop('Steam', axis=1, inplace=True)

# rell = {'A': 159.394495, 'B': 132.758275, 'C': 154.587740, 'E': 151.573186, 'F': 103.389248}
final_backtest_data = pd.merge(data, orig_data, how='inner', on=['PRO_Pad', 'Date']).infer_objects()
final_backtest_data['Tolerated_RMSE'] = final_backtest_data['rmse'] - final_backtest_data['rel_rmse']
final_backtest_data.columns = ['PRO_Pad',
                               'Reccomended_Steam',
                               'Prediced_Total_Fluid',
                               'exchange_rate',
                               'RMSE',
                               'Accuracy',
                               'Algorithm',
                               'Date',
                               'Relative_RMSE',
                               'PRO_Adj_Pump_Speed',
                               'PRO_Casing_Pressure',
                               'PRO_Heel_Pressure',
                               'PRO_Toe_Pressure',
                               'PRO_Heel_Temp',
                               'PRO_Toe_Temp',
                               'PRO_Adj_Alloc_Oil',
                               'PRO_Alloc_Oil',
                               'PRO_Total_Fluid',
                               'PRO_Alloc_Steam',
                               'PRO_Water_cut',
                               'Bin_1',
                               'Bin_2',
                               'Bin_3',
                               'Bin_4',
                               'Bin_5',
                               'Bin_6',
                               'Bin_7',
                               'Bin_8',
                               'weight',
                               'Steam_Original',
                               'Tolerated_RMSE']

_accessories.save_local_data_file(final_backtest_data.infer_objects(),
                                  'Optimization Reference Files/Backtests/Final_Some.csv')

final_backtest_data = final_backtest_data[final_backtest_data['PRO_Pad']
                                          == 'F'].reset_index(drop=True).sort_values('Date')
final_backtest_data = final_backtest_data[final_backtest_data['Date'] > '2012-12-30'].reset_index(drop=True)
final_backtest_data = final_backtest_data[final_backtest_data['Accuracy'] > 0]

_ = plt.figure(figsize=(20, 12))
_ = plt.title('PAD B Optimization Reccomendation')
_ = plt.xlabel('Days since 2019-12-30')
_ = plt.ylabel('Volume and Accuracy (Dual)')
_ = final_backtest_data['Reccomended_Steam'].plot(label="Reccomended Steam", legend=True)
_ = final_backtest_data['Steam_Original'].plot(label="Actual Steam", legend=True)
# _ = final_backtest_data['Total_Fluid'].plot(label="Predicted Total Fluid", legend=True)
# final_backtest_data['accuracy'].plot(secondary_y=True, label="Accuracy", legend=True)
_ = final_backtest_data['Relative_RMSE'].plot(secondary_y=True, label="Relative RMSE", legend=True)

# EOF

# EOF
