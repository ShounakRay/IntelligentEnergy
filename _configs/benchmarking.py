# @Author: Shounak Ray <Ray>
# @Date:   26-Apr-2021 10:04:89:899  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: test.py
# @Last modified by:   Ray
# @Last modified time: 26-Apr-2021 16:04:85:855  GMT-0600
# @License: [Private IP]


import glob
import os
import pickle
from io import StringIO

import _pickle as cPickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

plt.figure(figsize=(20, 12))
plt.title('PAD B Optimization Reccomendation')
plt.xlabel('Days since 2019-12-30')
plt.ylabel('Volume and Accuracy (Dual)')
data['Steam'].plot(label="Reccomended Steam", legend=True)
data['Total_Fluid'].plot(label="Predicted Total Fluid", legend=True)
# data['accuracy'].plot(secondary_y=True, label="Accuracy", legend=True)
data['rel_rmse'].plot(secondary_y=True, label="Relative RMSE", legend=True)


_ = """
#######################################################################################################################
#########################################   TEMPORAL + CONFIG BENCHMARKING   ##########################################
#######################################################################################################################
"""

with open('_configs/modeling_benchmarks.txt') as file:
    lines = file.readlines()[8:]
lines_obj = StringIO(''.join(lines))
temporal_benchmarks = pd.read_csv(lines_obj, sep=",").infer_objects()
temporal_benchmarks.columns = ['Math_Eng', 'Weighted', 'Run_Time', 'Duration', 'Run_Tag']

# # #

all_pickles = glob.glob("Modeling Reference Files/*/*pkl")
all_perf_files = []
for path in all_pickles:
    all_perf_files.append([path.split('/MODELS_')[-1].split('.pkl')[0], path])

data_storage = []
for run_tag, path in all_perf_files:
    with open(path, 'r') as file:
        lines = file.readlines()
        # lines = StringIO(''.join()
    data = pd.read_csv(StringIO(''.join(lines)), delim_whitespace=True).dropna(axis=1).reset_index(drop=True)
    data.columns = ['Name', 'RMSE', 'Rel_RMSE', 'Val_RMSE', 'Rel_Val_RMSE', 'Group', 'Tolerated RMSE', 'Run_Tag']
    data_storage.append(data.infer_objects())

aggregated_metrics = pd.concat(data_storage).reset_index(drop=True)
aggregated_metrics = aggregated_metrics[aggregated_metrics['Rel_Val_RMSE'] <= 0]
aggregated_metrics = aggregated_metrics.groupby(['Group', 'Run_Tag'],
                                                group_keys=False).apply(lambda x:
                                                                        x.sort_values(['Rel_Val_RMSE', 'RMSE'],
                                                                                      ascending=True))

# sns.heatmap(aggregated_metrics.select_dtypes(float)[aggregated_metrics.select_dtypes(float) < 1000])
# aggregated_metrics[aggregated_metrics['Group'] == 'B'].head(50)
# aggregated_metrics.drop_duplicates(subset=['Name', 'Group'])

benchmarks_combined = pd.merge(aggregated_metrics, temporal_benchmarks, 'inner', on='Run_Tag')
fig, ax = plt.subplots(figsize=(8, 6))
benchmarks_combined[(benchmarks_combined['Math_Eng'] == True) &
                    (benchmarks_combined['Weighted'] == True)].plot(x='Run_Time', y='Rel_Val_RMSE',
                                                                    ax=ax,
                                                                    kind='scatter', label='Eng + Weight',
                                                                    c='blue')
benchmarks_combined[(benchmarks_combined['Math_Eng'] == True) &
                    (benchmarks_combined['Weighted'] == False)].plot(x='Run_Time', y='Rel_Val_RMSE',
                                                                     ax=ax,
                                                                     kind='scatter', label='Eng',
                                                                     c='purple')
benchmarks_combined[(benchmarks_combined['Math_Eng'] == False) &
                    (benchmarks_combined['Weighted'] == True)].plot(x='Run_Time', y='Rel_Val_RMSE',
                                                                    ax=ax,
                                                                    kind='scatter', label='Weight',
                                                                    c='red')
benchmarks_combined[(benchmarks_combined['Math_Eng'] == False) &
                    (benchmarks_combined['Weighted'] == False)].plot(x='Run_Time', y='Rel_Val_RMSE',
                                                                     ax=ax,
                                                                     kind='scatter', label='Naive',
                                                                     c='green')
benchmarks_combined['Run_Time'] = benchmarks_combined['Run_Time'].astype(float)

# Associate run tag to model RMSE


model_info = pd.read_csv('Modeling Reference Files/5433 – ENG: False, WEIGHT: True, TIME: 60/MODELS_5433.csv')
model_info = model_info[model_info['group'] == 'B']
model_info = model_info.sort_values('Rel. Val. RMSE').reset_index(drop=True)
top_candidates = model_info[model_info['Rel. Val. RMSE'] <= 0].sort_values('Rel. RMSE')


fig, ax = plt.subplots(figsize=(8, 6))
temporal_benchmarks.groupby(['math_eng', 'weighted']).plot(x='run_time', y='duration', ax=ax)


# plt.plot(data[data['accuracy'] >= 0]['accuracy'])
