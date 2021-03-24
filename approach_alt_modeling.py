# @Author: Shounak Ray <Ray>
# @Date:   23-Mar-2021 12:03:13:138  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: approach_alt_modeling.py
# @Last modified by:   Ray
# @Last modified time: 24-Mar-2021 12:03:17:174  GMT-0600
# @License: [Private IP]

# @Author: Shounak Ray <Ray>
# @Date:   09-Mar-2021 11:03:94:947  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: IPC_Real_Modeling.py
# @Last modified by:   Ray
# @Last modified time: 24-Mar-2021 12:03:17:174  GMT-0600
# @License: [Private IP]

# HELPFUL NOTES:
# > https://github.com/h2oai/h2o-3/tree/master/h2o-docs/src/cheatsheets

import os
import pickle
import random
import subprocess
from collections import Counter
from itertools import chain
from typing import Final

import featuretools
import h2o
import matplotlib.pyplot as plt  # Req. dep. for h2o.estimators.random_forest.H2ORandomForestEstimator.varimp_plot()
import numpy as np
import pandas as pd  # Req. dep. for h2o.estimators.random_forest.H2ORandomForestEstimator.varimp()
import seaborn as sns
import util_traversal
from colorama import Fore, Style, init
from h2o.automl import H2OAutoML
from h2o.estimators import H2OTargetEncoderEstimator

init(convert=True)

_ = """
#######################################################################################################################
#########################################   VERIFY VERSIONS OF DEPENDENCIES   #########################################
#######################################################################################################################
"""

# Get the major java version in current environment
java_major_version = int(subprocess.check_output(['java', '-version'],
                                                 stderr=subprocess.STDOUT).decode().split('"')[1].split('.')[0])

# Check if environment's java version complies with H2O requirements
# http://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html#requirements
if not (java_major_version >= 8 and java_major_version <= 14):
    raise ValueError('STATUS: Java Version is not between 8 and 14 (inclusive).\n  \
                      h2o cluster will not be initialized.')

print("\x1b[32m" + f'STATUS: Dependency versions checked and confirmed.{Style.RESET_ALL}')

_ = """
#######################################################################################################################
#########################################   DEFINITIONS AND HYPERPARAMTERS   ##########################################
#######################################################################################################################
"""
# Aesthetic Console Output constants
OUT_BLOCK: Final = '»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»\n'

# H2O server constants
IP_LINK: Final = 'localhost'                                  # Initializing the server on the local host (temporary)
SECURED: Final = True if(IP_LINK != 'localhost') else False   # https protocol doesn't work locally, should be True
PORT: Final = 54321                                           # Always specify the port that the server should use
SERVER_FORCE: Final = True                                    # Tries to init new server if existing connection fails

# Experiment > Model Training Constants and Hyperparameters
MAX_EXP_RUNTIME: Final = 1200                                 # The longest that the experiment will run (seconds)
RANDOM_SEED: Final = 2381125                                  # To ensure reproducibility of experiments
EVAL_METRIC: Final = 'auto'                                   # The evaluation metric to discontinue model training
RANK_METRIC: Final = 'auto'                                   # Leaderboard ranking metric after all trainings
DATA_PATH: Final = 'Data/combined_ipc_aggregates.csv'         # Where the client-specific data is located

# Feature Engineering Constants
FIRST_WELL_STM: Final = 'CI06'                                # Used to splice column list to segregate responders
FOLD_COLUMN: Final = "kfold_column"                           # Aesthetic: name for specified CV fold assignment column

RUN_TAG: Final = random.randint(0, 10000)
while os.path.isdir('Modeling Reference Files/Round {}'.format(RUN_TAG)):
    RUN_TAG: Final = random.randint(0, 10000)
os.makedirs('Modeling Reference Files/Round {}'.format(RUN_TAG))

print(Fore.GREEN + 'STATUS: Directory created for Round {}'.format(RUN_TAG) + Style.RESET_ALL)

# Print file structure for reference every time this program is run
util_traversal.print_tree_to_txt()


def conditional_drop(data_frame, tbd_list):
    for tb_dropped in tbd_list:
        if(tb_dropped in data_frame.columns):
            data_frame = data_frame.drop(tb_dropped, axis=1)
        else:
            print(Fore.GREEN + '> STATUS: {} not in frame, skipping.'.format(tb_dropped) + Style.RESET_ALL)
    return data_frame


def snapshot(cluster: h2o.backend.cluster.H2OCluster, show: bool = True) -> dict:
    """Provides a snapshot of the H2O cluster and different status/performance indicators.

    Parameters
    ----------
    cluster : h2o.backend.cluster.H2OCluster
        The h2o cluster where the server was initialized.
    show : bool
        Whether details should be printed to screen. Uses h2os built-in method.py

    Returns
    -------
    dict
        Information about the status/performance of the specified cluster.

    """

    h2o.cluster().show_status() if(show) else None

    return {'cluster_name': cluster.cloud_name,
            'pid': cluster.get_status_details()['pid'][0],
            'version': [int(i) for i in cluster.version.split('.')],
            'run_status': cluster.is_running(),
            'branch_name': cluster.branch_name,
            'uptime_ms': cluster.cloud_uptime_millis,
            'health': cluster.cloud_healthy}


def shutdown_confirm(cluster: h2o.backend.cluster.H2OCluster) -> None:
    """Terminates the provided H2O cluster.

    Parameters
    ----------
    cluster : h2o.backend.cluster.H2OCluster
        The H2O cluster where the server was initialized.

    Returns
    -------
    None
        Nothing. ValueError may be raised during processing and cluster metrics may be printed.

    """
    # SHUT DOWN the cluster after you're done working with it
    cluster.shutdown()
    # Double checking...
    try:
        snapshot(cluster)
        raise ValueError('ERROR: H2O cluster improperly closed!')
    except Exception:
        pass


def exp_cumulative_varimps(aml_obj, tag=None, tag_name=None):
    """Determines variable importances for all models in an experiment.

    Parameters
    ----------
    aml_obj : h2o.automl.autoh2o.H2OAutoML
        The H2O AutoML experiment configuration.
    tag: iterable (of str) OR None
        The value(s) for the desired tag per experiment iteration.
        Must be in iterable AND len(tag) == len(tag_name).
    tag_name: iterable (of str) OR None
        The value(s) for the desired identifier for the specifc tag. Must be in iterable.
        Must be in iterable AND len(tag) == len(tag_name).

    Returns
    -------
    pandas.core.frame.DataFrame, list (of tuples)
        A concatenated DataFrame with all the model's variable importances for all the input features.
        A list of the models that did not have a variable importance. Format: (name, model object).

    """

    # Minor data sanitation
    if(tag is not None and tag_name is not None):
        if(len(tag) != len(tag_name)):
            raise ValueError('ERROR: Length of specified tag and tag names are not equal.')
        else:
            pass
    else:
        if(tag == tag_name):    # If both the arguments are None
            pass
        else:   # If one of the arguments is None but the other isn't
            raise ValueError("ERROR: One of the arguments is None but the other is not.")

    cumulative_varimps = []
    model_novarimps = []
    exp_leaderboard = aml_obj.leaderboard
    # Retrieve all the model objects from the given experiment
    exp_models = [h2o.get_model(exp_leaderboard[m_num, "model_id"]) for m_num in range(exp_leaderboard.shape[0])]
    for model in exp_models:
        model_name = model.params['model_id']['actual']['name']
        variable_importance = model.varimp(use_pandas=True)

        # Only conduct variable importance dataset manipulation if ranking data is available (eg. unavailable, stacked)
        if(variable_importance is not None):
            variable_importance = pd.pivot_table(variable_importance,
                                                 values='scaled_importance',
                                                 columns='variable').reset_index(drop=True)
            variable_importance['model_name'] = model_name
            variable_importance['model_object'] = model
            if(tag is not None and tag_name is not None):
                for i in range(len(tag)):
                    variable_importance[tag_name[i]] = tag[i]
            variable_importance.columns.name = None

            cumulative_varimps.append(variable_importance)
        else:
            # print('> WARNING: Variable importances unavailable for {MDL}'.format(MDL=model_name))
            model_novarimps.append((model_name, model))

    print(Fore.GREEN +
          '> STATUS: Determined variable importances of all models in {} experiment.'.format(aml_obj.project_name) +
          Style.RESET_ALL)

    return pd.concat(cumulative_varimps).reset_index(drop=True)  # , model_novarimps


# Diverging: sns.diverging_palette(240, 10, n=9, as_cmap=True)
# https://seaborn.pydata.org/generated/seaborn.color_palette.html
def correlation_matrix(df, FPATH, EXP_NAME, abs_arg=True, mask=True, annot=False,
                       type_corrs=['Pearson', 'Kendall', 'Spearman'],
                       cmap=sns.color_palette('flare', as_cmap=True), figsize=(24, 8), contrast_factor=1.0):
    """Outputs to console and saves to file correlation matrices given data with input features.
    Intended to represent the parameter space and different relationships within. Tool for modeling.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Input dataframe where each column is a feature that is to be correlated.
    FPATH : str
        Where the file should be saved. If directory is provided, it should already be created.
    abs_arg : bool
        Whether the absolute value of the correlations should be plotted (for strength magnitude).
        Impacts cmap, switches to diverging instead of sequential.
    mask : bool
        Whether only the bottom left triange of the matrix should be rendered.
    annot : bool
        Whether each cell should be annotated with its numerical value.
    type_corrs : list
        All the different correlations that should be provided. Default to all built-in pandas options for df.corr().
    cmap : child of matplotlib.colors
        The color map which should be used for the heatmap. Dependent on abs_arg.
    figsize : tuple
        The size of the outputted/saved heatmap (width, height).
    contrast_factor:
        The factor/exponent by which all the correlationed should be raised to the power of.
        Purpose is for high-contrast, better identification of highly correlated features.

    Returns
    -------
    pandas.core.frame.DataFrame
        The correlations given the input data, and other transformation arguments (abs_arg, contrast_factor)
        Furthermore, heatmap is saved in specifid format and printed to console.

    """
    input_data = {}

    # NOTE: Conditional assignment of sns.heatmap based on parameters
    # > Mask sns.heatmap parameter is conditionally controlled
    # > If absolute value is not chosen by the user, switch to divergent heatmap. Otherwise keep it as sequential.
    fig, ax = plt.subplots(ncols=len(type_corrs), sharey=True, figsize=figsize)
    for typec in type_corrs:
        input_data[typec] = (df.corr(typec.lower()).abs()**contrast_factor if abs_arg
                             else df.corr(typec.lower())**contrast_factor)
        sns_fig = sns.heatmap(input_data[typec],
                              mask=np.triu(df.corr().abs()) if mask else None,
                              ax=ax[type_corrs.index(typec)],
                              annot=annot,
                              cmap=cmap if abs_arg else sns.diverging_palette(240, 10, n=9, as_cmap=True)
                              ).set_title("{cortype}'s Correlation Matrix\n{EXP_NAME}".format(cortype=typec,
                                                                                              EXP_NAME=EXP_NAME))
    plt.tight_layout()
    sns_fig.get_figure().savefig(FPATH, bbox_inches='tight')

    plt.clf()

    return input_data


print(Fore.GREEN + 'STATUS: Hyperparameters assigned and functions defined.' + Style.RESET_ALL)

_ = """
#######################################################################################################################
##########################################   INITIALIZE SERVER AND SETUP   ############################################
#######################################################################################################################
"""

# Initialize the cluster
h2o.init(https=SECURED,
         ip=IP_LINK,
         port=PORT,
         start_h2o=SERVER_FORCE)

# Check the status of the cluster, just for reference
process_log = snapshot(h2o.cluster(), show=False)

# Confirm that the data path leads to an actual file
if not (os.path.isfile(DATA_PATH)):
    raise ValueError('ERROR: {data} does not exist in the specificied location.'.format(data=DATA_PATH))

# Import the data from the file and exclude any obvious features
data = h2o.import_file(DATA_PATH)

print(Fore.GREEN + 'STATUS: Server initialized and data imported.\n\n' + Style.RESET_ALL)

_ = """
#######################################################################################################################
##########################################   MINOR DATA MANIPULATION/PREP   ###########################################
#######################################################################################################################
"""
# Table diagnostics
# data = conditional_drop(data, ['C1', 'PRO_Alloc_Water'])
# data = conditional_drop(data, ['C1', 'PRO_Alloc_Water', 'PRO_Pump_Speed'])
data = conditional_drop(data, ['C1', 'PRO_Alloc_Water', 'PRO_Pump_Speed', 'Bin_1', 'Bin_5'])


PRODUCTION_PADS = data.as_data_frame()['PRO_Pad'].unique()

# Categorical Encoding Warning
categorical_names = list(data.as_data_frame().select_dtypes(object).columns)
if(len(categorical_names) > 0):
    print(Fore.LIGHTRED_EX +
          '> WARNING: {encoded} will be encoded by H2O model unless processed out.'.format(encoded=categorical_names)
          + Style.RESET_ALL)

RESPONDER = 'PRO_Alloc_Oil'

# NOTE: The model predictors a should only use the target encoded versions, and not the older versions
PREDICTORS = [col for col in data.columns
              if col not in [FOLD_COLUMN] + [RESPONDER] + [col.replace('_te', '')
                                                           for col in data.columns if '_te' in col]]

print(Fore.GREEN + 'STATUS: Experiment hyperparameters and data configured.\n\n' + Style.RESET_ALL)

_ = """
#######################################################################################################################
#########################################   MODEL TRAINING AND DEVELOPMENT   ##########################################
#######################################################################################################################
"""

print(Fore.GREEN + 'STATUS: Hyperparameter Overview:')
print(Fore.GREEN + '\t* max_runtime_secs\t-> ', MAX_EXP_RUNTIME,
      Fore.GREEN + '\t\tThe maximum runtime in seconds that you want to allot in order to complete the model.')
print(Fore.GREEN + '\t* stopping_metric\t-> ', EVAL_METRIC,
      Fore.GREEN + '\t\tThis option specifies the metric to consider when early stopping is specified')
print(Fore.GREEN + '\t* sort_metric\t\t-> ', RANK_METRIC,
      Fore.GREEN + '\t\tThis option specifies the metric used to sort the Leaderboard by at the end of an AutoML run.')
print(Fore.GREEN + '\t* seed\t\t\t-> ', RANDOM_SEED,
      Fore.GREEN + '\t\tRandom seed for reproducibility. There are caveats.')
print(Fore.GREEN + '\t* Predictors\t\t-> ', PREDICTORS,
      Fore.GREEN + '\t\tThese are the variables which will be used to predict the responder.')
print(Fore.GREEN + '\t* Responder\t\t-> ', RESPONDER,
      Fore.GREEN + '\tThis is what is being predicted.\n' + Style.RESET_ALL)
if(input('Proceed with given hyperparameters? (Y/N)') == 'Y'):
    pass
else:
    raise RuntimeError('\n\nSession forcefully terminated by user during review of hyperparamaters.')

cumulative_varimps = {}
for propad in PRODUCTION_PADS:
    print(Fore.GREEN + 'STATUS: Experiment -> Production Pad {}\n'.format(propad) + Style.RESET_ALL)
    aml_obj = H2OAutoML(max_runtime_secs=MAX_EXP_RUNTIME,     # How long should the experiment run for?
                        stopping_metric=EVAL_METRIC,          # The evaluation metric to discontinue model training
                        sort_metric=RANK_METRIC,              # Leaderboard ranking metric after all trainings
                        seed=RANDOM_SEED,
                        project_name="IPC_MacroPadModeling_{RESP}_{PPAD}".format(RESP=RESPONDER,
                                                                                 PPAD=propad))

    MODEL_DATA = data[data['PRO_Pad'] == propad]
    MODEL_DATA = MODEL_DATA.drop('PRO_Pad', axis=1)
    aml_obj.train(y=RESPONDER,                                # A single responder
                  training_frame=MODEL_DATA)                  # All the data is used for training, cross-validation

    varimps = exp_cumulative_varimps(aml_obj,
                                     tag=[propad, RESPONDER],
                                     tag_name=['production_pad', 'responder'])
    cumulative_varimps[propad] = varimps

print(Fore.GREEN + 'STATUS: Completed experiments\n\n' + Style.RESET_ALL)

# Concatenate all the individual model variable importances into one dataframe
final_cumulative_varimps = pd.concat(cumulative_varimps.values()).reset_index(drop=True)
# Exclude any features encoded by default (H2O puts a `.` in the column name of these features)
# final_cumulative_varimps = final_cumulative_varimps.loc[:, ~final_cumulative_varimps.columns.str.contains('.',
#                                                                                                           regex=False)]
final_cumulative_varimps.index = final_cumulative_varimps['model_name'] + \
    '___' + final_cumulative_varimps['production_pad']

print(Fore.GREEN + 'STATUS: Completed detecting variable importances.\n\n' + Style.RESET_ALL)

# NOTE: Save outputs for reference (so you don't have to wait an hour every time)
# with open('Modeling Pickles/model_novarimps.pkl', 'wb') as f:
#     pickle.dump(model_novarimps, f)
# final_cumulative_varimps.to_pickle('Modeling Pickles/final_cumulative_varimps.pkl')
# final_cumulative_varimps.to_html('Modeling Reference Files/final_cumulative_varimps.html')

# NOTE: FOR REFERENCE
# Filtering of the variable importance summary
# FILT_final_cumulative_varimps = final_cumulative_varimps[~final_cumulative_varimps['model_name'
#                                                                                    ].str.contains('XGBoost')
#                                                          ].reset_index(drop=True).select_dtypes(float)
FILT_final_cumulative_varimps = final_cumulative_varimps[(final_cumulative_varimps.mean(axis=1) > 0.0) &
                                                         (final_cumulative_varimps.mean(axis=1) < 1.0)
                                                         ].select_dtypes(float)

# Plot heatmap of variable importances across all model combinations
fig, ax = plt.subplots(figsize=(10, 20))
predictor_rank = FILT_final_cumulative_varimps.mean(axis=0).sort_values(ascending=False)
sns_fig = sns.heatmap(FILT_final_cumulative_varimps[predictor_rank.keys()], annot=True, annot_kws={"size": 4})
sns_fig.get_figure().savefig('Modeling Reference Files/Round {tag}/macropad_varimps_{tag}.pdf'.format(tag=RUN_TAG),
                             bbox_inches='tight')

plt.clf()

correlation_matrix(FILT_final_cumulative_varimps, EXP_NAME='Aggregated Experiment Results',
                   FPATH='Modeling Reference Files/Round {tag}/select_var_corrs_{tag}.pdf'.format(tag=RUN_TAG))

print(Fore.GREEN + 'STATUS: Saved variable importance configurations.' + Style.RESET_ALL)

_ = """
#######################################################################################################################
#########################################   EVALUATE MODEL PERFORMANCE   ##############################################
#######################################################################################################################
"""

perf_data = {}
for model_name, model_obj in zip(final_cumulative_varimps.index, final_cumulative_varimps['model_object']):
    perf_data[model_name] = {}
    perf_data[model_name]['R^2'] = model_obj.r2()
    # perf_data[model_name]['R'] = model_obj.r2() ** 0.5
    perf_data[model_name]['MSE'] = model_obj.mse()
    perf_data[model_name]['RMSE'] = model_obj.rmse()
    perf_data[model_name]['RMSLE'] = model_obj.rmsle()
    perf_data[model_name]['MAE'] = model_obj.mae()

mcmaps = {'R^2': sns.color_palette('rocket_r', as_cmap=True),
          # 'R': sns.color_palette('rocket_r'),
          'MSE': sns.color_palette("coolwarm", as_cmap=True),
          'RMSE': sns.color_palette("coolwarm", as_cmap=True),
          'RMSLE': sns.color_palette("coolwarm", as_cmap=True),
          'MAE': sns.color_palette("coolwarm", as_cmap=True)}
centers = {'R^2': None,
           'MSE': 25,
           'RMSE': 5,
           'RMSLE': None,
           'MAE': None}

# Structure model output and order
perf_data = pd.DataFrame(perf_data).T.sort_values('RMSE', ascending=False).infer_objects()
# Ensure correct data type
for col in perf_data.columns:
    perf_data[col] = perf_data[col].astype(float)

fig, ax = plt.subplots(figsize=(10, 30), ncols=len(perf_data.columns), sharey=True)
for col in perf_data.columns:
    cmap_local = mcmaps.get(col)
    center_local = centers.get(col)
    vmax_local = np.percentile(perf_data[col], 95)
    sns_fig = sns.heatmap(perf_data[[col]], ax=ax[list(perf_data.columns).index(col)],
                          annot=True, annot_kws={"size": 4}, cbar=False,
                          center=center_local, cmap=cmap_local, vmax=vmax_local)
    sns.set(font_scale=0.6)

sns_fig.get_figure().savefig('Modeling Reference Files/Round {tag}/model_performance_{tag}.pdf'.format(tag=RUN_TAG),
                             bbox_inches='tight')

print(Fore.GREEN + 'STATUS: Saved variable importance configurations.' + Style.RESET_ALL)

_ = """
#######################################################################################################################
########################################   SHUTDOWN THE SESSION/CLUSTER   #############################################
#######################################################################################################################
"""
if(input('Shutdown Cluster? (Y/N)') == 'Y'):
    shutdown_confirm(h2o.cluster())


# EOF

# EOF

# EOF
