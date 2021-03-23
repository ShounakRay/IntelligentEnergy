# @Author: Shounak Ray <Ray>
# @Date:   23-Mar-2021 12:03:13:138  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: approach_alt_modeling.py
# @Last modified by:   Ray
# @Last modified time: 23-Mar-2021 13:03:09:094  GMT-0600
# @License: [Private IP]

# @Author: Shounak Ray <Ray>
# @Date:   09-Mar-2021 11:03:94:947  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: IPC_Real_Modeling.py
# @Last modified by:   Ray
# @Last modified time: 23-Mar-2021 13:03:09:094  GMT-0600
# @License: [Private IP]

# HELPFUL NOTES:
# > https://github.com/h2oai/h2o-3/tree/master/h2o-docs/src/cheatsheets

import os
import pickle
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
from h2o.automl import H2OAutoML
from h2o.estimators import H2OTargetEncoderEstimator

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

print('STATUS: Dependency versions checked and confirmed.')

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
MAX_EXP_RUNTIME: Final = int(0.25 * 60)                       # The longest that the experiment will run (seconds)
RANDOM_SEED: Final = 2381125                                  # To ensure reproducibility of experiments
EVAL_METRIC: Final = 'auto'                                   # The evaluation metric to discontinue model training
RANK_METRIC: Final = 'auto'                                   # Leaderboard ranking metric after all trainings
DATA_PATH: Final = 'Data/combined_ipc.csv'                    # Where the client-specific data is located

# Feature Engineering Constants
FIRST_WELL_STM: Final = 'CI06'                                # Used to splice column list to segregate responders
FOLD_COLUMN: Final = "kfold_column"                           # Aesthetic: name for specified CV fold assignment column

# Print file structure for reference every time this program is run
util_traversal.print_tree_to_txt()


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
        # print('\n{block}> STATUS: Model {MDL}'.format(block=OUT_BLOCK,
        #                                               MDL=model_name))
        # Determine and store variable importances (for specific responder-model combination)
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

    print('> STATUS: Determined variable importances of all models in {} experiment.'.format(aml_obj.project_name))

    return pd.concat(cumulative_varimps).reset_index(drop=True), model_novarimps


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


print('STATUS: Hyperparameters assigned and functions defined.')

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

print('STATUS: Server initialized and data imported.')

_ = """
#######################################################################################################################
#########################################   MODEL TRAINING AND DEVELOPMENT   ##########################################
#######################################################################################################################
"""
# TODO: Determine best proxy-predictive model for EACH production well

# Each of the features individually predicted
# NOTE: The model predictors a should only use the target encoded versions, and not the older versions
PREDICTORS = [col for col in data.columns
              if col not in [FOLD_COLUMN] + [col.replace('_te', '') for col in data.columns if '_te' in col]]

categorical_names = list(data.as_data_frame().select_dtypes(object).columns)
if(len(categorical_names) > 0):
    print('WARNING: {encoded} will be encoded by H2O model.'.format(encoded=categorical_names))

# Each of the responders the features with the highest correlations to PRODUCTION_TARGET
# NOTE: The reason this is automated and not hard-coded is for a general solution, when the lack of domain-specific
#       engineering knowledge is unavailable
RESPONDERS = SENSOR_RANKING[:3].keys()
# For every predictor feature, run an experiment
cumulative_varimps = {}
model_novarimps = []
for responder in RESPONDERS:
    print('\n{block}{block}STATUS: Responder {RESP}'.format(block=OUT_BLOCK,
                                                            RESP=responder))
    cumulative_varimps[responder] = {}
    for prod_well in PRODUCTION_WELLS:
        # Configure experiment
        aml_obj = H2OAutoML(max_runtime_secs=MAX_EXP_RUNTIME,     # How long should the experiment run for?
                            stopping_metric=EVAL_METRIC,          # The evaluation metric to discontinue model training
                            sort_metric=RANK_METRIC,              # Leaderboard ranking metric after all trainings
                            seed=RANDOM_SEED,
                            project_name="IPC_MacroModeling_{RESP}_{PWELL}".format(RESP=responder,
                                                                                   PWELL=prod_well))

        TEST_DATA = data[data['Well'] == prod_well]
        # Run the experiment (for the specified responder)
        # NOTE: Fold column specified for cross validation to mitigate leakage (absent if no encoding was performed)
        # https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/_modules/h2o/automl/autoh2o.html
        aml_obj.train(x=PREDICTORS,                               # All the depedent variables in each model
                      y=responder,                                # A single responder
                      # fold_column=FOLD_COLUMN,                  # Fold column name, as specified from encoding
                      training_frame=TEST_DATA)                   # All the data is used for training, cross-validation

        # Calculate variable importance for these specific responders
        varimps = exp_cumulative_varimps(aml_obj,
                                         tag=[prod_well, responder],
                                         tag_name=['production_well', 'responder'])
        cumulative_varimps[responder][prod_well] = varimps
# Concatenate all the individual model variable importances into one dataframe
final_cumulative_varimps = pd.concat([list(pair.values())[0] for pair in list(cumulative_varimps.values())
                                      ]).reset_index(drop=True)
# Exclude any features encoded by default (H2O puts a `.` in the column name of these features)
final_cumulative_varimps = final_cumulative_varimps.loc[:, ~final_cumulative_varimps.columns.str.contains('.',
                                                                                                          regex=False)]
final_cumulative_varimps.index = final_cumulative_varimps['model_name'] + '___' + final_cumulative_varimps['responder']

print('STATUS: Completed experiments for all responders (oil proxies) and detecting variable importances.')

# NOTE: Save outputs for reference (so you don't have to wait an hour every time)
# with open('Modeling Pickles/model_novarimps.pkl', 'wb') as f:
#     pickle.dump(model_novarimps, f)
# final_cumulative_varimps.to_pickle('Modeling Pickles/final_cumulative_varimps.pkl')
# final_cumulative_varimps.to_html('Modeling Reference Files/final_cumulative_varimps.html')

# NOTE: FOR REFERENCE
# Filtering of the variable importance summary
FILT_final_cumulative_varimps = final_cumulative_varimps[~final_cumulative_varimps['model_name'
                                                                                   ].str.contains('XGBoost')
                                                         ].reset_index(drop=True).select_dtypes(float)
FILT_final_cumulative_varimps = final_cumulative_varimps[(final_cumulative_varimps.mean(axis=1) > 0.2) &
                                                         (final_cumulative_varimps.mean(axis=1) < 0.8)
                                                         ].select_dtypes(float)

# Plot heatmap of variable importances across all model combinations
fig, ax = plt.subplots(figsize=(64, 4))
predictor_rank = FILT_final_cumulative_varimps.select_dtypes(float).mean(axis=0).sort_values(ascending=False)
sns_fig = sns.heatmap(FILT_final_cumulative_varimps.select_dtypes(float)[predictor_rank.keys()], annot=False)
sns_fig.get_figure().savefig('Modeling Reference Files/PredictProxy/predictproxy_variable_importance.pdf',
                             bbox_inches='tight')

plt.clf()

correlation_matrix(FILT_final_cumulative_varimps, EXP_NAME='Aggregated Experiment Results',
                   FPATH='Modeling Reference Files/PredictProxy/predictproxy_correlation.pdf')

print('STATUS: Determined correlations and variable importance')
