# @Author: Shounak Ray <Ray>
# @Date:   09-Mar-2021 11:03:94:947  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: IPC_Real_Modeling.py
# @Last modified by:   Ray
# @Last modified time: 12-Mar-2021 14:03:28:283  GMT-0700
# @License: [Private IP]

# HELPFUL NOTES:
# > https://github.com/h2oai/h2o-3/tree/master/h2o-docs/src/cheatsheets

import os
import pickle
import subprocess
from typing import Final

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
MAX_EXP_RUNTIME: Final = 2 * 60                               # The longest that the experiment will run (seconds)
RANDOM_SEED: Final = 2381125                                  # To ensure reproducibility of experiments
EVAL_METRIC: Final = 'auto'                                   # The evaluation metric to discontinue model training
RANK_METRIC: Final = 'auto'                                   # Leaderboard ranking metric after all trainings
DATA_PATH: Final = 'Data/FINALE_INTERP.csv'                   # Where the client-specific data is located

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
        print('\n{block}> STATUS: Model {MDL}'.format(block=OUT_BLOCK,
                                                      MDL=model_name))
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
            print('\n{block}> WARNING: Variable importances unavailable for {MDL}'.format(block=OUT_BLOCK,
                                                                                          MDL=model_name))
            model_novarimps.append((model_name, model))

    return pd.concat(cumulative_varimps).reset_index(drop=True), model_novarimps


# Diverging: sns.diverging_palette(240, 10, n=9, as_cmap=True)
# https://seaborn.pydata.org/generated/seaborn.color_palette.html
def correlation_matrix(df, FPATH, abs_arg=True, mask=True, annot=False, type_corrs=['Pearson', 'Kendall', 'Spearman'],
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
                              ).set_title("{cortype}'s Correlation Matrix".format(cortype=typec))
    plt.tight_layout()
    sns_fig.get_figure().savefig(FPATH, box_inches="tight")

    return input_data


print('STATUS: Hyperparameters assigned and functions defined')

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
# NOTE: Reasoning on initial feature deletion:
# > C1          -> This is simply a consecutive counter column. No relation to target
# > unique_id   -> This is simply a consecutive counter column. No relation to target
# > 24_Fluid    -> High Correlation with `Fluid` feature in data
# > 24_Oil      -> High Correlation with `Oil` feature in data
# > 24_Water    -> High Correlation with `Water` feature in data
# > Pad         -> Pad not required, rather production well is enough
data = data.drop(['C1', 'unique_id', '24_Fluid', '24_Oil', '24_Water', 'Date', 'Pad'])

PRODUCTION_WELLS = list(data['Well'].unique().as_data_frame()['C1'])

print('STATUS: Server initialized and data imported.')

_ = """
#######################################################################################################################
########################################   ENGINEERING | BEST OIL PROXIES   ###########################################
#######################################################################################################################
"""
# TODO: Determine best oil proxies for EACH production well


aml_fe_obj = H2OAutoML(max_runtime_secs=MAX_EXP_RUNTIME,          # How long should the experiment run for?
                       stopping_metric=EVAL_METRIC,               # The evaluation metric to discontinue model training
                       sort_metric=RANK_METRIC,                   # Leaderboard ranking metric after all trainings
                       seed=RANDOM_SEED,
                       project_name="IPC_OilLimitedModeling")

SENSOR_PREDICTORS = ['Tubing_Pressure', 'Casing_Pressure', 'Heel_Pressure', 'Toe_Pressure', 'Heel_Temp', 'Toe_Temp']
PRODUCTION_TARGET = 'Oil'

# FOR EACH PRODUCTION WELL
proxy_correlations = {}
proxy_sensor_rankings = {}
for prod_well in PRODUCTION_WELLS:
    # Only look at data from test days (to minimize/eliminate missing values for PRODUCTION_TARGET)
    TEST_DATA = data[data['test_flag'].isin(['True'])]
    TEST_DATA = TEST_DATA[TEST_DATA['Well'] == prod_well]
    aml_fe_obj.train(x=SENSOR_PREDICTORS,
                     y=PRODUCTION_TARGET,                                   # A single responder
                     training_frame=TEST_DATA)                              # All the data is used for training, CV

    print('STATUS: Selective sensor-predictor model trained.')

    # Determine Variable Importance of Sensor Features
    cumulative_fe_varimps, excluded = exp_cumulative_varimps(aml_fe_obj)
    # Reindex output to name of model (for seaborn labeling)
    cumulative_fe_varimps.index = cumulative_fe_varimps['model_name']
    # Determine the feature rank depending on the mean of each predictor column
    SENSOR_RANKING = cumulative_fe_varimps.select_dtypes(float).mean(axis=0).sort_values(ascending=False)
    # Reorder columns by SENSOR_RANKING, Reorder rows by mean of model's variable importances
    cumulative_fe_varimps = cumulative_fe_varimps.select_dtypes(float)[SENSOR_RANKING.keys()]
    cumulative_fe_varimps = cumulative_fe_varimps.reindex(cumulative_fe_varimps.mean(axis=1).sort_values(axis=0).index,
                                                          axis=0)

    # Save oil-sensor correlations to PDF
    fig, ax = plt.subplots(figsize=(8, 16))
    sns_fig = sns.heatmap(cumulative_fe_varimps, annot=True).set_title('Feature Ranking:\nBest -> Worst')
    sns_fig.get_figure().savefig('Modeling Reference Files/cumulative_fe_varimps__{PWELL}.pdf'.format(PWELL=prod_well),
                                 bbox_inches="tight")

    # Correlation heatmaps
    proxy_correlations[prod_well] = correlation_matrix(cumulative_fe_varimps,
                                                       FPATH='Modeling Reference Files/cumulative_fe_varimps.corr()' +
                                                       '.abs()__{PWELL}.pdf'.format(PWELL=prod_well))
    proxy_sensor_rankings[prod_well] = SENSOR_RANKING

print('STATUS: Optimal sensor oil proxies determined. These will now be the responders for future predictive models.')

_ = """
#######################################################################################################################
#######################################   ENGINEERING | MEAN TARGET ENCODING   ########################################
#######################################################################################################################
"""
#
# # TODO: Mean Target Encoding isn't currently functional due to manual target spec. -> should be dynamically encoded
# # All the target features
# RESPONDERS = data.columns[data.columns.index(FIRST_WELL_STM):]
#
# # The columns should be encoded. Must be categorical, automatically select the categorical ones.
# temp_df = data.as_data_frame().select_dtypes([float, int]).columns
# encoded_columns = [col for col in data.columns if col not in temp_df]
# # Assign kfold assignment randomly (based on constant RANDOM_SEED for reproducibility)
# data[FOLD_COLUMN] = data.kfold_column(n_folds=5, seed=RANDOM_SEED)
#
# # Congifure auto-encoder
# # http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/_modules/h2o/estimators/targetencoder.html
# data_est_te = H2OTargetEncoderEstimator(data_leakage_handling="k_fold",  # enc. for a fold gen. on out-of-fold data
#                                         fold_column=FOLD_COLUMN,         # The column name with specific fold labels
#                                         blending=True,          # Helps in encoding low-cardinality categoricals
#                                         inflection_point=3,     # Inflection point of sigmoid used to blend pr.
#                                         smoothing=20,           # m^-1 @ inflection point on sigmoid used to blend pr
#                                         noise=0.01,             # Low data, needs more regularization; set to detault
#                                         seed=RANDOM_SEED)
# # Train the encoder model (this isn't predictive, its manipulative)
# data_est_te.train(x=encoded_columns,
#                   y=RESPONDERS[0],
#                   training_frame=data)
# # Encode the data based on the trained encoder
# data = data_est_te.transform(frame=data, as_training=True)
#
# print('STATUS: Completed Mean Target Encoding for all categorical features.')

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

# Configure experiment
aml_obj = H2OAutoML(max_runtime_secs=MAX_EXP_RUNTIME,           # How long should the experiment run for?
                    stopping_metric=EVAL_METRIC,                # The evaluation metric to discontinue model training
                    sort_metric=RANK_METRIC,                    # Leaderboard ranking metric after all trainings
                    seed=RANDOM_SEED,
                    project_name="IPC_MacroModeling")

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
sns_fig.get_figure().savefig('Modeling Reference Files/final_cumulative_varimps.pdf', bbox_inches="tight")

correlation_matrix(FILT_final_cumulative_varimps,
                   FPATH='Modeling Reference Files/FILT_final_cumulative_varimps.corr().abs().pdf')

print('STATUS: Determined correlations and variable importance')

_ = """
#######################################################################################################################
#########################################   EVALUATE MODEL PERFORMANCE   ##############################################
#######################################################################################################################
"""

ALL_MODELS = dict(zip(final_cumulative_varimps['responder'], final_cumulative_varimps['model_object']))
for model in ALL_MODELS:
    3

print('STATUS: Completed analysis of all models.')

_ = """
#######################################################################################################################
#################################################   SHUT DOWN H2O   ###################################################
#######################################################################################################################
"""

# Shutdown the cluster
shutdown_confirm(h2o.cluster())


# EOF
# EOF
