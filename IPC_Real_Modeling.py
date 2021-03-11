# @Author: Shounak Ray <Ray>
# @Date:   09-Mar-2021 11:03:94:947  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: IPC_Real_Modeling.py
# @Last modified by:   Ray
# @Last modified time: 11-Mar-2021 13:03:66:664  GMT-0700
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
MAX_EXP_RUNTIME: Final = 2 * 60                               # The longest that the experiment will tun
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
    """Provides a snapshot of the h2o cluster and different status/performance indicators.

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
    """Terminates the provided h2o cluster.

    Parameters
    ----------
    cluster : h2o.backend.cluster.H2OCluster
        The h2o cluster where the server was initialized.

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


def exp_cumulative_varimps(aml_obj):
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
            variable_importance.columns.name = None

            cumulative_varimps.append(variable_importance)
        else:
            print('\n{block}> WARNING: Variable importances unavailable for {MDL}'.format(block=OUT_BLOCK,
                                                                                          MDL=model_name))
            model_novarimps.append(model_name)

    return pd.concat(cumulative_varimps).reset_index(drop=True), model_novarimps


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
data = data.drop(['C1', 'unique_id', '24_Fluid', '24_Oil', '24_Water', 'Date'])

# Each of the features individually predicted
# NOTE: The model predictors a should only use the target encoded versions, and not the older versions
PREDICTORS = [col for col in data.columns
              if col not in [FOLD_COLUMN] + [col.replace('_te', '') for col in data.columns if '_te' in col]]


_ = """
#######################################################################################################################
########################################   ENGINEERING | BEST OIL PROXIES   ###########################################
#######################################################################################################################
"""

aml_fe_obj = H2OAutoML(max_runtime_secs=MAX_EXP_RUNTIME,          # How long should the experiment run for?
                       stopping_metric=EVAL_METRIC,               # The evaluation metric to discontinue model training
                       sort_metric=RANK_METRIC,                   # Leaderboard ranking metric after all trainings
                       seed=RANDOM_SEED,
                       project_name="IPC_OilLimitedModeling")

SENSOR_PREDICTORS = ['Tubing_Pressure', 'Casing_Pressure', 'Heel_Pressure', 'Toe_Pressure', 'Heel_Temp', 'Toe_Temp']
PRODUCTION_TARGET = 'Oil'
# pd_data_temp = data.as_data_frame()
TEST_DATA = data[data['test_flag'].isin(['True'])]
aml_fe_obj.train(x=SENSOR_PREDICTORS,
                 y=PRODUCTION_TARGET,                                   # A single responder
                 training_frame=TEST_DATA)                              # All the data is used for training, CV

# Determine Variable Importance
cumulative_fe_varimps, excluded = exp_cumulative_varimps(aml_fe_obj)
# Reindex output to name of model (for seaborn labeling)
cumulative_fe_varimps.index = cumulative_fe_varimps['model_name']
# Determine the feature rank depending on the mean of ach predictor column
ranking = cumulative_fe_varimps.select_dtypes(float).mean(axis=0).sort_values(ascending=False)
# Reorder columns by ranking, Reorder rows by mean of model's variable importances
cumulative_fe_varimps = cumulative_fe_varimps.select_dtypes(float)[ranking.keys()]
cumulative_fe_varimps = cumulative_fe_varimps.reindex(cumulative_fe_varimps.mean(axis=1).sort_values(axis=0).index,
                                                      axis=0)

# NOTE: For reference
# _ = plt.hist(cumulative_fe_varimps, bins=45, stacked=True)
# Save oil-sensor correlations to PDF
fig, ax = plt.subplots(figsize=(8, 16))
sns_fig = sns.heatmap(cumulative_fe_varimps, annot=False).set_title('Feature Ranking: Best -> Worst')
sns_fig.get_figure().savefig('Modeling Reference Files/cumulative_fe_varimps.pdf', bbox_inches="tight")

# Correlation heatmaps
fig, ax = plt.subplots(figsize=(8, 8))
cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)
sns_fig = sns.heatmap(cumulative_fe_varimps.abs().corr(), annot=True).set_title("Pearson's Correlation Matrix")
sns_fig.get_figure().savefig('Modeling Reference Files/cumulative_fe_varimps.abs().corr().pdf', bbox_inches="tight")

_ = """
#######################################################################################################################
#######################################   ENGINEERING | MEAN TARGET ENCODING   ########################################
#######################################################################################################################
"""

# TODO: Mean Target Encoding isn't currently functional due to manual target spec. -> should be dynamically encoded
# All the target features
RESPONDERS = data.columns[data.columns.index(FIRST_WELL_STM):]

# The columns should be encoded. Must be categorical, automatically select the categorical ones.
temp_df = data.as_data_frame().select_dtypes([float, int]).columns
encoded_columns = [col for col in data.columns if col not in temp_df]
# Assign kfold assignment randomly (based on constant RANDOM_SEED for reproducibility)
data[FOLD_COLUMN] = data.kfold_column(n_folds=5, seed=RANDOM_SEED)

# Congifure auto-encoder
# http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/_modules/h2o/estimators/targetencoder.html
data_est_te = H2OTargetEncoderEstimator(data_leakage_handling="k_fold",  # enc. for a fold gen. on out-of-fold data
                                        fold_column=FOLD_COLUMN,         # The column name with specific fold labels
                                        blending=True,          # Helps in encoding low-cardinality categoricals
                                        inflection_point=3,     # Inflection point of sigmoid used to blend pr.
                                        smoothing=20,           # m^-1 @ inflection point on sigmoid used to blend pr.
                                        noise=0.01,             # Low data, needs more regularization; set to detault
                                        seed=RANDOM_SEED)
# Train the encoder model (this isn't predictive, its manipulative)
data_est_te.train(x=encoded_columns,
                  y=RESPONDERS[0],
                  training_frame=data)
# Encode the data based on the trained encoder
data = data_est_te.transform(frame=data, as_training=True)

_ = """
#######################################################################################################################
#########################################   MODEL TRAINING AND DEVELOPMENT   ##########################################
#######################################################################################################################
"""

# Configure experiment
aml_obj = H2OAutoML(max_runtime_secs=MAX_EXP_RUNTIME,           # How long should the experiment run for?
                    stopping_metric=EVAL_METRIC,                # The evaluation metric to discontinue model training
                    sort_metric=RANK_METRIC,                    # Leaderboard ranking metric after all trainings
                    seed=RANDOM_SEED,
                    project_name="IPC_MacroModeling")

# Run the experiment (for the specified responder)
# NOTE: Fold column specified for cross validation to mitigate leakage
# https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/_modules/h2o/automl/autoh2o.html
aml_obj.train(x=PREDICTORS,                                 # All the depedent variables in each model
              y=PRO,                                  # A single responder
              fold_column=FOLD_COLUMN,                      # Fold column name, as specified from encoding
              training_frame=data)                          # All the data is used for training, cross-validation
exp_leaderboard = aml_obj.leaderboard

# Retrieve all the model objects from the given experiment
exp_models = [h2o.get_model(exp_leaderboard[m_num, "model_id"]) for m_num in range(exp_leaderboard.shape[0])]
cumulative_varimps = []
model_novarimps = []
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
        variable_importance['responder'] = responder
        variable_importance.columns.name = None

        cumulative_varimps.append(variable_importance)
    else:
        print('\n{block}> WARNING: Variable importances unavailable for {MDL}'.format(block=OUT_BLOCK,
                                                                                      MDL=model_name))
        model_novarimps.append((responder, model_name))


# Concatenate all the individual model variable importances into one dataframe
final_cumulative_varimps = pd.concat(cumulative_varimps).reset_index(drop=True)
final_cumulative_varimps.index = final_cumulative_varimps['model'] + '___' + final_cumulative_varimps['responder']

# NOTE: Save outputs for reference (so you don't have to wait an hour every time)
# with open('Modeling Pickles/model_novarimps.pkl', 'wb') as f:
#     pickle.dump(model_novarimps, f)
# final_cumulative_varimps.to_pickle('Modeling Pickles/final_cumulative_varimps.pkl')
# final_cumulative_varimps.to_html('Modeling Reference Files/final_cumulative_varimps.html')

# NOTE: FOR REFERENCE
# Filtering of the variable importance summary
final_cumulative_varimps = final_cumulative_varimps[~final_cumulative_varimps['model'].str.contains('XGBoost')
                                                    ].reset_index(drop=True).select_dtypes(float)
final_cumulative_varimps = final_cumulative_varimps[(final_cumulative_varimps.mean(axis=1) > 0.2) &
                                                    (final_cumulative_varimps.mean(axis=1) < 0.8)].select_dtypes(float)

# NOTE: FOR REFERENCE
# Plot heatmap of variable importances across all model combinations
fig, ax = plt.subplots(figsize=(32, 64))
sns_fig = sns.heatmap(final_cumulative_varimps, annot=False)
sns_fig.get_figure().savefig('Modeling Reference Files/svm_conf.pdf', bbox_inches="tight")
_ = """
#######################################################################################################################
#################################################   SHUT DOWN H2O   ###################################################
#######################################################################################################################
"""

# Shutdown the cluster
shutdown_confirm(h2o.cluster())


# EOF
# EOF
