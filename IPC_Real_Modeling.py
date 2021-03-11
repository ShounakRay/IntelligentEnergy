# @Author: Shounak Ray <Ray>
# @Date:   09-Mar-2021 11:03:94:947  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: IPC_Real_Modeling.py
# @Last modified by:   Ray
# @Last modified time: 10-Mar-2021 22:03:32:326  GMT-0700
# @License: [Private IP]

import os
import pickle
import subprocess

import h2o
import matplotlib.pyplot as plt  # Req. dep. for h2o.estimators.random_forest.H2ORandomForestEstimator.varimp_plot()
import numpy as np
import pandas as pd  # Req. dep. for h2o.estimators.random_forest.H2ORandomForestEstimator.varimp()
import seaborn as sns
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

if not (java_major_version >= 8 and java_major_version <= 14):
    raise ValueError('STATUS: Java Version is not between 8 and 15.\n  \
                      h2o instance will not be initialized')

print('STATUS: Dependency versions checked and confirmed.')


_ = """
#######################################################################################################################
#########################################   DEFINITIONS AND HYPERPARAMTERS   ##########################################
#######################################################################################################################
"""
OUT_BLOCK = '<><><><><><><><><><><><><><><><><><><><><><><>\n'

IP_LINK = 'localhost'                                   # Initializing the server ont he local host
SECURED = True if(IP_LINK != 'localhost') else False    # Set to False since https doesn't work locally, should be True
PORT = 54321                                            # Always specify the port that the server should use, tracking
SERVER_FORCE = True                                     # Attempts to init new server if an existing connection fails

MAX_RUNTIME = 1 * 60                                    # The longest that each model
RANDOM_SEED = 2381125
EVAL_METRIC = 'auto'
RANK_METRIC = 'auto'
DATA_PATH = 'Data/FINALE_INTERP.csv'

FIRST_WELL_STM = 'CI06'
FOLD_COLUMN = "kfold_column"


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


def shutdown_confirm(cluster: h2o.backend.cluster.H2OCluster):
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


_ = """
#######################################################################################################################
##########################################   INITIALIZE SERVER AND SETUP   ############################################
#######################################################################################################################
"""

# INITIALIZE the cluster
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
# NOTE: Reasoning on initial
data = data.drop(['C1', 'unique_id', '24_Fluid', '24_Oil', '24_Water', 'Date'])

# All the target features
RESPONDERS = data.columns[data.columns.index(FIRST_WELL_STM):]
# data.describe()

_ = """
#######################################################################################################################
#######################################   ENGINEERING | MEAN TARGET ENCODING   ########################################
#######################################################################################################################
"""

# Which columns should be encoded. Must be categorical.
encoded_columns = ["Pad", "Well", "test_flag"]
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
data = data_est_te.transform(frame=data, as_training=True)

_ = """
#######################################################################################################################
#########################################   MODEL TRAINING AND DEVELOPMENT   ##########################################
#######################################################################################################################
"""

# Configure experiment
aml_obj = H2OAutoML(max_runtime_secs=MAX_RUNTIME,               # How long should the experiment run for?
                    stopping_metric=EVAL_METRIC,                # The evaluation metric to disctinue model training
                    sort_metric=RANK_METRIC,                    # Leaderboard ranking metric after all trainings
                    seed=RANDOM_SEED,
                    project_name="IPC_MacroModeling")

# NOTE: The model predictors a should only use the target encoded versions, and not the older versions
PREDICTORS = [col for col in data.columns
              if col not in [FOLD_COLUMN] + [col.replace('_te', '') for col in data.columns if '_te' in col]]

# # Run the experiment
# # NOTE: Fold column specified for cross validation to mitigate leakage
# # https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/_modules/h2o/automl/autoh2o.html
# aml_obj.train(x=PREDICTORS,                                     # All the depedent variables in each model
#               y=RESPONDERS[0],                                  # A single responder
#               fold_column=FOLD_COLUMN,                          # Fold column name, as specified from encoding
#               training_frame=data)                              # All the data is used for training, cross-validation
#
# # View models leaderboard and extract desired model
# exp_leaderboard = aml_obj.leaderboard
# exp_leaderboard.head(rows=exp_leaderboard.nrows)
# specific_model = h2o.get_model(exp_leaderboard[0, "model_id"])


# <><><><><> <><><><><> <><><><><> <><><><><> <><><><><> <><><><><>
# <><><><><> <><><><><> <><><><><> <><><><><> <><><><><> <><><><><>
# <><><><><> <><><><><> <><><><><> <><><><><> <><><><><> <><><><><>


# For every predictor feature, run an experiment
cumulative_varimps = []
model_novarimps = []
for responder in RESPONDERS:
    print('\n{block}{block}STATUS: Responder {RESP}'.format(block=OUT_BLOCK,
                                                            RESP=responder))
    # Run the experiment (for the specified responder)
    # NOTE: Fold column specified for cross validation to mitigate leakage
    # https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/_modules/h2o/automl/autoh2o.html
    aml_obj.train(x=PREDICTORS,                                 # All the depedent variables in each model
                  y=responder,                                  # A single responder
                  fold_column=FOLD_COLUMN,                      # Fold column name, as specified from encoding
                  training_frame=data)                          # All the data is used for training, cross-validation
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
            variable_importance['model'] = model_name
            variable_importance['responder'] = responder
            variable_importance.columns.name = None

            cumulative_varimps.append(variable_importance)
        else:
            print('\n{block}> WARNING: Variable importances unavailable for {MDL}'.format(block=OUT_BLOCK,
                                                                                          MDL=model_name))
            model_novarimps.append((responder, model_name))

# with open('Modeling Pickles/model_novarimps.pkl', 'wb') as f:
#     pickle.dump(model_novarimps, f)

cumulative_varimps = pd.concat(cumulative_varimps)
final_cumulative_varimps = cumulative_varimps.reset_index(drop=True)
final_cumulative_varimps.index = final_cumulative_varimps['model'] + '___' + final_cumulative_varimps['responder']
# final_cumulative_varimps.to_pickle('Modeling Pickles/final_cumulative_varimps.pkl')

# final_cumulative_varimps.to_html('final_cumulative_varimps.html')
final_cumulative_varimps = final_cumulative_varimps[~final_cumulative_varimps['model'].str.contains('XGBoost')
                                                    ].reset_index(drop=True).select_dtypes(float)

final_cumulative_varimps = final_cumulative_varimps[(final_cumulative_varimps.mean(axis=1) > 0.2) &
                                                    (final_cumulative_varimps.mean(axis=1) < 0.8)].select_dtypes(float)

fig, ax = plt.subplots(figsize=(32, 64))
sns_fig = sns.heatmap(final_cumulative_varimps, annot=False)
sns_fig.get_figure().savefig('Modeling Reference Files/svm_conf.pdf', bbox_inches="tight")
_ = """
#######################################################################################################################
#################################################   SHUT DOWN H2O   ###################################################
#######################################################################################################################
"""

shutdown_confirm(h2o.cluster())


# EOF
# EOF
