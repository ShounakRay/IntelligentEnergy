# @Author: Shounak Ray <Ray>
# @Date:   09-Mar-2021 11:03:94:947  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: IPC_Real_Modeling.py
# @Last modified by:   Ray
# @Last modified time: 10-Mar-2021 12:03:65:659  GMT-0700
# @License: [Private IP]

import os
import subprocess

import h2o
import matplotlib as plt  # Required dependecy for h2o.estimators.random_forest.H2ORandomForestEstimator.varimp_plot()
import pandas as pd  # Required dependecy for h2o.estimators.random_forest.H2ORandomForestEstimator.varimp()
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
IP_LINK = 'localhost'                                   # Initializing the server ont he local host
SECURED = True if(IP_LINK != 'localhost') else False    # Set to False since https doesn't work locally, should be True
PORT = 54321                                            # Always specify the port that the server should use, tracking
SERVER_FORCE = True                                     # Attempts to init new server if an existing connection fails

MAX_RUNTIME = 15 * 60                                     # The longest that each model
RANDOM_SEED = 2381125
EVAL_METRIC = 'auto'
RANK_METRIC = 'auto'
DATA_PATH = 'Data/FINALE_INTERP.csv'

FIRST_WELL_STM = 'CI06'


def process_snapshot(cluster: h2o.backend.cluster.H2OCluster, show: bool = True) -> dict:
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
process_log = process_snapshot(h2o.cluster(), show=False)

# Confirm that the data path leads to an actual file
if not (os.path.isfile(DATA_PATH)):
    raise ValueError('ERROR: {data} does not exist in the specificied location.'.format(data=DATA_PATH))

# Import the data from the file and exclude any obvious features
data = h2o.import_file(DATA_PATH)
data = data.drop(['unique_id', '24_Fluid', '24_Oil', '24_Water'])

data.describe()

_ = """
#######################################################################################################################
#######################################   ENGINEERING | MEAN TARGET ENCODING   ########################################
#######################################################################################################################
"""
encoded_columns = ["Pad", "Well", "test_flag"]
fold_column = "kfold_column"
data[fold_column] = data.kfold_column(n_folds=5, seed=RANDOM_SEED)
data_te = H2OTargetEncoderEstimator(fold_column=fold_column,
                                    data_leakage_handling="k_fold",
                                    blending=True,
                                    inflection_point=3,
                                    smoothing=10,
                                    noise=0.15,     # In general, the less data you have the more regularization you need
                                    seed=RANDOM_SEED)

_ = """
#######################################################################################################################
#########################################   MODEL TRAINING AND DEVELOPMENT   ##########################################
#######################################################################################################################
"""

# Configure and train models
aml_obj = H2OAutoML(max_runtime_secs=MAX_RUNTIME,
                    stopping_metric=EVAL_METRIC,
                    sort_metric=RANK_METRIC,
                    seed=RANDOM_SEED,
                    project_name="IPC_MacroModeling")
PREDICTORS = data.columns[data.columns.index(FIRST_WELL_STM):]
aml_obj.train(y=PREDICTORS[0], training_frame=data)

# View models leaderboard and extract desired model
exp_leaderboard = aml_obj.leaderboard
exp_leaderboard.head(rows=exp_leaderboard.nrows)
specific_model = h2o.get_model(exp_leaderboard[0, "model_id"])

type(specific_model)
dir(h2o.estimators)

# Determine and store variable importances
specific_model.varimp(use_pandas=True)
_ = """
#######################################################################################################################
#################################################   SHUT DOWN H2O   ###################################################
#######################################################################################################################
"""

# SHUT DOWN the cluster after you're done working with it
h2o.cluster().shutdown()
# Double checking...
try:
    process_snapshot(h2o.cluster())
    raise ValueError('ERROR: H2O cluster improperly closed!')
except Exception:
    pass

# EOF
# EOF
