# @Author: Shounak Ray <Ray>
# @Date:   09-Mar-2021 11:03:94:947  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: IPC_Real_Modeling.py
# @Last modified by:   Ray
# @Last modified time: 09-Mar-2021 15:03:43:437  GMT-0700
# @License: [Private IP]

import os
import subprocess

import h2o
from h2o.automl import H2OAutoML

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


def process_snapshot(cluster: h2o.backend.cluster.H2OCluster) -> dict:
    """Provides a snapshot of the h2o cluster and different status/performance indicators.

    Parameters
    ----------
    cluster : h2o.backend.cluster.H2OCluster
        The h2o cluster where the server was initialized.

    Returns
    -------
    dict
        Information about the status/performance of the specified cluster.

    """
    return {'cluster_name': cluster.cloud_name,
            'pid': cluster.get_status_details()['pid'],
            'version': cluster.version,
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
h2o.init(https=False,        # Set to False since https doesn't work on localhost, should be True for docker
         ip='localhost',     # Initializing the server ont he local host
         port=54321,         # Always specify the port that the server will use, just to be safe
         start_h2o=True)     # Attempts to start a new h20 server if and when a connection to an existing one fails

# Check the status of the cluster, just for reference
process_snapshot(h2o.cluster())
h2o.cluster().show_status()
# Assign and confirm the data path
data_path = 'FINALE.csv'
if(os.path.isfile(data_path)):
    pass
else:
    raise ValueError('ERROR: {data} does not exist in the specificied location.'.format(data=data_path))

# Import the data from the file
data = h2o.import_file(data_path)

_ = """
#######################################################################################################################
#########################################   MODEL TRAINING AND DEVELOPMENT   ##########################################
#######################################################################################################################
"""

# Configure and train models
aml_obj = H2OAutoML(max_runtime_secs=60,
                    stopping_metric='auto',
                    sort_metric='auto',
                    seed=2381125,
                    project_name="IPC_MacroModeling")
aml_obj.train(y='Daily_Meter_Steam', training_frame=data)

# View models leaderboard and extract desired model
aml_obj.leaderboard.head(rows=aml_obj.leaderboard.nrows)
specific_model = h2o.get_model(aml_obj.leaderboard[2, "model_id"])

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
except Exception:
    raise ValueError('ERROR: H2O cluster improperly closed!')

# EOF
# EOF
