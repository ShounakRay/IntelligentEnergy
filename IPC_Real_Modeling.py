# @Author: Shounak Ray <Ray>
# @Date:   09-Mar-2021 11:03:94:947  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: IPC_Real_Modeling.py
# @Last modified by:   Ray
# @Last modified time: 09-Mar-2021 12:03:49:496  GMT-0700
# @License: [Private IP]

import os
import subprocess

import h2o

java_major_version = int(subprocess.check_output(['java', '-version'],
                                                 stderr=subprocess.STDOUT).decode().split('"')[1].split('.')[0])

if not (java_major_version >= 8 and java_major_version <= 14):
    raise ValueError('STATUS: Java Version is not between 8 and 15.\n  \
                      h2o instance will not be initialized')


def process_snapshot(cluster: h2o.backend.cluster.H2OCluster) -> dict:
    return {'cluster_name': cluster.cloud_name,
            'pid': cluster.get_status_details()['pid'],
            'version': cluster.version,
            'run_status': cluster.is_running(),
            'branch_name': cluster.branch_name,
            'uptime_ms': cluster.cloud_uptime_millis,
            'health': cluster.cloud_healthy}


def cmd_runprint(command: str, prnt_file: bool = True, prnt_scrn: bool = False, ret: bool = False):
    """Runs a command in Python script and uses remote_shell_output.txt or screen as stdout/console.

    Parameters
    ----------
    command : str
        The command to be executed.
    prnt_file : bool
        Whether output of command should be printed to remote_shell_output.txt
    prnt_scrn : bool
        Whether output of command should be printed to screen.
    ret : bool
        Whether output of command should be returned.

    Returns [Optional]
    -------
    None
        Nothing is returned if ret is False
    str
        The output of the command is returned if ret is True

    """
    exec_output = subprocess.Popen(command,
                                   shell=True,
                                   stdout=subprocess.PIPE).stdout.read().decode("utf-8")
    if(prnt_file):
        print(exec_output, file=open('remote_shell_output.txt', 'w'))
    if(prnt_scrn):
        print(exec_output)
    if(ret):
        return exec_output

# Convert the provided h2o demo file to a python file
# cmd_runprint(command="jupyter nbconvert --to script 'H2O Testing/automl_regression_powerplant_output.ipynb'",
#              prnt_file=False, prnt_scrn=True)


# INITIALIZE the cluster
h2o.init(https=False,        # Set to False since https doesn't work on localhost, should be True for docker
         ip='localhost',     # Initializing the server ont he local host
         port=54321,         # Always specify the port that the server will use, just to be safe
         start_h2o=True)     # Attempts to start a new h20 server if and when a connection to an existing one fails

# Check the status of the cluster, just for reference
process_snapshot(h2o.cluster())

# Assign and confirm the data path
data_path = 'FINALE.csv'
if(os.path.isfile(data_path)):
    pass
else:
    raise ValueError('{data} does not exist in the specificied location.'.format(data=data_path))

# Import the data from the file


# SHUT DOWN the cluster after you're done working with it
h2o.cluster().shutdown()
# EOF
# EOF
