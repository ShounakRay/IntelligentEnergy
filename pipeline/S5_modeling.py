# @Author: Shounak Ray <Ray>
# @Date:   23-Apr-2021 14:04:06:060  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: S5_modeling.py
# @Last modified by:   Ray
# @Last modified time: 26-Apr-2021 10:04:51:516  GMT-0600
# @License: [Private IP]

# HELPFUL NOTES:
# > https://github.com/h2oai/h2o-3/tree/master/h2o-docs/src/cheatsheets
# > https://www.h2o.ai/blog/h2o-release-3-30-zahradnik/#AutoML-Improvements
# > https://seaborn.pydata.org/generated/seaborn.color_palette.html

import datetime
import itertools
import os
import random
import subprocess
import sys
import time
from pprint import pprint
from typing import Final

import h2o
import matplotlib
import matplotlib.pyplot as plt  # Req. dep. for h2o.estimators.random_forest.H2ORandomForestEstimator.varimp_plot()
import numpy as np
import pandas as pd  # Req. dep. for h2o.estimators.random_forest.H2ORandomForestEstimator.varimp()
import seaborn as sns
from colorama import Fore, Style
from h2o.automl import H2OAutoML
from h2o.exceptions import H2OConnectionError

# from matplotlib.patches import Rectangle


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
    from S4_ft_eng_math import _FEATENG_MATH

    # Check java dependency
    check_java_dependency()

    # import _multiprocessed.defs as defs
    # import _traversal


# _traversal.print_tree_to_txt(PATH='_configs/FILE_STRUCTURE.txt')


_ = """
#######################################################################################################################
#############################################   HYPERPARAMETER SETTINGS   #############################################
#######################################################################################################################
"""
# Data Ingestion Constants
DATA_PATH_PAD: Final = 'Data/combined_ipc_engineered_math.csv'    # Where the client-specific pad data is located
DATA_PATH_PAD_vanilla = 'Data/combined_ipc_aggregates.csv'        # No feature engineering

# H2O Server Constants
IP_LINK: Final = 'localhost'                                  # Initializing the server on the local host (temporary)
SECURED: Final = True if(IP_LINK != 'localhost') else False   # https protocol doesn't work locally, should be True
PORT: Final = 54321                                           # Always specify the port that the server should use
SERVER_FORCE: Final = True                                    # Tries to init new server if existing connection fails

# Experiment > Model Training Constants and Hyperparameters
MAX_EXP_RUNTIME: Final = 20                                   # The longest that the experiment will run (seconds)
RANDOM_SEED: Final = 2381125                                  # To ensure reproducibility of experiments (caveats*)
EVAL_METRIC: Final = 'rmse'                                   # The evaluation metric to discontinue model training
RANK_METRIC: Final = 'rmse'                                   # Leaderboard ranking metric after all trainings
CV_FOLDS: Final = 50                                          # Number of cross validation folds
STOPPING_ROUNDS: Final = 4                                    # How many rounds to proceed until stopping_metric stops
WEIGHTS_COLUMN: Final = 'weight'                              # Name of the weights column
EXPLOIT_RATIO: Final = 0.0                                    # Exploit/Eploration ratio, see NOTES
MODELING_PLAN: Final = None                                   # Custom Modeling Plan
TRAIN_VAL_SPLIT: Final = 0.95                                 # Train test split proportion

# Feature Engineering Constants
FOLD_COLUMN: Final = "kfold_column"                           # Target encoding, must be consistent throughout training
TOP_MODELS = 15                                               # The top n number of models which should be filtered
PREFERRED_TOLERANCE = 0.1                                     # Tolerance applied on RMSE for allowable responders
TRAINING_VERBOSITY = 'warn'                                   # Verbosity of training (None, 'debug', 'info', d'warn')

# Miscellaneous Constants
MODEL_CMAPS = {'R^2': sns.color_palette('rocket_r', as_cmap=True),
               # 'R': sns.color_palette('rocket_r'),
               'MSE': sns.color_palette("coolwarm", as_cmap=True),
               'RMSE': sns.color_palette("coolwarm", as_cmap=True),
               'Rel. RMSE': sns.color_palette("coolwarm", as_cmap=True),
               'RMSLE': sns.color_palette("coolwarm", as_cmap=True),
               'MAE': sns.color_palette("coolwarm", as_cmap=True)}  # For heatmap visualization
HMAP_CENTERS = {'R^2': None,
                'MSE': 400,
                'RMSE': 20,
                'Rel. RMSE': 0,
                'RMSLE': None,
                'MAE': None}                                        # For heatmap visualization
CELL_RATIO = 4.666666666666667                                       # Scaling for visualizations

# Aesthetic Console Output constants
OUT_BLOCK: Final = '»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»\n'

_ = """
#######################################################################################################################
##############################################   FUNCTION DEFINITIONS   ###############################################
#######################################################################################################################
"""


def get_aml_objects(project_names):
    objs = []
    for project_name in project_names:
        objs.append(h2o.automl.get_automl(project_name))
    return objs


def record_hyperparameters(MAX_EXP_RUNTIME):
    # NOTE: Dependent on global, non-locally defined variables

    RUN_TAG: Final = random.randint(0, 10000)   # The identifying Key/ID for the specified run/config.
    # Ensure overwriting does not occur while making identifying experiment directory.
    while os.path.isdir(f'Modeling Reference Files/Round {RUN_TAG}/'):
        RUN_TAG: Final = random.randint(0, 10000)
    _accessories.auto_make_path(f'Modeling Reference Files/Round {RUN_TAG}/')

    # Compartmentalize Hyperparameters
    __LOCAL_VARS = globals().copy()
    _SERVER_HYPERPARAMS = ('IP_LINK', 'SECURED', 'PORT', 'SERVER_FORCE')
    _SERVER_HYPERPARAMS = {var: __LOCAL_VARS.get(var) for var in _SERVER_HYPERPARAMS}
    _TRAIN_HYPERPARAMS = ('MAX_EXP_RUNTIME', 'RANDOM_SEED', 'EVAL_METRIC', 'RANK_METRIC', 'CV_FOLDS',
                          'STOPPING_ROUNDS', 'WEIGHTS_COLUMN', 'EXPLOIT_RATIO', 'MODELING_PLAN')
    _TRAIN_HYPERPARAMS = {var: __LOCAL_VARS.get(var) for var in _TRAIN_HYPERPARAMS}
    _EXECUTION_HYPERPARAMS = ('FOLD_COLUMN', 'TOP_MODELS', 'PREFERRED_TOLERANCE', 'TRAINING_VERBOSITY')
    _EXECUTION_HYPERPARAMS = {var: __LOCAL_VARS.get(var) for var in _EXECUTION_HYPERPARAMS}
    _MISCELLANEOUS_HYPERPARAMS = ('MODEL_CMAPS', 'MODEL_CMAPS', 'HMAP_CENTERS', 'CELL_RATIO')
    _MISCELLANEOUS_HYPERPARAMS = {var: __LOCAL_VARS.get(var) for var in _MISCELLANEOUS_HYPERPARAMS}

    # Save hyperparameter configurations to file with TIMESTAMP
    with open(f'Modeling Reference Files/Round {RUN_TAG}/hyperparams.txt', 'wt') as out:
        print(OUT_BLOCK.replace('\n', '') + OUT_BLOCK.replace('\n', ''), file=out)
        print('JOB INITIALIZED: ', datetime.datetime.now(), file=out)
        print('RUN ID:', RUN_TAG, file=out)
        print(file=out)

        print(OUT_BLOCK.replace('\n', '') + OUT_BLOCK.replace('\n', ''), file=out)
        print('SERVER HYPERPARAMETERS:', file=out)
        pprint(_SERVER_HYPERPARAMS, stream=out)
        print(file=out)

        print(OUT_BLOCK.replace('\n', '') + OUT_BLOCK.replace('\n', ''), file=out)
        print('TRAINING HYPERPARAMETRS:', file=out)
        pprint(_TRAIN_HYPERPARAMS, stream=out)
        print(file=out)

        print(OUT_BLOCK.replace('\n', '') + OUT_BLOCK.replace('\n', ''), file=out)
        print('EXECUTION HYPERPARAMETRS:', file=out)
        pprint(_EXECUTION_HYPERPARAMS, stream=out)
        print(file=out)

        print(OUT_BLOCK.replace('\n', '') + OUT_BLOCK.replace('\n', ''), file=out)
        print('MISCELLANEOUS HYPERPARAMETRS:', file=out)
        pprint(_MISCELLANEOUS_HYPERPARAMS, stream=out)
        print(file=out)

    _accessories._print(f'STATUS: Directory created for Round {RUN_TAG}')

    return RUN_TAG


@_context_managers.server
def snapshot(cluster: h2o.backend.cluster.H2OCluster, show: bool = True) -> dict:
    """Provides a snapshot of the H2O cluster and different status/performance indicators.

    Parameters
    ----------
    cluster : h2o.backend.cluster.H2OCluster
        The h2o cluster where the server was initialized.
    show : bool
        Whether details should be printed to screen. Uses H2Os built-in method.py

    Returns
    -------
    dict
        Information about the status/performance of the specified cluster.

    """
    """DATA SANITATION"""
    _provided_args = locals()
    name = sys._getframe(0).f_code.co_name
    _expected_type_args = {'cluster': [h2o.backend.cluster.H2OCluster],
                           'show': [bool]}
    _expected_value_args = {'cluster': None,
                            'show': [True, False]}
    util_data_type_sanitation(_provided_args, _expected_type_args, name)
    util_data_range_sanitation(_provided_args, _expected_value_args, name)
    """END OF DATA SANITATION"""

    h2o.cluster().show_status() if(show) else None

    try:
        birds_eye = {'cluster_name': cluster.cloud_name,
                     'pid': cluster.get_status_details()['pid'][0],
                     'version': [int(i) for i in cluster.version.split('.')],
                     'run_status': cluster.is_running(),
                     'branch_name': cluster.branch_name,
                     'uptime_ms': cluster.cloud_uptime_millis,
                     'health': cluster.cloud_healthy}
    except H2OConnectionError as e:
        print('> STATUS: Connection to cluster seems to be closed.\n\t\t' +
              str(e).replace('H2OConnectionError: ', '')[:40] + '...')

    return


@_context_managers.server
def shutdown_confirm(h2o_instance: type(h2o)) -> None:
    """Terminates the provided H2O cluster.

    Parameters
    ----------
    cluster : type(h2o)
        The H2O instance where the server was initialized.

    Returns
    -------
    None
        Nothing. ValueError may be raised during processing and cluster metrics may be printed.

    """
    """DATA SANITATION"""
    _provided_args = locals()
    name = sys._getframe(0).f_code.co_name
    _expected_type_args = {'h2o_instance': [type(h2o)]}
    _expected_value_args = {'h2o_instance': None}
    util_data_type_sanitation(_provided_args, _expected_type_args, name)
    util_data_range_sanitation(_provided_args, _expected_value_args, name)
    """END OF DATA SANITATION"""

    # SHUT DOWN the cluster after you're done working with it
    h2o_instance.remove_all()
    h2o_instance.cluster().shutdown()
    h2o.shutdown()
    os.system('lsof -i tcp:54321')
    # Double checking...
    try:
        snapshot(h2o_instance.cluster)
        raise ValueError('ERROR: H2O cluster improperly closed!')
    except Exception:
        pass


@_context_managers.utility
def util_data_type_sanitation(val_and_inputs, expected, name):
    """Generic performance of data sanitation. Specifically cross-checks expected and actual data types.

    Parameters
    ----------
    val_and_inputs : dict
        Outlines relationship between inputted variable names and their literal values.
        That is, keys are "variable names" and values are "literal values."
    expected : dict
        Outlines relationship between expected variable names and their expected types.
        That is, keys are "variable names" and values are "expected dtypes."
    name : str
        The name of the parent function. Used for user clarity and reference in error-handling.

    Returns
    -------
    None
        Nothing! Simply a type check given inputs.

    """
    # Data sanitation for this function itself
    if(len(val_and_inputs) != len(expected)):
        raise ValueError(f'> Mismatched expected and received dicts during data_type_sanitation for {name}.')
    if not (type(val_and_inputs) == dict and type(expected) == dict and type(name) == str):
        raise ValueError(
            f'> One of the data type sanitations was unsucessful due to unexpected base inputs. Guess {str(name)}.')

    mismatched = {}
    try:
        for k, v in val_and_inputs.items():
            if type(v) not in expected.get(k):
                if(expected.get(k) is not None):
                    print(v)
                    mismatched[k] = {'recieved': type(v), 'expected': expected.get(k)}
    except Exception as e:
        print(f'>> WARNING: Exception (likely edge case) ignored in data_type_sanitation for {name}')
        print('>>\t' + str(e)[:50] + '...')

    if(len(mismatched) > 0):
        raise ValueError(
            f'> Invalid input for following pairs during data_type_sanitation for {name}.\n{mismatched}')


@_context_managers.utility
def util_data_range_sanitation(val_and_inputs, expected, name):
    """Generic performance of data sanitation. Specifically cross-checks expected and actual data ranges.

    Parameters
    ----------
    val_and_inputs : dict
        Outlines relationship between inputted variable names and their literal values.
        That is, keys are "variable names" and values are "literal values."
    expected : dict
        Outlines relationship between expected variable names and their expected ranges (if any).
        That is, keys are "variable names" and values are "expected ranges (if literally specified)."
    name : str
        The name of the parent function. Used for user clarity and reference in error-handling.

    Returns
    -------
    None
        Nothing! Simply a range check given inputs.

    """
    # Data sanitation for this function itself
    if(len(val_and_inputs) != len(expected)):
        raise ValueError(f'> ERROR: Mismatched expected and received dicts during data_range_sanitation for {name}.')
    if not (type(val_and_inputs) == dict and type(expected) == dict and type(name) == str):
        raise ValueError(
            f'> One of the data type sanitations was unsucessful due to unexpected base inputs. Guess {str(name)}.')

    mismatched = {}
    type(h2o.cluster()) == list
    try:
        for k, v in val_and_inputs.items():
            restriction = expected.get(k)
            if(type(restriction) == tuple or type(restriction) == list):
                if(len(restriction) > 2 or len(restriction) == 1):  # Check if input is one of the specified elements
                    if v not in restriction:
                        mismatched[k] = {'recieved': v, 'expected (inclusion I)': expected.get(k)}
                elif(len(restriction) == 2):  # Treat it as a range
                    if(type(restriction[0]) == float or type(restriction[0]) == int):
                        if not (v >= restriction[0] and v <= restriction[1]):
                            mismatched[k] = {'recieved': v, 'expected (num. range)': expected.get(k)}
                    else:
                        if v not in restriction:
                            mismatched[k] = {'recieved': v, 'expected (inclusion II)': expected.get(k)}
                else:
                    raise ValueError(f'Restriction for {k} improperly set in {name} range sanitation')
            elif(restriction is None):
                pass
    except Exception:
        print(f'>> WARNING: Exception (likely edge case) ignored in data_range_sanitation for {name}')

    if(len(mismatched) > 0):
        raise ValueError(
            f'> Invalid input for following pairs during data_range_sanitation for {name}.\n{mismatched}')


@_context_managers.analytics
def data_refinement(data, groupby, dropcols, responder, FOLD_COLUMN=FOLD_COLUMN):
    _provided_args = locals().copy()

    @_context_managers.utility
    def util_conditional_drop(data_frame, tbd_list):
        """Drops the specified column(s) from the H2O Frame if it exists in the Frame.

        Parameters
        ----------
        data_frame : h2o.frame.H2OFrame
            The H20 Frame to be traversed through and potentially modified.
        tbd_list : list (of str) OR any iterable (of str)
            The column names which should be conditionally dropped.

        Returns
        -------
        h2o.frame.H2OFrame
            The (potentially) modified H20 Frame.

        """
        """DATA SANITATION"""
        _provided_args = locals()
        name = sys._getframe(0).f_code.co_name
        _expected_type_args = {'data_frame': None,
                               'tbd_list': [list]}
        _expected_value_args = {'data_frame': None,
                                'tbd_list': None}
        util_data_type_sanitation(_provided_args, _expected_type_args, name)
        util_data_range_sanitation(_provided_args, _expected_value_args, name)
        """END OF DATA SANITATION"""

        for tb_dropped in tbd_list:
            if(tb_dropped in data_frame.columns):
                data_frame = data_frame.drop(tb_dropped, axis=1)
                print(Fore.GREEN + '> STATUS: {} dropped.'.format(tb_dropped) + Style.RESET_ALL)
            else:
                print(Fore.GREEN + '> STATUS: {} not in frame, skipping.'.format(tb_dropped) + Style.RESET_ALL)
        return data_frame
    """Return minimally modified H2OFrame, all possible categorical filtering options, and predictor variables.

    Parameters
    ----------
    data : h2o.frame.H2OFrame
        The H20 dataset used for model creation.
    groupby : str
        The name of the feature to determine groupings for.
    dropcols : list (of str)
        The names of the features of which to drop.
    responder : str
        The name of the dependent variable (which one will be predicted).
    FOLD_COLUMN : str
        The fold column used for target encoding (optional)

    Returns
    -------
    h2o.frame.H2OFrame, list (of str), list (of str)
        `data` -> The H2O frame with a couple dropped columns.
        `groupby_options` -> The list of filtering options in the upcoming experiment.
        `predictors` -> The list of predictors to be used in the model
                        (only for aesthetics, not inputted to actual experiment).

    """
    """DATA SANITATION"""
    name = sys._getframe(0).f_code.co_name
    _expected_type_args = {'data': None,
                           'groupby': [str],
                           'dropcols': [list],
                           'responder': [str],
                           'FOLD_COLUMN': [str, type(None)]}
    _expected_value_args = {'data': None,
                            'groupby': None,
                            'dropcols': None,
                            'responder': ['PRO_Alloc_Oil', 'PRO_Adj_Alloc_Oil', 'PRO_Total_Fluid'],
                            'FOLD_COLUMN': None}
    util_data_type_sanitation(_provided_args, _expected_type_args, name)
    util_data_range_sanitation(_provided_args, _expected_value_args, name)
    """END OF DATA SANITATION"""

    # Drop the specified columns before proceeding with the experiment
    data = util_conditional_drop(data, dropcols)

    # Determine all possible groups to filter the source data for when conducting the experiment
    print(data)
    print(data.describe())
    groupby_options = data.as_data_frame()[groupby].unique()

    # Warns user that certain categorical features will be auto-encoded by H2O if not dropped
    categorical_names = list(data.as_data_frame().select_dtypes(object).columns)
    if(len(categorical_names) > 0):
        print(Fore.LIGHTRED_EX +
              '> WARNING: {encoded} will be encoded by H2O model unless processed out.'.format(
                  encoded=categorical_names)
              + Style.RESET_ALL)

    # Determines the predictors which will be used in running the experiment. Based on configuration.
    # NOTE: The model predictors a should only use the target encoded versions, and not the older versions
    predictors = [col for col in data.columns
                  if col not in [FOLD_COLUMN] + [responder] + [col.replace('_te', '')
                                                               for col in data.columns if '_te' in col]]

    print(Fore.GREEN + 'STATUS: Experiment hyperparameters and data configured.' + Style.RESET_ALL)

    return data, groupby_options, predictors


@_context_managers.analytics
def run_experiment(data, groupby_options, responder, validation_frames_dict, weighting,
                   MAX_EXP_RUNTIME, EVAL_METRIC=EVAL_METRIC, RANK_METRIC=RANK_METRIC,
                   RANDOM_SEED=RANDOM_SEED, WEIGHTS_COLUMN=WEIGHTS_COLUMN, CV_FOLDS=CV_FOLDS,
                   STOPPING_ROUNDS=STOPPING_ROUNDS, EXPLOIT_RATIO=EXPLOIT_RATIO, MODELING_PLAN=MODELING_PLAN,
                   TRAINING_VERBOSITY=TRAINING_VERBOSITY, ENGINEER_FEATURES=True):
    """Runs an H2O Experiment given specific configurations.


    Parameters
    ----------
    data : h2o.frame.H2OFrame
        The data which will be used to train/test the models
    groupby_options : list
        The list of filtering options in the upcoming experiment.
    responder : str
        The name of the dependent variable (which one will be predicted).
    MAX_EXP_RUNTIME : int
        The maximum runtime in seconds that you want to allot in order to complete the model.
    EVAL_METRIC : str
        This option specifies the metric to consider when early stopping is specified.
    RANK_METRIC : str
        This option specifies the metric used to sort the Leaderboard by at the end of an AutoML run.
    RANDOM_SEED : int
        Random seed for reproducibility. There are caveats.
    WEIGHTS_COLUMN : str or None
        The name of the column in the H2O Frame that has the row-wise weights
        Note that this is different from an offset column (which is more corrective and bias-introducing)
    CV_FOLDS : int
        The number of fold for cross validation in training each model.
    STOPPING_ROUNDS : int
        The number of rounds which should pass depends on `EVAL_METRIC` aka `stopping_metric` parameter.
    EXPLOIT_RATIO : float
        The "budget ratio" dedicated to exploiting vs. exploring for fine-tuning XGBoost and GBM. **Experimental**
    MODELING_PLAN : list (of list !!!!!) OR None
        A specific modeling sequence to follow when running the experiments.
    TRAINING_VERBOSITY : str OR None
        The verbosity for which the experiment/model-training process is described.

    Returns
    -------
    list
        The names of the projects which were completed

    """
    # """DATA SANITATION"""
    # _provided_args = locals()
    # name = sys._getframe(0).f_code.co_name
    # _expected_type_args = {'data': None,
    #                        'groupby_options': [str],
    #                        'responder': [dict],
    #                        'validation_frames_dict': [dict, type(None)],
    #                        'MAX_EXP_RUNTIME': [dict],
    #                        'EVAL_METRIC': [list, type(None)],
    #                        'RANK_METRIC': [list, type(None)],
    #                        'RANDOM_SEED': [int, float],
    #                        'WEIGHTS_COLUMN': [str, type(None)],
    #                        'CV_FOLDS': [int],
    #                        'STOPPING_ROUNDS': [int],
    #                        'EXPLOIT_RATIO': [float],
    #                        'MODELING_PLAN': [list, type(None)],
    #                        'TRAINING_VERBOSITY': [str, type(None)]}
    # _expected_value_args = {'data': None,
    #                         'groupby_options': None,
    #                         'responder': None,
    #                         'validation_frames_dict': None,
    #                         'MAX_EXP_RUNTIME': [1, 10000],
    #                         'EVAL_METRIC': None,
    #                         'RANK_METRIC': None,
    #                         'RANDOM_SEED': [0, np.inf],
    #                         'WEIGHTS_COLUMN': None,
    #                         'CV_FOLDS': list(range(1, 100 + 1)),
    #                         'STOPPING_ROUNDS': list(range(1, 10 + 1)),
    #                         'EXPLOIT_RATIO': [0.0, 1.0],
    #                         'MODELING_PLAN': None,
    #                         'TRAINING_VERBOSITY': None}
    # util_data_type_sanitation(_provided_args, _expected_type_args, name)
    # util_data_range_sanitation(_provided_args, _expected_value_args, name)
    # """END OF DATA SANITATION"""

    try:
        # Determined the categorical variable to be dropped in TRAINING SET (should only be the groupby)
        tb_dropped = data.as_data_frame().select_dtypes(object).columns
        if not (len(tb_dropped) == 1):
            raise RuntimeError('Only and exactly one categorical variable was expected in the provided TRAIN data.' +
                               'However, ' + f'{len(tb_dropped)} were provided.')
        else:
            tb_dropped = tb_dropped[0]

        # Run all the H2O experiments for all the different groupby_options. Store in unique project name.
        initialized_projects = []

        if(ENGINEER_FEATURES):
            _accessories._print('No validation frame will be supplied to H20 since ' +
                                'mathematical features are dynamically engineered.')
            MODEL_DATASETS, CORE_FEATURES = _FEATENG_MATH(data.as_data_frame(), RESPONDER=responder)

        for group in groupby_options:
            print(Fore.GREEN + 'STATUS: Experiment -> Production Pad {}\n'.format(group) + Style.RESET_ALL)

            if(validation_frames_dict is not None and ENGINEER_FEATURES is False):
                validation_frame_groupby = validation_frames_dict[group]
                # Determined the categorical variable to be dropped in VALIDATION SET (should only be the groupby)
                tb_dropped_val = validation_frame_groupby.as_data_frame().select_dtypes(object).columns
                if not (len(tb_dropped_val) == 1):
                    raise RuntimeError('Only and exactly one categorical variable was expected in the provided VALID. data.' +
                                       'However, ' + f'{len(tb_dropped_val)} were provided.')
                else:
                    tb_dropped_val = tb_dropped_val[0]
                validation_frame_groupby = validation_frame_groupby.drop(tb_dropped_val, axis=1)
            else:
                validation_frame_groupby = None

            # Configure the experiment
            project_name = "IPC_MacroPadModeling__{RESP}__{PPAD}".format(RESP=responder, PPAD=group)
            aml_obj = H2OAutoML(max_runtime_secs=MAX_EXP_RUNTIME,     # How long should the experiment run for?
                                stopping_metric=EVAL_METRIC,          # The evaluation metric to discontinue model training
                                sort_metric=RANK_METRIC,              # Leaderboard ranking metric after all trainings
                                seed=RANDOM_SEED,
                                nfolds=CV_FOLDS,                      # Number of fold in cross-validation
                                stopping_rounds=STOPPING_ROUNDS,      # Rounds after which training stops w/o convergence
                                exploitation_ratio=EXPLOIT_RATIO,     # What budget proportion is set for exploitation?
                                modeling_plan=MODELING_PLAN,          # What is modeling order/plan to follow?
                                verbosity=TRAINING_VERBOSITY,         # What is the verbosity of training process?
                                project_name=project_name)
            initialized_projects.append(project_name)

            if(ENGINEER_FEATURES):
                validation_frame_groupby = None
                MODEL_DATA = MODEL_DATASETS[group][0].infer_objects()
                MODEL_DATA = h2o.H2OFrame(MODEL_DATA)
            # Filter the complete, provided source data to only include info for the current group
            MODEL_DATA = data[data[tb_dropped] == group]
            MODEL_DATA = MODEL_DATA.drop(tb_dropped, axis=1)

            if(weighting is False):
                WEIGHTS_COLUMN = None

            aml_obj.train(y=responder,                                # A single responder
                          weights_column=WEIGHTS_COLUMN,              # What is the weights column in the H2O frame?
                          training_frame=MODEL_DATA,                  # All the data is used for training, cross-validation
                          validation_frame=validation_frame_groupby)  # The validation dataset used to assess performance

        print(Fore.GREEN + 'STATUS: Completed experiments\n\n' + Style.RESET_ALL)
    except Exception as e:
        _accessories._print('EXCEPTION REACHED IN RUN_EXPERIMENT!', color='YELLOW')
        _accessories._print(str(e), color='YELLOW')
        # traceback.format_exception()
        return

    return initialized_projects


# @_context_managers.analytics
# def varimps(project_names):
#     _provided_args = locals().copy()
#
#     @_context_managers.server
#     def get_aml_objects(project_names):
#         objs = []
#         for project_name in project_names:
#             objs.append(h2o.automl.get_automl(project_name))
#         return objs
#
#     """Determines variable importances for all models in an experiment.
#
#     Parameters
#     ----------
#     project_names : id
#         The H2O AutoML experiment configuration.
#
#     Returns
#     -------
#     pandas.core.frame.DataFrame, list (of tuples)
#         A concatenated DataFrame with all the model's variable importances for all the input features.
#         [Optional] A list of the models that did not have a variable importance. Format: (name, model object).
#
#     """
#     """DATA SANITATION"""
#     name = sys._getframe(0).f_code.co_name
#     _expected_type_args = {'aml_obj': [h2o.automl.autoh2o.H2OAutoML]}
#     _expected_value_args = {'aml_obj': None}
#     util_data_type_sanitation(_provided_args, _expected_type_args, name)
#     util_data_range_sanitation(_provided_args, _expected_value_args, name)
#     """END OF DATA SANITATION"""
#
#     aml_objects = get_aml_objects(project_names=project_names)
#
#     variable_importances = []
#     for aml_obj in aml_objects:
#         responder, group = project_names[aml_objects.index(aml_obj)].split('__')[-2:]
#         tag = [group, responder]
#         tag_name = ['group', 'responder']
#
#         cumulative_varimps = []
#         model_novarimps = []
#         exp_leaderboard = aml_obj.leaderboard
#         # Retrieve all the model objects from the given experiment
#         exp_models = [h2o.get_model(exp_leaderboard[m_num, "model_id"]) for m_num in range(exp_leaderboard.shape[0])]
#         for model in exp_models:
#             model_name = model.params['model_id']['actual']['name']
#             variable_importance = model.varimp(use_pandas=True)
#
#             # Only conduct variable importance dataset manipulation if ranking data is available
#             # > (eg. unavailable, stacked)
#             if(variable_importance is not None):
#                 variable_importance = pd.pivot_table(variable_importance,
#                                                      values='scaled_importance',
#                                                      columns='variable').reset_index(drop=True)
#                 variable_importance['model_name'] = model_name
#                 variable_importance['model_object'] = model
#                 if(tag is not None and tag_name is not None):
#                     for i in range(len(tag)):
#                         variable_importance[tag_name[i]] = tag[i]
#                 variable_importance.index = variable_importance['model_name'] + \
#                     '___GROUP-' + variable_importance['group']
#                 variable_importance.drop('model_name', axis=1, inplace=True)
#                 variable_importance.columns.name = None
#
#                 cumulative_varimps.append(variable_importance)
#             else:
#                 # print('> WARNING: Variable importances unavailable for {MDL}'.format(MDL=model_name))
#                 model_novarimps.append((model_name, model))
#
#         varimp_all_models_in_run = pd.concat(cumulative_varimps)
#         variable_importances.append(varimp_all_models_in_run)  # , model_novarimps
#
#     requested_varimps = pd.concat(variable_importances)
#
#     print(Fore.GREEN +
#           '> STATUS: Determined variable importances of all models in given experiments: {}.'.format(project_names) +
#           Style.RESET_ALL)
#
#     return requested_varimps


@_context_managers.analytics
def model_performance(project_names_pad, adj_factor, validation_data_dict, RUN_TAG,
                      sort_by='RMSE', RESPONDER='PRO_Total_Fluid'):
    """Determine the model performance through a series of predefined metrics.

    Parameters
    ----------
    project_names_pad : list of str
        The names of the experiments/project run
    adj_factor : dict
        The benchlines for each grouping.
    sort_by : str
        The performance metric which should be used to sort the performance data in descending order.

    Returns
    -------
    pandas.core.frame.DataFrame
        `perf_data` -> The DataFrame with performance information for all of the inputted models.
                       Specifically, R^2, R (optionally), MSE, RMSE, RMSLE, and MAE.

    """
    # """DATA SANITATION"""
    # _provided_args = locals()
    # name = sys._getframe(0).f_code.co_name
    # _expected_type_args = {'project_names_pad': [list],
    #                        'adj_factor': [dict],
    #                        'sort_by': [str]}
    # _expected_value_args = {'project_names_pad': None,
    #                         'adj_factor': None,
    #                         'sort_by': ['R^2', 'MSE', 'RMSE', 'Rel. RMSE', 'RMSLE', 'MAE']}
    # util_data_type_sanitation(_provided_args, _expected_type_args, name)
    # util_data_range_sanitation(_provided_args, _expected_value_args, name)
    # """END OF DATA SANITATION"""

    perf_data = {}
    exp_objs = zip(project_names_pad, get_aml_objects(project_names_pad))

    for project_name, exp_obj in exp_objs:
        group = project_name.split('__')[-1]
        # Calculate validation RMSE
        validation_data = validation_data_dict.get(group)

        model_ids = exp_obj.leaderboard.as_data_frame()['model_id']
        for model_name in model_ids:
            model_obj = h2o.get_model(model_name)

            prediction_on_validation = model_obj.predict(validation_data).as_data_frame().infer_objects()
            val_rmse = np.sqrt(np.mean((validation_data.as_data_frame()[RESPONDER] -
                                        prediction_on_validation['predict'])**2))
            rel_val_rmse = val_rmse - adj_factor.get(group)

            perf_data[model_name] = {}
            # perf_data[model_name]['R^2'] = model_obj.r2()
            # perf_data[model_name]['R'] = model_obj.r2() ** 0.5
            # perf_data[model_name]['MSE'] = model_obj.mse()
            perf_data[model_name]['RMSE'] = model_obj.rmse()
            perf_data[model_name]['Rel. RMSE'] = model_obj.rmse() - adj_factor.get(group)
            perf_data[model_name]['Val. RMSE'] = val_rmse
            perf_data[model_name]['Rel. Val. RMSE'] = rel_val_rmse
            # if(model_obj.rmse(valid=True) is not None):
            #     perf_data[model_name]['Rel. Val. RMSE'] = rel_val_rmse
            # else:
            #     perf_data[model_name]['Rel. Val. RMSE'] = None
            perf_data[model_name]['group'] = group
            perf_data[model_name]['model_obj'] = model_obj
            # perf_data[model_name]['RMSLE'] = model_obj.rmsle()
            # perf_data[model_name]['MAE'] = model_obj.mae()

    # Structure model output and order
    perf_data = pd.DataFrame(perf_data).T.sort_values(sort_by, ascending=False).infer_objects()
    perf_data['tolerated RMSE'] = perf_data['group'].apply(lambda x: adj_factor.get(x))

    for model_obj in perf_data.sort_values(['RMSE', 'Rel. Val. RMSE'])['model_obj'][:10]:
        mpath = f'Modeling Reference Files/Round {RUN_TAG}/Models/'
        with _accessories.suppress_stdout():
            _accessories.auto_make_path(mpath)
        model_path = h2o.save_model(model=model_obj, path=f'Modeling Reference Files/Round {RUN_TAG}/Models',
                                    force=True)
        _accessories._print('MODEL PATH: ' + model_path)
        perf_data.at[model_name, 'model_obj'] = model_path
    perf_data['model_obj'] = perf_data['model_obj'].apply(lambda x: 'Not available' if type(x) != str else x)

    return perf_data


# @_context_managers.representation
# def varimp_heatmap(final_cumulative_varimps, FPATH, highlight=True,
#                    preferred='Steam', preferred_importance=0.7, mean_importance_threshold=0.7,
#                    top_color='green', chosen_color='cyan', RUN_TAG=RUN_TAG, FIGSIZE=(10, 55), annot=False,
#                    TOP_MODELS=TOP_MODELS):
#     """Plots a heatmap based on the inputted variable importance data.
#
#     Parameters
#     ----------
#     final_cumulative_varimps : pandas.core.frame.DataFrame
#         Data on each models variable importances, the model itself, and other identifying tags (like group_type)
#         Only numerical features are used to make the heatmap though.
#     FPATH : str
#         The file path where the heatmap should be saved.
#     highlight : bool
#         Whether or not Rectangles should be patched to show steam-preferred models and top models generated.
#     preferred : str
#         The name of the predictor that is preferred for the model to rank higher (for explainability).
#     preferred_importance : float
#         The relative, threshold value for which the `preferred` predictor should be at or above.
#         Used for filtering variable importances and identifying `ranked steam`.
#     mean_importance_threshold : float
#         The relative, average of each variable importance row (per model) should be at or above this value.
#         Used for filtering variable importances and identifying `ranked steam`.
#     top_color : str
#         The color the top model(s) should be. Specifically the Rectangle patched on.
#         Corresponds to `ranked_names`.
#     chosen_color : str
#         The color the high-steam model(s) should be. Specifically the Rectangle patched on.
#         Corresponds to `ranked_steam`.
#     RUN_TAG : int
#         The identifying ID/KEY for the global configuration.
#     FIGSIZE : tuple (of ints)
#         The size of the ouputted heatmap figure. (width x length)
#     annot : bool
#         Whether the heatmap should be annotated (aka variable importances should be labelled).
#         This should be turned off for very dense plots (lots of rows) or for minimzing runtime.
#     TOP_MODELS : int
#         The top n number of models which should be implicitly filtered.
#         Corresponds to `ranked_names`.
#
#     Returns
#     -------
#     list (of str), list (of str) OR None, None
#         `ranked_names` -> The indices of the models which are ranked according to
#                           `TOP_MODELS` and the predictor rank.
#         `ranked_steam` -> The indices of the models which are ranked according to
#                           `preferred` and `preferred_importance`.
#
#         NOTE RETURN DEPENDENCY: if `highlight` is False then both `ranked_names` and `ranked_steam` are None.
#
#     """
#     """DATA SANITATION"""
#     _provided_args = locals()
#     name = sys._getframe(0).f_code.co_name
#     _expected_type_args = {'final_cumulative_varimps': [pd.core.frame.DataFrame],
#                            'FPATH': [str],
#                            'highlight': [bool],
#                            'preferred': [str],
#                            'preferred_importance': [float],
#                            'mean_importance_threshold': [float],
#                            'top_color': [str],
#                            'chosen_color': [str],
#                            'RUN_TAG': [int],
#                            'FIGSIZE': [tuple],
#                            'annot': [bool],
#                            'TOP_MODELS': [int]}
#     _expected_value_args = {'final_cumulative_varimps': None,
#                             'FPATH': None,
#                             'highlight': [True, False],
#                             'preferred': None,
#                             'preferred_importance': [0.0, 1.0],
#                             'mean_importance_threshold': [0.0, 1.0],
#                             'top_color': None,
#                             'chosen_color': None,
#                             'RUN_TAG': None,
#                             'FIGSIZE': None,
#                             'annot': [True, False],
#                             'TOP_MODELS': [0, np.inf]}
#     util_data_type_sanitation(_provided_args, _expected_type_args, name)
#     util_data_range_sanitation(_provided_args, _expected_value_args, name)
#     """END OF DATA SANITATION"""
#
#     new = len(final_cumulative_varimps) / CELL_RATIO
#     if(new > 100):
#         print(f'> WARNING: The height of this figure is {int(new)} units. Render may take significant time...')
#     FIGSIZE = (FIGSIZE[0], new)
#
#     # Plot heatmap of variable importances across all model combinations
#     fig, ax = plt.subplots(figsize=FIGSIZE)
#     predictor_rank = final_cumulative_varimps.mean(axis=0).sort_values(ascending=False)
#     sns_fig = sns.heatmap(final_cumulative_varimps[predictor_rank.keys()], annot=annot, annot_kws={"size": 4})
#
#     if(highlight):
#         temp = final_cumulative_varimps[predictor_rank.keys()]
#
#         ranked_names = list(temp.index)[-TOP_MODELS:]
#         rect_width = len(predictor_rank)
#         bottom_y_loc: Final = len(temp) - 1
#         # For the top 10 models
#         for y_loc in range(TOP_MODELS + 1):
#             sns_fig.add_patch(Rectangle(xy=(0, bottom_y_loc - y_loc), width=rect_width, height=1,
#                                         edgecolor=top_color, fill=False, lw=3))
#
#         # For the models where steam is selected as the most important
#         ranked_steam = list(temp[(temp[preferred] >= preferred_importance) &
#                                  (temp.mean(axis=1) > mean_importance_threshold)].index)
#         y_locs = [list(range(len(temp))).index(list(temp.index).index(rstm)) for rstm in ranked_steam]
#         stm_loc = list(predictor_rank.index).index(preferred)
#         for y_loc in y_locs:
#             sns_fig.add_patch(Rectangle(xy=(stm_loc,  y_loc), width=1, height=1,
#                                         edgecolor=chosen_color, fill=False, lw=3))
#     else:
#         ranked_names = ranked_steam = None
#
#     sns_fig.get_figure().savefig(FPATH,
#                                  bbox_inches='tight')
#     plt.clf()
#
#     return ranked_names, ranked_steam


# @_context_managers.representation
# def correlation_matrix(df, FPATH, EXP_NAME, abs_arg=True, mask=True, annot=False,
#                        type_corrs=['Pearson', 'Kendall', 'Spearman'],
#                        cmap=sns.color_palette('flare', as_cmap=True), figsize=(24, 8), contrast_factor=1.0):
#     """Outputs to console and saves to file correlation matrices given data with input features.
#     Intended to represent the parameter space and different relationships within. Tool for modeling.
#
#     Parameters
#     ----------
#     df : pandas.core.frame.DataFrame
#         Input dataframe where each column is a feature that is to be correlated.
#     FPATH : str
#         Where the file should be saved. If directory is provided, it should already be created.
#     abs_arg : bool
#         Whether the absolute value of the correlations should be plotted (for strength magnitude).
#         Impacts cmap, switches to diverging instead of sequential.
#     mask : bool
#         Whether only the bottom left triange of the matrix should be rendered.
#     annot : bool
#         Whether each cell should be annotated with its numerical value.
#     type_corrs : list
#         All the different correlations that should be provided. Default to all built-in pandas options for df.corr().
#     cmap : child of matplotlib.colors
#         The color map which should be used for the heatmap. Dependent on abs_arg.
#     figsize : tuple
#         The size of the outputted/saved heatmap (width, height).
#     contrast_factor:
#         The factor/exponent by which all the correlationed should be raised to the power of.
#         Purpose is for high-contrast, better identification of highly correlated features.
#
#     Returns
#     -------
#     pandas.core.frame.DataFrame
#         The correlations given the input data, and other transformation arguments (abs_arg, contrast_factor)
#         Furthermore, heatmap is saved in specifid format and printed to console.
#
#     """
#     """DATA SANITATION"""
#     _provided_args = locals()
#     name = sys._getframe(0).f_code.co_name
#     _expected_type_args = {'df': [pd.core.frame.DataFrame],
#                            'FPATH': [str],
#                            'EXP_NAME': [str],
#                            'abs_arg': [bool],
#                            'mask': [bool],
#                            'annot': [bool],
#                            'type_corrs': [list],
#                            'cmap': [matplotlib.colors.ListedColormap, matplotlib.colors.LinearSegmentedColormap],
#                            'figsize': [tuple],
#                            'contrast_factor': [float]}
#     _expected_value_args = {'df': None,
#                             'FPATH': None,
#                             'EXP_PATH': None,
#                             'abs_arg': [True, False],
#                             'mask': [True, False],
#                             'annot': [True, False],
#                             'type_corrs': None,
#                             'cmap': None,
#                             'figsize': None,
#                             'contrast_factor': [0.0000001, np.inf]}
#     util_data_type_sanitation(_provided_args, _expected_type_args, name)
#     util_data_range_sanitation(_provided_args, _expected_value_args, name)
#     """END OF DATA SANITATION"""
#
#     input_data = {}
#
#     # NOTE: Conditional assignment of sns.heatmap based on parameters
#     # > Mask sns.heatmap parameter is conditionally controlled
#     # > If absolute value is not chosen by the user, switch to divergent heatmap. Otherwise keep it as sequential.
#     fig, ax = plt.subplots(ncols=len(type_corrs), sharey=True, figsize=figsize)
#     for typec in type_corrs:
#         input_data[typec] = (df.corr(typec.lower()).abs()**contrast_factor if abs_arg
#                              else df.corr(typec.lower())**contrast_factor)
#         sns_fig = sns.heatmap(input_data[typec],
#                               mask=np.triu(df.corr().abs()) if mask else None,
#                               ax=ax[type_corrs.index(typec)],
#                               annot=annot,
#                               cmap=cmap if abs_arg else sns.diverging_palette(240, 10, n=9, as_cmap=True)
#                               ).set_title("{cortype}'s Correlation Matrix\n{EXP_NAME}".format(cortype=typec,
#                                                                                               EXP_NAME=EXP_NAME))
#     plt.tight_layout()
#     sns_fig.get_figure().savefig(FPATH, bbox_inches='tight')
#
#     plt.clf()
#
#     return input_data


# @_context_managers.representation
# def plot_model_performance(perf_data, FPATH, mcmaps, centers, ranked_names, ranked_steam, RUN_TAG, extrema_thresh_pct=5,
#                            annot=True, annot_size=4, highlight=True, FIGSIZE=(10, 50)):
#     """Plots a heatmap based on the inputted model performance data.
#
#     Parameters
#     ----------
#     perf_data : pandas.core.frame.DataFrame
#         The DataFrame with performance information for all of the inputted models.
#         Specifically, R^2, R (optionally), MSE, RMSE, RMSLE, and MAE.
#     FPATH : str
#         The file path where the heatmap should be saved.
#     mcmaps : dict
#         Different heatmap/AxesPlot color schemes dependent on performance metric.
#         Stands for "macro color maps."
#     centers : dict
#         Description of parameter `centers`.
#     ranked_names : list (of ints) OR None
#         The indices of the models which are ranked according to `TOP_MODELS` and the predictor rank.
#         `TOP_MODELS` and the predictor rank may be found in `plot_varimp_heatmap`.
#         NOTE DEPENDENCY: If `highlight` is False in `plot_varimp_heatmap`, then it MUST be locally False here.
#                          If it is True in THIS function, it will raise an Exception since control doesn't know which
#                          models to highlight, including for mean-ranks. (determined in `plot_varimp_heatmap`).
#     ranked_steam : list (of ints) OR None
#         The indices of the models which are ranked according to `preferred` and `preferred_importance`.
#         `preferred` and `preferred_importance` may be found in `plot_varimp_heatmap`.
#         NOTE DEPENDENCY: If `highlight` is False in `plot_varimp_heatmap`, then it MUST be locally False here.
#                          If it is True in THIS function, it will raise an Exception since control doesn't know which
#                          models to highlight, including for steam-ranks. (determined in `plot_varimp_heatmap`).
#     extrema_thresh_pct : int
#         The top and botton n percentage points to set as the maximum and minimum values for the heatmap colors.
#         Stands for "extrema threshold percentage."
#     RUN_TAG : int
#         The identifying ID/KEY for the global configuration.
#     annot : bool
#         Whether the heatmap should be annotated (aka variable importances should be labelled).
#         This should be turned off for very dense plots (lots of rows) or for minimzing runtime.
#     annot_size : int
#         The font size of the annotation. Only works if `annot` is not False.
#     highlight : bool
#         Whether or not Rectangles should be patched to show steam-preferred models and top models generated.
#     FIGSIZE : tuple (of ints)
#         The size of the ouputted heatmap figure. (width x length)
#
#     Returns
#     -------
#     None
#         Nothing! Simply print out a heatmap.
#         There are no new "ranking indices" like `ranked_steam` or `ranked_names` to be determined, anyways.
#
#     """
#     """DATA SANITATION"""
#     _provided_args = locals()
#     name = sys._getframe(0).f_code.co_name
#     _expected_type_args = {'perf_data': [pd.core.frame.DataFrame],
#                            'FPATH': [str],
#                            'mcmaps': [dict],
#                            'centers': [dict],
#                            'ranked_names': [list, type(None)],
#                            'ranked_steam': [list, type(None)],
#                            'extrema_thresh_pct': [int, float],
#                            'RUN_TAG': [int],
#                            'annot': [bool],
#                            'annot_size': [int],
#                            'highlight': [bool],
#                            'FIGSIZE': [tuple]}
#     _expected_value_args = {'perf_data': None,
#                             'FPATH': None,
#                             'mcmaps': None,
#                             'centers': None,
#                             'ranked_names': None,
#                             'ranked_steam': None,
#                             'extrema_thresh_pct': (0, 99),
#                             'RUN_TAG': None,
#                             'annot': None,
#                             'annot_size': (0, np.inf),
#                             'highlight': None,
#                             'FIGSIZE': None}
#     util_data_type_sanitation(_provided_args, _expected_type_args, name)
#     util_data_range_sanitation(_provided_args, _expected_value_args, name)
#     """END OF DATA SANITATION"""
#
#     new = len(perf_data) / CELL_RATIO
#     if(new > 100):
#         print(f'> WARNING: The height of this figure is {int(new)} units. Render may take significant time...')
#     if(FIGSIZE[1] >= new):
#         pass
#     else:
#         FIGSIZE = (FIGSIZE[0], new)
#
#     # CUSTOM DATA SANITATION
#     if not (len(perf_data.select_dtypes(float).columns) <= len(perf_data.columns)):
#         cat_var = list(perf_data.select_dtypes(object).columns)
#         raise ValueError('> ERROR: Inputted data has categorical variables. {}'.format(cat_var))
#
#     fig, ax = plt.subplots(figsize=FIGSIZE, ncols=len(perf_data.columns), sharey=True)
#     for col in perf_data.columns:
#         cmap_local = mcmaps.get(col)
#         center_local = centers.get(col)
#         vmax_local = np.percentile(perf_data[col].dropna(), 100 - extrema_thresh_pct)
#         vmin_local = np.percentile(perf_data[col].dropna(), extrema_thresh_pct)
#
#         if not (vmin_local < vmax_local):
#             raise ValueError('> ERROR: Heatmap min, center, max have incompatible values {}-{}-{}'.format(vmin_local,
#                                                                                                           center_local,
#                                                                                                           vmax_local))
#
#         sns_fig = sns.heatmap(perf_data[[col]], ax=ax[list(perf_data.columns).index(col)],
#                               annot=annot, annot_kws={"size": annot_size}, cbar=False,
#                               center=center_local, cmap=cmap_local, mask=perf_data[[col]].isnull(),
#                               vmax=vmax_local, vmin=vmin_local)
#
#         if(highlight):
#             if(ranked_names is None or ranked_steam is None):
#                 raise ValueError('STATUS: Improper ranked rect inputs.')
#             for rtop in ranked_names:
#                 y_loc = list(perf_data.index).index(rtop)
#                 sns_fig.add_patch(Rectangle(xy=(0, y_loc), width=1, height=1,
#                                             edgecolor='green', fill=False, lw=3))
#             for rstm in ranked_steam:
#                 y_loc = list(perf_data.index).index(rtop)
#                 sns_fig.add_patch(Rectangle(xy=(0, y_loc), width=1, height=1,
#                                             edgecolor='cyan', fill=False, lw=4))
#         sns.set(font_scale=0.6)
#
#     sns_fig.get_figure().savefig(FPATH, bbox_inches='tight')
#
#     print(Fore.GREEN + 'STATUS: Saved variable importance configurations.' + Style.RESET_ALL)
#

@_context_managers.representation
def validate_models(perf_data, training_data_dict, benchline, validation_data_dict, RUN_TAG, RESPONDER,
                    order_by='Rel. RMSE', TOP_MODELS=TOP_MODELS):
    FIG_SCALAR = 5.33333333333333

    perf_data.sort_values(order_by, inplace=True)
    all_model_RMSE_H_to_L = [h2o.get_model(perf_data.index[i].split('___GROUP')[0]) for i in range(len(perf_data))]
    # all_model_RMSE_H_to_L = all_model_RMSE_H_to_L[-1 * TOP_MODELS:]
    if(TOP_MODELS is None):
        all_model_RMSE_H_to_L = all_model_RMSE_H_to_L[:]
    elif(TOP_MODELS < 0):
        all_model_RMSE_H_to_L = all_model_RMSE_H_to_L[TOP_MODELS:]
    else:
        all_model_RMSE_H_to_L = all_model_RMSE_H_to_L[:TOP_MODELS]

    demonstrations = ['Scoring History',
                      'Training Correlations', 'Testing Correlations',
                      'Training TS', 'Testing TS', 'Residual Plot']

    if(len(all_model_RMSE_H_to_L) * FIG_SCALAR > 150):
        print(f'WARNING: Figure length is: {int(len(all_model_RMSE_H_to_L) * FIG_SCALAR)}')

    fig, ax = plt.subplots(nrows=len(all_model_RMSE_H_to_L),
                           ncols=len(demonstrations),
                           figsize=(50, len(all_model_RMSE_H_to_L) * FIG_SCALAR))

    models_iterated = []
    for top_model in all_model_RMSE_H_to_L:
        # Extracting Model Configurations
        model_name = [c for c in list(perf_data.index) if top_model.key in c][0]
        model_group = perf_data[perf_data.index == model_name]['group'].values[0]
        _accessories._print(f'Accesssing configurations and testing model {model_name} for group {model_group}')
        model_type = top_model.algo
        model_position = ax[all_model_RMSE_H_to_L.index(top_model)]
        if(len(all_model_RMSE_H_to_L) == 1):
            raise ValueError('Only one model was created!!')
        models_iterated.append(model_name)
        RMSE_resp = f'{order_by}: ' + str(int(perf_data[perf_data.index == model_name][order_by][0]))

        training_data = training_data_dict.get(model_group)
        validation_data = validation_data_dict.get(model_group)

        # print(model_group)

        # Determining training and validation predictions
        prediction_on_training = top_model.predict(training_data).as_data_frame().infer_objects()
        prediction_on_validation = top_model.predict(validation_data).as_data_frame().infer_objects()

        # Loading existing objects
        pd_data = training_data.as_data_frame().infer_objects()
        pd_data_validation = validation_data.as_data_frame().infer_objects()

        val_accuracy = (pd_data_validation[RESPONDER] - prediction_on_validation['predict'])
        val_rmse = np.sqrt(np.mean((pd_data_validation[RESPONDER] - prediction_on_validation['predict'])**2))
        allowable_rmse = str(int(benchline.get(model_group)))
        rel_val_rmse = str(int(val_rmse - int(allowable_rmse)))

        # IDENTIFIERS FOR MODELING
        axis = model_position[0]
        # top_model = h2o.get_model(perf_data.index[-1].split('___GROUP')[0])
        # scoring_history = top_model.scoring_history()
        # # plt.figure(figsize=(15, 11.25))
        # axis.set_title(f'Scoring history\n{model_name}')
        # # perf_plot = scoring_history['training_rmse']
        # perf_plot = [0] * len(scoring_history)
        # axis.plot(scoring_history['timestamp'], perf_plot)
        # RMSE PRESENTATIONS
        axis.text(0.5, 0.5, model_type + f" {model_group}" + '\n' + RMSE_resp + f'\nValidation {order_by}: {rel_val_rmse}' +
                  f'\nAllowable RMSE: {allowable_rmse}',
                  horizontalalignment='center', verticalalignment='center',
                  transform=axis.transAxes, fontsize=24)

        # Correlations for training
        axis = model_position[1]
        # plt.figure(figsize=(10, 10))
        axis.set_title(f'Model Predicting Training Dataset\n{model_name}')
        axis.scatter(pd_data[RESPONDER], prediction_on_training['predict'])
        # Correlations for testing
        axis = model_position[2]
        # plt.figure(figsize=(10, 10))
        axis.set_title(f'Model Predicting Holdout Dataset\n{model_name}')
        axis.scatter(pd_data_validation[RESPONDER], prediction_on_validation['predict'])

        # Time Series comparison for training
        axis = model_position[3]
        plt.figure(figsize=(30, 20))
        axis.set_title(f'Model Predicting Training Time Series\n{model_name}')
        axis.plot(pd_data[RESPONDER], linewidth=0.7)
        axis.plot(prediction_on_training['predict'], linewidth=0.7)
        # Time Series comparison for training
        axis = model_position[4]
        # plt.figure(figsize=(30, 20))
        axis.set_title(f'Model Predicting Holdout Time Series\n{model_name}')
        axis.plot(pd_data_validation[RESPONDER], linewidth=0.7)
        axis.plot(prediction_on_validation['predict'], linewidth=0.7)

        axis = model_position[5]
        axis.set_title(f'Residual Plot\n{model_name}')
        axis.plot(val_accuracy, linewidth=0.7)
        axis.plot(val_accuracy.rolling(window=7).mean().fillna(val_accuracy.mean()), linewidth=0.4)

        plt.tight_layout()
    # for ax, row in zip(ax[:, 0], models_iterated):
    #     ax.set_ylabel(row, rotation=90, size='large')
    plt.tight_layout()

    fig.savefig(f'Modeling Reference Files/Round {RUN_TAG}/model_analytics_{RUN_TAG}_top{TOP_MODELS}.pdf',
                bbox_inches='tight')


_ = """
#######################################################################################################################
#####################################################   WRAPPERS  #####################################################
#######################################################################################################################
"""


def setup_and_server(paths_to_check=[DATA_PATH_PAD, DATA_PATH_PAD_vanilla],
                     SECURED=SECURED, IP_LINK=IP_LINK, PORT=PORT, SERVER_FORCE=SERVER_FORCE):
    # Initialize the cluster
    try:
        shutdown_confirm(h2o)
    except Exception:
        pass
    h2o.init(https=SECURED,
             ip=IP_LINK,
             port=PORT,
             start_h2o=SERVER_FORCE)
    # # Check the status of the cluster, just for reference
    # process_log = snapshot(h2o.cluster(), show=False)

    # Confirm that the data path leads to an actual file
    for path in paths_to_check:
        if not (os.path.isfile(path)):
            raise ValueError('ERROR: {data} does not exist in the specificied location.'.format(data=path))


def create_validation_splits(DATA_PATH_PAD, pd_data_pad, group_colname='PRO_Pad', TRAIN_VAL_SPLIT=TRAIN_VAL_SPLIT):
    # NOTE: Global Dependencies:

    # Split into train/test (CV) and holdout set (per each class of grouping)
    # pd_data_pad = pd.read_csv(DATA_PATH_PAD_vanilla).drop('Unnamed: 0', axis=1)
    unique_pads = list(pd_data_pad[group_colname].unique())
    grouped_data_split = {}
    for u_pad in unique_pads:
        filtered_by_group = pd_data_pad[pd_data_pad[group_colname] == u_pad].sort_values('Date').reset_index(drop=True)
        data_pad_loop, data_pad_validation_loop = [dat.reset_index(drop=True).infer_objects()
                                                   for dat in np.split(filtered_by_group,
                                                                       [int(TRAIN_VAL_SPLIT *
                                                                            len(filtered_by_group))])]
        grouped_data_split[u_pad] = (data_pad_loop, data_pad_validation_loop)

    # Holdout and validation reformatting
    data_pad = pd.concat([v[0] for k, v in grouped_data_split.items()]).reset_index(drop=True).infer_objects()
    wanted_types = {k: 'real' if v == float or v == int else 'enum' for k, v in dict(data_pad.dtypes).items()}
    data_pad = h2o.H2OFrame(data_pad, column_types=wanted_types)
    data_pad_validation = pd.concat([v[1] for k, v in grouped_data_split.items()]
                                    ).reset_index(drop=True).infer_objects()
    data_pad_validation = h2o.H2OFrame(data_pad_validation, column_types=wanted_types)

    # Create pad validation relationships
    pad_relationship_validation = {}
    for u_pad in unique_pads:
        df_loop = data_pad_validation.as_data_frame()
        df_loop = df_loop[df_loop[group_colname] == u_pad].drop(
            ['Date'], axis=1).infer_objects().reset_index(drop=True)
        local_wanted_types = {k: v for k, v in wanted_types.items() if k in df_loop.columns}
        df_loop = h2o.H2OFrame(df_loop, column_types=local_wanted_types)
        pad_relationship_validation[u_pad] = df_loop
    # Create pad training relationships
    pad_relationship_training = {}
    for u_pad in unique_pads:
        df_loop = data_pad.as_data_frame()
        df_loop = df_loop[df_loop[group_colname] == u_pad].drop(
            [group_colname], axis=1).infer_objects().reset_index(drop=True)
        local_wanted_types = {k: v for k, v in wanted_types.items() if k in df_loop.columns}
        df_loop = h2o.H2OFrame(df_loop, column_types=local_wanted_types)
        pad_relationship_training[u_pad] = df_loop

    _accessories._print('STATUS: Server initialized and data imported.')
    print(OUT_BLOCK)

    return data_pad, pad_relationship_validation, pad_relationship_training


def manual_selection_and_processing(data_pad, RUN_TAG, RESPONDER='PRO_Total_Fluid',
                                    EXCLUDE=['Bin_1', 'Bin_8',
                                             'PRO_Pump_Speed', 'PRO_Alloc_Oil',
                                             'PRO_Adj_Pump_Speed', 'PRO_Adj_Alloc_Oil'],
                                    weighting=False, weights_column=WEIGHTS_COLUMN, FOLD_COLUMN=FOLD_COLUMN):
    EXCLUDE.extend(['C1', 'Date'])
    if(weighting is False):
        EXCLUDE.extend([WEIGHTS_COLUMN])
    EXCLUDE = list(set(EXCLUDE))

    data_pad, groupby_options_pad, PREDICTORS = data_refinement(data=data_pad,
                                                                groupby='PRO_Pad',
                                                                dropcols=EXCLUDE,
                                                                responder=RESPONDER,
                                                                FOLD_COLUMN=FOLD_COLUMN)

    with open(f'Modeling Reference Files/Round {RUN_TAG}/hyperparams.txt', 'a') as out:
        print(OUT_BLOCK.replace('\n', '') + OUT_BLOCK.replace('\n', ''), file=out)
        print('PREDICTORS', file=out)
        pprint(PREDICTORS, stream=out)
        print(OUT_BLOCK.replace('\n', '') + OUT_BLOCK.replace('\n', ''), file=out)
        print('RESPONDER', file=out)
        pprint(RESPONDER, stream=out)

        return data_pad, groupby_options_pad, PREDICTORS


def hyp_overview(PREDICTORS, RESPONDER, MAX_EXP_RUNTIME):
    print(Fore.GREEN + 'STATUS: Hyperparameter Overview:')
    print(Fore.GREEN + '\t* max_runtime_secs\t-> ', MAX_EXP_RUNTIME,
          Fore.GREEN + '\t\tThe maximum runtime in seconds that you want to allot in order to complete the model.')
    print(Fore.GREEN + '\t* stopping_metric\t-> ', EVAL_METRIC,
          Fore.GREEN + '\t\tThis option specifies the metric to consider when early stopping is specified')
    print(Fore.GREEN + '\t* sort_metric\t\t-> ', RANK_METRIC,
          Fore.GREEN + '\t\tThis option specifies the metric used to sort the Leaderboard by at the end of an AutoML run.')
    print(Fore.GREEN + '\t* weights_column\t-> ', WEIGHTS_COLUMN,
          Fore.GREEN + '\t\tThe name of the column with the weights')
    print(Fore.GREEN + '\t* n_folds\t\t-> ', CV_FOLDS,
          Fore.GREEN + '\t\t\tThis is the number of cross-validation folds for each model in the experiment.')
    print(Fore.GREEN + '\t* stopping_rounds\t-> ', STOPPING_ROUNDS,
          Fore.GREEN + '\t\t\tThis is the tolerated number of training rounds until `stopping_metric` stops improving.')
    print(Fore.GREEN + '\t* exploitation\t\t-> ', EXPLOIT_RATIO,
          Fore.GREEN + '\t\t\tThe "budget ratio" dedicated to exploiting vs. exploring for fine-tuning XGBoost and GBM. ' +
          '**Experimental**')
    print(Fore.GREEN + '\t* tolerance\t\t-> ', PREFERRED_TOLERANCE,
          Fore.GREEN + '\t\tThis is the two-tailed value for RMSE.')
    print(Fore.GREEN + '\t* top_models\t\t-> ', TOP_MODELS,
          Fore.GREEN + '\t\t\tThis is the top "n" models to visually filter when plotting the variable importance heatmap.')
    print(Fore.GREEN + '\t* seed\t\t\t-> ', RANDOM_SEED,
          Fore.GREEN + '\t\tRandom seed for reproducibility. There are caveats.')
    print(Fore.GREEN + '\t* Predictors\t\t-> ', PREDICTORS,
          Fore.GREEN + '\t\tThese are the variables which will be used to predict the responder.')
    print(Fore.GREEN + '\t* Responder\t\t-> ', RESPONDER,
          Fore.GREEN + '\tThis is what is being predicted.\n' + Style.RESET_ALL)


def get_benchlines(data_pad_pd, RESPONDER, grouper='PRO_Pad', PREFERRED_TOLERANCE=PREFERRED_TOLERANCE):
    benchline_pad = list(data_pad_pd[data_pad_pd[RESPONDER] > 0].groupby(
        ['Date', grouper])[RESPONDER].sum().reset_index().groupby(grouper).median().to_dict().values())[0]
    benchline_pad.update((x, y * PREFERRED_TOLERANCE) for x, y in benchline_pad.items())

    return benchline_pad


_ = """
#######################################################################################################################
##################################################   CORE EXECUTION  ##################################################
######################################################################################################################
"""

# TODO: Validation setup when feature engineering occurs
# TODO: See iPhone notes for next steps.


def _MODELING(math_eng=False, weighting=False, MAX_EXP_RUNTIME=20, plot_for_ref=False):
    RESPONDER = 'PRO_Total_Fluid'
    _accessories._print(f'DATA_PATH_PAD: {DATA_PATH_PAD}', color='LIGHTCYAN_EX')
    _accessories._print(f'DATA_PATH_PAD_vanilla: {DATA_PATH_PAD_vanilla}', color='LIGHTCYAN_EX')
    _accessories._print(f'IP_LINK: {IP_LINK}', color='LIGHTCYAN_EX')
    _accessories._print(f'SECURED: {SECURED}', color='LIGHTCYAN_EX')
    _accessories._print(f'PORT: {PORT}', color='LIGHTCYAN_EX')
    _accessories._print(f'SERVER_FORCE: {SERVER_FORCE}', color='LIGHTCYAN_EX')
    _accessories._print(f'MAX_EXP_RUNTIME: {MAX_EXP_RUNTIME}', color='LIGHTCYAN_EX')
    _accessories._print(f'RANDOM_SEED: {RANDOM_SEED}', color='LIGHTCYAN_EX')
    _accessories._print(f'EVAL_METRIC: {EVAL_METRIC}', color='LIGHTCYAN_EX')
    _accessories._print(f'RANK_METRIC: {RANK_METRIC}', color='LIGHTCYAN_EX')
    _accessories._print(f'CV_FOLDS: {CV_FOLDS}', color='LIGHTCYAN_EX')
    _accessories._print(f'STOPPING_ROUNDS: {STOPPING_ROUNDS}', color='LIGHTCYAN_EX')
    _accessories._print(f'WEIGHTS_COLUMN: {WEIGHTS_COLUMN}', color='LIGHTCYAN_EX')
    _accessories._print(f'EXPLOIT_RATIO: {EXPLOIT_RATIO}', color='LIGHTCYAN_EX')
    _accessories._print(f'MODELING_PLAN: {MODELING_PLAN}', color='LIGHTCYAN_EX')
    _accessories._print(f'TRAIN_VAL_SPLIT: {TRAIN_VAL_SPLIT}', color='LIGHTCYAN_EX')
    _accessories._print(f'FOLD_COLUMN: {FOLD_COLUMN}', color='LIGHTCYAN_EX')
    _accessories._print(f'TOP_MODELS: {TOP_MODELS}', color='LIGHTCYAN_EX')
    _accessories._print(f'PREFERRED_TOLERANCE: {PREFERRED_TOLERANCE}', color='LIGHTCYAN_EX')
    _accessories._print(f'TRAINING_VERBOSITY: {TRAINING_VERBOSITY}', color='LIGHTCYAN_EX')
    _accessories._print(f'MODEL_CMAPS: {MODEL_CMAPS}', color='LIGHTCYAN_EX')
    _accessories._print(f'HMAP_CENTERS: {HMAP_CENTERS}', color='LIGHTCYAN_EX')
    _accessories._print(f'CELL_RATIO: {CELL_RATIO}', color='LIGHTCYAN_EX')

    _accessories._print('Initializing server and checking data...')
    RUN_TAG = record_hyperparameters(MAX_EXP_RUNTIME)
    setup_and_server()

    _accessories._print('Retreiving pre-model data...')
    PATH_PAD = 'Data/combined_ipc_aggregates.csv'
    pd_data_pad = _accessories.retrieve_local_data_file(PATH_PAD)

    _accessories._print('Creating validation sets...')
    data_pad, pad_relationship_validation, pad_relationship_training = create_validation_splits(DATA_PATH_PAD=PATH_PAD,
                                                                                                pd_data_pad=pd_data_pad.copy(),
                                                                                                group_colname='PRO_Pad')

    _accessories._print('Manual feature deletion and automatic processing...')
    data_pad, groupby_options_pad, PREDICTORS = manual_selection_and_processing(data_pad=h2o.deep_copy(data_pad, '_' + str(RUN_TAG)),
                                                                                RUN_TAG=RUN_TAG,
                                                                                RESPONDER=RESPONDER,
                                                                                weighting=weighting,
                                                                                weights_column=WEIGHTS_COLUMN,
                                                                                FOLD_COLUMN=FOLD_COLUMN)

    hyp_overview(PREDICTORS, RESPONDER, MAX_EXP_RUNTIME)
    # if(input('Proceed with given hyperparameters? (Y/N)') != 'Y'):
    #     raise RuntimeError('Session forcefully terminated by user during review of hyperparamaters.')

    _accessories._print('Running the experiment...')
    project_names_pad = run_experiment(data=data_pad,
                                       groupby_options=groupby_options_pad,
                                       responder=RESPONDER,
                                       validation_frames_dict=pad_relationship_validation,
                                       MAX_EXP_RUNTIME=MAX_EXP_RUNTIME,
                                       WEIGHTS_COLUMN=WEIGHTS_COLUMN,
                                       ENGINEER_FEATURES=math_eng,
                                       weighting=weighting)  # pad_relationship_validation

    _accessories._print(str(project_names_pad), color='YELLOW')

    # _accessories._print('Calculating variable importance...')
    # varimps_pad = varimps(project_names_pad)
    # _accessories._print('Visualizing heatmap...')
    # ranked_names_pad, ranked_steam_pad = varimp_heatmap(varimps_pad,
    #                                                     'Modeling Reference Files/Round ' +
    #                                                     '{tag}/variable_importances_PAD{tag}.pdf'.format(tag=RUN_TAG),
    #                                                     FIGSIZE=(10, 50),
    #                                                     highlight=False,
    #                                                     annot=False)
    # _accessories._print('Plot predictor correlation matrix...')
    # with _accessories.suppress_stdout():
    #     correlation_matrix(varimps_pad,
    #                        EXP_NAME='Aggregated Experiment Results - Pad-Level',
    #                        FPATH='Modeling Reference Files/Round {tag}/cross-correlations_PAD{tag}.pdf'.format(tag=RUN_TAG))

    _accessories._print('Determing RMSE Benchlines...')
    benchline_pad = get_benchlines(data_pad_pd=pd_data_pad.copy(),
                                   RESPONDER=RESPONDER,
                                   PREFERRED_TOLERANCE=PREFERRED_TOLERANCE)

    _accessories._print('Calculating model performance...')
    perf_pad = model_performance(project_names_pad=project_names_pad,
                                 adj_factor=benchline_pad,
                                 validation_data_dict=pad_relationship_validation,
                                 RESPONDER=RESPONDER,
                                 sort_by='Rel. RMSE',
                                 RUN_TAG=RUN_TAG)

    # _accessories._print('Plotting model performance metrics...')
    # with _accessories.suppress_stdout():
    #     plot_model_performance(perf_pad.select_dtypes(float),
    #                            'Modeling Reference Files/Round {tag}/model_performance_PAD{tag}.pdf'.format(tag=RUN_TAG),
    #                            MODEL_CMAPS, HMAP_CENTERS, ranked_names_pad, ranked_steam_pad,
    #                            highlight=False,
    #                            annot=True,
    #                            annot_size=6,
    #                            FIGSIZE=(10, 1))

    _accessories._print('Saving performance data to file...')
    with _accessories.suppress_stdout():
        _accessories.save_local_data_file(perf_pad.drop('model_obj', axis=1),
                                          f'Modeling Reference Files/Round {RUN_TAG}/MODELS_{RUN_TAG}.pkl')

    if(plot_for_ref is False):
        # OPTIONAL VISUALIZATION: Validation metrics
        with _accessories.suppress_stdout():
            validate_models(perf_data=perf_pad,
                            training_data_dict=pad_relationship_training,
                            benchline=benchline_pad,
                            validation_data_dict=pad_relationship_validation,
                            TOP_MODELS=30,
                            order_by='Rel. Val. RMSE',
                            RUN_TAG=RUN_TAG,
                            RESPONDER=RESPONDER)

    _accessories._print('Renaming this file...')
    new_config = f"ENG: {locals().get('math_eng')}, WEIGHT: {locals().get('weighting')}, TIME: {locals().get('MAX_EXP_RUNTIME')}"
    os.rename(f'Modeling Reference Files/Round {RUN_TAG}',
              f'Modeling Reference Files/{RUN_TAG} – {new_config}')

    # if(input('Shutdown Cluster? (Y/N)') == 'Y'):
    #     shutdown_confirm(h2o)
    _accessories._print('Shutting down server...')
    shutdown_confirm(h2o)

    return RUN_TAG


if __name__ == '__main__':
    _MODELING(math_eng=False, weighting=False, MAX_EXP_RUNTIME=200, plot_for_ref=False)


# def benchmark(math_eng, weighting, MAX_EXP_RUNTIME):
#     path = '_configs/modeling_benchmarks.csv'
#
#     combos = list(itertools.product(*[math_eng, weighting, MAX_EXP_RUNTIME]))
#     _accessories._print(f'{len(combos)} hyperparameter combinations to run...', color='LIGHTCYAN_EX')
#     for math_eng, weighting, MAX_EXP_RUNTIME in combos:
#         _accessories._print(f'Engineered: {math_eng}, Weighting: {weighting}, Run-Time: {MAX_EXP_RUNTIME}',
#                             color='LIGHTCYAN_EX')
#         t1 = time.time()
#         tag = _MODELING(math_eng=math_eng,
#                         weighting=weighting,
#                         MAX_EXP_RUNTIME=MAX_EXP_RUNTIME.item(),
#                         plot_for_ref=False)
#         t2 = time.time()
#
#         _accessories.auto_make_path(path)
#         with open(path, 'a') as file:
#             content = str(math_eng) + ',' + str(weighting) + ',' + str(MAX_EXP_RUNTIME) + ',' + str(t2 - t1) + \
#                 ',' + str(tag)
#             file.write(content)
#
#
# if __name__ == '__main__':
#     math_eng_options = [False, True]
#     weighting_options = [True, False]
#     MAX_EXP_RUNTIME_options = np.arange(10, 210, 10)
#     benchmark(math_eng_options, weighting_options, MAX_EXP_RUNTIME_options)


# CSOR
# Chlorides total pad
# Constriants
# 7200 cubes/day water
# 305 cubes/hour
# 288 cubes/hour
# 14000 barells/day

# EOF

# EOF

# EOF
