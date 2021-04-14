# @Author: Shounak Ray <Ray>
# @Date:   13-Apr-2021 12:04:82:829  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: modeling_bare_minimum.py
# @Last modified by:   Ray
# @Last modified time: 13-Apr-2021 21:04:73:738  GMT-0600
# @License: [Private IP]

# HELPFUL NOTES:
# > https://github.com/h2oai/h2o-3/tree/master/h2o-docs/src/cheatsheets
# > https://www.h2o.ai/blog/h2o-release-3-30-zahradnik/#AutoML-Improvements
# > https://seaborn.pydata.org/generated/seaborn.color_palette.html

import datetime
import os
import random
import subprocess
import sys
from pprint import pprint
from typing import Final

import h2o
import numpy as np
import pandas as pd
import util_traversal
from colorama import Fore, Style
from h2o.automl import H2OAutoML
from h2o.exceptions import H2OConnectionError  # KEEP THIS

# Data Ingestion Constants
DATA_PATH_PAD: Final = 'Data/combined_ipc_engineered_math.csv'    # Where the client-specific pad data is located


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
    # """DATA SANITATION"""
    # _provided_args = locals()
    # name = sys._getframe(0).f_code.co_name
    # _expected_type_args = {'cluster': [h2o.backend.cluster.H2OCluster],
    #                        'show': [bool]}
    # _expected_value_args = {'cluster': None,
    #                         'show': [True, False]}
    # util_data_type_sanitation(_provided_args, _expected_type_args, name)
    # util_data_range_sanitation(_provided_args, _expected_value_args, name)
    # """END OF DATA SANITATION"""

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
    return birds_eye


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
    # """DATA SANITATION"""
    # _provided_args = locals()
    # name = sys._getframe(0).f_code.co_name
    # _expected_type_args = {'h2o_instance': [type(h2o)]}
    # _expected_value_args = {'h2o_instance': None}
    # util_data_type_sanitation(_provided_args, _expected_type_args, name)
    # util_data_range_sanitation(_provided_args, _expected_value_args, name)
    # """END OF DATA SANITATION"""

    # SHUT DOWN the cluster after you're done working with it
    h2o_instance.remove_all()
    h2o_instance.cluster().shutdown()
    # Double checking...
    try:
        snapshot(h2o_instance.cluster)
        raise ValueError('ERROR: H2O cluster improperly closed!')
    except Exception:
        pass


def run_model(DATA_PATH_PAD, skip_track=True):
    _ = """
    #######################################################################################################################
    #########################################   VERIFY VERSIONS OF DEPENDENCIES   #########################################
    #######################################################################################################################
    """
    # Aesthetic Console Output constants
    OUT_BLOCK: Final = '»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»\n'

    # Get the major java version in current environment
    java_major_version = int(subprocess.check_output(['java', '-version'],
                                                     stderr=subprocess.STDOUT).decode().split('"')[1].split('.')[0])

    # Check if environment's java version complies with H2O requirements
    # http://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html#requirements
    if not (java_major_version >= 8 and java_major_version <= 14):
        raise ValueError('STATUS: Java Version is not between 8 and 14 (inclusive).\nH2O cluster will not be initialized.')

    print("\x1b[32m" + 'STATUS: Java dependency versions checked and confirmed.')
    print(OUT_BLOCK)

    _ = """
    #######################################################################################################################
    ############################################  ASSIGNING HYPERPARAMTERS   ##############################################
    #######################################################################################################################
    """

    # H2O Server Constants
    # Initializing the server on the local host (temporary)
    IP_LINK: Final = 'localhost'
    SECURED: Final = True if(IP_LINK != 'localhost') else False   # https protocol doesn't work locally, should be True
    PORT: Final = 54321                                           # Always specify the port that the server should use
    SERVER_FORCE: Final = True                                    # Tries to init new server if existing connection fails

    # Experiment > Model Training Constants and Hyperparameters
    MAX_EXP_RUNTIME: Final = 5                                   # The longest that the experiment will run (seconds)
    RANDOM_SEED: Final = 2381125                                  # To ensure reproducibility of experiments (caveats*)
    EVAL_METRIC: Final = 'rmse'                                   # The evaluation metric to discontinue model training
    RANK_METRIC: Final = 'rmse'                                   # Leaderboard ranking metric after all trainings
    CV_FOLDS: Final = 5                                           # Number of cross validation folds
    STOPPING_ROUNDS: Final = 3                                    # How many rounds to proceed until stopping_metric stops
    WEIGHTS_COLUMN: Final = 'weight'                              # Name of the weights column
    EXPLOIT_RATIO: Final = 0.2                                    # Exploit/Eploration ratio, see NOTES
    MODELING_PLAN: Final = None                                   # Custom Modeling Plan
    TRAIN_VAL_SPLIT: Final = 0.90                                 # Train test split proportion

    # Feature Engineering Constants
    FOLD_COLUMN: Final = "kfold_column"                           # Target encoding, must be consistent throughout training
    PREFERRED_TOLERANCE = 0.1                                     # Tolerance applied on RMSE for allowable responders
    # Verbosity of training (None, 'debug', 'info', d'warn')
    TRAINING_VERBOSITY = 'warn'

    # Miscellaneous Constants
    # The identifying Key/ID for the specified run/config.
    RUN_TAG: Final = random.randint(0, 10000)
    # SOME CODE DELETE HERE

    _ = """
    #######################################################################################################################
    ##############################################   SAVING HYPERPARAMTERS   ##############################################
    #######################################################################################################################
    """

    if not skip_track:
        # Ensure overwriting does not occur while making identifying experiment directory.
        while os.path.isdir(f'Modeling Reference Files/Round {RUN_TAG}'):
            RUN_TAG: Final = random.randint(0, 10000)
        os.makedirs(f'Modeling Reference Files/Round {RUN_TAG}')

        # Compartmentalize Hyperparameters
        __LOCAL_VARS = locals().copy()
        _SERVER_HYPERPARAMS = ('IP_LINK', 'SECURED', 'PORT', 'SERVER_FORCE')
        _SERVER_HYPERPARAMS = {var: __LOCAL_VARS.get(var) for var in _SERVER_HYPERPARAMS}
        _TRAIN_HYPERPARAMS = ('MAX_EXP_RUNTIME', 'RANDOM_SEED', 'EVAL_METRIC', 'RANK_METRIC', 'CV_FOLDS', 'STOPPING_ROUNDS',
                              'WEIGHTS_COLUMN', 'EXPLOIT_RATIO', 'MODELING_PLAN')
        _TRAIN_HYPERPARAMS = {var: __LOCAL_VARS.get(var) for var in _TRAIN_HYPERPARAMS}
        _EXECUTION_HYPERPARAMS = ('FOLD_COLUMN', 'PREFERRED_TOLERANCE', 'TRAINING_VERBOSITY')
        _EXECUTION_HYPERPARAMS = {var: __LOCAL_VARS.get(var) for var in _EXECUTION_HYPERPARAMS}

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

    print(Fore.GREEN + 'STATUS: Directory created for Round {}'.format(RUN_TAG) + Style.RESET_ALL)

    # Print file structure for reference every time this program is run
    util_traversal.print_tree_to_txt(skip_git=True)

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
            raise ValueError(
                f'> ERROR: Mismatched expected and received dicts during data_range_sanitation for {name}.')
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
        # """DATA SANITATION"""
        # _provided_args = locals()
        # name = sys._getframe(0).f_code.co_name
        # _expected_type_args = {'data_frame': None,
        #                        'tbd_list': [list]}
        # _expected_value_args = {'data_frame': None,
        #                         'tbd_list': None}
        # util_data_type_sanitation(_provided_args, _expected_type_args, name)
        # util_data_range_sanitation(_provided_args, _expected_value_args, name)
        # """END OF DATA SANITATION"""

        for tb_dropped in tbd_list:
            if(tb_dropped in data_frame.columns):
                data_frame = data_frame.drop(tb_dropped, axis=1)
                print(Fore.GREEN + '> STATUS: {} dropped.'.format(tb_dropped) + Style.RESET_ALL)
            else:
                print(Fore.GREEN + '> STATUS: {} not in frame, skipping.'.format(tb_dropped) + Style.RESET_ALL)
        return data_frame

    def data_refinement(data, groupby, dropcols, responder, FOLD_COLUMN=FOLD_COLUMN):
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
        # """DATA SANITATION"""
        # _provided_args = locals()
        # name = sys._getframe(0).f_code.co_name
        # _expected_type_args = {'data': None,
        #                        'groupby': [str],
        #                        'dropcols': [list],
        #                        'responder': [str],
        #                        'FOLD_COLUMN': [str, type(None)]}
        # _expected_value_args = {'data': None,
        #                         'groupby': None,
        #                         'dropcols': None,
        #                         'responder': ['PRO_Alloc_Oil', 'PRO_Adj_Alloc_Oil', 'PRO_Total_Fluid'],
        #                         'FOLD_COLUMN': None}
        # util_data_type_sanitation(_provided_args, _expected_type_args, name)
        # util_data_range_sanitation(_provided_args, _expected_value_args, name)
        # """END OF DATA SANITATION"""

        # Drop the specified columns before proceeding with the experiment
        data = util_conditional_drop(data, dropcols)

        # Determine all possible groups to filter the source data for when conducting the experiment
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

    def run_experiment(data, groupby_options, responder, validation_frames_dict,
                       MAX_EXP_RUNTIME=MAX_EXP_RUNTIME, EVAL_METRIC=EVAL_METRIC, RANK_METRIC=RANK_METRIC,
                       RANDOM_SEED=RANDOM_SEED, WEIGHTS_COLUMNS=WEIGHTS_COLUMN, CV_FOLDS=CV_FOLDS,
                       STOPPING_ROUNDS=STOPPING_ROUNDS, EXPLOIT_RATIO=EXPLOIT_RATIO, MODELING_PLAN=MODELING_PLAN,
                       TRAINING_VERBOSITY=TRAINING_VERBOSITY):
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
        #                        'WEIGHTS_COLUMNS': [str, type(None)],
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
        #                         'WEIGHTS_COLUMNS': None,
        #                         'CV_FOLDS': list(range(1, 10 + 1)),
        #                         'STOPPING_ROUNDS': list(range(1, 10 + 1)),
        #                         'EXPLOIT_RATIO': [0.0, 1.0],
        #                         'MODELING_PLAN': None,
        #                         'TRAINING_VERBOSITY': None}
        # util_data_type_sanitation(_provided_args, _expected_type_args, name)
        # util_data_range_sanitation(_provided_args, _expected_value_args, name)
        # """END OF DATA SANITATION"""

        # Determined the categorical variable to be dropped in TRAINING SET (should only be the groupby)
        tb_dropped = data.as_data_frame().select_dtypes(object).columns
        if not (len(tb_dropped) == 1):
            raise RuntimeError('Only and exactly one categorical variable was expected in the provided TRAIN data.' +
                               'However, ' + f'{len(tb_dropped)} were provided.')
        else:
            tb_dropped = tb_dropped[0]

        # Run all the H2O experiments for all the different groupby_options. Store in unique project name.
        cumulative_varimps = {}
        initialized_projects = []
        for group in groupby_options:
            print(Fore.GREEN + 'STATUS: Experiment -> Production Pad {}\n'.format(group) + Style.RESET_ALL)

            if(validation_frames_dict is not None):
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
            # Filter the complete, provided source data to only include info for the current group
            MODEL_DATA = data[data[tb_dropped] == group]
            MODEL_DATA = MODEL_DATA.drop(tb_dropped, axis=1)
            print(MODEL_DATA)
            aml_obj.train(y=responder,                                # A single responder
                          weights_column=WEIGHTS_COLUMN,              # What is the weights column in the H2O frame?
                          training_frame=MODEL_DATA,                  # All the data is used for training, cross-validation
                          validation_frame=validation_frame_groupby)  # The validation dataset used to assess performance

        print(Fore.GREEN + 'STATUS: Completed experiments\n\n' + Style.RESET_ALL)

        return initialized_projects

    def varimps(project_names):
        """Determines variable importances for all models in an experiment.

        Parameters
        ----------
        project_names : id
            The H2O AutoML experiment configuration.

        Returns
        -------
        pandas.core.frame.DataFrame, list (of tuples)
            A concatenated DataFrame with all the model's variable importances for all the input features.
            [Optional] A list of the models that did not have a variable importance. Format: (name, model object).

        """
        # """DATA SANITATION"""
        # _provided_args = locals()
        # name = sys._getframe(0).f_code.co_name
        # _expected_type_args = {'aml_obj': [h2o.automl.autoh2o.H2OAutoML]}
        # _expected_value_args = {'aml_obj': None}
        # util_data_type_sanitation(_provided_args, _expected_type_args, name)
        # util_data_range_sanitation(_provided_args, _expected_value_args, name)
        # """END OF DATA SANITATION"""

        aml_objects = get_aml_objects(project_names)

        variable_importances = []
        for aml_obj in aml_objects:
            responder, group = project_names[aml_objects.index(aml_obj)].split('__')[-2:]
            tag = [group, responder]
            tag_name = ['group', 'responder']

            cumulative_varimps = []
            model_novarimps = []
            exp_leaderboard = aml_obj.leaderboard
            # Retrieve all the model objects from the given experiment
            exp_models = [h2o.get_model(exp_leaderboard[m_num, "model_id"])
                          for m_num in range(exp_leaderboard.shape[0])]
            for model in exp_models:
                model_name = model.params['model_id']['actual']['name']
                variable_importance = model.varimp(use_pandas=True)

                # Only conduct variable importance dataset manipulation if ranking data is available
                # > (eg. unavailable, stacked)
                if(variable_importance is not None):
                    variable_importance = pd.pivot_table(variable_importance,
                                                         values='scaled_importance',
                                                         columns='variable').reset_index(drop=True)
                    variable_importance['model_name'] = model_name
                    variable_importance['model_object'] = model
                    if(tag is not None and tag_name is not None):
                        for i in range(len(tag)):
                            variable_importance[tag_name[i]] = tag[i]
                    variable_importance.index = variable_importance['model_name'] + \
                        '___GROUP-' + variable_importance['group']
                    variable_importance.drop('model_name', axis=1, inplace=True)
                    variable_importance.columns.name = None

                    cumulative_varimps.append(variable_importance)
                else:
                    # print('> WARNING: Variable importances unavailable for {MDL}'.format(MDL=model_name))
                    model_novarimps.append((model_name, model))

            varimp_all_models_in_run = pd.concat(cumulative_varimps)
            variable_importances.append(varimp_all_models_in_run)  # , model_novarimps

        requested_varimps = pd.concat(variable_importances)

        print(Fore.GREEN +
              '> STATUS: Determined variable importances of all models in given experiments: {}.'.format(project_names) +
              Style.RESET_ALL)

        return requested_varimps

    def model_performance(project_names_pad, adj_factor, sort_by='RMSE'):
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
            model_ids = exp_obj.leaderboard.as_data_frame()['model_id']
            for model_name in model_ids:
                model_obj = h2o.get_model(model_name)
                perf_data[model_name] = {}
                # perf_data[model_name]['R^2'] = model_obj.r2()
                # perf_data[model_name]['R'] = model_obj.r2() ** 0.5
                # perf_data[model_name]['MSE'] = model_obj.mse()
                perf_data[model_name]['RMSE'] = model_obj.rmse()
                perf_data[model_name]['Rel. RMSE'] = model_obj.rmse() - adj_factor.get(group)
                perf_data[model_name]['Val. RMSE'] = model_obj.rmse(valid=True)
                if(model_obj.rmse(valid=True) is not None):
                    perf_data[model_name]['Rel. Val. RMSE'] = model_obj.rmse(valid=True) - adj_factor.get(group)
                else:
                    perf_data[model_name]['Rel. Val. RMSE'] = None
                perf_data[model_name]['group'] = group
                perf_data[model_name]['model_obj'] = model_obj
                # perf_data[model_name]['RMSLE'] = model_obj.rmsle()
                # perf_data[model_name]['MAE'] = model_obj.mae()

        # Structure model output and order
        perf_data = pd.DataFrame(perf_data).T.sort_values(sort_by, ascending=False).infer_objects()
        perf_data['tolerated RMSE'] = perf_data['group'].apply(lambda x: adj_factor.get(x))

        return perf_data

    print(Fore.GREEN + 'STATUS: Hyperparameters assigned and functions defined.' + Style.RESET_ALL)
    print(OUT_BLOCK)

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
    # Filter the complete, provided source data to only include info f')
    # Check the status of the cluster, just for reference
    process_log = snapshot(h2o.cluster(), show=False)

    # Confirm that the data path leads to an actual file
    for path in [DATA_PATH_PAD]:
        if not (os.path.isfile(path)):
            raise ValueError('ERROR: {data} does not exist in the specificied location.'.format(data=path))

    _ = """
    ####################################
    #######  VALIDATION SPLITING #######
    ####################################
    """
    # Split into train/test (CV) and holdout set (per each class of grouping)
    pd_data_pad = pd.read_csv(DATA_PATH_PAD).drop('Unnamed: 0', axis=1)
    unique_pads = list(pd_data_pad['PRO_Pad'].unique())
    grouped_data_split = {}
    for u_pad in unique_pads:
        filtered_by_group = pd_data_pad[pd_data_pad['PRO_Pad'] == u_pad].sort_values('Date').reset_index(drop=True)
        data_pad_loop, data_pad_validation_loop = [dat.reset_index(drop=True).infer_objects()
                                                   for dat in np.split(filtered_by_group,
                                                                       [int(TRAIN_VAL_SPLIT * len(filtered_by_group))])]
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
        df_loop = df_loop[df_loop['PRO_Pad'] == u_pad].drop(['Date'], axis=1).infer_objects().reset_index(drop=True)
        local_wanted_types = {k: v for k, v in wanted_types.items() if k in df_loop.columns}
        df_loop = h2o.H2OFrame(df_loop, column_types=local_wanted_types)
        pad_relationship_validation[u_pad] = df_loop
    # Create pad training relationships
    pad_relationship_training = {}
    for u_pad in unique_pads:
        df_loop = data_pad.as_data_frame()
        df_loop = df_loop[df_loop['PRO_Pad'] == u_pad].drop(['PRO_Pad'], axis=1).infer_objects().reset_index(drop=True)
        local_wanted_types = {k: v for k, v in wanted_types.items() if k in df_loop.columns}
        df_loop = h2o.H2OFrame(df_loop, column_types=local_wanted_types)
        pad_relationship_training[u_pad] = df_loop

    print(Fore.GREEN + 'STATUS: Server initialized and data imported.' + Style.RESET_ALL)
    print(OUT_BLOCK)

    _ = """
    #######################################################################################################################
    ##########################################   MINOR DATA MANIPULATION/PREP   ###########################################
    #######################################################################################################################
    """
    # Table diagnostics
    RESPONDER = 'PRO_Adj_Alloc_Oil'  # OR 'PRO_Total_Fluid'

    EXCLUDE = ['C1', 'Bin_1', 'Bin_5', 'Date']
    EXCLUDE.extend(['PRO_Alloc_Oil', 'PRO_Pump_Speed', 'PRO_Adj_Pump_Speed'])

    data_pad, groupby_options_pad, PREDICTORS = data_refinement(data_pad, 'PRO_Pad', EXCLUDE, RESPONDER)

    if not skip_track:
        with open(f'Modeling Reference Files/Round {RUN_TAG}/hyperparams.txt', 'a') as out:
            print(OUT_BLOCK.replace('\n', '') + OUT_BLOCK.replace('\n', ''), file=out)
            print('PREDICTORS', file=out)
            pprint(PREDICTORS, stream=out)
            print(OUT_BLOCK.replace('\n', '') + OUT_BLOCK.replace('\n', ''), file=out)
            print('RESPONDER', file=out)
            pprint(RESPONDER, stream=out)

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
    print(Fore.GREEN + '\t* seed\t\t\t-> ', RANDOM_SEED,
          Fore.GREEN + '\t\tRandom seed for reproducibility. There are caveats.')
    print(Fore.GREEN + '\t* Predictors\t\t-> ', PREDICTORS,
          Fore.GREEN + '\t\tThese are the variables which will be used to predict the responder.')
    print(Fore.GREEN + '\t* Responder\t\t-> ', RESPONDER,
          Fore.GREEN + '\tThis is what is being predicted.\n' + Style.RESET_ALL)

    # Run the experiment
    project_names_pad = run_experiment(data_pad, groupby_options_pad, RESPONDER,
                                       validation_frames_dict=pad_relationship_validation)  # pad_relationship_validation

    # VARIABLE IMPORTANCE CALCULATION SKIPPED
    # CORRELATION MATRIX: CODE SKIPPED
    # VARIABLE IMPROTANCE HEATMAP: CODE SKIPPED

    print(Fore.GREEN + 'STATUS: Saved variable importance configurations.' + Style.RESET_ALL)
    print(OUT_BLOCK)
    _ = """
    #######################################################################################################################
    #########################################   EVALUATE MODEL PERFORMANCE   ##############################################
    #######################################################################################################################
    """
    # Calculate Benchlines
    data_pad_pd = pd_data_pad.copy()  # pd.read_csv(DATA_PATH_PAD)
    benchline_pad = list(data_pad_pd[data_pad_pd[RESPONDER] > 0].groupby(
        ['Date', 'PRO_Pad'])[RESPONDER].sum().reset_index().groupby('PRO_Pad').median().to_dict().values())[0]
    benchline_pad.update((x, y * PREFERRED_TOLERANCE) for x, y in benchline_pad.items())

    # Calculate model performance metrics
    perf_pad = model_performance(project_names_pad, benchline_pad, sort_by='Rel. RMSE')

    # PERFORMANCE VISUALIZATION: CODE SKIPPED

    _ = """
    #######################################################################################################################
    ###############################   EVALUATE MODEL PERFORMANCE | VIZUALIZATION   ########################################
    #######################################################################################################################
    """

    # MODEL PREDICTIONS, PERFORMANCE VISUALIZATION: CODE SKIPPED

    return perf_pad


output = run_model(DATA_PATH_PAD)

_ = """
#######################################################################################################################
########################################   SHUTDOWN THE SESSION/CLUSTER   #############################################
#######################################################################################################################
"""

shutdown_confirm(h2o)

# EOF
