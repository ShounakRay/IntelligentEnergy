# @Author: Shounak Ray <Ray>
# @Date:   23-Mar-2021 12:03:13:138  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: approach_alt_modeling.py
# @Last modified by:   Ray
# @Last modified time: 26-Mar-2021 22:03:11:117  GMT-0600
# @License: [Private IP]

# HELPFUL NOTES:
# > https://github.com/h2oai/h2o-3/tree/master/h2o-docs/src/cheatsheets
# > https://www.h2o.ai/blog/h2o-release-3-30-zahradnik/#AutoML-Improvements
# > https://seaborn.pydata.org/generated/seaborn.color_palette.html

import os
import random
import subprocess
import sys
from contextlib import contextmanager
from typing import Final

import h2o
import matplotlib
import matplotlib.pyplot as plt  # Req. dep. for h2o.estimators.random_forest.H2ORandomForestEstimator.varimp_plot()
import numpy as np
import pandas as pd  # Req. dep. for h2o.estimators.random_forest.H2ORandomForestEstimator.varimp()
import seaborn as sns
import util_traversal
from colorama import Fore, Style
from h2o.automl import H2OAutoML
from matplotlib.patches import Rectangle

# from h2o.estimators import H2OTargetEncoderEstimator


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
#########################################   DEFINITIONS AND HYPERPARAMTERS   ##########################################
#######################################################################################################################
"""

# H2O Server Constants
IP_LINK: Final = 'localhost'                                  # Initializing the server on the local host (temporary)
SECURED: Final = True if(IP_LINK != 'localhost') else False   # https protocol doesn't work locally, should be True
PORT: Final = 54321                                           # Always specify the port that the server should use
SERVER_FORCE: Final = True                                    # Tries to init new server if existing connection fails

# Experiment > Model Training Constants and Hyperparameters
MAX_EXP_RUNTIME: Final = 10                                       # The longest that the experiment will run (seconds)
RANDOM_SEED: Final = 2381125                                      # To ensure reproducibility of experiments (caveats*)
EVAL_METRIC: Final = 'auto'                                       # The evaluation metric to discontinue model training
RANK_METRIC: Final = 'rmse'                                       # Leaderboard ranking metric after all trainings
CV_FOLDS: Final = 5
STOPPING_ROUNDS: Final = 3
WEIGHTS_COLUMN: Final = None
EXPLOIT_RATIO: Final = 0
MODELING_PLAN: Final = None

# Data Ingestion Constants
DATA_PATH_PAD: Final = 'Data/combined_ipc_aggregates.csv'         # Where the client-specific pad data is located
DATA_PATH_WELL: Final = 'Data/combined_ipc_aggregates_PWELL.csv'  # Where the client-specific well data is located

# Feature Engineering Constants
FOLD_COLUMN: Final = "kfold_column"                           # Target encoding, must be consistent throughout training
TOP_MODELS = 15                                               # The top n number of models which should be filtered.
PREFERRED_TOLERANCE = 0.1                                     # Tolerance applied on RMSE for allowable responders
TRAINING_VERBOSITY = 'warn'                                   # Verbosity of training (None, 'debug', 'info', d'warn')

# Miscellaneous Constants
RUN_TAG: Final = random.randint(0, 10000)                     # The identifying Key/ID for the specified run/config.

# Ensure overwriting does not occur while making identifying experiment directory.
while os.path.isdir(f'Modeling Reference Files/Round {RUN_TAG}'):
    RUN_TAG: Final = random.randint(0, 10000)
os.makedirs(f'Modeling Reference Files/Round {RUN_TAG}')

print(Fore.GREEN + 'STATUS: Directory created for Round {}'.format(RUN_TAG) + Style.RESET_ALL)

# Print file structure for reference every time this program is run
util_traversal.print_tree_to_txt(skip_git=True)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def typical_manipulation_h20(data, groupby, dropcols, responder, FOLD_COLUMN=FOLD_COLUMN):
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
    _provided_args = locals()
    name = sys._getframe(0).f_code.co_name
    _expected_type_args = {'data': None,
                           'groupby': [str],
                           'dropcols': [list],
                           'responder': [str],
                           'FOLD_COLUMN': [str, type(None)]}
    _expected_value_args = {'data': None,
                            'groupby': None,
                            'dropcols': None,
                            'responder': ['PRO_Alloc_Oil'],
                            'FOLD_COLUMN': None}
    data_type_sanitation(_provided_args, _expected_type_args, name)
    data_range_sanitation(_provided_args, _expected_value_args, name)
    """END OF DATA SANITATION"""

    # Drop the specified columns before proceeding with the experiment
    data = conditional_drop(data, dropcols)

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

    print(Fore.GREEN + 'STATUS: Experiment hyperparameters and data configured.\n\n' + Style.RESET_ALL)

    return data, groupby_options, predictors


def run_experiment(data, groupby_options, responder,
                   MAX_EXP_RUNTIME=MAX_EXP_RUNTIME, EVAL_METRIC=EVAL_METRIC, RANK_METRIC=RANK_METRIC,
                   RANDOM_SEED=RANDOM_SEED, CV_FOLDS=CV_FOLDS, STOPPING_ROUNDS=STOPPING_ROUNDS,
                   EXPLOIT_RATIO=EXPLOIT_RATIO, MODELING_PLAN=MODELING_PLAN, TRAINING_VERBOSITY=TRAINING_VERBOSITY):
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
    pandas.core.DataFrame, h2o.automl.autoh2o.H2OAutoML
        `final_cumulative_varimps` -> The data with aggregated variable importances per model in all groups provided.
        `aml_obj` ->  The H2O object with information for all the experiments

    """
    """DATA SANITATION"""
    _provided_args = locals()
    name = sys._getframe(0).f_code.co_name
    _expected_type_args = {'data': None,
                           'groupby_options': [str],
                           'responder': [dict],
                           'MAX_EXP_RUNTIME': [dict],
                           'EVAL_METRIC': [list, type(None)],
                           'RANK_METRIC': [list, type(None)],
                           'RANDOM_SEED': [int, float],
                           'CV_FOLDS': [int],
                           'STOPPING_ROUNDS': [int],
                           'EXPLOIT_RATIO': [float],
                           'MODELING_PLAN': [list, type(None)],
                           'TRAINING_VERBOSITY': [str, type(None)]}
    _expected_value_args = {'data': None,
                            'groupby_options': None,
                            'responder': None,
                            'MAX_EXP_RUNTIME': [1, 10000],
                            'EVAL_METRIC': None,
                            'RANK_METRIC': None,
                            'RANDOM_SEED': [0, np.inf],
                            'CV_FOLDS': list(range(1, 10 + 1)),
                            'STOPPING_ROUNDS': list(range(1, 10 + 1)),
                            'EXPLOIT_RATIO': [0.0, 1.0],
                            'MODELING_PLAN': None,
                            'TRAINING_VERBOSITY': None}
    data_type_sanitation(_provided_args, _expected_type_args, name)
    data_range_sanitation(_provided_args, _expected_value_args, name)
    """END OF DATA SANITATION"""

    # Determined the categorical variable to be dropped (should only be the groupby)
    tb_dropped = data.as_data_frame().select_dtypes(object).columns
    if not (len(tb_dropped) == 1):
        raise RuntimeError('Only and exactly one categorical variable was expected in the provided data. However, ' +
                           f'{len(tb_dropped)} were provided.')
    else:
        tb_dropped = tb_dropped[0]

    # Run all the H2O experiments for all the different groupby_options. Store in unique project name.
    cumulative_varimps = {}
    initialized_projects = []
    for group in groupby_options:
        print(Fore.GREEN + 'STATUS: Experiment -> Production Pad {}\n'.format(group) + Style.RESET_ALL)
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
        aml_obj.train(y=responder,                                # A single responder
                      weights_column=WEIGHTS_COLUMN,              # What is the weights column in the H2O frame?
                      training_frame=MODEL_DATA)                  # All the data is used for training, cross-validation

    print(Fore.GREEN + 'STATUS: Completed experiments\n\n' + Style.RESET_ALL)

    return initialized_projects


def plot_varimp_heatmap(final_cumulative_varimps, FPATH, highlight=True,
                        preferred='Steam', preferred_importance=0.7, mean_importance_threshold=0.7,
                        top_color='green', chosen_color='cyan', RUN_TAG=RUN_TAG, FIGSIZE=(10, 55), annot=False,
                        TOP_MODELS=TOP_MODELS):
    """Plots a heatmap based on the inputted variable importance data.

    Parameters
    ----------
    final_cumulative_varimps : pandas.core.frame.DataFrame
        Data on each models variable importances, the model itself, and other identifying tags (like group_type)
        Only numerical features are used to make the heatmap though.
    FPATH : str
        The file path where the heatmap should be saved.
    highlight : bool
        Whether or not Rectangles should be patched to show steam-preferred models and top models generated.
    preferred : str
        The name of the predictor that is preferred for the model to rank higher (for explainability).
    preferred_importance : float
        The relative, threshold value for which the `preferred` predictor should be at or above.
        Used for filtering variable importances and identifying `ranked steam`.
    mean_importance_threshold : float
        The relative, average of each variable importance row (per model) should be at or above this value.
        Used for filtering variable importances and identifying `ranked steam`.
    top_color : str
        The color the top model(s) should be. Specifically the Rectangle patched on.
        Corresponds to `ranked_names`.
    chosen_color : str
        The color the high-steam model(s) should be. Specifically the Rectangle patched on.
        Corresponds to `ranked_steam`.
    RUN_TAG : int
        The identifying ID/KEY for the global configuration.
    FIGSIZE : tuple (of ints)
        The size of the ouputted heatmap figure. (width x length)
    annot : bool
        Whether the heatmap should be annotated (aka variable importances should be labelled).
        This should be turned off for very dense plots (lots of rows) or for minimzing runtime.
    TOP_MODELS : int
        The top n number of models which should be implicitly filtered.
        Corresponds to `ranked_names`.

    Returns
    -------
    list (of str), list (of str) OR None, None
        `ranked_names` -> The indices of the models which are ranked according to
                          `TOP_MODELS` and the predictor rank.
        `ranked_steam` -> The indices of the models which are ranked according to
                          `preferred` and `preferred_importance`.

        NOTE RETURN DEPENDENCY: if `highlight` is False then both `ranked_names` and `ranked_steam` are None.

    """
    """DATA SANITATION"""
    _provided_args = locals()
    name = sys._getframe(0).f_code.co_name
    _expected_type_args = {'final_cumulative_varimps': [pd.core.frame.DataFrame],
                           'FPATH': [str],
                           'highlight': [bool],
                           'preferred': [str],
                           'preferred_importance': [float],
                           'mean_importance_threshold': [float],
                           'top_color': [str],
                           'chosen_color': [str],
                           'RUN_TAG': [int],
                           'FIGSIZE': [tuple],
                           'annot': [bool],
                           'TOP_MODELS': [int]}
    _expected_value_args = {'final_cumulative_varimps': None,
                            'FPATH': None,
                            'highlight': [True, False],
                            'preferred': None,
                            'preferred_importance': [0.0, 1.0],
                            'mean_importance_threshold': [0.0, 1.0],
                            'top_color': None,
                            'chosen_color': None,
                            'RUN_TAG': None,
                            'FIGSIZE': None,
                            'annot': [True, False],
                            'TOP_MODELS': [0, np.inf]}
    data_type_sanitation(_provided_args, _expected_type_args, name)
    data_range_sanitation(_provided_args, _expected_value_args, name)
    """END OF DATA SANITATION"""

    # Plot heatmap of variable importances across all model combinations
    fig, ax = plt.subplots(figsize=FIGSIZE)
    predictor_rank = final_cumulative_varimps.mean(axis=0).sort_values(ascending=False)
    sns_fig = sns.heatmap(final_cumulative_varimps[predictor_rank.keys()], annot=annot, annot_kws={"size": 4})

    if(highlight):
        temp = final_cumulative_varimps[predictor_rank.keys()]

        ranked_names = list(temp.index)[-TOP_MODELS:]
        rect_width = len(predictor_rank)
        bottom_y_loc: Final = len(temp) - 1
        # For the top 10 models
        for y_loc in range(TOP_MODELS + 1):
            sns_fig.add_patch(Rectangle(xy=(0, bottom_y_loc - y_loc), width=rect_width, height=1,
                                        edgecolor=top_color, fill=False, lw=3))

        # For the models where steam is selected as the most important
        ranked_steam = list(temp[(temp[preferred] >= preferred_importance) &
                                 (temp.mean(axis=1) > mean_importance_threshold)].index)
        y_locs = [list(range(len(temp))).index(list(temp.index).index(rstm)) for rstm in ranked_steam]
        stm_loc = list(predictor_rank.index).index(preferred)
        for y_loc in y_locs:
            sns_fig.add_patch(Rectangle(xy=(stm_loc,  y_loc), width=1, height=1,
                                        edgecolor=chosen_color, fill=False, lw=3))
    else:
        ranked_names = ranked_steam = None

    sns_fig.get_figure().savefig(FPATH,
                                 bbox_inches='tight')
    plt.clf()

    return ranked_names, ranked_steam

# TODO: Add "normalized" RMSE value according to a benchline


def model_performance(tracker_with_modelobj, sort_by='RMSE', modelobj_colname='model_object'):
    """Determine the model performance through a series of predefined metrics.

    Parameters
    ----------
    tracker_with_modelobj : pandas.core.frame.DataFrame
        Data on each models variable importances, the model itself, and other identifying tags (like group_type)
        Only numerical features are used to make the heatmap though.
    sort_by : str
        The performance metric which should be used to sort the performance data in descending order.
    modelobj_colname : str
        The name of the column in the DataFrame which contains the H2O models.
        The default value is dependent on the defintion of `exp_cumulative_varimps`.

    Returns
    -------
    pandas.core.frame.DataFrame
        `perf_data` -> The DataFrame with performance information for all of the inputted models.
                       Specifically, R^2, R (optionally), MSE, RMSE, RMSLE, and MAE.

    """
    """DATA SANITATION"""
    _provided_args = locals()
    name = sys._getframe(0).f_code.co_name
    _expected_type_args = {'tracker_with_modelobj': [pd.core.frame.DataFrame],
                           'sort_by': [str],
                           'modelobj_colname': [str]}
    _expected_value_args = {'tracker_with_modelobj': None,
                            'sort_by': ['R^2', 'MSE', 'RMSE', 'RMSLE', 'MAE'],
                            'modelobj_colname': None}
    data_type_sanitation(_provided_args, _expected_type_args, name)
    data_range_sanitation(_provided_args, _expected_value_args, name)
    """END OF DATA SANITATION"""

    perf_data = {}
    for model_name, model_obj in zip(tracker_with_modelobj.index, tracker_with_modelobj[modelobj_colname]):
        perf_data[model_name] = {}
        perf_data[model_name]['R^2'] = model_obj.r2()
        # perf_data[model_name]['R'] = model_obj.r2() ** 0.5
        perf_data[model_name]['MSE'] = model_obj.mse()
        perf_data[model_name]['RMSE'] = model_obj.rmse()
        perf_data[model_name]['RMSLE'] = model_obj.rmsle()
        perf_data[model_name]['MAE'] = model_obj.mae()

    # Structure model output and order
    perf_data = pd.DataFrame(perf_data).T.sort_values(sort_by, ascending=False).infer_objects()
    # Ensure correct data type of ther performance metric columns.
    # Ensures proper heatmap functionality in `plot_model_performance`
    for col in perf_data.columns:
        perf_data[col] = perf_data[col].astype(float)

    return perf_data


def data_type_sanitation(val_and_inputs, expected, name):
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
        print('>> WARNING: Exception (likely edge case) ignored in data_type_sanitation')
        print('>>\t' + str(e)[:50] + '...')

    if(len(mismatched) > 0):
        raise ValueError(
            f'> Invalid input for following pairs during data_type_sanitation for {name}.\n{mismatched}')


def data_range_sanitation(val_and_inputs, expected, name):
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
        print('>> WARNING: Exception (likely edge case) ignored in data_range_sanitation')

    if(len(mismatched) > 0):
        raise ValueError(
            f'> Invalid input for following pairs during data_range_sanitation for {name}.\n{mismatched}')


def plot_model_performance(perf_data, FPATH, mcmaps, centers, ranked_names, ranked_steam, extrema_thresh_pct=5,
                           RUN_TAG=RUN_TAG, annot=True, annot_size=4, highlight=True, FIGSIZE=(10, 50)):
    """Plots a heatmap based on the inputted model performance data.

    Parameters
    ----------
    perf_data : pandas.core.frame.DataFrame
        The DataFrame with performance information for all of the inputted models.
        Specifically, R^2, R (optionally), MSE, RMSE, RMSLE, and MAE.
    FPATH : str
        The file path where the heatmap should be saved.
    mcmaps : dict
        Different heatmap/AxesPlot color schemes dependent on performance metric.
        Stands for "macro color maps."
    centers : dict
        Description of parameter `centers`.
    ranked_names : list (of ints) OR None
        The indices of the models which are ranked according to `TOP_MODELS` and the predictor rank.
        `TOP_MODELS` and the predictor rank may be found in `plot_varimp_heatmap`.
        NOTE DEPENDENCY: If `highlight` is False in `plot_varimp_heatmap`, then it MUST be locally False here.
                         If it is True in THIS function, it will raise an Exception since control doesn't know which
                         models to highlight, including for mean-ranks. (determined in `plot_varimp_heatmap`).
    ranked_steam : list (of ints) OR None
        The indices of the models which are ranked according to `preferred` and `preferred_importance`.
        `preferred` and `preferred_importance` may be found in `plot_varimp_heatmap`.
        NOTE DEPENDENCY: If `highlight` is False in `plot_varimp_heatmap`, then it MUST be locally False here.
                         If it is True in THIS function, it will raise an Exception since control doesn't know which
                         models to highlight, including for steam-ranks. (determined in `plot_varimp_heatmap`).
    extrema_thresh_pct : int
        The top and botton n percentage points to set as the maximum and minimum values for the heatmap colors.
        Stands for "extrema threshold percentage."
    RUN_TAG : int
        The identifying ID/KEY for the global configuration.
    annot : bool
        Whether the heatmap should be annotated (aka variable importances should be labelled).
        This should be turned off for very dense plots (lots of rows) or for minimzing runtime.
    annot_size : int
        The font size of the annotation. Only works if `annot` is not False.
    highlight : bool
        Whether or not Rectangles should be patched to show steam-preferred models and top models generated.
    FIGSIZE : tuple (of ints)
        The size of the ouputted heatmap figure. (width x length)

    Returns
    -------
    None
        Nothing! Simply print out a heatmap.
        There are no new "ranking indices" like `ranked_steam` or `ranked_names` to be determined, anyways.

    """
    """DATA SANITATION"""
    _provided_args = locals()
    name = sys._getframe(0).f_code.co_name
    _expected_type_args = {'perf_data': [pd.core.frame.DataFrame],
                           'FPATH': [str],
                           'mcmaps': [dict],
                           'centers': [dict],
                           'ranked_names': [list, type(None)],
                           'ranked_steam': [list, type(None)],
                           'extrema_thresh_pct': [int, float],
                           'RUN_TAG': [int],
                           'annot': [bool],
                           'annot_size': [int],
                           'highlight': [bool],
                           'FIGSIZE': [tuple]}
    _expected_value_args = {'perf_data': None,
                            'FPATH': None,
                            'mcmaps': None,
                            'centers': None,
                            'ranked_names': None,
                            'ranked_steam': None,
                            'extrema_thresh_pct': (0, 99),
                            'RUN_TAG': None,
                            'annot': None,
                            'annot_size': (0, np.inf),
                            'highlight': None,
                            'FIGSIZE': None}
    data_type_sanitation(_provided_args, _expected_type_args, name)
    data_range_sanitation(_provided_args, _expected_value_args, name)
    """END OF DATA SANITATION"""

    # CUSTOM DATA SANITATION
    if not (len(perf_data.select_dtypes(float).columns) <= len(perf_data.columns)):
        cat_var = list(perf_data.select_dtypes(object).columns)
        raise ValueError('> ERROR: Inputted data has categorical variables. {}'.format(cat_var))

    fig, ax = plt.subplots(figsize=FIGSIZE, ncols=len(perf_data.columns), sharey=True)
    for col in perf_data.columns:
        cmap_local = mcmaps.get(col)
        center_local = centers.get(col)
        vmax_local = np.percentile(perf_data[col].dropna(), 100 - extrema_thresh_pct)
        vmin_local = np.percentile(perf_data[col].dropna(), extrema_thresh_pct)

        if not (vmin_local < vmax_local):
            raise ValueError('> ERROR: Heatmap min, center, max have incompatible values {}-{}-{}'.format(vmin_local,
                                                                                                          center_local,
                                                                                                          vmax_local))

        sns_fig = sns.heatmap(perf_data[[col]], ax=ax[list(perf_data.columns).index(col)],
                              annot=annot, annot_kws={"size": annot_size}, cbar=False,
                              center=center_local, cmap=cmap_local, mask=perf_data[[col]].isnull(),
                              vmax=vmax_local, vmin=vmin_local)

        if(highlight):
            if(ranked_names is None or ranked_steam is None):
                raise ValueError('STATUS: Improper ranked rect inputs.')
            for rtop in ranked_names:
                y_loc = list(perf_data.index).index(rtop)
                sns_fig.add_patch(Rectangle(xy=(0, y_loc), width=1, height=1,
                                            edgecolor='green', fill=False, lw=3))
            for rstm in ranked_steam:
                y_loc = list(perf_data.index).index(rtop)
                sns_fig.add_patch(Rectangle(xy=(0, y_loc), width=1, height=1,
                                            edgecolor='cyan', fill=False, lw=4))
        sns.set(font_scale=0.6)

    sns_fig.get_figure().savefig(FPATH, bbox_inches='tight')

    print(Fore.GREEN + 'STATUS: Saved variable importance configurations.' + Style.RESET_ALL)


def conditional_drop(data_frame, tbd_list):
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
    data_type_sanitation(_provided_args, _expected_type_args, name)
    data_range_sanitation(_provided_args, _expected_value_args, name)
    """END OF DATA SANITATION"""

    for tb_dropped in tbd_list:
        if(tb_dropped in data_frame.columns):
            data_frame = data_frame.drop(tb_dropped, axis=1)
            print(Fore.GREEN + '> STATUS: {} dropped.'.format(tb_dropped) + Style.RESET_ALL)
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
    data_type_sanitation(_provided_args, _expected_type_args, name)
    data_range_sanitation(_provided_args, _expected_value_args, name)
    """END OF DATA SANITATION"""

    h2o.cluster().show_status() if(show) else None

    return {'cluster_name': cluster.cloud_name,
            'pid': cluster.get_status_details()['pid'][0],
            'version': [int(i) for i in cluster.version.split('.')],
            'run_status': cluster.is_running(),
            'branch_name': cluster.branch_name,
            'uptime_ms': cluster.cloud_uptime_millis,
            'health': cluster.cloud_healthy}


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
    data_type_sanitation(_provided_args, _expected_type_args, name)
    data_range_sanitation(_provided_args, _expected_value_args, name)
    """END OF DATA SANITATION"""

    # SHUT DOWN the cluster after you're done working with it
    h2o_instance.remove_all()
    h2o_instance.cluster().shutdown()
    # Double checking...
    try:
        snapshot(h2o_instance.cluster)
        raise ValueError('ERROR: H2O cluster improperly closed!')
    except Exception:
        pass


def get_aml_objects(project_names):
    objs = []
    for project_name in project_names:
        objs.append(h2o.automl.get_automl(project_name))
    return objs


def exp_cumulative_varimps(project_names):
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
    """DATA SANITATION"""
    _provided_args = locals()
    name = sys._getframe(0).f_code.co_name
    _expected_type_args = {'aml_obj': [h2o.automl.autoh2o.H2OAutoML]}
    _expected_value_args = {'aml_obj': None}
    data_type_sanitation(_provided_args, _expected_type_args, name)
    data_range_sanitation(_provided_args, _expected_value_args, name)
    """END OF DATA SANITATION"""

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
        exp_models = [h2o.get_model(exp_leaderboard[m_num, "model_id"]) for m_num in range(exp_leaderboard.shape[0])]
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
# Diverging: sns.diverging_palette(240, 10, n=9, as_cmap=True)


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
    """DATA SANITATION"""
    _provided_args = locals()
    name = sys._getframe(0).f_code.co_name
    _expected_type_args = {'df': [pd.core.frame.DataFrame],
                           'FPATH': [str],
                           'EXP_NAME': [str],
                           'abs_arg': [bool],
                           'mask': [bool],
                           'annot': [bool],
                           'type_corrs': [list],
                           'cmap': [matplotlib.colors.ListedColormap, matplotlib.colors.LinearSegmentedColormap],
                           'figsize': [tuple],
                           'contrast_factor': [float]}
    _expected_value_args = {'df': None,
                            'FPATH': None,
                            'EXP_PATH': None,
                            'abs_arg': [True, False],
                            'mask': [True, False],
                            'annot': [True, False],
                            'type_corrs': None,
                            'cmap': None,
                            'figsize': None,
                            'contrast_factor': [0.0000001, np.inf]}
    data_type_sanitation(_provided_args, _expected_type_args, name)
    data_range_sanitation(_provided_args, _expected_value_args, name)
    """END OF DATA SANITATION"""

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
for path in [DATA_PATH_PAD, DATA_PATH_WELL]:
    if not (os.path.isfile(path)):
        raise ValueError('ERROR: {data} does not exist in the specificied location.'.format(data=path))

# Import the data from the file
# data_well = h2o.import_file(DATA_PATH_WELL)
data_pad = h2o.import_file(DATA_PATH_PAD)

print(Fore.GREEN + 'STATUS: Server initialized and data imported.' + Style.RESET_ALL)
print(OUT_BLOCK)

_ = """
#######################################################################################################################
##########################################   MINOR DATA MANIPULATION/PREP   ###########################################
#######################################################################################################################
"""
# Table diagnostics
# data_pad = conditional_drop(data_pad, ['C1', 'PRO_Alloc_Water'])
# data_pad = conditional_drop(data_pad, ['C1', 'PRO_Alloc_Water', 'PRO_Pump_Speed'])
# data_pad = conditional_drop(data_pad, ['C1', 'PRO_Alloc_Water', 'PRO_Pump_Speed', 'Bin_1', 'Bin_5'])
RESPONDER = 'PRO_Alloc_Oil'

EXCLUDE = ['C1', 'PRO_Alloc_Water', 'Bin_1', 'Bin_5', 'Date']
data_pad, groupby_options_pad, PREDICTORS = typical_manipulation_h20(data_pad, 'PRO_Pad', EXCLUDE, RESPONDER)
# data_well, groupby_options_well, PREDICTORS = typical_manipulation_h20(data_well, 'PRO_Well', EXCLUDE, RESPONDER)

print(OUT_BLOCK)

_ = """
#######################################################################################################################
#########################################   MODEL TRAINING AND DEVELOPMENT   ##########################################
#######################################################################################################################
"""

print(Fore.GREEN + 'STATUS: Hyperparameter Overview:')
print(Fore.GREEN + '\t* max_runtime_secs\t-> ', MAX_EXP_RUNTIME,
      Fore.GREEN + '\t\t\tThe maximum runtime in seconds that you want to allot in order to complete the model.')
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
    raise RuntimeError('Session forcefully terminated by user during review of hyperparamaters.')

project_names_pad = run_experiment(data_pad, groupby_options_pad, RESPONDER)
# final_cumulative_varimps_well = run_experiment(data_well, groupby_options_well, RESPONDER)


varimps_pad = exp_cumulative_varimps(project_names_pad)

mask_pad = (varimps_pad.mean(axis=1) > 0.0) & (varimps_pad.mean(axis=1) < 1.0)
selective_varimps_pad = varimps_pad[mask_pad].select_dtypes(float)
# mask_well = (varimps_well.mean(axis=1) > 0.0) & (varimps_well.mean(axis=1) < 1.0)
# FILT_final_cumulative_varimps_well = final_cumulative_varimps_well[mask_well]


ranked_names_pad, ranked_steam_pad = plot_varimp_heatmap(selective_varimps_pad,
                                                         'Modeling Reference Files/Round ' +
                                                         '{tag}/macropad_varimps_PAD{tag}.pdf'.format(tag=RUN_TAG))
# ranked_names_well, ranked_steam_well = plot_varimp_heatmap(selective_varimps_well,
#                                                            'Modeling Reference Files/Round ' +
#                                                            '{tag}/macropad_varimps_PWELL{tag}.pdf'.format(tag=RUN_TAG),
#                                                            FIGSIZE=(10, 100),
#                                                            highlight=False,
#                                                            annot=False)

with suppress_stdout():
    correlation_matrix(selective_varimps_pad,
                       EXP_NAME='Aggregated Experiment Results - Pad-Level',
                       FPATH='Modeling Reference Files/Round {tag}/select_var_corrs_{tag}.pdf'.format(tag=RUN_TAG))
# correlation_matrix(selective_varimps_well, EXP_NAME='Aggregated Experiment Results - Well-Level',
#                    FPATH='Modeling Reference Files/Round {tag}/select_var_corrs_PWELL{tag}.pdf'.format(tag=RUN_TAG))


print(Fore.GREEN + 'STATUS: Saved variable importance configurations.' + Style.RESET_ALL)
print(OUT_BLOCK)

_ = """
#######################################################################################################################
#########################################   EVALUATE MODEL PERFORMANCE   ##############################################
#######################################################################################################################
"""

get_aml_objects(project_names_pad)[1].leaderboard
dir(h2o.get_model('GBM_3_AutoML_20210326_170904'))
h2o.get_model('GBM_3_AutoML_20210326_170904').shap_summary_plot(data_pad)

# data_well_pd = pd.read_csv(DATA_PATH_WELL)
# benchline = data_well_pd[data_well_pd[RESPONDER] > 0].groupby(
#     ['Date', 'PRO_Well'])[RESPONDER].sum().reset_index().groupby('PRO_Well').median()
# grouped_tolerance = (PREFERRED_TOLERANCE * benchline).to_dict()[RESPONDER]
perf_pad = model_performance(varimps_pad)

# perf_well = model_performance(varimps_well)
# perf_well['group_type'] = [t[1] for t in perf_well.index.str.split('___GROUP')]

mcmaps = {'R^2': sns.color_palette('rocket_r', as_cmap=True),
          # 'R': sns.color_palette('rocket_r'),
          'MSE': sns.color_palette("coolwarm", as_cmap=True),
          'RMSE': sns.color_palette("coolwarm", as_cmap=True),
          'RMSLE': sns.color_palette("coolwarm", as_cmap=True),
          'MAE': sns.color_palette("coolwarm", as_cmap=True)}
centers = {'R^2': None,
           'MSE': 400,
           'RMSE': 20,
           'RMSLE': None,
           'MAE': None}

plot_model_performance(perf_pad.select_dtypes(float),
                       'Modeling Reference Files/Round {tag}/model_performance_PAD{tag}.pdf'.format(tag=RUN_TAG),
                       mcmaps, centers, ranked_names_pad, ranked_steam_pad,
                       highlight=False,
                       annot=True,
                       annot_size=6,
                       FIGSIZE=(10, 10))
# plot_model_performance(perf_well.select_dtypes(float),
#                        'Modeling Reference Files/Round {tag}/model_performance_PWELL{tag}.pdf'.format(tag=RUN_TAG),
#                        mcmaps, centers, ranked_names_well, ranked_steam_well,
#                        highlight=False,
#                        annot=True,
#                        annot_size=2,
#                        FIGSIZE=(10, 100))


_ = """
#######################################################################################################################
########################################   SHUTDOWN THE SESSION/CLUSTER   #############################################
#######################################################################################################################
"""

if(input('Shutdown Cluster? (Y/N)') == 'Y'):
    shutdown_confirm(h2o)
print(OUT_BLOCK)

# EOF

# EOF

# EOF
