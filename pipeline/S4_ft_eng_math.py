# @Author: Shounak Ray <Ray>
# @Date:   07-Apr-2021 09:04:99:992  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: ft_eng_math.py
# @Last modified by:   Ray
# @Last modified time: 17-May-2021 11:05:23:234  GMT-0600
# @License: [Private IP]

import itertools
import math
import os
import sys
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sympy
from autofeat import AutoFeatRegressor


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


if True:
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

    # import _context_managers
    # import _multiprocessed.defs as defs
    # import _traversal


# _traversal.print_tree_to_txt(PATH='_configs/FILE_STRUCTURE.txt')

_ = """
#######################################################################################################################
#############################################   HYPERPARAMETER SETTINGS   #############################################
#######################################################################################################################
"""

# HYPERPARAMS
# TRAIN_PCT = 0.8
# TOP_F = 5

_ = """
#######################################################################################################################
###############################################   FUNCTION DEFINITIONS   ##############################################
#######################################################################################################################
"""


def minor_processing(df):
    if('Unnamed: 0' in df.columns):
        df.drop('Unnamed', axis=1, inplace=True)
    if('Date' in df.columns):
        df = df.sort_values('Date').reset_index(drop=True)

    return df


def generate_new_features(df, RESPONDER, group_colname='PRO_Pad', save=True, **kwargs):
    CORE_FEATURES = list(df.columns).copy()

    groupers = df[group_colname].unique()
    new_ft = {}
    for group in groupers:
        subset_df = df[df[group_colname] == group].reset_index(drop=True).drop(group_colname, 1)

        interpolated = subset_df.interpolate('linear').fillna(subset_df.mean())

        SOURCE_DF = interpolated.copy()

        if 'weight' in SOURCE_DF.columns:
            SOURCE_DF.drop('weight', 1, inplace=True)
        for c in SOURCE_DF.select_dtypes(object).columns:
            SOURCE_DF.drop(c, 1, inplace=True)

        # msk = np.random.rand(len(SOURCE_DF)) < kwargs.get(TRAIN_PCT)
        # TRAIN = SOURCE_DF[msk].reset_index(drop=True)
        # TEST = SOURCE_DF[~msk].reset_index(drop=True)
        TRAIN_X, TRAIN_Y = SOURCE_DF[[c for c in SOURCE_DF.columns if c not in [RESPONDER]]], SOURCE_DF[[RESPONDER]]
        # TEST_X, TEST_Y = TEST[[c for c in SOURCE_DF.columns if c not in [RESPONDER]]], TEST[[RESPONDER]]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _accessories._print(f'Performing feature engineering and selection for group "{group}"...')
            model_feateng = AutoFeatRegressor(feateng_steps=2, verbose=0)
            feateng_X_TRAIN = model_feateng.fit_transform(TRAIN_X, TRAIN_Y)

        new_ft[group] = (feateng_X_TRAIN, model_feateng.new_feat_cols_)

    if(save):
        _accessories.save_local_data_file(new_ft, f'Data/Pickles/FEATENGS_[{group_colname}, Engineered].pkl')

    return new_ft, CORE_FEATURES


def extract_specs(new_ft, group_colname, CORE_FEATURES):
    # REFORMATTING
    all_new_fts = list(itertools.chain.from_iterable([tup[1] for tup in new_ft.values()]))
    if(group_colname == 'PRO_Pad'):
        all_new_dfs = pd.concat([tup[0].assign(PRO_Pad=key) for key, tup in new_ft.items()], 0).reset_index(drop=True)
    elif(group_colname == 'PRO_Well'):
        all_new_dfs = pd.concat([tup[0].assign(PRO_Well=key) for key, tup in new_ft.items()], 0).reset_index(drop=True)
    all_new_dfs = all_new_dfs.T.drop_duplicates().T
    all_new_dfs.reset_index(drop=False, inplace=True)

    NEW_FEATURES = [c for c in all_new_dfs.columns if c not in CORE_FEATURES]

    return all_new_dfs, all_new_fts, NEW_FEATURES


def naive_merge_all(df, all_new_dfs, NEW_FEATURES):
    _accessories._print('WARNING: All generated features were merged with provided source data.\n' +
                        'This is a quick and dirty way to proceed with engineered features, but may results in ' +
                        'spurious results...', color='LIGHTRED_EX')
    df = df.reset_index(drop=False, inplace=False)
    concatenated = pd.merge(df, all_new_dfs[NEW_FEATURES], on='index', how='inner').drop('index', 1)
    concatenated = concatenated.T.drop_duplicates().T.infer_objects()
    if 'level_0' in concatenated:
        concatenated.drop('level_0', axis=1, inplace=True)

    return concatenated


# def view_generated_eqs(df, all_new_fts, TOP_F):
#     def divisorGenerator(n):
#         large_divisors = []
#         for i in range(1, int(math.sqrt(n) + 1)):
#             if n % i == 0:
#                 yield i
#                 if i * i != n:
#                     large_divisors.append(n / i)
#         for divisor in reversed(large_divisors):
#             yield int(divisor)
#
#     with _accessories.suppresss_stdout():
#         plt.figure(figsize=(25, 25))
#         sns.heatmap(df.select_dtypes(np.number).corr())
#         plt.savefig('Manipulation Reference Files/ft_end_correlations.png', bbox_inches='tight')
#
#         # TOP FORMULAS
#         _temp = pd.DataFrame(Counter(all_new_fts).most_common()[:TOP_F], columns=['Transformation', 'Frequency'])
#         _temp.index = _temp['Transformation']
#         _temp = _temp.drop('Transformation', 1)
#         plt.xticks(rotation='vertical')
#         plt.bar(_temp.index, _temp['Frequency'])
#         plt.title(f'Top {TOP_F} Engineered Features')
#
#         # REMOVE DUPLICATES
#         all_new_fts = list(set(all_new_fts))
#         all_new_fts_ltx = [sympy.latex(sympy.sympify(e.replace('_', '').replace('PRO', '')),
#                                        mul_symbol='dot') for e in all_new_fts]
#
#         # PLOTTING ALL GENERATED EQUATIONS
#         total_cells = len(all_new_fts_ltx)
#         divisors = list(divisorGenerator(total_cells))
#         # Find squarest combination
#         track = {}
#         for i in divisors:
#             divided = int(total_cells / i)
#             divisors.remove(divided) if divided in divisors else None
#             track[(divided, i)] = abs(divided - i)
#
#         nrows, ncols = sorted(sorted(track, reverse=False)[0], reverse=True)
#
#         fig, ax = plt.subplots(figsize=(2 * nrows, 2 * ncols), nrows=nrows, ncols=ncols)
#         relation = np.arange(0, total_cells).reshape(nrows, ncols).tolist()
#         for row in relation:
#             row_pos = relation.index(row)
#             for val in row:
#                 col_pos = row.index(val)
#                 axis_now = ax[row_pos][col_pos]
#                 content = all_new_fts_ltx[val]
#                 exec(f'axis_now.text(0.45, 0.5, r"${content}$", fontsize=14)')
#                 axis_now.axes.get_xaxis().set_visible(False)
#                 axis_now.axes.get_yaxis().set_visible(False)
#                 axis_now.spines['top'].set_visible(False)
#                 axis_now.spines['right'].set_visible(False)
#                 axis_now.spines['left'].set_visible(False)
#                 axis_now.spines['bottom'].set_visible(False)
#                 plt.tight_layout()
#             plt.tight_layout()
#         plt.show()
#         fig.savefig('Manipulation Reference Files/Engineered Features.png', dpi=500,
#                     facecolor='white', bbox_inches='tight')


_ = """
#######################################################################################################################
##################################################   CORE EXECUTION  ##################################################
#######################################################################################################################
"""


def _FEATENG_MATH(data=None, RESPONDER='PRO_Total_Fluid', skip_save=True):
    if data is None:
        _accessories._print('Ingesting PHYSICS ENGINEERD, WEIGHTED data...', color='LIGHTYELLOW_EX')
        _accessories._print('WARNING: Mathematical feature engineering is isolated from modelling component.\n' +
                            'This is not representative of the production pipeline.')
        loaded_data = _accessories.retrieve_local_data_file('Data/S3 Files/combined_ipc_aggregates.csv')
    elif type(data) is pd.core.frame.DataFrame:
        _accessories._print('Model-group specific data is loaded for feature engineering...', color='LIGHTYELLOW_EX')
        loaded_data = data.infer_objects()
    else:
        raise ValueError('Incorrectly formatted data inputted. Not DataFrame or defaulted None.')

    DATASETS = {'WEIGHTED': loaded_data}

    _accessories._print('Minor processing on data...', color='LIGHTYELLOW_EX')
    DATASETS['WEIGHTED'] = minor_processing(DATASETS['WEIGHTED'])

    _accessories._print('Generating new features...', color='LIGHTYELLOW_EX')
    groups_engdfs, CORE_FEATURES = generate_new_features(DATASETS['WEIGHTED'], RESPONDER, 'PRO_Pad')

    if(not skip_save):
        _accessories._print('Extracting specs of generated features...', color='LIGHTYELLOW_EX')
        all_new_dfs, all_new_fts, NEW_FEATURES = extract_specs(groups_engdfs, 'PRO_Pad', CORE_FEATURES)

        _accessories._print('Naively merging and saving data...', color='LIGHTYELLOW_EX')
        DATASETS['CONCATENATED'] = naive_merge_all(DATASETS['WEIGHTED'], all_new_dfs.copy(), NEW_FEATURES)

        _accessories._print('Merging and saving...', color='LIGHTYELLOW_EX')
        _accessories.finalize_all(DATASETS, skip=[])
        _accessories.save_local_data_file(DATASETS['CONCATENATED'], 'Data/S4 Files/combined_ipc_engineered_math.csv')
    else:
        return groups_engdfs, CORE_FEATURES


if __name__ == '__main__':
    _FEATENG_MATH()

# EOF

# EOF
