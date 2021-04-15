# @Author: Shounak Ray <Ray>
# @Date:   07-Apr-2021 09:04:99:992  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: ft_eng_math.py
# @Last modified by:   Ray
# @Last modified time: 14-Apr-2021 20:04:78:780  GMT-0600
# @License: [Private IP]

import itertools
import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sympy
from autofeat import AutoFeatRegressor


def divisorGenerator(n):
    large_divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i * i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield int(divisor)


# HYPERPARAMS
RESPONDER = 'PRO_Total_Fluid'
TRAIN_PCT = 0.8
TOP_F = 5

# IMPORTS
df = pd.read_csv('Data/combined_ipc_aggregates.csv').drop('Unnamed: 0', 1)
df = df.sort_values('Date').reset_index(drop=True)

CORE_FEATURES = list(df.columns).copy()

# FEATURE ENGINEERING
groupers = df['PRO_Pad'].unique()
new_ft = {}
for group in groupers:
    subset_df = df[df['PRO_Pad'] == group].reset_index(drop=True).drop('PRO_Pad', 1)

    # # Plotting
    # fig, ax = plt.subplots(figsize=(25, 85), nrows=len(subset_df.select_dtypes(np.number).columns), ncols=1)
    #
    # iter_cols = subset_df.select_dtypes(np.number).columns
    #
    # for col in iter_cols:
    #     subset_df[col].plot(title=col + ' Original', ax=ax[list(iter_cols).index(col)])
    # plt.tight_layout()
    # # Plotting Fixed
    interpolated = subset_df.interpolate('linear').fillna(subset_df.mean()).sort_values('Date')
    # for col in iter_cols:
    #     interpolated[col].plot(title=col + ' Filled', ax=ax[list(iter_cols).index(col)], linewidth=0.9)
    # plt.tight_layout()
    # plt.show()

    SOURCE_DF = interpolated.copy()

    SOURCE_DF.drop('weight', 1, inplace=True)
    for c in SOURCE_DF.select_dtypes(object).columns:
        SOURCE_DF.drop(c, 1, inplace=True)

    msk = np.random.rand(len(SOURCE_DF)) < TRAIN_PCT

    TRAIN = SOURCE_DF[msk].reset_index(drop=True)
    # TEST = SOURCE_DF[~msk].reset_index(drop=True)
    TRAIN_X, TRAIN_Y = SOURCE_DF[[c for c in SOURCE_DF.columns if c not in [RESPONDER]]], SOURCE_DF[[RESPONDER]]
    # TEST_X, TEST_Y = TEST[[c for c in SOURCE_DF.columns if c not in [RESPONDER]]], TEST[[RESPONDER]]

    model_feateng = AutoFeatRegressor(feateng_steps=2, verbose=3)
    feateng_X_TRAIN = model_feateng.fit_transform(TRAIN_X, TRAIN_Y)

    new_ft[group] = (feateng_X_TRAIN, model_feateng.new_feat_cols_)


# REFORMATTING
all_new_fts = list(itertools.chain.from_iterable([tup[1] for tup in new_ft.values()]))
all_new_dfs = pd.concat([tup[0].assign(PRO_Pad=key) for key, tup in new_ft.items()], 0).reset_index(drop=True)
all_new_dfs = all_new_dfs.T.drop_duplicates().T
all_new_dfs.reset_index(drop=False, inplace=True)

NEW_FEATURES = [c for c in all_new_dfs.columns if c not in CORE_FEATURES]

df.reset_index(drop=False, inplace=True)
concatenated = pd.merge(df, all_new_dfs[NEW_FEATURES], on='index', how='inner').drop('index', 1)
concatenated = concatenated.T.drop_duplicates().T.infer_objects()

concatenated.to_csv('Data/combined_ipc_engineered_math.csv')


""""""""""""""""""""""""""""""""""""""""""""" PLOTTING """""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""__"""""""""""""""""""""""""""""""""""""""""""""""""

plt.figure(figsize=(25, 25))
sns.heatmap(concatenated.select_dtypes(np.number).corr())
plt.savefig('Manipulation Reference Files/ft_end_correlations.png', bbox_inches='tight')

# TOP FORMULAS
_temp = pd.DataFrame(Counter(all_new_fts).most_common()[:5], columns=['Transformation', 'Frequency'])
_temp.index = _temp['Transformation']
_temp = _temp.drop('Transformation', 1)
plt.xticks(rotation='vertical')
plt.bar(_temp.index, _temp['Frequency'])
plt.title(f'Top {TOP_F} Engineered Features')

# REMOVE DUPLICATES
all_new_fts = list(set(all_new_fts))
all_new_fts_ltx = [sympy.latex(sympy.sympify(e.replace('_', '').replace('PRO', '')),
                               mul_symbol='dot') for e in all_new_fts]

"""
# PLOTTING ALL GENERATED EQUATIONS
"""
total_cells = len(all_new_fts_ltx)
divisors = list(divisorGenerator(total_cells))
# Find squarest combination
track = {}
for i in divisors:
    divided = int(total_cells / i)
    divisors.remove(divided) if divided in divisors else None
    track[(divided, i)] = abs(divided - i)

nrows, ncols = sorted(sorted(track, reverse=False)[0], reverse=True)

fig, ax = plt.subplots(figsize=(2 * nrows, 2 * ncols), nrows=nrows, ncols=ncols)
relation = np.arange(0, total_cells).reshape(nrows, ncols).tolist()
for row in relation:
    row_pos = relation.index(row)
    for val in row:
        col_pos = row.index(val)
        axis_now = ax[row_pos][col_pos]
        content = all_new_fts_ltx[val]
        exec(f'axis_now.text(0.45, 0.5, r"${content}$", fontsize=14)')
        axis_now.axes.get_xaxis().set_visible(False)
        axis_now.axes.get_yaxis().set_visible(False)
        axis_now.spines['top'].set_visible(False)
        axis_now.spines['right'].set_visible(False)
        axis_now.spines['left'].set_visible(False)
        axis_now.spines['bottom'].set_visible(False)
        plt.tight_layout()
    plt.tight_layout()
plt.show()
fig.savefig('Manipulation Reference Files/Engineered Features.png', dpi=500,
            facecolor='white', bbox_inches='tight')
"""
END
"""


# EOF

# EOF
