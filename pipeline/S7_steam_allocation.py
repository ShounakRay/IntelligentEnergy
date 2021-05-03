# @Author: Shounak Ray <Ray>
# @Date:   14-Apr-2021 09:04:63:632  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: S6_steam_allocation.py
# @Last modified by:   Ray
# @Last modified time: 03-May-2021 14:05:84:848  GMT-0600
# @License: [Private IP]

import os
import pickle
import random
import sys
from io import StringIO
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pipeline.S3_weighting as S3
import seaborn as sns


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


_ = """
#######################################################################################################################
##################################################   HYPERPARAMETERS   ################################################
#######################################################################################################################
"""
# Only needed to get available production wells
DATA_PATH_WELL: Final = 'Data/combined_ipc_engineered_phys.csv'    # Where the client-specific pad data is located
DATA_PATH_WELL: Final = 'Data/combined_ipc_engineered_phys.csv'    # Where the client-specific pad data is located
DATA_PATH_DMATRIX: Final = 'Data/Pickles/DISTANCE_MATRIX.csv'

CLOSENESS_THRESH: Final = 0.1

_ = """
#######################################################################################################################
##################################################   EXPERIMENTATION   ################################################
#######################################################################################################################
"""


def inj_dist_matrix(INJECTOR_COORDINATES):
    df_matrix = pd.DataFrame([], columns=INJECTOR_COORDINATES.keys(), index=INJECTOR_COORDINATES.keys())
    for iwell in df_matrix.columns:
        iwell_coord = INJECTOR_COORDINATES.get(iwell)
        dists = [S3.euclidean_2d_distance(iwell_coord, dyn_coord) for dyn_coord in INJECTOR_COORDINATES.values()]
        df_matrix[iwell] = dists
    np.fill_diagonal(df_matrix.values, np.nan)
    return df_matrix.infer_objects()


def pro_dist_matrix(PRODUCER_COORDINATES, scaled=False):
    per_pwell = {}
    for pwell in PRODUCER_COORDINATES.keys():
        # Get the coordinates (plural) for the specific producer
        pwell_coords = PRODUCER_COORDINATES.get(pwell)
        avg_distance_cd = {}
        for cd in pwell_coords:
            # Get each SPECIFIC coordinate for the specific producer
            avg_distance = {}
            for pwell_name, pwell_coords_again in PRODUCER_COORDINATES.items():
                # Access the coordinates (plural) for the specific producer again
                # print(f'PWELL_FIRST: {pwell}, PWELL_SECOND: {pwell_name}')
                vals = [S3.euclidean_2d_distance(cd, dyn_coord)
                        for dyn_coord in pwell_coords_again]
                # Average distance between one SPECIFIC coordinate of upper-level producer AND all the coordinates
                #   of all the other producers
                avg_distance[pwell_name] = np.mean(vals)
            avg_distance_cd[pwell_coords.index(cd)] = avg_distance
        per_pwell[pwell] = avg_distance_cd

    for pwell in per_pwell.keys():
        per_pwell[pwell] = pd.DataFrame(per_pwell.get(pwell)).T.min().to_dict()
    df_matrix = pd.DataFrame(per_pwell)
    np.fill_diagonal(df_matrix.values, np.nan)

    if(scaled):
        df_matrix = df_matrix / df_matrix.max()

    return df_matrix.infer_objects()


CANDIDATES = _accessories.retrieve_local_data_file('Data/Pickles/WELL_Candidates.pkl', mode=2)
PI_DIST_MATRIX = _accessories.retrieve_local_data_file(DATA_PATH_DMATRIX)
II_DIST_MATRIX = inj_dist_matrix(S3.get_coordinates(data_group='INJECTION'))
PP_DIST_MATRIX = pro_dist_matrix(S3.get_coordinates(data_group='PRODUCTION'))

fig, ax = plt.subplots(ncols=len(CANDIDATES.keys()), nrows=max([len(i) for p, i in CANDIDATES.items()]),
                       figsize=(40, 20))
impact_tracker = {}
for pwell, candidates in CANDIDATES.items():
    impact_tracker[pwell] = {}
    for iwell in candidates:
        axis = ax[candidates.index(iwell)][list(CANDIDATES.keys()).index(pwell)]
        # This are the shortest distances from the current injector to all the producers
        ip_distances = PI_DIST_MATRIX[['PRO_Well', iwell]].set_index('PRO_Well').to_dict().get(iwell)
        ip_distances = dict(sorted(ip_distances.items(), key=lambda tup: tup[1]))
        top_pwell, closest_distance = next(iter(ip_distances.items()))
        search_scope = closest_distance * (1 + CLOSENESS_THRESH)
        impacts_on = {k: v for k, v in ip_distances.items() if v <= search_scope}
        impact_tracker[pwell][iwell] = {k: (v / sum(impacts_on.values())) for k, v in impacts_on.items()}
        axis.set_title(f'{pwell}, {iwell}')
        axis.bar(ip_distances.keys(), ip_distances.values())


# plt.figure(figsize=(15, 5))
# sns.heatmap(PI_DIST_MATRIX.set_index('PRO_Well').select_dtypes(float))

_ = """
#######################################################################################################################
##############################################   NAIVE DISTANCE APPROACH   ############################################
#######################################################################################################################
"""

# Only needed to get available production wells
model_data_agg = pd.read_csv(DATA_PATH_WELL).infer_objects()

pwells = list(model_data_agg['PRO_Well'].unique())
pwell_allocation = {well_name: random.randint(100, 200) for well_name in pwells}
# NOTE: Load in pickled distance matrix
dist_matrix = pd.read_pickle('Data/Pickles/DISTANCE_MATRIX.pkl').infer_objects()
with open('Data/Pickles/DISTANCE_MATRIX.pkl', 'r') as file:
    lines = file.readlines()
    df = pd.read_csv(StringIO(''.join(lines)), delim_whitespace=True)
all_injs = list(dist_matrix.columns)[1:]
# NOTE: Load in producer well candidates
# candidates_by_prodpad = pickle.load(open('Data/candidates_by_prodpad.pkl', 'rb'))
candidates_by_prodwell = pickle.load(open('Data/candidates_by_prodwell.pkl', 'rb'))

# NOTE: Get sliced row from distance matrix for the specific producer well
allocated_steam_values = {}
allocated_steam_props = {}
for pwell in dist_matrix['PRO_Well'].unique():
    # Get candidates for current producer well
    pwell_candidates = candidates_by_prodwell[pwell]
    pwell_allocated_steam = pwell_allocation[pwell]

    # Get sliced row from producer well pad for the specific producer well
    sliced_row = dist_matrix[dist_matrix['PRO_Well'] == pwell][pwell_candidates].reset_index(drop=True).T
    # Normalize sliced row (with all injectors) (0 to 1)
    max = float(sliced_row.max())
    range = max - float(sliced_row.min())
    const = range / 2
    sliced_row[pwell] = [(max - i + const) / (1.5 * range) for i in sliced_row[0]]
    sliced_row[pwell] = sliced_row[pwell] / sliced_row[pwell].sum()
    sliced_row.drop(0, axis=1, inplace=True)

    # Multiply sliced row (with all injectors) by pad allocation and store
    allocated_steam_values[pwell] = list((sliced_row * pwell_allocated_steam).to_dict().values())[0]
    allocated_steam_props[pwell] = list(sliced_row.to_dict().values())[0]

# Reformatting
allocated_steam_values = pd.DataFrame(allocated_steam_values).infer_objects()
allocated_steam_props = pd.DataFrame(allocated_steam_props).infer_objects()

# Reordering
allocated_steam_props = allocated_steam_props.reset_index().sort_values(by='index')
allocated_steam_props.index = allocated_steam_props['index']
allocated_steam_props.drop('index', axis=1, inplace=True)
allocated_steam_props.index.name = None
allocated_steam_values = allocated_steam_values.reset_index().sort_values(by='index')
allocated_steam_values.index = allocated_steam_values['index']
allocated_steam_values.drop('index', axis=1, inplace=True)
allocated_steam_values.index.name = None

# Plotting
fig, ax = plt.subplots(ncols=2, figsize=(12, 10))

sns.heatmap(allocated_steam_props, ax=ax[0])
ax[0].set_title('Relative Allocated Steam values')
ax[0].set_xlabel('Producer Wells')
ax[0].set_ylabel('Candidate Injectors')
sns.heatmap(allocated_steam_values, ax=ax[1])
ax[1].set_title('Absolute Allocated Steam values')
ax[1].set_xlabel('Producer Wells')
ax[1].set_ylabel('Candidate Injectors')
fig.tight_layout()
fig.savefig('Modeling Reference Files/Steam Allocation.pdf', bbox_inches='tight')

# >

# NOTE: Now you have your allocated steam values

# EOF
