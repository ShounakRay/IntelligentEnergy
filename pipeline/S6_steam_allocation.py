# @Author: Shounak Ray <Ray>
# @Date:   14-Apr-2021 09:04:63:632  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: S6_steam_allocation.py
# @Last modified by:   Ray
# @Last modified time: 14-Apr-2021 20:04:72:721  GMT-0600
# @License: [Private IP]

import pickle
import random
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Only needed to get available production wells
DATA_PATH_WELL: Final = 'Data/combined_ipc_engineered_phys.csv'    # Where the client-specific pad data is located
DATA_PATH_WELL: Final = 'Data/combined_ipc_engineered_phys.csv'    # Where the client-specific pad data is located

# Only needed to get available production wells
model_data_agg = pd.read_csv(DATA_PATH_WELL).drop('Unnamed: 0', axis=1).infer_objects()

pwells = list(model_data_agg['PRO_Well'].unique())
pwell_allocation = {well_name: random.randint(100, 200) for well_name in pwells}
# NOTE: Load in pickled distance matrix
dist_matrix = pd.read_pickle('Data/injector_producer_dist_matrix.pkl').infer_objects()
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
