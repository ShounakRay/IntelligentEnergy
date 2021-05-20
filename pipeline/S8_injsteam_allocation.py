# @Author: Shounak Ray <Ray>
# @Date:   14-Apr-2021 09:04:63:632  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: S6_steam_allocation.py
# @Last modified by:   Ray
# @Last modified time: 20-May-2021 11:05:91:918  GMT-0600
# @License: [Private IP]

# import os
# import sys
from typing import Final

if __name__ == '__main__':
    # import matplotlib
    # matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt

import _references._accessories as _accessories
import numpy as np
import pandas as pd
import pipeline.S3_weighting as S3
# import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# TODO: Heel constraints
# TODO: Fiber Consideration


# def ensure_cwd(expected_parent):
#     init_cwd = os.getcwd()
#     sub_dir = init_cwd.split('/')[-1]
#
#     if(sub_dir != expected_parent):
#         new_cwd = init_cwd
#         print(f'\x1b[91mWARNING: "{expected_parent}" folder was expected to be one level ' +
#               f'lower than parent directory! Project CWD: "{sub_dir}" (may already be properly configured).\x1b[0m')
#     else:
#         new_cwd = init_cwd.replace('/' + sub_dir, '')
#         print(f'\x1b[91mWARNING: Project CWD will be set to "{new_cwd}".')
#         os.chdir(new_cwd)
#
#
# if True:
#     try:
#         _EXPECTED_PARENT_NAME = os.path.abspath(__file__ + "/..").split('/')[-1]
#     except Exception:
#         _EXPECTED_PARENT_NAME = 'pipeline'
#         print('\x1b[91mWARNING: Seems like you\'re running this in a Python interactive shell. ' +
#               f'Expected parent is manually set to: "{_EXPECTED_PARENT_NAME}".\x1b[0m')
#     ensure_cwd(_EXPECTED_PARENT_NAME)
#     sys.path.insert(1, os.getcwd() + '/_references')
#     sys.path.insert(1, os.getcwd() + '/' + _EXPECTED_PARENT_NAME)
#     import _accessories
#     import _traversal
#
# _traversal.print_tree_to_txt(PATH='_configs/FILE_STRUCTURE.txt')

_ = """
#######################################################################################################################
#############################################   HYPERPARAMETER SETTINGS   #############################################
#######################################################################################################################
"""

CLOSENESS_THRESH_PI: Final = 0.1
CLOSENESS_THRESH_II: Final = 0.1
RESOLUTION = 0.01

_ = """
#######################################################################################################################
##############################################   FUNCTION DEFINITIONS   ###############################################
#######################################################################################################################
"""


def distance_matrix(request, data, scaled=False):
    if(request == 'II'):
        INJECTOR_COORDINATES = data
        df_matrix = pd.DataFrame([], columns=INJECTOR_COORDINATES.keys(), index=INJECTOR_COORDINATES.keys())
        for iwell in df_matrix.columns:
            iwell_coord = INJECTOR_COORDINATES.get(iwell)
            dists = [S3.euclidean_2d_distance(iwell_coord, dyn_coord) for dyn_coord in INJECTOR_COORDINATES.values()]
            df_matrix[iwell] = dists
        np.fill_diagonal(df_matrix.values, np.nan)

        if(scaled):
            df_matrix = df_matrix / df_matrix.max()

        df_matrix = df_matrix.infer_objects()
    elif(request == 'PP'):
        PRODUCER_COORDINATES = data
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
        df_matrix = df_matrix.infer_objects()
    else:
        raise ValueError('Improper argument for `request` in `distance_matrix`')

    return df_matrix


def PI_imapcts(CANDIDATES, PI_DIST_MATRIX, CLOSENESS_THRESH_PI=CLOSENESS_THRESH_PI, plot=False):
    if(plot):
        fig, ax = plt.subplots(ncols=len(CANDIDATES.keys()), nrows=max([len(i) for p, i in CANDIDATES.items()]),
                               figsize=(60, 30))
    impact_tracker = {}
    for pwell, candidates in CANDIDATES.items():
        impact_tracker[pwell] = {}
        for iwell in candidates:
            # This are the shortest distances from the current injector to all the producers
            ip_distances = PI_DIST_MATRIX[['PRO_Well', iwell]].set_index('PRO_Well').to_dict().get(iwell)
            ip_distances = dict(sorted(ip_distances.items(), key=lambda tup: tup[1]))
            top_pwell, closest_distance = next(iter(ip_distances.items()))
            search_scope = closest_distance * (1 + CLOSENESS_THRESH_PI)
            impacts_on = {k: v for k, v in ip_distances.items() if v <= search_scope}
            impact_tracker[pwell][iwell] = {k: (v / sum(impacts_on.values())) for k, v in impacts_on.items()}

            if(plot):
                axis = ax[candidates.index(iwell)][list(CANDIDATES.keys()).index(pwell)]
                axis.set_title(f'{pwell}, {iwell}')
                axis.bar({k: v for k, v in ip_distances.items() if k in impacts_on.keys()}.keys(),
                         {k: v for k, v in ip_distances.items() if k in impacts_on.keys()}.values(), color='red')
                axis.bar({k: v for k, v in ip_distances.items() if k not in impacts_on.keys()}.keys(),
                         {k: v for k, v in ip_distances.items() if k not in impacts_on.keys()}.values(), color='blue')
    if(plot):
        plt.tight_layout()
        plt.savefig('Manipulation Reference Files/Final Schematics/Candidate_Impacts.png', bbox_inches='tight')

    cleaned_imapct_tracker = dict(sorted(pd.DataFrame(impact_tracker).fillna(method='bfill',
                                                                             axis=1).iloc[:, 0].to_dict().items(),
                                         key=lambda tup: tup[0]))

    # Identify injectors that are isolated to a single well.
    isolated = []
    for name, impacts in cleaned_imapct_tracker.items():
        if len(impacts) == 1:
            isolated.append(name)

    return cleaned_imapct_tracker, isolated


def II_impacts(II_DIST_MATRIX, CLOSENESS_THRESH_II=CLOSENESS_THRESH_II):
    # NOTE: The purpose of this is to ballpark steam connections between injectors
    links = {}
    for iwell in II_DIST_MATRIX.columns:
        # This is the distance between the iterated injector and all the other injectors
        slice = dict(sorted(II_DIST_MATRIX[iwell].dropna().to_dict().items(), key=lambda x: x[1]))
        thresh = list(slice.values())[0] * (1 + CLOSENESS_THRESH_II)
        slice = {iwell: dist for iwell, dist in slice.items() if dist <= thresh}
        slice = {iwell: dist / sum(list(slice.values())) for iwell, dist in slice.items() if dist <= thresh}
        links[iwell] = slice

    isolates = []
    for iwell, options in links.items():
        if len(options) == 1:
            isolates.append(iwell)

    return links, isolates


# def produce_search_space(CANDIDATES, PI_DIST_MATRIX, II_DIST_MATRIX, RESOLUTION=RESOLUTION):
#     def optimal_injectors(isolates_PI, isolates_II):
#         return tuple(set([e for e in isolates_PI if e in isolates_II] + [e for e in isolates_II if e in isolates_PI]))
#
#     # 3 minutes and 15 seconds: RES = 0.025
#     # _ : RES = 0.01
#     search_space = {}
#     for thresh_PI in np.arange(0.0, 1 + RESOLUTION, RESOLUTION):
#         print(f'thresh_PI: {thresh_PI}')
#         search_space[thresh_PI] = {}
#         for thresh_II in np.arange(0.0, 1 + RESOLUTION, RESOLUTION):
#             impact_tracker_PI, isolates_PI = PI_imapcts(CANDIDATES, PI_DIST_MATRIX,
#                                                         CLOSENESS_THRESH_PI=thresh_PI)
#             impact_tracker_II, isolates_II = II_impacts(II_DIST_MATRIX,
#                                                         CLOSENESS_THRESH_II=thresh_II)
#             optimals = optimal_injectors(isolates_PI, isolates_II)
#             search_space[thresh_PI][thresh_II] = optimals
#     search_space_df = pd.DataFrame(search_space).reset_index().infer_objects()
#     _accessories.save_local_data_file(search_space_df, 'Data/S8 Files/threshold_search_space.csv')
#     print('SAVED')
#
#     return search_space_df

# def retrieve_search_space(min_bound=0.5, early=False):
#     search_space = _accessories.retrieve_local_data_file(
#         'Data/S8 Files/threshold_search_space.csv').drop('Unnamed: 0', 1)
#
#     if early:
#         return search_space
#
#     search_space_df = search_space.set_index('index').applymap(lambda x: len(x)).reset_index()
#     search_space_df = pd.melt(search_space_df, id_vars='index', value_vars=list(search_space_df)[1:]).infer_objects()
#     search_space_df.columns = ['thresh_PI', 'thresh_II', 'n_optimal']
#     search_space_df['thresh_II'] = search_space_df['thresh_II'].astype(float)
#     search_space_df['thresh_PI'] = search_space_df['thresh_PI'].astype(float)
#     search_space_df = search_space_df[(search_space_df['thresh_PI'] < min_bound) &
#                                       (search_space_df['thresh_II'] < min_bound)].reset_index(drop=True)
#
#     return search_space_df


def plot_search_space(search_space_df, cmap=cm.turbo):
    ax = Axes3D(plt.figure())
    ax.plot_trisurf(search_space_df['thresh_PI'], search_space_df['thresh_II'], search_space_df['n_optimal'],
                    cmap=cmap)
    plt.title('Search Space When Finding Optimal Injectors')
    ax.set_xlabel('thresh_PI')
    ax.set_ylabel('thresh_II')
    ax.set_zlabel('n_optimal')
    plt.tight_layout()
    plt.show()


def naive_distance_allocation(PI_DIST_MATRIX, CANDIDATES, pwell_allocation, format='dict',
                              min_steam=0.08, base_start=0, base_max=10**4):
    def reformat_to_df(allocated_steam_values, allocated_steam_props):
        # Reformatting
        allocated_steam_values = pd.DataFrame(allocated_steam_values).infer_objects()
        allocated_steam_props = pd.DataFrame(allocated_steam_props).infer_objects()
        # Reordering
        allocated_steam_props = allocated_steam_props.reset_index().sort_values(by='index').set_index('index')
        allocated_steam_props.index.name = None
        allocated_steam_values = allocated_steam_values.reset_index().sort_values(by='index').set_index('index')
        allocated_steam_values.index.name = None

        return allocated_steam_values, allocated_steam_props

    allocated_steam_values = {}
    allocated_steam_props = {}
    for pwell in PI_DIST_MATRIX['PRO_Well'].unique():
        # Get candidates for current producer well
        pwell_candidates = CANDIDATES[pwell]
        pwell_allocated_steam = pwell_allocation[pwell]

        # Get sliced row from producer well pad for the specific producer well
        sliced_row = PI_DIST_MATRIX[PI_DIST_MATRIX['PRO_Well'] == pwell][pwell_candidates].reset_index(drop=True).T
        sliced_row = sliced_row.to_dict()[0]
        # Normalize sliced row (with all injectors) (0 to 1)
        for base in np.arange(base_start, base_max + 1, 1):
            maximum = float(max(sliced_row.values()))
            minimum = float(min(sliced_row.values()))
            range = maximum - minimum
            transformed = {inj: (maximum - i + base) / (range + base) for inj, i in sliced_row.items()}
            summed = sum(transformed.values())
            transformed = {inj: i / summed for inj, i in transformed.items()}

            if(min(transformed.values()) < min_steam):
                if(base == base_max):
                    _accessories._print(f'Minimum: {min(transformed.values())} for {pwell}\n' +
                                        f'The maximum bound of base was too low at {base_max}. Moving forward...',
                                        color='LIGHTRED_EX')
                    break
                continue
            else:  # elif(min(transformed.values()) >= min_steam)
                break

        # Multiply sliced row (with all injectors) by pad allocation and store
        allocated_steam_values[pwell] = {inj: i * pwell_allocated_steam for inj, i in transformed.items()}
        allocated_steam_props[pwell] = transformed
    out = reformat_to_df(allocated_steam_values,
                         allocated_steam_props) if format == 'df' else (allocated_steam_values, allocated_steam_props)
    return out


def plot_relative_allocations(suggestions):
    fig, ax = plt.subplots(figsize=(10, 28), nrows=len(suggestions['PRO_Well'].unique()))
    for pwell, group_df in suggestions.groupby('PRO_Well'):
        axis = ax[list(suggestions['PRO_Well'].unique()).index(pwell)]
        axis.set_title(f'Production Well: {pwell}')
        group_df.plot(x='Candidate_Injector', y='Candidate_Proportion', ax=axis, kind='bar')
    plt.tight_layout()


def plot_producer_delta(suggestions):
    _temp = suggestions[['PRO_Well', 'Delta']].drop_duplicates().reset_index(drop=True).set_index('PRO_Well')
    fig, ax = plt.subplots(figsize=(12, 8))
    _temp.plot(kind='bar', ax=ax)
    plt.title('Delta from original allocation to revised allocation')


_ = """
#######################################################################################################################
#####################################################   WRAPPERS  #####################################################
#######################################################################################################################
"""


def initial_allocations(allocated_steam_props: dict,
                        impact_tracker_PI: pd.core.frame.DataFrame, impact_tracker_II: pd.core.frame.DataFrame):
    # Determining injector scores and rankings
    injector_tracks = []
    for pwell, allocations in allocated_steam_props.items():
        for inj, alloc_prop in allocations.items():
            # Get producer impacts
            impact_on_producer = impact_tracker_PI.get(inj)
            importance_candidate = impact_on_producer.get(pwell)
            isolate_candidate = True if len(impact_on_producer) == 1 else False

            if(isolate_candidate):
                state_on_candidate = 'AMAZING'
            else:
                state_on_candidate = 'OKAY'
            score_on_candidate = importance_candidate
            dist_on_candidate = 1 / len(impact_on_producer)

            # IDEA: Get injector impacts
            secondary_impact_on_injector = impact_tracker_II.get(inj)
            for ext_inj, importance_secondary_injector in secondary_impact_on_injector.items():
                secondary_impact_on_producer = impact_tracker_PI.get(inj)
                # Does this external injector impact the current producer?
                secondary_isolate_producer = True if pwell in secondary_impact_on_producer else False
                if(secondary_isolate_producer):
                    if len(secondary_impact_on_producer) == 1:
                        status_on_external = 'AMAZING'
                    elif len(secondary_impact_on_producer) > 1:
                        status_on_external = 'OKAY'
                    score_on_external = 1 / len(secondary_impact_on_producer)
                    dist_on_external = importance_secondary_injector
                else:
                    status_on_external = 'POOR'
                    score_on_external = 0
                    dist_on_external = 0
                injector_tracks.append([pwell, inj, alloc_prop,
                                        state_on_candidate, score_on_candidate, dist_on_candidate,
                                        ext_inj, status_on_external, score_on_external, dist_on_external])

    decisions = pd.DataFrame(injector_tracks, columns=['PRO_Well',
                                                       'Candidate_Injector', 'Naive_Allocation', 'Candidate_Decision',
                                                       'Candidate_Distance_Importance',
                                                       'Candidate_Distributed_Importance',
                                                       'External_Injector', 'External_Decision',
                                                       'External_Distance_Importance',
                                                       'External_Distributed_Importance'])
    # NOTE: Any NaN values in the 'Candidate_Distance_Importance' column means that the original candidates for the
    # producer well included this injector. However, the producer–injector matrix did not due to a low
    #   `CLOSENESS_THRESH_PI` threshold. This column will be forced to 1.0 (since candidates matter the most) and the
    #   respective 'Candidate_Decision' column value will be switched from 'AMAZING' to 'FORCED'
    decisions['Candidate_Decision'] = decisions.apply(lambda row: 'FORCED'
                                                      if pd.isna(row['Candidate_Distance_Importance'])
                                                      else row['Candidate_Decision'], axis=1)
    decisions['Candidate_Distance_Importance'] = decisions.apply(lambda row: 1.0
                                                                 if pd.isna(row['Candidate_Distance_Importance'])
                                                                 else row['Candidate_Distance_Importance'], axis=1)
    decisions['Final_Factor'] = decisions.apply(lambda row: np.average([row['Candidate_Distance_Importance'],
                                                                        row['Candidate_Distributed_Importance'],
                                                                        row['External_Distance_Importance'],
                                                                        row['External_Distributed_Importance']],
                                                                       weights=[2, 2, 1, 1]), axis=1)
    decisions['Revised_Allocation'] = decisions['Naive_Allocation'] * decisions['Final_Factor']
    decisions['DELTA'] = decisions['Naive_Allocation'] - decisions['Revised_Allocation']
    # sns.histplot(decisions['DELTA'])

    return decisions.infer_objects()


def accounted_for(decisions, pwell_allocation, group_name='PRO_Well', pad_filter=['A', 'B']):
    # Check how much steam is accounted and unaccounted for.
    # accounted_proportions = {}
    accounted_units = {}
    for pwell, group_df in decisions.groupby([group_name]):
        proportion_covered = group_df.groupby('Candidate_Injector')['Revised_Allocation'].mean().sum()
        # accounted_proportions[pwell] = proportion_covered
        accounted_units[pwell] = proportion_covered * pwell_allocation.get(pwell)

    pwell_allocation = {k: v for k, v in pwell_allocation.items() if k in ['AP2', 'AP3', 'AP4',
                                                                           'AP5', 'AP6', 'AP7', 'AP8',
                                                                           'BP1', 'BP2', 'BP3', 'BP4',
                                                                           'BP5', 'BP6']}
    accounted_units = {k: v for k, v in accounted_units.items() if k in ['AP2', 'AP3', 'AP4',
                                                                                       'AP5', 'AP6', 'AP7', 'AP8',
                                                                                       'BP1', 'BP2', 'BP3', 'BP4',
                                                                                       'BP5', 'BP6']}
    available_REAL = sum(pwell_allocation.values())
    available_FINAL = sum(accounted_units.values())
    units_remaining = available_REAL - available_FINAL

    return accounted_units, units_remaining


def maximize_allocations(pwell_allocation, accounted_units, units_remaining, decisions):
    pwell_trimmed_dist = {k: v / sum(accounted_units.values())
                          for k, v in accounted_units.items()}
    FINALE = []
    if units_remaining > 0:
        _accessories._print('Units remaining to be allocated.')
        # These are allocations on the producer well level
        additional_units = {k: v * units_remaining for k, v in pwell_trimmed_dist.items()}
        final_allocations = {k: v + accounted_units.get(k) for k, v in additional_units.items()}
        # These are allocations on the injector well level (from the producer level)
        # NOTE: How to distribute `pwell_trimmed_dist` to to the injectors
        for pwell, group_df in decisions.groupby(['PRO_Well']):
            proportion_per_iwell = group_df.groupby('Candidate_Injector')['Revised_Allocation'].mean().to_dict()
            proportion_per_iwell = {k: v / sum(proportion_per_iwell.values()) for k, v in proportion_per_iwell.items()}
            units_to_allocate = final_allocations.get(pwell)
            final_unit_allocation = {k: v * units_to_allocate for k, v in proportion_per_iwell.items()}
            for iwell in proportion_per_iwell.keys():
                FINALE.append((pwell, additional_units.get(pwell), final_allocations.get(pwell), iwell,
                               proportion_per_iwell.get(iwell), final_unit_allocation.get(iwell)))
    elif units_remaining == 0:
        _accessories._print('All units were already allocated.')
    else:
        _accessories._print('ERROR: Units were over-allocated')

    suggestions = pd.DataFrame(FINALE, columns=['PRO_Well', 'PRO_Well_Additional_Units', 'PRO_Well_Final_Allocation',
                                                'Candidate_Injector', 'Candidate_Proportion', 'Candidate_Units'])
    suggestions['PRO_Well_Initial_Allocation'] = [pwell_allocation.get(x) for x in suggestions['PRO_Well']]
    suggestions['Delta'] = suggestions['PRO_Well_Final_Allocation'] - suggestions['PRO_Well_Initial_Allocation']
    suggestions = suggestions[['PRO_Well', 'PRO_Well_Additional_Units', 'PRO_Well_Initial_Allocation', 'Delta',
                               'PRO_Well_Final_Allocation', 'Candidate_Injector', 'Candidate_Proportion',
                               'Candidate_Units']]

    return suggestions.infer_objects()


def constrain_allocations(constraints, suggestions):
    for pro_well, group_df in suggestions.groupby('PRO_Well'):
        subset = group_df.drop('PRO_Well', axis=1).set_index('Candidate_Injector')['Candidate_Units'].to_dict()
        diff_store = {}
        room_up_store = {}
        room_down_store = {}
        for injector, initial_allocation in subset.items():
            lower_bound, upper_bound = constraints.get(injector)
            room_up = np.nan
            room_down = np.nan
            if (lower_bound <= initial_allocation) and (initial_allocation <= upper_bound):  # Within bounds
                difference = 0
                room_up = upper_bound - initial_allocation
                room_down = initial_allocation - lower_bound
            elif initial_allocation <= lower_bound:  # Too low
                difference = initial_allocation - lower_bound
            elif initial_allocation >= upper_bound:  # Too high
                difference = initial_allocation - upper_bound
            diff_store[injector] = difference
            room_up_store[injector] = room_up
            room_down_store[injector] = room_down
        # This value could be positive (too above constraints) or negative (too below constraints)
        delta_to_distribute = sum(list(diff_store.values()))
        suggestions.loc[suggestions.index[group_df.index], 'Room_Up'] = list(room_up_store.values())
        suggestions.loc[suggestions.index[group_df.index], 'Room_Down'] = list(room_down_store.values())
        suggestions.loc[suggestions.index[group_df.index], ['Constraint_Delta']] = list(diff_store.values())
        suggestions.loc[suggestions.index[group_df.index], 'Well_Net_Delta'] = delta_to_distribute

        # Determine producer-level injector adjustment (manipulate the ones within constraint)
        local_deltas = suggestions.loc[suggestions.index[group_df.index], ['Candidate_Injector', 'Constraint_Delta',
                                                                           'Room_Up', 'Room_Down',
                                                                           'Candidate_Proportion', 'Candidate_Units']]
        # Injectors which originally fell out of bounds
        to_be_changed = local_deltas[~(local_deltas['Constraint_Delta'] == 0.0)
                                     ].dropna(axis=1).set_index('Candidate_Injector')['Constraint_Delta'].to_dict()
        # Injects which originally fell out of bounds
        stars_to_manipulate = local_deltas[local_deltas['Constraint_Delta'] == 0.0].set_index('Candidate_Injector')
        # Re-normalize the stars
        injector_importances = stars_to_manipulate['Candidate_Proportion'].to_dict()
        # The values here could be positive (too above constraints) or negative (too below constraints)
        naive_distribution = {k: delta_to_distribute * v / sum(injector_importances.values())
                              for k, v in injector_importances.items()}

        if len(to_be_changed) == 0 or to_be_changed is None:
            continue
        if len(stars_to_manipulate) == 0 or stars_to_manipulate is None:
            continue

        # For the injectors that are within contraints and MAXIMALLY a candidate, change them a lot, proportionally
        print(f'WELL: {pro_well}: {delta_to_distribute}')
        if delta_to_distribute > 0:
            # This means that there is "extra" steam to be added to valid inejctors
            # Deduct reccomended steam values to available steam in contraints
            available_room = stars_to_manipulate['Room_Up'].to_dict()
            delta_from_rooms = {k: v - available_room.get(k) for k, v in naive_distribution.items()}
            # Only for positive values (not enough steam available given contraints to push so high)
            delta_filtered = {k: v for k, v in delta_from_rooms.items() if v > 0}
            could_not_allocate = sum(delta_filtered.values())

            # Aggregate the allocations (for star wells and non-star wells)
            stars_finalized_allocations = {k: naive_distribution.get(k) - v for k, v in delta_filtered.items()}
            stars_finalized_allocations = dict(naive_distribution, **stars_finalized_allocations)
            original_base = local_deltas.set_index('Candidate_Injector')['Candidate_Units'].to_dict()
            stars_finalized_allocations = {k: original_base.get(k) + v for k, v in stars_finalized_allocations.items()}
            to_be_changed = {k: original_base.get(k) + (could_not_allocate * v /
                                                        sum(to_be_changed.values())) for k, v in to_be_changed.items()}
            finalized_allocations = dict(to_be_changed, **stars_finalized_allocations)

            # Change the dataset
            _just_for_order = list(local_deltas['Candidate_Injector'].values)
            finalized_allocations = dict(sorted(finalized_allocations.items(),
                                                key=lambda pair: _just_for_order.index(pair[0])))

            suggestions.loc[suggestions.index[group_df.index],
                            'Finalized_Allocations'] = list(finalized_allocations.values())
            suggestions.loc[suggestions.index[group_df.index],
                            'Unable_to_Allocate'] = could_not_allocate

        elif delta_to_distribute < 0:
            # This means that there is steam that must be taken away from valid injectors
            naive_distribution = {k: -1 * v for k, v in naive_distribution.items()}
            # Deduct reccomended steam values to available steam in contraints
            available_room = stars_to_manipulate['Room_Down'].to_dict()
            delta_from_rooms = {k: v + available_room.get(k) for k, v in naive_distribution.items()}
            # Only for negative values (not enough steam available given contraints to push so low)
            delta_filtered = {k: v for k, v in delta_from_rooms.items() if v < 0}
            could_not_allocate = sum(delta_filtered.values())

            # Aggregate the allocations (for star wells and non-star wells)
            stars_finalized_allocations = {k: v - naive_distribution.get(k) for k, v in delta_filtered.items()}
            stars_finalized_allocations = dict(naive_distribution, **stars_finalized_allocations)
            original_base = local_deltas.set_index('Candidate_Injector')['Candidate_Units'].to_dict()
            stars_finalized_allocations = {k: original_base.get(k) + v for k, v in stars_finalized_allocations.items()}
            to_be_changed = {k: original_base.get(k) + (could_not_allocate * v / sum(to_be_changed.values()))
                             for k, v in to_be_changed.items()}
            finalized_allocations = dict(to_be_changed, **stars_finalized_allocations)

            # Change the dataset
            suggestions.loc[suggestions.index[group_df.index],
                            'Finalized_Allocations'] = list(finalized_allocations.values())
            suggestions.loc[suggestions.index[group_df.index],
                            'Unable_to_Allocate'] = could_not_allocate
            1
        else:
            # This is very unlikely, but possible, and great! This means that valid injectors are not touched.
            # Change the dataset
            suggestions.loc[suggestions.index[group_df.index],
                            'Finalized_Allocations'] = suggestions.loc[suggestions.index[group_df.index],
                                                                       'Candidate_Units']
            suggestions.loc[suggestions.index[group_df.index],
                            'Unable_to_Allocate'] = 0

    # fig, ax = plt.subplots(figsize=(20, 20))
    # suggestions['Candidate_Units'].plot(ax=ax)
    # suggestions['Finalized_Allocations'].plot(ax=ax)
    # suggestions['Constraint_Delta'].plot(ax=ax)
    # suggestions.tail(10)

    # Determine least-changed wells (after within-producer adjustment)
    # minimally_changed = suggestions[['PRO_Well', 'Well_Net_Delta']].drop_duplicates().set_index('PRO_Well').to_dict()
    # minimally_changed = minimally_changed['Well_Net_Delta']
    # minimally_changed = dict(sorted(minimally_changed.items(), key=lambda tup: abs(tup[1])))
    #
    # sum(suggestions['Finalized_Allocations'])

    return suggestions


def configure_minor(suggestions):
    suggestions.columns = ['PRO_Well',
                           'PRO_Well_Additional_Units',
                           'PRO_Well_Initial_Allocation',
                           'Delta',
                           'PRO_Well_Final_Allocation',
                           'Cand_Injector',
                           'Cand_Proportion',
                           'Cand_Units',
                           'Room_Up',
                           'Room_Down',
                           'Constraint_Delta',
                           'Well_Net_Delta',
                           'Finalized_Allocations',
                           'Unable_to_Allocate']
    suggestions['PRO_Pad'] = suggestions['PRO_Well'].str[0]

    return suggestions.infer_objects()


_ = """
#######################################################################################################################
##################################################   CORE EXECUTION   #################################################
#######################################################################################################################
"""


def _INJECTOR_ALLOCATION(data=None, candidates=None, PI_distances=None,
                         _return=True, flow_ingest=True,
                         CLOSENESS_THRESH_PI=0.1, CLOSENESS_THRESH_II=0.1):
    _accessories._print('Ingesting the positional data matrixes and candidate relationships...')

    DATASETS = {}
    if flow_ingest:
        DATASETS['CANDIDATES'] = candidates
        DATASETS['PI_DIST_MATRIX'] = PI_distances
        DATASETS['II_DIST_MATRIX'] = distance_matrix('II', S3.get_coordinates(data_group='INJECTION'), scaled=False)
        DATASETS['PP_DIST_MATRIX'] = distance_matrix('PP', S3.get_coordinates(data_group='PRODUCTION'), scaled=False)
        DATASETS['PRO_CONSTRAINTS'] = data
    else:
        DATA_PATH_DMATRIX: Final = 'Data/Pickles/DISTANCE_MATRIX.csv'
        DATA_PATH_CANDIDATES: Final = 'Data/Pickles/WELL_Candidates.pkl'
        DATA_PATH_ALLOCATIONS: Final = 'Data/Pickles/pwell_allocations.pkl'

        DATASETS['CANDIDATES'] = _accessories.retrieve_local_data_file(DATA_PATH_CANDIDATES, mode=2)
        DATASETS['PI_DIST_MATRIX'] = _accessories.retrieve_local_data_file(DATA_PATH_DMATRIX)
        DATASETS['II_DIST_MATRIX'] = distance_matrix('II', S3.get_coordinates(data_group='INJECTION'), scaled=False)
        DATASETS['PP_DIST_MATRIX'] = distance_matrix('PP', S3.get_coordinates(data_group='PRODUCTION'), scaled=False)
        DATASETS['PRO_CONSTRAINTS'] = _accessories.retrieve_local_data_file(DATA_PATH_ALLOCATIONS, mode=2)

    _wells_available = DATASETS['PI_DIST_MATRIX']['PRO_Well'].unique()
    candidates = {k: v for k, v in candidates.items() if k[0] in _wells_available}

    # TEMP: Arbitrary contraints generation
    # constraints = {inj: (random.randint(5, 29), random.randint(30, 60)) for inj in list(DATASETS['II_DIST_MATRIX'])}
    constraints = data.copy()

    _accessories._print('Engineering impact area/overlap datasets...')
    impact_tracker_PI, isolates_PI = PI_imapcts(DATASETS['CANDIDATES'].copy(),
                                                DATASETS['PI_DIST_MATRIX'].copy(),
                                                CLOSENESS_THRESH_PI=CLOSENESS_THRESH_PI)
    impact_tracker_II, isolates_II = II_impacts(DATASETS['II_DIST_MATRIX'].copy(),
                                                CLOSENESS_THRESH_II=CLOSENESS_THRESH_II)

    _accessories._print('Determining over-allocated naïve solution...')
    allocated_steam_values, allocated_steam_props = naive_distance_allocation(DATASETS['PI_DIST_MATRIX'].copy(),
                                                                              DATASETS['CANDIDATES'].copy(),
                                                                              DATASETS['PRO_CONSTRAINTS'].copy(),
                                                                              format='dict')

    _accessories._print('Taking positional relationship into account and adjusting allocations...')
    decisions = initial_allocations(allocated_steam_props, impact_tracker_PI, impact_tracker_II)

    _accessories._print('Proportionally adding steam to reach maximum capacity...')
    accounted_units, units_remaining = accounted_for(decisions, DATASETS['PRO_CONSTRAINTS'].copy())
    # Fill the remaining steam allocation per injector based on pad level contraints
    suggestions = maximize_allocations(DATASETS['PRO_CONSTRAINTS'], accounted_units, units_remaining, decisions)
    _accessories._print('Tuning steam allocations to fit in constraints...')
    suggestions = constrain_allocations(constraints, suggestions)

    _accessories._print('Finalizing and saving injection suggestion data...')
    suggestions = configure_minor(suggestions)
    _accessories.finalize_all(DATASETS, coerce_date=False)

    if _return:
        return suggestions
    else:
        _accessories.save_local_data_file(suggestions, 'Data/S8 Files/final_suggestions.csv')


if __name__ == '__main__':
    _INJECTOR_ALLOCATION()

# SEARCH_SPACE = produce_search_space(CANDIDATES, PI_DIST_MATRIX, II_DIST_MATRIX, RESOLUTION=0.001)
# plot_search_space(retrieve_search_space(min_bound=0.3, early=False), cmap=cm.turbo)

# EOF
