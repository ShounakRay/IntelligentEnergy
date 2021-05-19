# @Author: Shounak Ray <Ray>
# @Date:   19-Mar-2021 08:03:81:813  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: approach_alternative.py
# @Last modified by:   Ray
# @Last modified time: 19-May-2021 12:05:76:760  GMT-0600
# @License: [Private IP]

import math
import os
import sys
import warnings
from datetime import datetime
from itertools import chain
from multiprocessing import Pool

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    import _multiprocessed.defs as defs
    import _positional.coordinates as coordinates

    # import _traversal

# _traversal.print_tree_to_txt(PATH='_configs/FILE_STRUCTURE.txt')

_ = """
#######################################################################################################################
#############################################   HYPERPARAMETER SETTINGS   #############################################
#######################################################################################################################
"""
plot_geo = False

# Display hyperparams
relative_radius = 50
dimless_radius = relative_radius * 950
focal_period = 20

# Scaling, window constants
POS_TL = (478, 71)
POS_TR = (1439, 71)
POS_BL = (478, 1008)
POS_BR = (1439, 1008)
x_delta = POS_TL[0] | POS_BL[0]
y_delta = POS_TR[0] | POS_BR[0]

OLD_DUPLICATES = ['PRO_Alloc_Oil', 'PRO_Pump_Speed']

aggregation_dict = {'PRO_Adj_Pump_Speed': 'sum',
                    # 'PRO_Pump_Speed': 'sum',
                    # 'PRO_Time_On': 'mean',
                    'PRO_Casing_Pressure': 'mean',
                    'PRO_Heel_Pressure': 'mean',
                    'PRO_Toe_Pressure': 'mean',
                    'PRO_Heel_Temp': 'mean',
                    'PRO_Toe_Temp': 'mean',
                    'PRO_Adj_Alloc_Oil': 'sum',
                    'PRO_Alloc_Oil': 'sum',
                    'PRO_Adj_Pump_Efficiency': 'mean',
                    'PRO_Adj_Alloc_Water': 'sum',
                    # 'PRO_Duration': 'mean',
                    # 'PRO_Alloc_Oil': 'sum',
                    'PRO_Total_Fluid': 'sum',
                    'PRO_Water': 'sum',
                    'PRO_Chlorides': 'sum',
                    # 'PRO_Alloc_Water': 'sum',
                    'PRO_Alloc_Steam': 'sum',
                    'PRO_Water_cut': 'sum',
                    'Bin_1': 'mean',
                    'Bin_2': 'mean',
                    'Bin_3': 'mean',
                    'Bin_4': 'mean',
                    'Bin_5': 'mean',
                    'Bin_6': 'mean',
                    'Bin_7': 'mean',
                    'Bin_8': 'mean',
                    'weight': 'mean'}

_ = """
#######################################################################################################################
###############################################   FUNCTION DEFINITIONS   ##############################################
#######################################################################################################################
"""


def filter_negatives(df, columns, out=True, placeholder=0):
    """Filter out all the rows in the DataFrame whose rows contain negatives.
    TODO: PENDING!!!! Groupby when finding means

    Parameters
    ----------
    df : DataFrame
        The intitial, unfiltered DataFrame.
    columns : pandas Index, or any iterable
        Which columns to consider.
    out : boolean
        Whether the "negative rows" should be deleted from the DataFrame.
    placeholder : int OR str
        The value or strategy used in "treating" the "negative rows."

    Returns
    -------
    DataFrame
        The filtered DataFrame, may contain NaNs (intentionally).

    """
    for num_col in columns:
        if(out):
            df = df[~(df[num_col] < 0)].reset_index(drop=True)
        else:
            if(placeholder == 'mean'):
                placeholder = df[num_col].mean()
            elif(placeholder == 'median'):
                placeholder = df[num_col].median()
            elif(type(placeholder) == int or type(placeholder) == float):
                pass
            else:
                raise ValueError(f'Incompatible argument entered for `placeholder`: {placeholder}')
            df.loc[df[num_col] < 0, num_col] = placeholder
    return df


def drop_singles(df):
    dropped = []
    for col in df.columns:
        if len(df[col].unique()) <= 1:
            dropped.append(col)
            df.drop(col, axis=1, inplace=True)
    return df, dropped


def minor_processing(FINALE, PRO_PAD_KEYS):
    FINALE['PRO_Pad'] = FINALE['PRO_Well'].apply(lambda x: PRO_PAD_KEYS.get(x))
    FINALE = FINALE.dropna(subset=['PRO_Well']).reset_index(drop=True)

    FINALE = filter_negatives(FINALE, FINALE.select_dtypes(float), out=True)

    return FINALE


def specialized_anomaly_detection(FINALE):
    def perform_detection(TEMP, injector_names, all_prod_wells, benchmark_plot=False):
        all_continuous_columns = TEMP.select_dtypes(np.number).columns

        # Conduct Anomaly Detection
        cumulative_tagged = []          # Stores the outputs from the customized, anomaly detection function
        temporal_benchmarks = []        # Used to track multi-process benchmarking for utlimate vizualization
        process_init = datetime.now()
        for cont_col in all_continuous_columns:
            rep_multiplier = len(all_prod_wells) if cont_col not in injector_names else 1
            arguments = list(zip([TEMP] * rep_multiplier, all_prod_wells, [cont_col] * rep_multiplier))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if __name__ == '__main__':
                    with Pool(os.cpu_count() - 1) as pool:
                        data_outputs = pool.starmap(defs.process_local_anomalydetection, arguments)
            cumulative_tagged.extend(data_outputs)

            process_final = datetime.now()
            temporal_benchmarks.append((cont_col, (process_final - process_init).total_seconds()))
            _accessories._print('> STATUS: {} % progress with `{}`'.format(
                round(100.0 * (list(all_continuous_columns).index(cont_col) + 1) / len(all_continuous_columns), 3),
                cont_col), color='CYAN')

        if(benchmark_plot):
            with _accessories.suppress_stdout():
                fig_path = 'Manipulation Reference Files/Benchmarking Anomaly Detection.png'
                # Save benchmarking materials
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot([t[1] for t in temporal_benchmarks])
                ax.set_xlabel('Feature (the middle are injectors)')
                ax.set_ylabel('Cumulative Time (s)')
                ax.set_title('Multiprocessing – Benchmarking Anomaly Detection')
                _accessories.auto_make_path(fig_path)
                fig.savefig(fig_path)

        # Reformat the tagged anomalies to a format more accessible DataFrame
        anomaly_tag_tracker = pd.concat(cumulative_tagged).reset_index(drop=True)

        return anomaly_tag_tracker

    def coalesce_datasets(anomaly_tag_tracker, FINALE, injector_names):
        sole_injector_data = anomaly_tag_tracker[anomaly_tag_tracker['feature'].isin(   # Only the injector data
            injector_names)].reset_index(drop=True).drop('group', axis=1)

        # Duplicate the injector data for all available producer wells
        rep_tracker = []
        for pwell in all_prod_wells:
            sole_injector_data['group'] = pwell
            rep_tracker.append(sole_injector_data.copy())
        sole_injector_data_rep = pd.concat(rep_tracker).reset_index(drop=True)
        sole_injector_data_rep['feature'] = sole_injector_data_rep['feature'].apply(lambda x: col_reference.get(x))
        sole_injector_data_rep.drop(['selection', 'detection_iter', 'anomaly'], axis=1, inplace=True)

        # NOTE: Do not consider old pump speed and allocated oil value in the weighting transformations
        _temp_OLD_DUPLICATES = [c.lower() for c in OLD_DUPLICATES]
        anomaly_tag_tracker = anomaly_tag_tracker[
            ~anomaly_tag_tracker['feature'].isin(_temp_OLD_DUPLICATES)].reset_index(drop=True)
        anomaly_tag_tracker.drop(['selection', 'detection_iter', 'anomaly'], axis=1, inplace=True)

        # Concatenate the non-injector and the duplicated-injector tables together
        cumulative_anomalies = pd.concat([sole_injector_data_rep, anomaly_tag_tracker], axis=0).reset_index(drop=True)

        # Find the daily, unweighted average of the column-specific weights
        reformatted_anomalies = cumulative_anomalies.groupby(['group', 'date'])[['updated_score']].mean().reset_index()
        reformatted_anomalies.columns = ['PRO_Well', 'Date', 'weight']
        reformatted_anomalies = reformatted_anomalies.infer_objects()

        # Date Formatting (just to be sure)
        FINALE['Date'] = pd.to_datetime(FINALE['Date'])
        reformatted_anomalies['Date'] = pd.to_datetime(reformatted_anomalies['Date'])

        # Merge this anomaly data into the original, highest-resolution base table
        FINALE = pd.merge(FINALE.infer_objects(), reformatted_anomalies, 'inner', on=['Date', 'PRO_Well'])

        return FINALE

    # Anomaly Detection Preparation
    TEMP = FINALE.copy()
    TEMP.columns = list(TEMP.columns.str.lower())
    col_reference = dict(zip(TEMP.columns, FINALE.columns))
    lc_injectors = [k for k, v in col_reference.items() if 'I' in v]
    all_prod_wells = list(FINALE['PRO_Well'].unique())

    anomaly_tag_tracker = perform_detection(TEMP, injector_names=lc_injectors,
                                            all_prod_wells=all_prod_wells, benchmark_plot=True)

    anom_original_merged = coalesce_datasets(anomaly_tag_tracker, FINALE, injector_names=lc_injectors)

    return anom_original_merged


def produce_injection_aggregates(FINALE, candidates_by_prodtype, group_type, INJ_PAD_KEYS, naive_selection=False):
    # This is just to delete the implicit duplicates found in the data
    # > (each production well has the same injection data)
    some_random_pwell = list(FINALE['PRO_Well'])[0]
    injector_data = FINALE[FINALE['PRO_Well'] == some_random_pwell].reset_index(drop=True).drop('PRO_Well', 1)
    all_injs = [c for c in injector_data.columns if 'I' in c and '_' not in c]

    INJECTOR_AGGREGATES = {}
    if(not naive_selection):
        for category, cat_candidates in candidates_by_prodtype.items():
            # Select candidates (not all the wells)
            local_candidates = cat_candidates.copy()
            absence = []
            for cand in local_candidates:
                if cand not in all_injs:
                    _accessories._print(f'> STATUS: Candidate {cand} removed assessing {category}, ' +
                                        'unavailable in initial data', color='CYAN')
                    absence.append(cand)
            local_candidates = [el for el in local_candidates if el not in absence]

            melted_inj = pd.melt(injector_data, id_vars=['Date'], value_vars=local_candidates,
                                 var_name='Injector', value_name='Steam')
            melted_inj['INJ_Pad'] = melted_inj['Injector'].apply(lambda x: INJ_PAD_KEYS.get(x))
            melted_inj = melted_inj[~melted_inj['INJ_Pad'].isna()].reset_index(drop=True)
            # To groupby injector pads, by=['Date', 'INJ_Pad']
            agg_inj = melted_inj.groupby(by=['Date'], axis=0, sort=False, as_index=False).sum()
            agg_inj[group_type] = category
            INJECTOR_AGGREGATES[category] = agg_inj
        INJECTOR_AGGREGATES = pd.concat(INJECTOR_AGGREGATES.values()).reset_index(drop=True)
    else:
        FINALE = FINALE[['Date', 'PRO_Alloc_Steam', 'PRO_Pad']]
        INJECTOR_AGGREGATES = FINALE.groupby(by=['Date', group_type], axis=0, sort=False, as_index=False).sum()
        INJECTOR_AGGREGATES = INJECTOR_AGGREGATES[['Date', 'PRO_Alloc_Steam', 'PRO_Pad']]
        INJECTOR_AGGREGATES.columns = ['Date', 'Steam', 'PRO_Pad']

    return INJECTOR_AGGREGATES


def produce_production_aggregates(FINALE, aggregation_dict, group_name='PRO_Pad', include_weights=True):
    if('weight' not in FINALE.columns):
        _accessories._print('WARNING: The `weight` key does not exist in the supplied data. ' +
                            '`include_weights` is now coerced to False.',
                            color='LIGHTRED_EX')
        include_weights = False
        del aggregation_dict['weight']

    all_pro_data = list(aggregation_dict.keys())

    subset_cols = all_pro_data + ['Date'] + [group_name]
    if include_weights:
        subset_cols.extend(['weight'])

    FINALE_pro = FINALE[subset_cols]
    FINALE_pro, dropped_cols = drop_singles(FINALE_pro)

    FINALE_agg_pro = FINALE_pro.groupby(by=['Date', group_name], axis=0,
                                        sort=False, as_index=False).agg(aggregation_dict)

    return FINALE_agg_pro


def euclidean_2d_distance(c1, c2):
    x1 = c1[0]
    x2 = c2[0]
    y1 = c1[1]
    y2 = c2[1]
    return math.hypot(x2 - x1, y2 - y1)


def get_coordinates(data_group):
    """Work to be done here provided new positional data"""
    def get_injector_coordinates():
        if True:
            INJ_relcoords = {}
            INJ_relcoords = {'I02': '(757, 534)',
                             'I03': '(709, 519)',
                             'I04': '(760, 488)',
                             'I05': '(708, 443)',
                             'I06': '(825, 537)',
                             'I07': '(823, 461)',
                             'I08': '(997, 571)',
                             'I09': '(940, 516)',
                             'I10': '(872, 489)',
                             'I11': '(981, 477)',
                             'I12': '(1026, 495)',
                             'I13': '(1034, 444)',
                             'I14': '(935, 440)',
                             'I15': '(709, 686)',
                             'I16': '(694, 611)',
                             'I17': '(758, 649)',
                             'I18': '(760, 571)',
                             'I19': '(818, 684)',
                             'I20': '(880, 645)',
                             'I21': '(817, 606)',
                             'I22': '(881, 565)',
                             'I23': '(946, 682)',
                             'I24': '(1066, 679)',
                             'I25': '(1063, 604)',
                             'I26': '(995, 643)',
                             'I27': '(940, 604)',
                             'I28': '(758, 801)',
                             'I29': '(701, 766)',
                             'I30': '(825, 763)',
                             'I31': '(759, 736)',
                             'I32': '(871, 716)',
                             'I33': '(939, 739)',
                             'I34': '(873, 801)',
                             'I35': '(1023, 727)',
                             'I36': '(996, 789)',
                             'I37': '(1061, 782)',
                             'I38': '(982, 529)',
                             'I86': '(792, 385)',
                             'I80': '(880, 416)',
                             'I61': '(928, 370)',
                             'I59': '(928, 334)',
                             'I60': '(1036, 374)',
                             'I47': '(1085, 411)',
                             'I44': '(1144, 409)'}
            # for inj in [k for k in INJ_PAD_KEYS.keys() if INJ_PAD_KEYS[k] in ['A', '15-05', '16-05', '11-05', '10-05',
            #                                                                   '09-05', '06-05', '08-05']]:
            #     INJ_relcoords[inj] = input(prompt='Please enter coordinates for injector {}'.format(inj))
            for k, v in INJ_relcoords.items():
                # String to tuple
                INJ_relcoords[k] = eval(v)
                v = INJ_relcoords[k]
                # Re-scaling
                INJ_relcoords[k] = (v[0] - x_delta, y_delta - v[1])
        else:
            None
        return INJ_relcoords

    def get_producer_coordinates(nb_points=8):
        def intermediates(p1, p2, nb_points=nb_points):
            """"Return a list of nb_points equally spaced points
            between p1 and p2"""
            # If we have 8 intermediate points, we have 8+1=9 spaces
            # between p1 and p2
            x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
            y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

            return [[p1[0] + i * x_spacing, p1[1] + i * y_spacing]
                    for i in range(0, nb_points + 2)]
        PRO_relcoords = {}
        PRO_relcoords = {'AP2': '(616, 512) <> (683, 557) <> (995, 551)',
                         'AP3': '(601, 522) <> (690, 582) <> (995, 582)',
                         'AP4': '(621, 504) <> (691, 526) <> (1052, 523)',
                         'AP5': '(616, 483) <> (688, 505) <> (759, 507) <> (1058, 501)',
                         'AP6': '(606, 470) <> (685, 478) <> (827, 472) <> (910, 472)',
                         'AP7': '(602, 461) <> (674, 456) <> (846, 452) <> (992, 450)',
                         'AP8': '(593, 456) <> (674, 429) <> (1009, 427)',
                         'BP1': '(541, 733) <> (609, 654) <> (674, 633) <> (916, 636) <> (992, 635) <> (1015, 629)',
                         'BP2': '(541, 747) <> (630, 670) <> (1014, 668)',
                         'BP3': '(541, 760) <> (647, 704) <> (1016, 697)',
                         'BP4': '(555, 772) <> (691, 752) <> (908, 750)',
                         'BP5': '(555, 784) <> (838, 786) <> (1010, 748)',
                         'BP6': '(555, 803) <> (690, 821) <> (1026, 817)'}
        # Get relative position inputs
        # for well in [pw for pw in PRO_PAD_KEYS.keys() if 'A' in pw or 'B' in pw]:
        #     PRO_relcoords[well] = input(prompt='Please enter coordinates for producer {}'.format(well))
        # k, v = list(PRO_relcoords.items())[0]
        for k, v in PRO_relcoords.items():
            # Re-format relative positions
            # Parsing
            PRO_relcoords[k] = [chunk.strip() for chunk in v.split('<>')]
            # String to tuple
            PRO_relcoords[k] = [eval(chunk) for chunk in PRO_relcoords[k]]

            # Re-scale relative positions (cartesian, not jS, system)
            transformed = []
            for coordinate in PRO_relcoords[k]:
                transformed.append((coordinate[0] - x_delta, y_delta - coordinate[1]))

            # Find n points connecting the points
            discrete_links = []
            for coordinate_i in range(len(transformed) - 1):
                c1 = transformed[coordinate_i]
                c2 = transformed[coordinate_i + 1]
                num_points = int(euclidean_2d_distance(c1, c2) / focal_period)
                discrete_ind = intermediates(c1, c2, nb_points=num_points)
                discrete_links.append(discrete_ind)
            PRO_relcoords[k] = list(chain.from_iterable(discrete_links))

        return PRO_relcoords

    if(data_group == 'PRODUCTION'):
        return get_producer_coordinates()
    elif(data_group == 'INJECTION'):
        return get_injector_coordinates()
    else:
        raise ValueError('Improperly entered `data_group` in `get_coordinates`.')


get_coordinates('INJECTION')


def distance_matrix(injector_coordinates, producer_coordinates, save=False):
    pro_inj_distance = pd.DataFrame([], columns=injector_coordinates.keys(), index=producer_coordinates.keys())
    pro_inj_distance = pro_inj_distance.rename_axis('PRO_Well').reset_index()
    operat = 'mean'
    for injector in injector_coordinates.keys():
        iwell_coord = injector_coordinates.get(injector)
        PRO_Well_uniques = pro_inj_distance['PRO_Well']
        inj_specific_distances = []
        for pwell in PRO_Well_uniques:
            pwell_coords = producer_coordinates.get(pwell)
            point_distances = [euclidean_2d_distance(pwell_coord, iwell_coord) for pwell_coord in pwell_coords]
            dist_store = np.mean(point_distances) if operat == 'mean' else min(point_distances)
            inj_specific_distances.append(dist_store)
        pro_inj_distance[injector] = inj_specific_distances
    pro_inj_distance = pro_inj_distance.infer_objects()

    if(save):
        _accessories.save_local_data_file(pro_inj_distance, 'Data/Pickles/DISTANCE_MATRIX.csv')

    return pro_inj_distance


def get_all_candidates(injection_coordinates, production_coordinates,
                       available_pads_transformed, available_pwells_transformed, pro_well_pad_relationship,
                       rel_rad=50, save=False, plot=False, **kwargs):
    def injector_candidates(production_pad, production_well,
                            injector_coordinates, production_coordinates, relative_radius=50,
                            pro_well_pad_relationship=pro_well_pad_relationship):

        if(production_pad is not None or production_well is not None):
            inclusion = pd.DataFrame([], columns=injector_coordinates.keys(), index=production_coordinates.keys())
            for iwell, iwell_point in injector_coordinates.items():
                x_test = iwell_point[0]
                y_test = iwell_point[1]
                inj_allpro_tracker = []
                for pwell, pwell_points in production_coordinates.items():
                    inj_pro_level_tracker = []
                    for pwell_point in pwell_points:
                        center_x = pwell_point[0]
                        center_y = pwell_point[1]
                        inclusion_condition = (x_test - center_x)**2 + (y_test - center_y)**2 < relative_radius**2
                        inj_pro_level_tracker.append(inclusion_condition)
                    state = any(inj_pro_level_tracker)
                    inj_allpro_tracker.append(state)
                inclusion[iwell] = inj_allpro_tracker

            inclusion = inclusion.rename_axis('PRO_Well').reset_index()
            if(production_pad is not None):
                inclusion['PRO_Pad'] = [pro_well_pad_relationship.get(pwell) for pwell in inclusion['PRO_Well']]

            if(production_pad is not None):
                type_filtered_inclusion = inclusion[inclusion['PRO_Pad'] == production_pad].reset_index(drop=True)
            elif(production_well is not None):
                type_filtered_inclusion = inclusion[inclusion['PRO_Well'] == production_well].reset_index(drop=True)

            all_possible_candidates = type_filtered_inclusion.select_dtypes(bool).columns
            candidate_injectors_full = dict([(candidate, any(type_filtered_inclusion[candidate]))
                                             for candidate in all_possible_candidates])
            candidate_injectors = [k for k, v in candidate_injectors_full.items() if v is True]

            return candidate_injectors, candidate_injectors_full
        else:
            raise ValueError('ERROR: Fatal parameter inputs for `injector_candidates`')

    def plot_geo_candidates(candidate_dict, group_type, PRO_finalcoords, INJ_relcoords,
                            POS_TL=POS_TL, POS_BR=POS_BR, POS_TR=POS_TR):
        # Producer connections
        aratio = (POS_TR[0] - POS_TL[0]) / (POS_BR[1] - POS_TR[1])
        fig, ax = plt.subplots(figsize=(20 * aratio, 20))
        colors = cm.rainbow(np.linspace(0, 1, len(PRO_finalcoords.keys())))
        for k, v in PRO_finalcoords.items():
            all_x = [c[0] for c in v]
            all_y = [c[1] for c in v]
            plt.scatter(all_x, all_y, linestyle='solid', color=colors[list(PRO_finalcoords.keys()).index(k)])
            plt.plot(all_x, all_y, color=colors[list(PRO_finalcoords.keys()).index(k)])
            plt.scatter(all_x, all_y, color=list(
                (*colors[list(PRO_finalcoords.keys()).index(k)][:3], *[0.1])), s=dimless_radius)
        # Injector connections
        all_x = [t[0] for t in INJ_relcoords.values()]
        all_y = [t[1] for t in INJ_relcoords.values()]
        ax.scatter(all_x, all_y)
        for i, txt in enumerate(INJ_relcoords.keys()):
            if(txt in candidate_dict.get(group_type)):
                ax.scatter(all_x[i], all_y[i], color='green', s=200)
            else:
                ax.scatter(all_x[i], all_y[i], color='red', s=200)
            ax.annotate(txt, (all_x[i] + 2, all_y[i] + 2))
        plt.title('Producer Well and Injector Space, Overlaps')
        plt.tight_layout()
        plt.savefig('Modeling Reference Files/Producer-Injector Overlap for {}.png'.format(group_type))
    candidates_by_prodpad = {}
    reports_by_prodpad = {}
    for pad in available_pads_transformed:
        candidates, report = injector_candidates(production_pad=pad,
                                                 production_well=None,
                                                 pro_well_pad_relationship=pro_well_pad_relationship,
                                                 injector_coordinates=injection_coordinates,
                                                 production_coordinates=production_coordinates,
                                                 relative_radius=rel_rad)
        candidates_by_prodpad[pad] = candidates
        reports_by_prodpad[pad] = report

    candidates_by_prodwell = {}
    reports_by_prodwell = {}
    for pwell in available_pwells_transformed:
        candidates, report = injector_candidates(production_pad=None,
                                                 production_well=pwell,
                                                 pro_well_pad_relationship=None,
                                                 injector_coordinates=injection_coordinates,
                                                 production_coordinates=production_coordinates,
                                                 relative_radius=rel_rad)
        candidates_by_prodwell[pwell] = candidates
        reports_by_prodwell[pwell] = report

    if(plot_geo):
        group = kwargs.get('group')
        name = kwargs.get('name')
        if(group is None or name is None):
            raise ValueError('Improper kwargs for `get_all_candidates`. Please check `group` and `name`.')
        elif(group == 'PAD'):
            plot_geo_candidates(candidates_by_prodpad, name,
                                production_coordinates, injection_coordinates)
        elif(group == 'WELL'):
            plot_geo_candidates(candidates_by_prodwell, name,
                                production_coordinates, injection_coordinates)

    if(save):
        _accessories.save_local_data_file(candidates_by_prodpad, 'Data/Pickles/PAD_Candidates.pkl')
        _accessories.save_local_data_file(candidates_by_prodwell, 'Data/Pickles/WELL_Candidates.pkl')

    return candidates_by_prodpad, candidates_by_prodwell


def merge(datasets, available_pads):
    PRODUCER_AGGREGATES = datasets['PRODUCTION_AGGREGATES'][datasets['PRODUCTION_AGGREGATES']
                                                            ['PRO_Pad'].isin(available_pads)]
    COMBINED_AGGREGATES = pd.merge(PRODUCER_AGGREGATES, datasets['INJECTION_AGGREGATES'],
                                   how='inner', on=['Date', 'PRO_Pad'])
    COMBINED_AGGREGATES, dropped = drop_singles(COMBINED_AGGREGATES)

    return COMBINED_AGGREGATES


_ = """
#######################################################################################################################
##################################################   CORE EXECUTION  ##################################################
#######################################################################################################################
"""


def _INTELLIGENT_AGGREGATION(data=None, taxonomy=None, _return=True, flow_ingest=True, weights=True):
    """PURPOSE: TO GENERATE WEIGHTS + AGGREGATE PRODUCER DATA TO PAD-LEVEL DATA
       INPUTS:
       1 – ONE MERGED DATASET, IDEALLY UNDERWENT PHYSICS ENGINEERING (BUT NOT A DEPENDENCY)
       2 – A DICTIONARY OF WELL-PAD RELATIONSHIPS, BOTH FOR INJECTORS AND PRODUCERS
       PROCESSING:
       OUTPUT: 1 - A DATASET W/ SOME NEW PHYSICS-ENGINEERED FEATURES
                   RESOLUTION: Per day, per producer pad, what is aggregate producer, injector steam, and fiber data?
                               And – optionally – what are the weights of each instance?
               2 – DISTANCE MATRIX
                   This are the distances between all the injectors to each other.
               3 – CANDIDATES
                   This is a double-nested dictionary of candidates, on a producer-pad-level and producer-well-level
    """
    if flow_ingest:
        _accessories._print('Ingesting PHYSICS ENGINEERED data from LAST STEP...',
                            color='LIGHTYELLOW_EX')
        DATASETS = {'FINALE': data}
        _accessories._print('Ingesting well relationship data from saved data...', color='LIGHTYELLOW_EX')
        INJ_PAD_KEYS = taxonomy['INJECTION']
        PRO_PAD_KEYS = taxonomy['PRODUCTION']
    else:
        _accessories._print('Ingesting PHYSICS ENGINEERED data from SAVED DATA...', color='LIGHTYELLOW_EX')
        DATASETS = {'FINALE': _accessories.retrieve_local_data_file(
            'Data/S2 Files/combined_ipc_engineered_phys_ALL.csv')}
        _accessories._print('Ingesting well relationship data from saved data...', color='LIGHTYELLOW_EX')
        INJ_PAD_KEYS = _accessories.retrieve_local_data_file('Data/Pickles/INJECTION_[Well, Pad].pkl', mode=2)
        PRO_PAD_KEYS = _accessories.retrieve_local_data_file('Data/Pickles/PRODUCTION_[Well, Pad].pkl', mode=2)

    _accessories._print('Minor processing...', color='LIGHTYELLOW_EX')
    DATASETS['FINALE'] = minor_processing(DATASETS['FINALE'], PRO_PAD_KEYS)

    if weights:
        _accessories._print('Performing anomaly detection for weight calculation...', color='LIGHTYELLOW_EX')
        DATASETS['FINALE'] = specialized_anomaly_detection(DATASETS['FINALE'])

    _accessories._print('Getting producer and injector coordinates...', color='LIGHTYELLOW_EX')
    injector_coords = get_coordinates('INJECTION')
    producer_coords = get_coordinates('PRODUCTION')

    _accessories._print('Determining candidates and distance matrix...', color='LIGHTYELLOW_EX')
    available_pads_transformed = ['A', 'B']
    available_pwells_transformed = [k for k, v in PRO_PAD_KEYS.items() if v in available_pads_transformed]
    candidates_by_prodpad, candidates_by_prodwell = get_all_candidates(injector_coords, producer_coords,
                                                                       available_pads_transformed,
                                                                       available_pwells_transformed,
                                                                       rel_rad=relative_radius,
                                                                       save=False,
                                                                       plot=plot_geo,
                                                                       pro_well_pad_relationship=PRO_PAD_KEYS)
    CANDIDATES = {'BY_PAD': candidates_by_prodpad,
                  'BY_WELL': candidates_by_prodwell}

    DISTANCE_MATRIX = distance_matrix(injector_coords, producer_coords, save=False)

    _accessories._print('Determining PRODUCTION data aggregates...', color='LIGHTYELLOW_EX')
    DATASETS['PRODUCTION_AGGREGATES'] = produce_production_aggregates(DATASETS['FINALE'],
                                                                      aggregation_dict.copy(),
                                                                      include_weights=False)
    _accessories._print('Determining INJECTION data aggregates...', color='LIGHTYELLOW_EX')
    DATASETS['INJECTION_AGGREGATES'] = produce_injection_aggregates(DATASETS['FINALE'],
                                                                    None,
                                                                    'PRO_Pad', INJ_PAD_KEYS,
                                                                    naive_selection=True)

    # fig, ax = plt.subplots(figsize=(24, 14))
    # for name, group in DATASETS['INJECTION_AGGREGATES'].groupby('PRO_Pad'):
    #     if(name in ['A', 'B']):
    #         group.plot(x='Date', y='Steam', ax=ax, label='Steam: PAD ' + name)

    _accessories._print('Merging and saving...', color='LIGHTYELLOW_EX')
    _accessories.finalize_all(DATASETS, skip=[])
    merged_df = merge(DATASETS, ['A', 'B', 'C', 'E', 'F'])

    if _return:
        return merged_df, DISTANCE_MATRIX, CANDIDATES
    else:
        _accessories.save_local_data_file(merged_df, 'Data/S3 Files/combined_ipc_aggregates_ALL.csv')


if __name__ == '__main__':
    _INTELLIGENT_AGGREGATION()


# _ = """
# ####################################
# ###########  WEIGHT EDA ############
# ####################################
# """
# MAX = 150.0
# def plot_weights_eda(df, groupby_val, groupby_col, time_col='Date', weight_col='weight', col_thresh=None):
#     plt.figure(figsize=(30, 20))
#     _temp = df[df[groupby_col] == groupby_val].sort_values(time_col).reset_index(drop=True)
#     if(col_thresh is None):
#         iter_cols = _temp.columns
#     else:
#         iter_cols = _temp.columns[:col_thresh]
#     # Plot all features (normalized to 100 max)
#     for col in [c for c in iter_cols if c not in ['Date', 'PRO_Pad', 'PRO_Well', 'weight', 'PRO_Alloc_Oil']]:
#         __temp = _temp[['Date', col]].copy().fillna(_temp[col].mean())
#         __temp[col] = MAX * __temp[col] / (MAX if max(__temp[col]) is np.nan else max(__temp[col]))
#         # plt.hist(__temp[col], bins=100)
#         if(col in ['PRO_Adj_Alloc_Oil', 'Steam', 'PRO_Adj_Pump_Speed']):
#             lw = 0.75
#         else:
#             lw = 0.3
#         plt.plot(__temp[time_col], __temp[col], linewidth=lw, label=col)
#     plt.legend(loc='upper left', ncol=2)
#
#     # # Plot weight
#     plt.plot(_temp[time_col], _temp[weight_col])
#     plt.title(f'Weight Time Series for {groupby_col} = {groupby_val}')
#     plt.savefig(f'Manipulation Reference Files/Weight TS {groupby_col} = {groupby_val}.png')
# for pad in available_pads_transformed:
#     plot_weights_eda(COMBINED_AGGREGATES, groupby_val=pad, groupby_col='PRO_Pad',
#                      time_col='Date', weight_col='weight')
# for pwell in available_pwells_transformed:
#     plot_weights_eda(COMBINED_AGGREGATES_PWELL, groupby_val=pwell,
#                      groupby_col='PRO_Well', time_col='Date', weight_col='weight')
# os.system('say finished weight exploratory analysis')


# EOF
