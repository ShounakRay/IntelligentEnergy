# @Author: Shounak Ray <Ray>
# @Date:   17-May-2021 10:05:35:354  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: master_execution.py
# @Last modified by:   Ray
# @Last modified time: 20-May-2021 11:05:42:423  GMT-0600
# @License: [Private IP]

# from collections import Counter

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from pipeline import S7_prosteam_allocation as S7_WALL
# from pipeline import S4_ft_eng_math as S4_MATH --> Ignored: Used directly in S5_MODL
# from pipeline import S5_modeling as S5_MODL --> Ignored: This is for model generation, not creation
from pipeline import S1_base_generation as S1_BASE
from pipeline import S2_ft_eng_physics as S2_PHYS
from pipeline import S3_weighting as S3_WGHT
from pipeline import S6_optimization as S6_OPTM
from pipeline import S8_injsteam_allocation as S8_SALL

_ = """
#######################################################################################################################
#############################################   HYPERPARAMETER SETTINGS   #############################################
#######################################################################################################################
"""

new = {'FP1': ['I37', 'I72', 'I70'],
       'FP2': ['I64', 'I73', 'I69', 'I37', 'I72', 'I70'],
       'FP3': ['I64', 'I74', 'I73', 'I69', 'I71'],
       'FP4': ['I74', 'I71', 'I75', 'I76'],
       'FP5': ['I67', 'I75', 'I77', 'I76', 'I66'],
       'FP6': ['I67', 'I65', 'I78', 'I77', 'I79', 'I68'],
       'FP7': ['I65', 'I68', 'I79'],
       'CP1': ['I25', 'I24', 'I26', 'I08'],
       'CP2': ['I24', 'I49', 'I45', 'I46', 'I39', 'I47'],
       'CP3': ['I47', 'I39', 'I46', 'I45', 'I49'],
       'CP4': ['I44', 'I43', 'I45', 'I51', 'I48'],
       'CP5': ['I40', 'I43', 'I51', 'I50'],
       'CP6': ['I40', 'I41', 'I50', 'CI06'],
       'CP7': ['I42', 'I41', 'CI06'],
       'CP8': ['I41', 'I42', 'CI06'],
       'EP2': ['I61', 'I60', 'I53'],
       'EP3': ['I59', 'I52', 'I61', 'I60', 'I53'],
       'EP4': ['I59', 'I52', 'I57', 'I54'],
       'EP5': ['I62', 'I57', 'I56', 'I54'],
       'EP6': ['I62', 'I56', 'I58', 'I55'],
       'EP7': ['I63', 'I56', 'I55']}
# compare_df = pd.read_csv('Data/field_data.csv')
taxonomy = {'INJECTION': {'CI06': 'C',
                          'CI07': 'C',
                          'CI08': 'C',
                          'I02': 'A',
                          'I03': 'A',
                          'I04': 'A',
                          'I05': 'A',
                          'I06': '15-05',
                          'I07': '15-05',
                          'I08': '16-05',
                          'I09': '16-05',
                          'I10': '16-05',
                          'I11': '16-05',
                          'I12': '16-05',
                          'I13': '16-05',
                          'I14': '16-05',
                          'I15': '11-05',
                          'I16': '11-05',
                          'I17': '11-05',
                          'I18': '11-05',
                          'I19': '10-05',
                          'I20': '10-05',
                          'I21': '10-05',
                          'I22': '10-05',
                          'I23': '09-05',
                          'I24': '09-05',
                          'I25': '09-05',
                          'I26': '09-05',
                          'I27': '09-05',
                          'I28': '06-05',
                          'I29': '06-05',
                          'I30': '06-05',
                          'I31': '06-05',
                          'I32': '08-05',
                          'I33': '08-05',
                          'I34': '08-05',
                          'I35': '08-05',
                          'I36': '08-05',
                          'I37': '08-05',
                          'I38': '16-05',
                          'I39': 'C1',
                          'I40': 'C1',
                          'I41': 'C1',
                          'I42': 'C1',
                          'I43': 'C1',
                          'I44': 'C1',
                          'I45': 'C1',
                          'I46': 'C1',
                          'I47': 'C1',
                          'I48': 'C2',
                          'I49': 'C2',
                          'I50': 'C2',
                          'I51': 'C2',
                          'I52': 'E1',
                          'I53': 'E1',
                          'I54': 'E1',
                          'I55': 'E1',
                          'I56': 'E2',
                          'I57': 'E2',
                          'I58': 'E2',
                          'I59': 'E2',
                          'I60': 'E2',
                          'I61': 'E3',
                          'I62': 'E3',
                          'I63': 'E3',
                          'I64': 'F1',
                          'I65': 'F1',
                          'I66': 'F1',
                          'I67': 'F1',
                          'I68': 'F1',
                          'I69': 'F2',
                          'I70': 'F2',
                          'I71': 'F2',
                          'I72': 'F2',
                          'I73': 'F2',
                          'I74': 'F2',
                          'I75': 'F2',
                          'I76': 'F2',
                          'I77': 'F2',
                          'I78': 'F2',
                          'I79': 'F2',
                          'I80': 'E3',
                          'I82': 'E3',
                          'I83': 'E3',
                          'I84': 'E3',
                          'I85': 'D2',
                          'I86': 'D2',
                          'I87': 'D2',
                          'I88': 'D2',
                          'I89': 'D2',
                          'I90': 'D3',
                          'I91': 'D3',
                          'I92': 'D3',
                          'I93': 'D3'},
            'PRODUCTION': {'AP2': 'A',
                           'AP3': 'A',
                           'AP4': 'A',
                           'AP5': 'A',
                           'AP6': 'A',
                           'AP7': 'A',
                           'AP8': 'A',
                           'BP1': 'B',
                           'BP2': 'B',
                           'BP3': 'B',
                           'BP4': 'B',
                           'BP5': 'B',
                           'BP6': 'B',
                           'CP1': 'C',
                           'CP2': 'C',
                           'CP3': 'C',
                           'CP4': 'C',
                           'CP5': 'C',
                           'CP6': 'C',
                           'CP7': 'C',
                           'CP8': 'C',
                           'EP2': 'E',
                           'EP3': 'E',
                           'EP4': 'E',
                           'EP5': 'E',
                           'EP6': 'E',
                           'EP7': 'E',
                           'FP1': 'F',
                           'FP2': 'F',
                           'FP3': 'F',
                           'FP4': 'F',
                           'FP5': 'F',
                           'FP6': 'F',
                           'FP7': 'F'}}

_ = """
#######################################################################################################################
##################################################   CORE EXECUTION   #################################################
#######################################################################################################################
"""


def MASTER_PIPELINE(all_data, skip_ingestion=True, weights=False, date='2020-01-01', model_plan='SKLEARN'):
    # NOTE: GET DATA
    if not skip_ingestion:
        all_data, taxonomy = S1_BASE._INGESTION()
        # all_data.to_csv('starting_joined_data.csv')
    else:
        # Data imported
        # Taxonomy is hyper parameter
        pass

    # NOTE: CONDUCT PHYSICS FEATURE ENGINEERING
    phys_engineered = S2_PHYS._FEATENG_PHYS(data=all_data)

    # NOTE: CONDUCT WEIGHTING (weights:=False for time speed-up)
    aggregated, PI_distances, candidates = S3_WGHT._INTELLIGENT_AGGREGATION(data=phys_engineered,
                                                                            taxonomy=taxonomy,
                                                                            weights=False)

    # chl = compare_df.groupby(['date', 'producer_well'])['chloride_contrib'].sum().reset_index()
    # chl['Date'] = pd.to_datetime(chl['Date'])
    # chl = chl.rename(columns={'date': 'Date', 'producer_well': 'PRO_Well'})
    # chl.to_csv('Data/temp_chloride_contribution_dependency.csv')

    aggregated.rename(columns={'Steam': 'PRO_Alloc_Steam'}, inplace=True)

    # NOTE: CONDUCT OPTIMIZATION
    # TODO: Engineering Chloride Contribution
    # phys_engineered['chloride_contrib'] = 0.5
    # WARNING: This dictionary addition doesn't actually matter if `PI_distances` is incomplete
    candidates['BY_WELL'] = dict(candidates['BY_WELL'], **new)

    well_allocations, well_sol, pad_sol, field_kpi = S6_OPTM._OPTIMIZATION(data=phys_engineered,
                                                                           date='2020-01-01',
                                                                           well_interactions=candidates['BY_WELL'],
                                                                           model_plan='SKLEARN')  # OR H2O

    # CREATING SCENARIO TABLE FOR: pad A
    # NOTE: CONDUCT WELL-ALLOCATION
    # well_allocation = S7_SALL._PRODUCER_ALLOCATION()

    # CONDUCT INJECTOR-ALLOCATION

    # Only for A and B since positional data for injectors is not parsed yet
    injector_allocation = S8_SALL._INJECTOR_ALLOCATION(data=well_allocations.copy(),
                                                       candidates=candidates['BY_WELL'].copy(),
                                                       PI_distances=PI_distances.copy())

    return pad_sol, well_sol, injector_allocation, field_kpi

    # store = {}
    # for group, df in injector_allocation.groupby('PRO_Well'):
    #     store[group] = df.set_index('Cand_Injector')['Cand_Proportion'].to_dict()


pad_sol, well_sol, injector_allocation, field_kpi = MASTER_PIPELINE(weights=False,
                                                                    date='2020-01-01',
                                                                    model_plan='SKLEARN')
