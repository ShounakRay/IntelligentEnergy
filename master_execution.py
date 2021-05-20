# @Author: Shounak Ray <Ray>
# @Date:   17-May-2021 10:05:35:354  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: master_execution.py
# @Last modified by:   Ray
# @Last modified time: 20-May-2021 01:05:45:451  GMT-0600
# @License: [Private IP]

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
# well_allocations = {'AP2': 168.50638133970068,
#                     'AP3': 158.77670562530514,
#                     'AP4': 218.1557318198097,
#                     'AP5': 184.56816243995024,
#                     'AP6': 193.2713740126074,
#                     'AP7': 169.2051413545468,
#                     'AP8': 172.03388085040683,
#                     'BP1': 275.1545015663883,
#                     'BP2': 199.0340338530178,
#                     'BP3': 181.0507502880953,
#                     'BP4': 108.6806995783778,
#                     'BP5': 265.3717097531363,
#                     'BP6': 214.94264461323525,
#                     'CP1': 299.8409604774457,
#                     'CP2': 101.34910855792296,
#                     'CP3': 228.57072034063148,
#                     'CP4': 151.64662099501112,
#                     'CP5': 239.7184561911505,
#                     'CP6': 184.13507125912707,
#                     'CP7': 133.30833235009698,
#                     'CP8': 42.308273124784144,
#                     'EP2': 176.40490759899953,
#                     'EP3': 244.58464252535686,
#                     'EP4': 54.901433696492944,
#                     'EP5': 78.25684769901943,
#                     'EP6': 106.2963343631096,
#                     'EP7': 250.1620315560527,
#                     'FP1': 197.11805473487084,
#                     'FP2': 219.36129581897654,
#                     'FP3': 192.32563480401095,
#                     'FP4': 117.3281237235291,
#                     'FP5': 207.15013466550303,
#                     'FP6': 120.53492563970951,
#                     'FP7': 145.94637278362194}

_ = """
#######################################################################################################################
##################################################   CORE EXECUTION   #################################################
#######################################################################################################################
"""

if __name__ == '__main__':
    # NOTE: GET DATA
    all_data, taxonomy = S1_BASE._INGESTION()
    # all_data.to_csv('S1_works.csv')
    # all_data = pd.read_csv('S1_works.csv').drop('Unnamed: 0', axis=1)

    # NOTE: CONDUCT PHYSICS FEATURE ENGINEERING
    phys_engineered = S2_PHYS._FEATENG_PHYS(data=all_data)
    # phys_engineered.to_csv('S2_works.csv')
    # phys_engineered = pd.read_csv('S2_works.csv').drop('Unnamed: 0', axis=1)

    # NOTE: CONDUCT WEIGHTING (weights:=False for time speed-up)
    aggregated, PI_distances, candidates = S3_WGHT._INTELLIGENT_AGGREGATION(data=phys_engineered,
                                                                            taxonomy=taxonomy,
                                                                            weights=False)
    # aggregated.to_csv('S3_works.csv')
    # aggregated = pd.read_csv('S3_works.csv').drop('Unnamed: 0', axis=1)
    # aggregated = pd.read_csv('Data/S3 Files/combined_ipc_aggregates_PWELL.csv').drop('Unnamed: 0', axis=1)
    aggregated.rename(columns={'Steam': 'PRO_Alloc_Steam'}, inplace=True)

    # NOTE: CONDUCT OPTIMIZATION
    # TODO: Engineering Chloride Contribution
    phys_engineered['chloride_contrib'] = 0.5
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
    injector_allocation = S8_SALL._INJECTOR_ALLOCATION(data=well_allocations,
                                                       candidates=candidates['BY_WELL'].copy(),
                                                       PI_distances=PI_distances)

    return pad_sol, well_sol, injector_allocation, field_kpi

    # store = {}
    # for group, df in injector_allocation.groupby('PRO_Well'):
    #     store[group] = df.set_index('Cand_Injector')['Cand_Proportion'].to_dict()
