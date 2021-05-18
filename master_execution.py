# @Author: Shounak Ray <Ray>
# @Date:   17-May-2021 10:05:35:354  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: master_execution.py
# @Last modified by:   Ray
# @Last modified time: 17-May-2021 23:05:25:250  GMT-0600
# @License: [Private IP]

import pandas as pd
# from pipeline import S4_ft_eng_math as S4_MATH --> Ignored: Used directly in S5_MODL
# from pipeline import S5_modeling as S5_MODL --> Ignored: This is for model generation, not creation
from pipeline import S1_base_generation as S1_BASE
from pipeline import S2_ft_eng_physics as S2_PHYS
from pipeline import S3_weighting as S3_WGHT
from pipeline import S6_optimization as S6_OPTM
from pipeline import S7_prosteam_allocation as S7_WALL
from pipeline import S8_injsteam_allocation as S8_SALL

_ = """
#######################################################################################################################
#############################################   HYPERPARAMETER SETTINGS   #############################################
#######################################################################################################################
"""


class Optimization_Params:
    def __init__(self, date, steam_available, steam_variance, pad_steam_constraint, well_steam_constraint,
                 well_pump_constraint, watercut_source, chl_steam_percent, pres_steam_percent, recent_days,
                 hist_days, target, features, group):
        self.date = date
        self.steam_available = steam_available
        self.steam_variance = steam_variance
        self.pad_steam_constraint = pad_steam_constraint
        self.well_steam_constraint = well_steam_constraint
        self.well_pump_constraint = well_pump_constraint
        self.watercut_source = watercut_source
        self.chl_steam_percent = chl_steam_percent
        self.pres_steam_percent = pres_steam_percent
        self.recent_days = recent_days
        self.hist_days = hist_days
        self.target = target
        self.features = features
        self.group = group


Op_Params = {
    "date": "2020-12-01",
    "steam_available": 6000,
    "steam_variance": 0.25,
    "pad_steam_constraint": {
        "A": {"min": 0, "max": 2000},
        "B": {"min": 0, "max": 2000},
        "C": {"min": 0, "max": 2000},
        "E": {"min": 0, "max": 2000},
        "F": {"min": 1200, "max": 2000},
    },
    "well_steam_constraint": {
        "AP2": {"min": 0, "max": 400},
        "CP7": {"min": 0, "max": 400},
        "CP8": {"min": 0, "max": 400},
        "EP2": {"min": 0, "max": 400},
        "EP3": {"min": 0, "max": 400},
        "EP4": {"min": 0, "max": 400},
        "EP5": {"min": 0, "max": 400},
        "CP6": {"min": 0, "max": 400},
        "EP7": {"min": 0, "max": 400},
        "FP2": {"min": 0, "max": 400},
        "FP3": {"min": 0, "max": 400},
        "FP4": {"min": 0, "max": 400},
        "FP5": {"min": 0, "max": 400},
        "FP6": {"min": 0, "max": 400},
        "FP7": {"min": 0, "max": 400},
        "FP1": {"min": 0, "max": 400},
        "CP5": {"min": 0, "max": 400},
        "EP6": {"min": 0, "max": 400},
        "CP3": {"min": 0, "max": 400},
        "CP4": {"min": 0, "max": 400},
        "AP3": {"min": 0, "max": 400},
        "AP4": {"min": 0, "max": 400},
        "AP5": {"min": 0, "max": 400},
        "AP6": {"min": 0, "max": 400},
        "AP8": {"min": 0, "max": 400},
        "BP1": {"min": 0, "max": 400},
        "AP7": {"min": 0, "max": 400},
        "BP3": {"min": 0, "max": 400},
        "BP4": {"min": 0, "max": 400},
        "BP5": {"min": 0, "max": 400},
        "BP6": {"min": 0, "max": 400},
        "CP1": {"min": 0, "max": 400},
        "CP2": {"min": 0, "max": 400},
        "BP2": {"min": 0, "max": 400},
    },
    "well_pump_constraint": {
        "AP2": {"min": 1, "max": 4.5},
        "CP7": {"min": 1, "max": 4.5},
        "CP8": {"min": 1, "max": 4.5},
        "EP2": {"min": 1, "max": 4.5},
        "EP3": {"min": 1, "max": 4.5},
        "EP4": {"min": 1, "max": 4.5},
        "EP5": {"min": 1, "max": 4.5},
        "CP6": {"min": 1, "max": 4.5},
        "EP7": {"min": 1, "max": 4.5},
        "FP2": {"min": 1, "max": 4.5},
        "FP3": {"min": 1, "max": 4.5},
        "FP4": {"min": 1, "max": 4.5},
        "FP5": {"min": 1, "max": 4.5},
        "FP6": {"min": 1, "max": 4.5},
        "FP7": {"min": 1, "max": 4.5},
        "FP1": {"min": 1, "max": 4.5},
        "CP5": {"min": 1, "max": 4.5},
        "EP6": {"min": 1, "max": 4.5},
        "CP3": {"min": 1, "max": 4.5},
        "CP4": {"min": 1, "max": 4.5},
        "AP3": {"min": 1, "max": 4.5},
        "AP4": {"min": 1, "max": 4.5},
        "AP5": {"min": 1, "max": 4.5},
        "AP6": {"min": 1, "max": 4.5},
        "AP8": {"min": 1, "max": 4.5},
        "BP1": {"min": 1, "max": 4.5},
        "AP7": {"min": 1, "max": 4.5},
        "BP3": {"min": 1, "max": 4.5},
        "BP4": {"min": 1, "max": 4.5},
        "BP5": {"min": 1, "max": 4.5},
        "BP6": {"min": 1, "max": 4.5},
        "CP1": {"min": 1, "max": 4.5},
        "CP2": {"min": 1, "max": 4.5},
        "BP2": {"min": 1, "max": 4.5},
    },
    "watercut_source": "meas_water_cut",
    "chl_steam_percent": 0.1,
    "pres_steam_percent": 0.15,
    "recent_days": 45,
    "hist_days": 365,
    "target": "total_fluid",
    "features": [
        "prod_casing_pressure",
        "prod_bhp_heel",
        "prod_bhp_toe",
        "alloc_steam",
    ],
    "group": "pad",
}


MAPPING = {'date': 'Date',
           'uwi': 'PRO_UWI',
           'producer_well': 'PRO_Well',
           'spm': 'PRO_Adj_Pump_Speed',
           'hours_on_prod': 'PRO_Time_On',
           'prod_casing_pressure': 'PRO_Casing_Pressure',
           'prod_bhp_heel': 'PRO_Heel_Pressure',
           'prod_bhp_toe': 'PRO_Toe_Pressure',
           'oil': 'PRO_Alloc_Oil',
           'water': 'PRO_Alloc_Water',
           'bin_1': 'Bin_1',
           'bin_2': 'Bin_2',
           'bin_3': 'Bin_3',
           'bin_4': 'Bin_4',
           'bin_5': 'Bin_5',
           'ci06_steam': '',
           'ci07_steam': '',
           'ci08_steam': '',
           'i02_steam': '',
           'i03_steam': '',
           'i04_steam': '',
           'i05_steam': '',
           'i06_steam': '',
           'i07_steam': '',
           'i08_steam': '',
           'i09_steam': '',
           'i10_steam': '',
           'i11_steam': '',
           'i12_steam': '',
           'i13_steam': '',
           'i14_steam': '',
           'i15_steam': '',
           'i16_steam': '',
           'i17_steam': '',
           'i18_steam': '',
           'i19_steam': '',
           'i20_steam': '',
           'i21_steam': '',
           'i22_steam': '',
           'i23_steam': '',
           'i24_steam': '',
           'i25_steam': '',
           'i26_steam': '',
           'i27_steam': '',
           'i28_steam': '',
           'i29_steam': '',
           'i30_steam': '',
           'i31_steam': '',
           'i32_steam': '',
           'i33_steam': '',
           'i34_steam': '',
           'i35_steam': '',
           'i36_steam': '',
           'i37_steam': '',
           'i38_steam': '',
           'i39_steam': '',
           'i40_steam': '',
           'i41_steam': '',
           'i42_steam': '',
           'i43_steam': '',
           'i44_steam': '',
           'i45_steam': '',
           'i46_steam': '',
           'i47_steam': '',
           'i48_steam': '',
           'i49_steam': '',
           'i50_steam': '',
           'i51_steam': '',
           'i52_steam': '',
           'i53_steam': '',
           'i54_steam': '',
           'i55_steam': '',
           'i56_steam': '',
           'i57_steam': '',
           'i58_steam': '',
           'i59_steam': '',
           'i60_steam': '',
           'i61_steam': '',
           'i62_steam': '',
           'i63_steam': '',
           'i64_steam': '',
           'i65_steam': '',
           'i66_steam': '',
           'i67_steam': '',
           'i68_steam': '',
           'i69_steam': '',
           'i70_steam': '',
           'i71_steam': '',
           'i72_steam': '',
           'i73_steam': '',
           'i74_steam': '',
           'i75_steam': '',
           'i76_steam': '',
           'i77_steam': '',
           'i78_steam': '',
           'i79_steam': '',
           'pad': 'PRO_Pad',
           'test_oil': 'PRO_Oil',
           'test_water': 'PRO_Water',
           'test_chlorides': 'PRO_Chlorides',
           'test_spm': '',
           'pump_size': '',
           'pump_efficiency': 'PRO_Pump_Efficiency',
           'op_approved': 'op_approved',
           'op_comment': 'op_comment',
           'eng_approved': 'PRO_Engineering_Approved',
           'eng_comment': 'eng_comment',
           'test_total_fluid': 'test_total_fluid',
           'total_fluid': 'total_fluid',
           'volume_per_stroke': 'volume_per_stroke',
           'theoretical_fluid': 'theoretical_fluid',
           'test_water_cut': 'test_water_cut',
           'theoretical_water': 'theoretical_water',
           'theoretical_oil': 'theoretical_water',
           'alloc_steam': 'alloc_steam',
           'sor': 'sor',
           'chlorides': 'chlorides',
           'meas_water_cut': 'meas_water_cut',
           'field_chloride': 'field_chloride',
           'chloride_contrib': 'chloride_contrib',
           'field': 'field',
           'pressure_average': 'pressure_average',
           'ci06_pressure': '',
           'ci07_pressure': '',
           'ci08_pressure': '',
           'i02_pressure': '',
           'i03_pressure': '',
           'i04_pressure': '',
           'i05_pressure': '',
           'i06_pressure': '',
           'i07_pressure': '',
           'i08_pressure': '',
           'i09_pressure': '',
           'i10_pressure': '',
           'i11_pressure': '',
           'i12_pressure': '',
           'i13_pressure': '',
           'i14_pressure': '',
           'i15_pressure': '',
           'i16_pressure': '',
           'i17_pressure': '',
           'i18_pressure': '',
           'i19_pressure': '',
           'i20_pressure': '',
           'i21_pressure': '',
           'i22_pressure': '',
           'i23_pressure': '',
           'i24_pressure': '',
           'i25_pressure': '',
           'i26_pressure': '',
           'i27_pressure': '',
           'i28_pressure': '',
           'i29_pressure': '',
           'i30_pressure': '',
           'i31_pressure': '',
           'i32_pressure': '',
           'i33_pressure': '',
           'i34_pressure': '',
           'i35_pressure': '',
           'i36_pressure': '',
           'i37_pressure': '',
           'i38_pressure': '',
           'i39_pressure': '',
           'i40_pressure': '',
           'i41_pressure': '',
           'i42_pressure': '',
           'i43_pressure': '',
           'i44_pressure': '',
           'i45_pressure': '',
           'i46_pressure': '',
           'i47_pressure': '',
           'i48_pressure': '',
           'i49_pressure': '',
           'i50_pressure': '',
           'i51_pressure': '',
           'i52_pressure': '',
           'i53_pressure': '',
           'i54_pressure': '',
           'i55_pressure': '',
           'i56_pressure': '',
           'i57_pressure': '',
           'i58_pressure': '',
           'i59_pressure': '',
           'i60_pressure': '',
           'i61_pressure': '',
           'i62_pressure': '',
           'i63_pressure': '',
           'i64_pressure': '',
           'i65_pressure': '',
           'i66_pressure': '',
           'i67_pressure': '',
           'i68_pressure': '',
           'i69_pressure': '',
           'i70_pressure': '',
           'i71_pressure': '',
           'i72_pressure': '',
           'i73_pressure': '',
           'i74_pressure': '',
           'i75_pressure': '',
           'i76_pressure': '',
           'i77_pressure': '',
           'i78_pressure': '',
           'i79_pressure': '',
           'i80': '',
           'i82': '',
           'i83': '',
           'i84': '',
           'i85': '',
           'i86': '',
           'i87': '',
           'i88': '',
           'i89': '',
           'i90': '',
           'i91': '',
           'i92': '',
           'i93': ''}


_ = """
#######################################################################################################################
##################################################   CORE EXECUTION   #################################################
#######################################################################################################################
"""

if __name__ == '__main__':
    # GET DATA
    # all_data = S1_BASE._INGESTION()
    # all_data.to_csv('S1_works.csv')
    all_data = pd.read_csv('S1_works.csv').drop('Unnamed: 0', axis=1)

    # CONDUCT PHYSICS FEATURE ENGINEERING
    # phys_engineered = S2_PHYS._FEATENG_PHYS(data=all_data)
    # phys_engineered.to_csv('S2_works.csv')
    phys_engineered = pd.read_csv('S2_works.csv').drop('Unnamed: 0', axis=1)

    # CONDUCT WEIGHTING
    # aggregated = S3_WGHT._INTELLIGENT_AGGREGATION(data=phys_engineered, weights=False)
    # aggregated.to_csv('S3_works.csv')
    # aggregated = pd.read_csv('S3_works.csv').drop('Unnamed: 0', axis=1)
    aggregated = pd.read_csv('Data/S3 Files/combined_ipc_aggregates_PWELL.csv').drop('Unnamed: 0', axis=1)
    aggregated.rename(columns={'Steam': 'PRO_Alloc_Steam'}, inplace=True)

    # CONDUCT OPTIMIZATION
    aggregated['chloride_contrib'] = 0.5
    aggregated['PRO_Chlorides'] = 2000
    phys_engineered['chloride_contrib'] = 0.5
    # list(phys_engineered)
    macro_results, chloride_output = S6_OPTM._OPTIMIZATION(data=phys_engineered, engineered=False,
                                                           today=False, singular_date='2020-12-01')

    # CONDUCT WELL-ALLOCATION
    # TODO: This needs to be formatted as a dict
    well_allocation = {'AP2': 168.50638133970068,
                       'AP3': 158.77670562530514,
                       'AP4': 218.1557318198097,
                       'AP5': 184.56816243995024,
                       'AP6': 193.2713740126074,
                       'AP7': 169.2051413545468,
                       'AP8': 172.03388085040683,
                       'BP1': 275.1545015663883,
                       'BP2': 199.0340338530178,
                       'BP3': 181.0507502880953,
                       'BP4': 108.6806995783778,
                       'BP5': 265.3717097531363,
                       'BP6': 214.94264461323525,
                       'CP1': 299.8409604774457,
                       'CP2': 101.34910855792296,
                       'CP3': 228.57072034063148,
                       'CP4': 151.64662099501112,
                       'CP5': 239.7184561911505,
                       'CP6': 184.13507125912707,
                       'CP7': 133.30833235009698,
                       'CP8': 42.308273124784144,
                       'EP2': 176.40490759899953,
                       'EP3': 244.58464252535686,
                       'EP4': 54.901433696492944,
                       'EP5': 78.25684769901943,
                       'EP6': 106.2963343631096,
                       'EP7': 250.1620315560527,
                       'FP1': 197.11805473487084,
                       'FP2': 219.36129581897654,
                       'FP3': 192.32563480401095,
                       'FP4': 117.3281237235291,
                       'FP5': 207.15013466550303,
                       'FP6': 120.53492563970951,
                       'FP7': 145.94637278362194}
    well_allocation = None  # S7_SALL._PRODUCER_ALLOCATION()

    # CONDUCT INJECTOR-ALLOCATION

    injector_allocation = S8_SALL._INJECTOR_ALLOCATION(data=well_allocation)
    store = {}
    for group, df in injector_allocation.groupby('PRO_Well'):
        store[group] = df.set_index('Cand_Injector')['Cand_Proportion'].to_dict()
