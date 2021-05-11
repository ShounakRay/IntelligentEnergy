# @Author: Shounak Ray <Ray>
# @Date:   10-May-2021 12:05:14:149  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: coordinate_playground.py
# @Last modified by:   Ray
# @Last modified time: 11-May-2021 16:05:78:787  GMT-0600
# @License: [Private IP]

import os
import re
from io import StringIO

import _references._accessories as _accessories
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# DONE: Coordinate Injestion and Processing
# DONE: Producer Re-Scaling
# TODO: Injector Re-Scaling
# TODO: Injector Level Toggling

_ = """
#######################################################################################################################
################################################   DATA REQUIREMENTS   ################################################
#######################################################################################################################
"""
# FILE PATHS AND RELATIONSHIPS
LINER_PATH = 'Data/Coordinates/Liner Depths (measured depth).xlsx'
INJECTOR_COORDINATE_PATH = 'Data/Coordinates/OLT Verical Injector Bottom Hole Coordinates.xlsx'

_ = """
#######################################################################################################################
###############################################   FUNCTION DEFINITIONS   ##############################################
#######################################################################################################################
"""


def get_producer_positions(rell_data, liner_path=LINER_PATH, cut_liner=True, up_scalar=100):
    liner_bounds = pd.read_excel(liner_path).infer_objects().set_index('Well')
    f_paths = [f for f in sorted(['Data/Coordinates/' + c for c in list(os.walk('Data/Coordinates'))[0][2]])
               if ('BPRI' in f)]
    rell_data = _accessories.retrieve_local_data_file('Data/Pickles/PRODUCTION_[Well, Pad].pkl', mode=2)
    all_wells = sorted(list(rell_data.keys()))

    subsets = {}
    all_positions = {}
    for file_path in f_paths:
        well_group = str([group for group in all_wells + ['I2'] if group in file_path][0])
        lines = open(file_path, 'r', errors='ignore').readlines()

        try:
            data_line = [line.split('\n') for line in ''.join(
                map(str, lines)).split('\n\n') if 'Local Coordinates' in line][0]
            data_line = [line.split('\t') for line in data_line if line != '']
        except Exception:
            data_line = [line[0].split('\t') for line in data_line if line != '']

        data_start_index = sorted([data_line.index(line) for line in data_line if '0.0' in line[0]])[0]
        data_string = data_line[data_start_index:]
        data_string = [re.sub(' +', ' ', line[0].strip()) + '\n' for line in data_string]
        dummy_columns = ''.join(map(str, ['col_' + str(i) + ' '
                                          for i in range(len(data_string[0].split(' ')))])) + '\n'
        str_obj_input = StringIO(dummy_columns + ''.join(map(str, data_string)))
        df = pd.read_csv(str_obj_input, delim_whitespace=True, error_bad_lines=False)
        # df = pd.read_csv(str_obj_input, sep=' ', error_bad_lines=False).dropna(1).infer_objects()
        df = df.select_dtypes(np.number)
        df.columns = ['Depth', 'Incl', 'Azim', 'SubSea_Depth', 'Vertical_Depth', 'Local_Northing',
                      'Local_Easting', 'UTM_Northing', 'UTM_Easting', 'Vertical_Section', 'Dogleg']

        if cut_liner:
            # Constrain data based on liner bounds
            start_bound = liner_bounds.loc[well_group, 'Liner Start (mD)']
            end_bound = liner_bounds.loc[well_group, 'Liner End (mD)']
            final_df = df[(df['Depth'] > start_bound) & (df['Depth'] < end_bound)]
        else:
            final_df = df
        all_positions[well_group] = final_df.sort_values('Depth').reset_index(drop=True)

        # Store subset coordinates for normalization
        subsets[well_group] = final_df[['UTM_Northing', 'UTM_Easting']].values.tolist()

    # Normalize the coordinates
    subsets = _accessories.norm_base(subsets, out_of_scope=True, up_scalar=up_scalar)
    for well_group, df in all_positions.items():
        df['UTM_Northing'] = [tup[0] for tup in subsets.get(well_group)]
        df['UTM_Easting'] = [tup[1] for tup in subsets.get(well_group)]

    return all_positions, subsets


def get_injector_coordinates(path=INJECTOR_COORDINATE_PATH, up_scalar=100):
    inj_coords = pd.read_excel(path)
    inj_coords.columns = ['Injector', 'Lat_Y', 'Long_X']
    inj_coords['Lat_Y'] = inj_coords['Lat_Y'].apply(lambda x: float(str(x)[:-1]))
    inj_coords['Long_X'] = inj_coords['Long_X'].apply(lambda x: float(str(x)[:-1]))
    inj_coords = inj_coords.set_index('Injector').to_dict()
    all_injs = sorted(list(inj_coords['Lat_Y'].keys()))

    # CALCULATE INJECTOR POSITIONS
    injector_coordinates = {}
    for inj in all_injs:
        map_x, map_y = -1, 1
        coord = (inj_coords['Long_X'].get(inj) * map_x, inj_coords['Lat_Y'].get(inj) * map_y)
        injector_coordinates[inj] = coord
    # Scale down to 0–1 range
    injector_coordinates = _accessories.norm_base(injector_coordinates, up_scalar=up_scalar)

    return injector_coordinates


def plot_positions(rell_data, producer_data=None, injector_data=None,
                   for_pads=['A', 'B', 'C', 'D', 'E', 'F', 'I2'], annotate='PI'):
    fig_opened = False
    if producer_data is not None:
        _temp = {k: v for k, v in producer_data.items() if rell_data.get(k) in for_pads}
        fig_2, ax_2 = plt.subplots(nrows=1, ncols=1, figsize=(15, 12))
        fig_opened = True
        ax_2.set_ylabel('UTM_Northing')
        ax_2.set_ylabel('Local_Northing')
        for well, df in _temp.items():
            last_point = tuple(df[['UTM_Easting', 'UTM_Northing']].tail(1).reset_index(drop=True).iloc[0])
            if 'P' in annotate:
                ax_2.annotate(well, last_point)
            df.plot(x='UTM_Easting', y='UTM_Northing', ax=ax_2, label=well, legend=None)
        plt.tight_layout()

    if injector_data is not None:
        if not fig_opened:
            fig_2, ax_2 = plt.subplots(nrows=1, ncols=1, figsize=(15, 12))
        for inj, coord in injector_coordinates.items():
            ax_2.scatter(*coord, c='red')
            if 'I' in annotate:
                ax_2.annotate(inj, coord)
        plt.tight_layout()


_ = """
#######################################################################################################################
#############################################   GET COORDINATES AND PLOT   ############################################
#######################################################################################################################
"""

if __name__ == '__main__':
    # FILE PATHS AND RELATIONSHIPS
    rell_data = _accessories.retrieve_local_data_file('Data/Pickles/PRODUCTION_[Well, Pad].pkl', mode=2)

    # CALCULATE PRODUCER POSITIONS
    all_positions, subsets = get_producer_positions(rell_data, cut_liner=False, up_scalar=100)
    injector_coordinates = get_injector_coordinates(up_scalar=100)

    # PLOT POSITIONS
    plot_positions(producer_data=all_positions, injector_data=injector_coordinates,
                   annotate='I', rell_data=rell_data)


# EOF

# EOF
