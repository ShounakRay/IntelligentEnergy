# @Author: Shounak Ray <Ray>
# @Date:   10-May-2021 12:05:14:149  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: coordinate_playground.py
# @Last modified by:   Ray
# @Last modified time: 10-May-2021 15:05:93:934  GMT-0600
# @License: [Private IP]

import os
import re
from io import StringIO

import _references._accessories as _accessories
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# FILE PATHS AND RELATIONSHIPS
all_file_paths = [f for f in sorted(
    ['Data/Coordinates/' + c for c in list(os.walk('Data/Coordinates'))[0][2]]) if ('BPRI' in f)]
rell_data = _accessories.retrieve_local_data_file('Data/Pickles/PRODUCTION_[Well, Pad].pkl', mode=2)
all_wells = sorted(list(rell_data.keys()))
all_pads = list(set(list(rell_data.values())))

# LINER BOUNDS
liner_bounds = pd.read_excel('Data/Coordinates/Liner Depths (measured depth).xlsx').infer_objects()

# INJECTOR COORDINATES
inj_coords = pd.read_excel('Data/Coordinates/OLT Verical Injector Bottom Hole Coordinates.xlsx')
inj_coords.columns = ['Injector', 'Lat_Y', 'Long_X']
inj_coords['Lat_Y'] = inj_coords['Lat_Y'].apply(lambda x: float(str(x)[:-1]))
inj_coords['Long_X'] = inj_coords['Long_X'].apply(lambda x: float(str(x)[:-1]))
inj_coords = inj_coords.set_index('Injector').to_dict()
all_injs = sorted(list(inj_coords['Lat_Y'].keys()))

# PARSE PRODUCER POSITIONS
all_files = {}
all_positions = {}
for file_path in all_file_paths:
    well_group = str([group for group in all_wells + ['I2'] if group in file_path][0])
    lines = open(file_path, 'r', errors='ignore').readlines()
    all_files[file_path] = lines

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
    _accessories._print(f'Well: {well_group}')
    print(df)
    # df = pd.read_csv(str_obj_input, sep=' ', error_bad_lines=False).dropna(1).infer_objects()
    df = df.select_dtypes(np.number)
    df.columns = ['Depth', 'Incl', 'Azim', 'SubSea_Depth', 'Vertical_Depth', 'Local_Northing',
                  'Local_Easting', 'UTM_Northing', 'UTM_Easting', 'Vertical_Section', 'Dogleg']
    # df = df[['UTM_Easting', 'UTM_Northing']]

    # start_bound = float(liner_bounds[liner_bounds['Well'] == well_group]['Liner Start (mD)'])
    # end_bound = float(liner_bounds[liner_bounds['Well'] == well_group]['Liner End (mD)'])
    # final_df = df[(df['Depth'] > start_bound) & (df['Depth'] < end_bound)]
    final_df = df
    all_positions[well_group] = final_df.sort_values('Depth').reset_index(drop=True)


# fig, ax = plt.subplots(nrows=len(all_positions.keys()), ncols=2, figsize=(15, 100))
# for well in all_positions.keys():
#     group_df = all_positions[well]  # /all_positions[well].max()
#     axes_1 = ax[list(all_positions.keys()).index(well)][0]
#     axes_1.plot(group_df['UTM_Easting'], group_df['UTM_Northing'])
#     axes_1.set_xlabel('UTM Easting')
#     axes_1.set_ylabel('UTM Northing')
#     axes_1.set_title(well + ' UTM Coordinates')
#     # axes_1.set_ylim(1000000 + 3.963 * 10**6, 1000000 + 5.963 * 10**6)
#
#     axes_2 = ax[list(all_positions.keys()).index(well)][1]
#     axes_2.plot(group_df['Local_Easting'], group_df['Local_Northing'])
#     axes_2.set_xlabel('Local Easting')
#     axes_2.set_ylabel('Local Northing')
#     axes_2.set_title(well + ' Local Coordinates')
#     # axes_2.set_ylim(0, 225)
# plt.tight_layout()
# fig.suptitle('Coordinates Bounded by Provided Liner Depths XLSX File')
# plt.savefig('Modeling Reference Files/Candidate Selection Images/provided_coordinate_plots.png',
#             bbox_inches='tight')

# PLOT PRODUCER POSITIONS
_temp = {k: v for k, v in all_positions.items() if rell_data.get(k) in ['A', 'B', 'C', 'D', 'E', 'F', 'I2']}
fig_2, ax_2 = plt.subplots(nrows=1, ncols=2, figsize=(30, 12))
ax_2[0].set_ylabel('UTM_Northing')
ax_2[1].set_ylabel('Local_Northing')
for well, df in _temp.items():
    last_point = tuple(df[['UTM_Easting', 'UTM_Northing']].tail(1).reset_index(drop=True).iloc[0])
    ax_2[0].annotate(well, last_point)
    df.plot(x='UTM_Easting', y='UTM_Northing', ax=ax_2[0], label=well, legend=None)
    last_point = tuple(df[['Local_Easting', 'Local_Northing']].tail(1).reset_index(drop=True).iloc[0])
    ax_2[1].annotate(well, last_point)
    df.plot(x='Local_Easting', y='Local_Northing', ax=ax_2[1], label=well, legend=None)
plt.tight_layout()
# PLOT INJECTOR POSITIONS
for inj in all_injs:
    coord = (inj_coords['Long_X'].get(inj), inj_coords['Lat_Y'].get(inj))
    ax_2[0].plot(*coord)
    ax_2[0].annotate(inj, coord)
    ax_2[1].plot(*coord)
    ax_2[1].annotate(inj, coord)


# EOF

# EOF
