# @Author: Shounak Ray <Ray>
# @Date:   10-May-2021 12:05:14:149  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: coordinate_playground.py
# @Last modified by:   Ray
# @Last modified time: 03-Jun-2021 13:06:20:204  GMT-0600
# @License: [Private IP]

import os
import re
import sys
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# DONE: Coordinate Injestion and Processing
# DONE: Producer Re-Scaling
# TODO: Injector Re-Scaling
# TODO: Injector Level Toggling


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

if True:
    import _accessories

_ = """
#######################################################################################################################
################################################   DATA REQUIREMENTS   ################################################
#######################################################################################################################
"""
# FILE PATHS AND RELATIONSHIPS
LINER_PATH = 'Data/Coordinates/Liner Depths (measured depth).xlsx'
INJECTOR_COORDINATE_PATH = 'Data/Coordinates/OLT Verical Injector Bottom Hole Coordinates.xlsx'
with _accessories.suppress_stdout():
    rell_data = _accessories.retrieve_local_data_file('Data/Pickles/PRODUCTION_[Well, Pad].pkl', mode=2)

_ = """
#######################################################################################################################
###############################################   FUNCTION DEFINITIONS   ##############################################
#######################################################################################################################
"""


def replace_tab(s, tabstop=20, look_for='\t'):
    result = str()
    for c in s:
        if c == look_for:
            while (len(result) % tabstop != 0):
                result += ' '
        else:
            result += c
    return result


def custom_convert_dict_to_df(diction):
    # REFORMAT
    dfs = []
    for key, value in diction.items():
        df_local = pd.DataFrame(value, columns=['UTM_Northing', 'UTM_Easting'])
        df_local['PRO_Well'] = key
        dfs.append(df_local)
    final = pd.concat(dfs)
    final['PRO_Pad'] = final['PRO_Well'].apply(lambda x: rell_data.get(x))

    return final


def new_get_producer_positions(rell_data=rell_data, up_scalar=100, ignore=[]):
    prowells = list(_accessories.retrieve_local_data_file('Data/Pickles/PRODUCTION_[Well, Pad].pkl', mode=2).keys())
    path = '/Users/Ray/Documents/GitHub/IntelligentEnergy/Data/Horizontal Well Liner Coordinates (NAD 83, 12N).xlsx'
    all = []
    rename_dict = {'MD': 'Depth',
                   'UTM X': 'Eastings',
                   'UTM Y': 'Northings'}
    liner_bounds = pd.read_excel(LINER_PATH).infer_objects().set_index('Well')

    for pwell in prowells:
        # pwell = 'BP3'
        print(f'Well: {pwell}')
        data = pd.read_excel(path, sheet_name=pwell).rename(columns=rename_dict).dropna(axis=0, how='any')
        data['PRO_Well'] = pwell

        start_bound = liner_bounds.loc[pwell, 'Liner Start (mD)']
        end_bound = liner_bounds.loc[pwell, 'Liner End (mD)']

        if pwell in ['BP3', 'BP4', 'BP5']:
            data = data[(data['Eastings'] > 566697.025)]
        else:
            data = data[(data['Depth'] > start_bound) & (data['Depth'] < end_bound)]

        # data['Northings'].plot()
        # data['Eastings'].plot()
        # data.set_index('Eastings')['Northings'].plot()

        print(data.head(5))

        all.append(data.infer_objects())

    cumulative = pd.concat(all).reset_index(drop=True).dropna(
        axis=0, how='all').drop('Depth', axis=1).set_index('PRO_Well')
    cumulative['Coord'] = tuple(zip(cumulative['Northings'], cumulative['Eastings']))
    cumulative.drop(['Northings', 'Eastings'], axis=1, inplace=True)
    cumulative = cumulative.groupby('PRO_Well').sum()
    cumulative['Coord'] = cumulative['Coord'].apply(lambda x: [(x[i], x[i + 1])
                                                               for i in range(len(x) - 1) if i % 2 == 0])
    subsets = cumulative['Coord'].to_dict()
    for pwell, coordinates in subsets.copy().items():
        if pwell in ignore:
            del subsets[pwell]
    subsets = _accessories.norm_base(subsets, out_of_scope=True, up_scalar=up_scalar)

    return subsets


def old_get_producer_positions(rell_data=rell_data, liner_path=LINER_PATH, cut_liner=True, up_scalar=100,
                               only=[]):
    liner_bounds = pd.read_excel(liner_path).infer_objects().set_index('Well')
    f_paths = [f for f in sorted(['Data/Coordinates/' + c for c in list(os.walk('Data/Coordinates'))[0][2]])
               if ('BPRI' in f)]
    rell_data = _accessories.retrieve_local_data_file('Data/Pickles/PRODUCTION_[Well, Pad].pkl', mode=2)
    all_wells = sorted(list(rell_data.keys())) + ['I2']

    # keywords = ['measured', 'vertical', 'dogleg']
    expected_columns = ['Depth', 'Incl', 'Azim', 'SubSea_Depth', 'Vertical_Depth', 'Local_Northing',
                        'Local_Easting', 'UTM_Northing', 'UTM_Easting', 'Vertical_Section', 'Dogleg']

    subsets = {}
    all_positions = {}
    # file_path = f_paths[2]

    for file_path in f_paths:
        well_group = str([group for group in all_wells if group in file_path][0])
        lines = open(file_path, 'r', errors='ignore').readlines()
        # lines = [line.replace('\t', ',') for line in lines]
        # lines = [line.replace('\n', '') for line in lines
        #          if ((len(set(line)) > 12) & (len(set(line)) < 30)) | any([kw in line.lower() for kw in keywords])]
        # lines = [line for line in lines if line.startswith(' ')]
        # _ = [print(len(set(line))) for line in lines]

        try:
            reached = False
            data_line = [line.split('\n') for line in ''.join(
                map(str, lines)).split('\n\n') if 'Local Coordinates' in line][0]
            reached = True
            data_line = [line.split('\t') for line in data_line if line != '']
            skip = False
            print(f'SUCCESSFUL {well_group}')
        except Exception:
            if reached:
                print(f'EXCEPTION: {well_group}')
                data_line = [line[0].split('\t') for line in data_line if line != '']
                skip = False
            else:
                print(f'ISSUE: {well_group}')
                # # TAB-DELIMITED
                # # header = [replace_tab(''.join(map(str, [col + '\t' for col in expected_columns])))]
                # lines = [replace_tab(line) for line in lines if line != '']
                # lines = ''.join(lines).split('\n\n')[0].split('\n')
                # min_length = min([len(line) for line in lines]) - 1
                # lines = [line[:min_length] + '\n' for line in lines]
                #
                # col_n = len([i for i in list(lines[0].strip()) if i != ' '])
                # dummy_columns = ''.join(map(str, ['col_' + str(i) + ' ' for i in range(col_n)])) + '\n'
                # str_obj_input = StringIO(dummy_columns + ''.join(map(str, lines)))
                # df = pd.read_csv(str_obj_input, delim_whitespace=True, error_bad_lines=False).infer_objects()
                # df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
                # df = df[df.columns[df.nunique() > 2]]
                #
                # revised_columns = expected_columns + ['dummy' for i in range(len(list(df)) - len(expected_columns))]
                # df.columns = revised_columns
                #
                # skip = True
                # raise ValueError()

        if not skip:
            data_start_index = sorted([data_line.index(line) for line in data_line if '0.0' in line[0]])[0]
            data_string = data_line[data_start_index:]
            data_string = [re.sub(' +', ' ', line[0].strip()) + '\n' for line in data_string]
            dummy_columns = ''.join(map(str, ['col_' + str(i) + ' '
                                              for i in range(len(data_string[0].split(' ')))])) + '\n'
            str_obj_input = StringIO(dummy_columns + ''.join(map(str, data_string)))
            df = pd.read_csv(str_obj_input, delim_whitespace=True, error_bad_lines=False)
            # df = pd.read_csv(str_obj_input, sep=' ', error_bad_lines=False).dropna(1).infer_objects()
            df = df.select_dtypes(np.number)
            df.columns = expected_columns

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
    only = all_wells if len(only) == 0 else only
    for pwell, coordinates in subsets.copy().items():
        if pwell not in only:
            del subsets[pwell]
    subsets_FINAL = _accessories.norm_base(subsets, out_of_scope=True, up_scalar=up_scalar)

    return subsets_FINAL


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


def plot_positions(rell_data=rell_data, producer_data=None, injector_data=None,
                   for_pads=['A', 'B', 'C', 'D', 'E', 'F', 'I2'], annotate='PI'):
    fig_opened = False
    if producer_data is not None:
        producer_data = producer_data[producer_data['PRO_Pad'].isin(for_pads)]
        fig_2, ax_2 = plt.subplots(nrows=1, ncols=1, figsize=(15, 12))
        fig_opened = True
        ax_2.set_xlabel('UTM_Easting')
        ax_2.set_ylabel('UTM_Northing')
        for well, df in producer_data.groupby('PRO_Well'):
            last_point = tuple(df[['UTM_Easting', 'UTM_Northing']].iloc[-1])
            if 'P' in annotate:
                ax_2.annotate(well, last_point)
            df.plot(x='UTM_Easting', y='UTM_Northing', label=well, legend=None, ax=ax_2)
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
    # CALCULATE PRODUCER POSITIONS
    subsets_new = new_get_producer_positions(rell_data=rell_data, up_scalar=100, ignore=['CP1'])
    subsets_cp1 = {'CP1': old_get_producer_positions(up_scalar=100)['CP1']}
    subsets_final = dict(subsets_new, **subsets_cp1)
    producer_coordinates = custom_convert_dict_to_df(subsets_final)
    injector_coordinates = get_injector_coordinates(up_scalar=100)
    injector_coordinates['I71'] = (77.8037, 5.97304)

    # PLOT POSITIONS
    plot_positions(producer_data=producer_coordinates, injector_data=injector_coordinates,
                   annotate='IP', rell_data=rell_data)

    _accessories.save_local_data_file(injector_coordinates, 'Data/Coordinates/inj_coordinates.pkl')
    _accessories.save_local_data_file(producer_coordinates, 'Data/Coordinates/prod_coordinates.csv')


_ = """
#######################################################################################################################
###############################################   INJECTOR EXPLORATION   ##############################################
#######################################################################################################################
"""

INJECTION_DATA = _accessories.retrieve_local_data_file('Data/Isolated/OLT injection data.xlsx')

INJECTION_DATA[INJECTION_DATA['Well'] == 'I28'].sort_values(
    'Date (yyyy/mm/dd)').set_index('Date (yyyy/mm/dd)')['Metered Steam (m3)'].plot()

filt = INJECTION_DATA[(INJECTION_DATA['Metered Steam (m3)'] > 5) & (INJECTION_DATA['Metered Steam (m3)'] < 250)]
plt.figure(figsize=(26, 16))
_ = filt.groupby('Well')['Metered Steam (m3)'].plot(kind='hist', bins=200, legend=False, stacked=False)


# EOF

# EOF
