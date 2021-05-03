# @Author: Shounak Ray <Ray>
# @Date:   16-Apr-2021 11:04:04:041  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: acessory.py
# @Last modified by:   Ray
# @Last modified time: 03-May-2021 11:05:92:929  GMT-0600
# @License: [Private IP]

import ast
import functools
import os
import pickle
import sys
import time
from contextlib import contextmanager
from io import StringIO

import pandas as pd
from colorama import Fore, Style


def _print(txt: str, color: str = 'LIGHTGREEN_EX') -> None:
    """Custom print function with optional colored text.

    Parameters
    ----------
    txt : str
        The content of the print statement (must be a text, not any other data structure).
    color : str
        The desired color of the message. This must be compatible with the colorama.Fore package.
        SEE: https://pypi.org/project/colorama/

    Returns
    -------
    None
        While nothing is returned, this function prints to the console.

    """
    fcolor = color.upper()
    if(type(txt) == str):
        # Format the provided string
        txt = txt.replace("'", "\\'").replace('"', '\\"')
        output = f'print(Fore.{fcolor} + """{txt}""" + Style.RESET_ALL)'
        # Print the specified string to the console
    else:
        output = f'print(Fore.{fcolor}, {txt}, Style.RESET_ALL)'
    exec(output)


def retrieve_local_data_file(filedir, mode=1):
    data = None
    filename = filedir.split('/')[-1:][0]
    try:
        if(filename.endswith('.csv')):
            data = pd.read_csv(filedir, error_bad_lines=False, warn_bad_lines=False).infer_objects()
        elif(filename.endswith(('.xlsx', '.xls', '.xlsm', '.xlsb', '.odf', '.ods', '.odt'))):
            data = pd.read_excel(filedir).infer_objects()
        elif(filename.endswith('.pkl')):
            if mode == 1:
                data = pickle.load(filedir).infer_objects()
            elif mode == 2:
                with open(filedir, 'rb') as f:
                    data = f.readlines()[0]
                    data = ast.literal_eval(data.decode("utf-8").replace('\n', ''))
            elif mode == 3:
                with open(filedir, 'r') as file:
                    lines = file.readlines()
                    data = pd.read_csv(StringIO(''.join(lines)), delim_whitespace=True).infer_objects()
        if(data is None):
            raise Exception
        _print(f'> Imported "{filename}"...', color='GREEN')
    except Exception as e:
        _print('Unable to retrieve local data', color='RED')
        raise e
    return data


def save_local_data_file(data, filepath, **kwargs):
    auto_make_path(filepath)
    if(filepath.endswith('.csv')):
        data.to_csv(filepath, index=kwargs.get('index'))
    elif(filepath.endswith('.pkl')):
        with open(filepath, 'w') as file:
            print(data, file=file)
    _print(f'> Saved data to "{filepath}"')


def finalize_all(datasets, skip=[], coerce_date=True, nan_check=True, **kwargs):
    for name, df in datasets.items():
        if name in skip:
            continue
        _temp = df.infer_objects()
        if(nan_check):
            if('PRO_Pad' in df.columns):
                _print(f'NaN Checks: for {name} dataset', color='CYAN')
                na_eda(df, 'PRO_Pad', threshold=kwargs.get('threshold'))
            elif('PRO_Well' in df.columns):
                _print(f'NaN Checks: for {name} dataset', color='CYAN')
                na_eda(df, 'PRO_Well', threshold=kwargs.get('threshold'))
            else:
                _print('WARNING: No expected group in data to perform NaN check. Skipping...')
        if(coerce_date):
            _temp['Date'] = pd.to_datetime(_temp['Date'])
        datasets[name] = _temp


def auto_make_path(path: str, **kwargs: bool) -> None:
    """Create the specified directories and nested file. Custom actions based on **kwargs.

    Parameters
    ----------
    path : str
        The path containing [optional] directories and the file name with extenstion.
        Should not begin with backslash.
    **kwargs : bool
        Any keyword arguments to be processed inside `auto_make_path`.

    Returns
    -------
    None
        While nothing is returned, this function makes and [potentially] prints to a file.

    """
    # Sequentially create the directories and files specified in `path`, respectively
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if('.' in path):
        open(path, 'a').close()
    # Confirm creation (waterfall)
    if not os.path.exists(path=path):
        raise Exception('Something is TERRIBLY wrong.')
    _print(f'>> Created: \"{path}\"', color='green')


@contextmanager
def suppress_stdout():
    # _print('Attemping to suppress output using context manager...', color='LIGHTRED_EX')
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def na_eda(df, group, show='selective', threshold=0.7):
    if(threshold is None):
        threshold = 0.7
    df = df.copy()
    for g in df[group].unique():
        gdf = df[df[group] == g].reset_index(drop=True)
        _print(f'"{g}" grouped by "{group}"', color='CYAN')
        content = dict(gdf.isna().sum() / len(gdf))
        if(show == "all"):
            print(content)
        elif(show == 'selective'):
            for col, prop in content.items():
                if(prop > threshold):
                    _print(f'WARNING: Column "{col}" in group "{g}" exceeded threshold of {threshold} with ' +
                           f'a NaN proportion of {prop}', color='LIGHTRED_EX')


def _distinct_colors():
    return ["#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
            "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87", "#5A0007", "#809693",
            "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80", "#61615A", "#BA0900", "#6B7900",
            "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100", "#7B4F4B", "#A1C299",
            "#0AA6D8", "#00846F", "#FFB500", "#C2FFED", "#A079BF", "#CC0744",
            "#C0B9B2", "#C2FF99", "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68",
            "#7A87A1", "#788D66", "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED",
            "#886F4C", "#34362D", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
            "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00", "#7900D7",
            "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700", "#549E79", "#FFF69F",
            "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329", "#5B4534", "#FDE8DC", "#404E55",
            "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C", "#83AB58", "#D1F7CE", "#004B28",
            "#C8D0F6", "#A3A489", "#806C66", "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59",
            "#8ADBB4", "#5B4E51", "#C895C5", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94",
            "#7ED379", "#7A7BFF", "#D68E01", "#353339", "#78AFA1", "#FEB2C6", "#75797C", "#837393",
            "#943A4D", "#B5F4FF", "#D2DCD5", "#9556BD", "#6A714A", "#02525F", "#0AA3F7", "#E98176",
            "#5EBCD1", "#3D4F44", "#7E6405", "#02684E", "#962B75", "#8D8546", "#9695C5", "#E773CE",
            "#D86A78", "#3E89BE", "#CA834E", "#518A87", "#5B113C", "#55813B", "#E704C4", "#A97399",
            "#4B8160", "#59738A", "#FF5DA7", "#F7C9BF", "#643127", "#513A01", "#6B94AA", "#51A058", "#A45B02",
            "#E20027", "#E7AB63", "#4C6001", "#9C6966", "#64547B", "#97979E", "#006A66",
            "#F4D749", "#0045D2", "#006C31", "#DDB6D0", "#7C6571", "#9FB2A4", "#00D891", "#15A08A", "#BC65E9",
            "#C6DC99", "#671190", "#6B3A64", "#FFA0F2", "#CCAA35", "#374527",
            "#8BB400", "#797868", "#C6005A", "#3B000A", "#C86240", "#29607C", "#402334", "#7D5A44", "#CCB87C",
            "#B88183", "#AA5199", "#B5D6C3", "#A38469", "#9F94F0", "#A74571", "#B894A6", "#71BB8C", "#00B433",
            "#789EC9", "#6D80BA", "#953F00", "#5EFF03", "#E4FFFC", "#1BE177", "#BCB1E5", "#76912F", "#003109",
            "#0060CD", "#D20096", "#895563", "#A76F42", "#89412E", "#1A3A2A", "#494B5A",
            "#A88C85", "#F4ABAA", "#A3F3AB", "#00C6C8", "#EA8B66", "#958A9F", "#BDC9D2", "#9FA064", "#BE4700",
            "#658188", "#83A485", "#453C23", "#47675D", "#3A3F00", "#DFFB71", "#868E7E", "#98D058",
            "#6C8F7D", "#D7BFC2", "#3C3E6E", "#D83D66", "#2F5D9B", "#6C5E46", "#D25B88", "#5B656C", "#00B57F",
            "#545C46", "#866097", "#365D25", "#252F99", "#00CCFF", "#674E60", "#FC009C", "#92896B"]


class timeit:
    """decorator for benchmarking"""

    def __init__(self, fmt='Completed {:s} in {:.3f} seconds', track_type='modeling_benchmarks'):
        # there is no need to make a class for a decorator if there are no parameters
        self.fmt = fmt
        self.track_type = track_type

    def __call__(self, fn):
        # returns the decorator itself, which accepts a function and returns another function
        # wraps ensures that the name and docstring of 'fn' is preserved in 'wrapper'
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # the wrapper passes all parameters to the function being decorated
            t1 = time.time()
            res = fn(*args, **kwargs)
            t2 = time.time()
            _print(self.fmt.format(fn.__name__, t2 - t1), color='LIGHTBLUE_EX')

            path = f'_configs/{self.track_type}.csv'
            auto_make_path(path)
            with open(path, 'a') as file:
                content = str(kwargs.get('math_eng')) + ',' + \
                    str(kwargs.get('weighting')) + ',' + str(kwargs.get('MAX_EXP_RUNTIME')) + ',' + str(t2 - t1) + \
                    ',' + str(res)
                file.write(content)

            return res
        return wrapper
