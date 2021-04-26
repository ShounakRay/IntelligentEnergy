# @Author: Shounak Ray <Ray>
# @Date:   16-Apr-2021 11:04:04:041  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: acessory.py
# @Last modified by:   Ray
# @Last modified time: 26-Apr-2021 11:04:83:837  GMT-0600
# @License: [Private IP]

import ast
import functools
import os
import sys
import time
from contextlib import contextmanager

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
        output = f'print(Fore.{fcolor} + {txt} + Style.RESET_ALL)'
    exec(output)


def retrieve_local_data_file(filedir):
    data = None
    filename = filedir.split('/')[-1:][0]
    try:
        if(filename.endswith('.csv')):
            data = pd.read_csv(filedir, error_bad_lines=False, warn_bad_lines=False).infer_objects()
        elif(filename.endswith(('.xlsx', '.xls', '.xlsm', '.xlsb', '.odf', '.ods', '.odt'))):
            data = pd.read_excel(filedir).infer_objects()
        elif(filename.endswith('.pkl')):
            with open(filedir, 'rb') as f:
                data = f.readlines()[0]
                data = ast.literal_eval(data.decode("utf-8").replace('\n', ''))
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
        open(path, 'w').close()
    # Confirm creation (waterfall)
    if not os.path.exists(path=path):
        raise Exception('Something is TERRIBLY wrong.')
    _print(f'>> Created: \"{path}\"', color='green')


@contextmanager
def suppress_stdout():
    _print('Attemping to suppress output using context manager...', color='LIGHTRED_EX')
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
