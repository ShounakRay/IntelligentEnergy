# @Author: Shounak Ray <Ray>
# @Date:   16-Apr-2021 11:04:04:041  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: acessory.py
# @Last modified by:   Ray
# @Last modified time: 20-Apr-2021 13:04:53:538  GMT-0600
# @License: [Private IP]

import os

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
    # Format the provided string
    fcolor = color.upper()
    txt = txt.replace("'", "\\'").replace('"', '\\"')
    output = f'print(Fore.{fcolor} + """{txt}""" + Style.RESET_ALL)'
    # Print the specified string to the console
    exec(output)


def retrieve_local_data_file(filedir):
    data = None
    filename = filedir.split('/')[-1:][0]
    try:
        if(filename.endswith('.csv')):
            data = pd.read_csv(filedir, error_bad_lines=False, warn_bad_lines=False)
            _print(f'> Importing "{filename}"...', color='GREEN')
        elif(filename.endswith(('.xlsx', '.xls', '.xlsm', '.xlsb', '.odf', '.ods', '.odt'))):
            data = pd.read_excel(filedir)
            _print(f'> Importing "{filename}"...', color='GREEN')
        if(data is None):
            raise Exception
    except Exception as e:
        _print('Unable to retrieve local data', color='RED')
        raise e
    return data.infer_objects()


def save_local_data_file(data, filepath, **kwargs):
    data.to_csv(filepath, index=kwargs.get('index'))
    _print(f'> Saved data to "{filepath}"')


def finalize_all(datasets, skip=[], coerce_date=True):
    for name, df in datasets.items():
        if name in skip:
            continue
        _temp = df.infer_objects()
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
        Currently supports:
        > `exceed_json` : bool –> Indicates whether an empty dictionary should be printed to the opened JSON file.

    Returns
    -------
    None
        While nothing is returned, this function makes and [potentially] prints to a file.

    """
    # Sequentially create the directories and files specified in `path`, respectively
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w').close()
    # Confirm creation (waterfall)
    if not os.path.exists(path=path):
        raise Exception(message='Something is TERRIBLY wrong.')
    _print(f'>> Created: \"{path}\"', color='green')
