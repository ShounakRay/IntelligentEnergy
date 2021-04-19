# @Author: Shounak Ray <Ray>
# @Date:   16-Apr-2021 11:04:04:041  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: acessory.py
# @Last modified by:   Ray
# @Last modified time: 19-Apr-2021 16:04:33:334  GMT-0600
# @License: [Private IP]

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
            _print(f'> Importing "{filename}"...', color='GREEN')
            data = pd.read_csv(filedir, error_bad_lines=False, warn_bad_lines=False)
        elif(filename.endswith(('.xlsx', '.xls', '.xlsm', '.xlsb', '.odf', '.ods', '.odt'))):
            _print(f'> Importing "{filename}"...', color='GREEN')
            data = pd.read_excel(filedir)
        if(data is None):
            raise Exception
    except Exception as e:
        _print('Unable to retrieve local data', color='RED')
        raise e
    return data.infer_objects()


def save_local_data_file(data, filepath, **kwargs):
    data.to_csv(filepath, index=kwargs.get('index'))
    _print(f'> Saved data to {filepath}')
