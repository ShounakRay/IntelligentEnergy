# @Author: Shounak Ray <Ray>
# @Date:   09-Mar-2021 11:03:94:947  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: IPC_Real_Modeling.py
# @Last modified by:   Ray
# @Last modified time: 09-Mar-2021 12:03:98:984  GMT-0700
# @License: [Private IP]

import subprocess

import h2o


def cmd_runprint(command: str, prnt_file: bool = True, prnt_scrn: bool = False, ret: bool = False):
    """Runs a command in Python script and uses remote_shell_output.txt or screen as stdout/console.

    Parameters
    ----------
    command : str
        The command to be executed.
    prnt_file : bool
        Whether output of command should be printed to remote_shell_output.txt
    prnt_scrn : bool
        Whether output of command should be printed to screen.
    ret : bool
        Whether output of command should be returned.

    Returns [Optional]
    -------
    None
        Nothing is returned if ret is False
    str
        The output of the command is returned if ret is True

    """
    exec_output = subprocess.Popen(command,
                                   shell=True,
                                   stdout=subprocess.PIPE).stdout.read().decode("utf-8")
    if(prnt_file):
        print(exec_output, file=open('remote_shell_output.txt', 'w'))
    if(prnt_scrn):
        print(exec_output)
    if(ret):
        return exec_output


# Convert the provided h2o demo file to a python file
# cmd_runprint(command="jupyter nbconvert --to script 'H2O Testing/automl_regression_powerplant_output.ipynb'",
#              prnt_file=False, prnt_scrn=True)

# EOF

# EOF
