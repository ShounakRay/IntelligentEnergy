# @Author: Shounak Ray <Ray>
# @Date:   16-Apr-2021 11:04:04:041  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: acessory.py
# @Last modified by:   Ray
# @Last modified time: 16-Apr-2021 13:04:87:870  GMT-0600
# @License: [Private IP]

import functools
import inspect


def get_default_args(func):
    signature = inspect.signature(func)
    default_args = {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}
    return default_args


def wrapper(func):
    @functools.wraps(func)
    def wrap(self, *args, **kwargs):
        print("inside wrap")
        return func(self, *args, **kwargs)
    return wrap
