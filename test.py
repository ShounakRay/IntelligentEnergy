# @Author: Shounak Ray <Ray>
# @Date:   26-Apr-2021 10:04:89:899  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: test.py
# @Last modified by:   Ray
# @Last modified time: 26-Apr-2021 11:04:87:876  GMT-0600
# @License: [Private IP]


import os

import pandas as pd

pd.read_csv('Modeling Reference Files/1360 – ENG: False, WEIGHT: True, TIME: 10/MODELS_1360.csv')

directory = 'Modeling Reference Files'
[x[0] for x in os.walk(directory)]
