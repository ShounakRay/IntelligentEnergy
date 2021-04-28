# @Author: Shounak Ray <Ray>
# @Date:   27-Apr-2021 09:04:39:394  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: test.py
# @Last modified by:   Ray
# @Last modified time: 27-Apr-2021 22:04:43:435  GMT-0600
# @License: [Private IP]


import _references._accessories as _accessories
import matplotlib.pyplot as plt
import pandas as pd

data_1 = pd.read_csv('Data/combined_ipc_engineered_phys_ALL.csv')
fig, ax = plt.subplots(figsize=(20, 20))
_ = data_1.groupby('PRO_Well').plot(x='Date', y='PRO_Water_cut',
                                    ax=ax, kind='line')

PATH_PAD = 'Data/combined_ipc_aggregates_ALL.csv'
_accessories.retrieve_local_data_file(PATH_PAD)
