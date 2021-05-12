# @Author: Shounak Ray <Ray>
# @Date:   12-May-2021 11:05:04:041  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: minor_fix.py
# @Last modified by:   Ray
# @Last modified time: 12-May-2021 15:05:72:725  GMT-0600
# @License: [Private IP]
import pandas as pd

data = pd.read_csv('Data/S6 Files/Right_Aggregates_2015-04-01_2020-12-20.csv')
data = data[data['PRO_Pad'] == 'A'].sort_values('Date').reset_index(drop=True)

data[data['accuracy'] > 0].plot(x='Date', y='accuracy')
data.dtypes
