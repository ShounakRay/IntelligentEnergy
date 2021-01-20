# Purpose:
#   to mimic IPC testing data where they see production levels for each well every few weeks
#   understand what features are the most important predicting output steam (daily steam)


from stringcase import snakecase
import pandas as pd
import numpy as np
# from Anomaly_Detection_PKG import reset_df_index
import datetime
from datetime import timedelta

# Imports
__FOLDER__ = r'/Users/Ray/Documents/Python/9 - Oil and Gas/Husky'
PATH_SANDALL = __FOLDER__ + r'/Data/Sandall/sandall.csv'
sandall_data = pd.read_csv(PATH_SANDALL)

# Data Processing
sandall_data = reset_df_index(sandall_data.dropna(inplace = False))
ALL_FEATURES = [
 'production_date',
 'pad_name',
 'pair_name',
 'dly_stm',
 'oil_sales',
 'water_sales',
 'gas_sales',
]
sandall_data = sandall_data[ALL_FEATURES]
sandall_data['production_date'] = pd.to_datetime(sandall_data['production_date'])
sandall_data['production_date'] = sandall_data['production_date'].apply(lambda x: x.date())
# filter df by date range to most closely emulate IPC data
sandall_data = reset_df_index(sandall_data[(sandall_data['production_date'] >= datetime.date(2018, 9, 11)) & (sandall_data['production_date'] <= datetime.date(2020, 9, 19))].sort_values(by = 'production_date'))

# *****PIVOT*****
table = pd.pivot_table(sandall_data,
                        index = ['production_date', 'pad_name'],
                        values = ['dly_stm', 'oil_sales', 'water_sales', 'gas_sales'],
                        # columns = ['pair_name'],
                        aggfunc = {'dly_stm': np.sum,
                                   'oil_sales': np.sum,
                                   'water_sales': np.sum,
                                   'gas_sales': np.sum},
                        dropna = True)
# You'd look through every day

# Extraneous code
{
    # pads= list(sandall_data['pad_name'].unique())
    # pairs = list(sandall_data['pair_name'].unique())
    # dates = list(sandall_data['production_date'].unique())

    # for pad in pads:
    #     filt = reset_df_index(sandall_data[sandall_data['pad_name'] == pad])
    #     for date in dates:
    #         filt = reset_df_index(filt[filt['production_date'] == date])
    # for pair in pairs:
    #     filt = reset_df_index(sandall_data[sandall_data['pair_name'] == pair])
    #     filt['production_date'] = pd.to_datetime(filt['production_date'])
    #     ### To replace date with count from minimum
    #     # min_date = min(filt['production_date'])
    #     # filt['production_date'] = filt['production_date'].apply(lambda x: (x - min_date).days + 1).tolist()
    #     # sandall_data[sandall_data['pair_name'] == pair]['production_date'] = filt['production_date']

}

# Next Steps
# Perform ft. selection, regression + anomaly detection (?) there
https://scikit-learn.org/stable/modules/feature_selection.html
Regression Algorithm to predict dly_stm dynamically




#
