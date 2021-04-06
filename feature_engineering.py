# @Author: Shounak Ray <Ray>
# @Date:   30-Mar-2021 11:03:53:534  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: feature_engineering.py
# @Last modified by:   Ray
# @Last modified time: 06-Apr-2021 15:04:63:638  GMT-0600
# @License: [Private IP]

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("Data/combined_ipc.csv")
df = df.sort_values('Date').reset_index(drop=True)

df['PRO_Total_Fluid'] = df['PRO_Alloc_Oil'] + df['PRO_Alloc_Water']
df['PRO_Fluid'] = df['PRO_Oil'] + df['PRO_Water']
df['PRO_Alloc_Water_Cut'] = df['PRO_Alloc_Water'] / df['PRO_Total_Fluid']
df['PRO_Water_cut'] = df['PRO_Water'] / df['PRO_Fluid']
df['PRO_Adj_Pump_Speed'] = (df['PRO_Pump_Speed'] * df['PRO_Time_On']) / 24.0
# df = df[df['PRO_Oil'] > 0][['Date', 'PRO_Well', 'PRO_Alloc_Oil', 'PRO_Alloc_Water', 'PRO_Total_Fluid',
#                             'PRO_Alloc_Water_Cut', 'PRO_Oil', 'PRO_Water', 'PRO_Fluid', 'PRO_Water_cut',
#                             'PRO_Pump_Speed', 'PRO_Pump_Efficiency']].dropna()

injectors = [c for c in df.columns if 'I' in c][1:]
# fiber_segments = [c for c in df.columns if 'Bin' in c]

theoretical_df = []
for producer in df['PRO_Well'].dropna().unique():
    producer_df = df[df['PRO_Well'] == producer].reset_index(drop=True)
    producer_df.loc[producer_df['PRO_Engineering_Approved'] == False, 'PRO_Water'] = np.nan
    producer_df = producer_df.sort_values('Date').interpolate('linear')
    producer_df['PRO_Theo_Fluid'] = (producer_df['PRO_Fluid'] / producer_df['PRO_Pump_Efficiency']
                                     ) * (100 / producer_df['PRO_Adj_Pump_Speed'])
    producer_df = producer_df.replace([np.inf, -np.inf], np.nan)
    producer_df['PRO_Theo_Fluid'] = producer_df['PRO_Theo_Fluid'].dropna().median() * \
        producer_df['PRO_Adj_Pump_Speed']
    theoretical_df.append(producer_df)


theoretical_df = pd.concat(theoretical_df).reset_index(drop=True)
theoretical_df.loc[theoretical_df['PRO_Total_Fluid'] < 0, 'PRO_Total_Fluid'] = 0
combined = theoretical_df.groupby(['Date']).sum().reset_index()
combined['PRO_Alloc_Factor'] = combined['PRO_Theo_Fluid'] / combined['PRO_Total_Fluid']

theoretical_df = pd.merge(theoretical_df, combined[['Date', 'PRO_Alloc_Factor']], how='left', on=['Date'])
theoretical_df['adj_PRO_Theo_Fluid'] = theoretical_df['PRO_Theo_Fluid'] / theoretical_df['PRO_Alloc_Factor']
theoretical_df['PRO_Adj_Alloc_Oil'] = theoretical_df['adj_PRO_Theo_Fluid'] * (1 - theoretical_df['PRO_Water_cut'])
theoretical_df['PRO_Adj_Alloc_Water'] = theoretical_df['adj_PRO_Theo_Fluid'] * theoretical_df['PRO_Water_cut']
theoretical_df['PRO_Adj_Pump_Efficiency'] = theoretical_df['adj_PRO_Theo_Fluid'] / \
    theoretical_df['PRO_Adj_Pump_Speed'] * 10
theoretical_df['Field_Steam'] = theoretical_df[injectors].sum(axis=1)

theoretical_df.drop(['PRO_Pump_Efficiency', 'PRO_Engineering_Approved', 'PRO_Total_Fluid', 'PRO_Alloc_Water_Cut',
                     'PRO_Water_cut', 'PRO_Theo_Fluid', 'PRO_Alloc_Factor', 'adj_PRO_Theo_Fluid',
                     'PRO_Adj_Alloc_Water', 'PRO_Adj_Pump_Efficiency', 'Field_Steam', 'PRO_Pump_Speed', 'PRO_UWI'],
                    axis=1, inplace=True)

theoretical_df.to_csv('Data/combined_ipc_engineered.csv')
