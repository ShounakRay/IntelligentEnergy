# @Author: Shounak Ray <Ray>
# @Date:   30-Mar-2021 11:03:53:534  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: feature_engineering.py
# @Last modified by:   Ray
# @Last modified time: 30-Mar-2021 14:03:29:291  GMT-0600
# @License: [Private IP]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("Data/combined_ipc.csv")
df = df.sort_values('Date')

df['total_fluid'] = df['PRO_Alloc_Oil'] + df['PRO_Alloc_Water']
df['PRO_Fluid'] = df['PRO_Oil'] + df['PRO_Water']
df['water_cut'] = df['PRO_Water'] / df['total_fluid']
df['PRO_Water_cut'] = df['PRO_Water'] / df['PRO_Fluid']
df[df['PRO_Oil'] > 0][['Date', 'PRO_Well', 'PRO_Alloc_Oil', 'PRO_Alloc_Water', 'total_fluid', 'water_cut',
                       'PRO_Oil', 'PRO_Water', 'PRO_Fluid', 'PRO_Water_cut', 'PRO_Pump_Speed', 'pump_efficiency']].dropna()

injectors = [c for c in df.columns if c[0] == 'I']
fiber_segments = [c for c in df.columns if c[0:3] == 'bin']

df.groupby('Date')['oil'].sum().plot()
df.groupby('Date')['PRO_Alloc_Water'].sum().plot()
df.groupby('Date')['total_fluid'].sum().plot()
df.groupby('Date')['PRO_Pump_Speed'].sum().plot()

theoretical_df = []
for producer in df['PRO_Well'].dropna().unique():
    producer_df = df[df['PRO_Well'] == producer]
    # producer_df.loc[producer_df['eng_approved'] == False, 'PRO_Water'] = np.nan
    # producer_df.loc[producer_df['eng_approved'] == False, 'PRO_Oil'] = np.nan
    # producer_df.loc[producer_df['eng_approved'] == False, 'PRO_Fluid'] = np.nan
    # producer_df.loc[producer_df['eng_approved'] == False, 'PRO_Water_cut'] = np.nan
    # producer_df.loc[producer_df['eng_approved'] == False, 'pump_efficiency'] = np.nan
    producer_df.loc[producer_df['eng_approved'] == False, 'PRO_Water'] = np.nan
    producer_df = producer_df.sort_values('Date').interpolate('linear')
    producer_df['theoretical_fluid'] = producer_df['PRO_Fluid'] / \
        producer_df['pump_efficiency'] * 100 / producer_df['PRO_Pump_Speed']
    producer_df = producer_df.replace([np.inf, -np.inf], np.nan)
    producer_df['theoretical_fluid'] = producer_df['theoretical_fluid'].dropna().median() * \
        producer_df['PRO_Pump_Speed']
    theoretical_df.append(producer_df)
    # producer_df['pump_efficiency'].plot()
    # producer_df['PRO_Pump_Speed'].plot()
    # producer_df['theoretical_fluid'].plot()
    # producer_df['total_fluid'].plot()

theoretical_df = pd.concat(theoretical_df)
theoretical_df.loc[theoretical_df['total_fluid'] < 0, 'total_fluid'] = 0
combined = theoretical_df.groupby(['Date']).sum().reset_index()
combined['alloc_factor'] = combined['theoretical_fluid'] / combined['total_fluid']

theoretical_df = pd.merge(theoretical_df, combined[['Date', 'alloc_factor']], how='left', on=['Date'])
theoretical_df['adj_theoretical_fluid'] = theoretical_df['theoretical_fluid'] / theoretical_df['alloc_factor']
theoretical_df['adj_oil'] = theoretical_df['adj_theoretical_fluid'] * (1 - theoretical_df['PRO_Water_cut'])
theoretical_df['adj_water'] = theoretical_df['adj_theoretical_fluid'] * theoretical_df['PRO_Water_cut']
theoretical_df['prod_eff'] = theoretical_df['adj_theoretical_fluid'] / theoretical_df['PRO_Pump_Speed'] * 10
theoretical_df['field_steam'] = theoretical_df[injectors].sum(axis=1)
