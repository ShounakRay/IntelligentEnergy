# @Author: Shounak Ray <Ray>
# @Date:   19-Mar-2021 10:03:97:973  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: base_generation.py
# @Last modified by:   Ray
# @Last modified time: 19-Mar-2021 11:03:56:560  GMT-0600
# @License: [Private IP]


import os

import pandas as pd
from matplotlib import pyplot as plt

data_dir = 'Data/Isolated/'
fiber_dir = 'Data/DTS/'

injectors = pd.read_excel(data_dir + "OLT injection data.xlsx")
producers = pd.read_excel(data_dir + "OLT production data (rev 1).xlsx")
well_test = pd.read_excel(data_dir + "OLT well test data.xlsx")
​
# well_test.columns
well_test = well_test[['Start Date/Time (yyyy/mm/dd HH:mm)', 'Pad', 'Well', '24 Hour Oil (m3)', '24 Hour Water (m3)',
                       'Pump Speed (SPM)', 'Pump Size in', 'Pump Efficiency (%)', 'Operator Approved',
                       'Engineering Approved']]
well_test.columns = ['date', 'pad', 'producer_well', 'test_oil',
                     'test_water', 'test_spm',
                     'pump_size', 'pump_efficiency', 'op_approved',
                     'eng_approved']
well_test['date'] = well_test['date'].astype(str).str.split(" ").str[0]
​


def combine_data(producer):
    filedir = fiber_dir + producer + "/"
    files = os.listdir(filedir)
    combined = []
    for f in files:
        print(f)
        try:
            fiber_data = pd.read_csv(filedir + f, error_bad_lines=False).T.iloc[1:]
        except Exception:
            fiber_data = pd.read_excel(filedir + f).T.iloc[1:]
        combined.append(fiber_data)
    return pd.concat(combined)


def get_fiber_data(producer, bins=5):
    combined = combine_data(producer)
    combined = combined.iloc[:, 9:]
    combined = combined.apply(pd.to_numeric)
    combined = combined.reset_index()
    combined = combined.sort_values('index')
    combined['index'] = combined['index'].str.split(" ").str[0]

    max_length = int(max(combined.columns[1:]))
    segment_length = int(max_length / bins)

    condensed = []
    for i, s in enumerate(range(0, max_length + 1, segment_length)):
        condensed.append(pd.DataFrame(
            {'bin_' + str(i + 1): list(combined.iloc[:, s:s + segment_length].mean(axis=1))}))

    condensed = pd.concat(condensed, axis=1)
    condensed['date'] = combined['index']
    condensed = condensed.sort_values('date').set_index('date')

    return condensed


producer_wells = [p for p in os.listdir(fiber_dir) if p[0] != '.']
​
aggregated_fiber = []
for producer in producer_wells[1:]:
    condensed = get_fiber_data(producer)
    condensed['producer_well'] = producer
    aggregated_fiber.append(condensed)
​
aggregated_fiber = pd.concat(aggregated_fiber)
aggregated_fiber = aggregated_fiber.dropna(how='all', axis=1)
​
injectors.columns = ['date', 'pad', 'injector_well', 'uwi', 'hours_on_inj', 'alloc_steam', 'steam',
                     'inj_bhp', 'inj_tubing_pressure', 'Reason', 'Comment']
inj_cols = ['date', 'injector_well', 'steam', 'inj_bhp', 'inj_tubing_pressure']
​
producers.columns = ['date', 'pad', 'producer_well', 'uwi', 'hours_on_prod',
                     'Downtime Reason Code', 'oil', 'water',
                     'gas', 'Allocated Injector Steam (m3)',
                     'Allocated Total Steam To Producer (m3)', 'Metered Steam (m3/hr)',
                     'Metered Steam (m3/day)', 'spm', 'prod_tubing_pressure',
                     'prod_casing_pressure', 'prod_bhp_heel', 'prod_bhp_toe',
                     'subcool_heel', 'subcool_toe', 'last_test_date',
                     'Reason Code', 'Comment']
​
prod_cols = ['date', 'uwi', 'producer_well', 'spm', 'hours_on_prod',
             'prod_casing_pressure', 'prod_bhp_heel', 'prod_bhp_toe', 'oil', 'water']
​
​
injector_table = pd.pivot_table(injectors, values='steam', index='date', columns='injector_well').reset_index()
injector_table['date'] = pd.to_datetime(injector_table['date']).astype(str)
​
producers['date'] = pd.to_datetime(producers['date']).astype(str)
​
aggregated_fiber = aggregated_fiber.reset_index()
aggregated_fiber['date'] = pd.to_datetime(aggregated_fiber['date']).astype(str)
​
​
df = pd.merge(producers[prod_cols], aggregated_fiber, how='outer', on=['date', 'producer_well'])
df = pd.merge(df, injector_table, how='outer', on=['date'])
df = pd.merge(df, well_test, how='left', on=['date', 'producer_well'])
​
df.drop(['op_approved', 'eng_approved', 'uwi'], 1, inplace=True)

df.to_csv('combined_ipc.csv', index=False)
list(df.columns)
