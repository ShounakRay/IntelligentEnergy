# @Author: Shounak Ray <Ray>
# @Date:   20-Feb-2021 23:02:81:814  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: extracontent.py
# @Last modified by:   Ray
# @Last modified time: 09-Mar-2021 13:03:04:044  GMT-0700
# @License: [Private IP]

# DATA_INJECTION_STEAM.set_index('Date')[well].plot(figsize=(24, 8))
# Counter(DATA_INJECTION_STEAM_NANBENCH['OUTCOME'])

# sensor_pivoted.dropna()
# Counter(DATA_PRODUCTION[['Date', col, 'Well']]['Well']) == Counter(sensor_pivoted['Well'])
#
# DATA_PRODUCTION[['Date', col, 'Well']]
# df_diff = pd.concat([DATA_PRODUCTION[['Date', col, 'Well']],
#                      sensor_pivoted]).drop_duplicates(keep=False)
#
# len(set(DATA_PRODUCTION[['Date', col, 'Well']]['Well']))
# len(set(sensor_pivoted['Well']))

# _ = plt.hist(FIBER_DATA['Bin_1'], bins=1000)
# _ = plt.hist(FIBER_DATA['Bin_2'], bins=1000)
# _ = plt.hist(FIBER_DATA['Bin_3'], bins=1000)
# _ = plt.hist(FIBER_DATA['Bin_4'], bins=1000)
# _ = plt.hist(FIBER_DATA['Bin_5'], bins=1000)

# Convert the provided h2o demo file to a python file
# cmd_runprint(command="jupyter nbconvert --to script 'H2O Testing/automl_regression_powerplant_output.ipynb'",
#              prnt_file=False, prnt_scrn=True)
