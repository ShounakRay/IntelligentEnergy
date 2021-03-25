# @Author: Shounak Ray <Ray>
# @Date:   20-Feb-2021 23:02:81:814  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: extracontent.py
# @Last modified by:   Ray
# @Last modified time: 24-Mar-2021 17:03:51:512  GMT-0600
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

# variable_importance.melt(id_vars=['variable'],
#                          value_vars='scaled_importance',
#                          var_name='injector',
#                          value_name='importance')

# plt.figure(figsize=(12, 8))
# plt.plot(history['timestamp'], history['training_deviance'])[0]

# # Run the experiment
# # NOTE: Fold column specified for cross validation to mitigate leakage
# # https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/_modules/h2o/automl/autoh2o.html
# aml_obj.train(x=PREDICTORS,                                     # All the depedent variables in each model
#               y=RESPONDERS[0],                                  # A single responder
#               fold_column=FOLD_COLUMN,                          # Fold column name, as specified from encoding
#               training_frame=data)                              # All the data is used for training, cross-validation
#
# # View models leaderboard and extract desired model
# exp_leaderboard = aml_obj.leaderboard
# exp_leaderboard.head(rows=exp_leaderboard.nrows)
# specific_model = h2o.get_model(exp_leaderboard[0, "model_id"])

# Last login: Tue Mar  9 11:13:07 on ttys000
# Ray@Whites-MacBook-Pro ~ % sudo xcodebuild -license
# Password:
# xcode-select: error: tool 'xcodebuild' requires Xcode, but active developer directory '/Library/Developer/CommandLineTools' is a command line tools instance
# Ray@Whites-MacBook-Pro ~ % brew install openssl
# Updating Homebrew...
# ==> Auto-updated Homebrew!
# Updated 3 taps (homebrew/core, homebrew/cask and caskroom/cask).
# ==> New Formulae
# cyrus-sasl                gopass-jsonapi            python-tabulate           qt-mariadb                qt-percona-server         qt-unixodbc               xray
# enzyme                    klee                      qt-libiodbc               qt-mysql                  qt-postgresql             wllvm
# ==> Updated Formulae
# Updated 143 formulae.
# ==> New Casks
# veepn                                                                                       veepn
# ==> Updated Casks
# Updated 106 casks.
# xcrun: error: invalid active developer path (/Library/Developer/CommandLineTools), missing xcrun at: /Library/Developer/CommandLineTools/usr/bin/xcrun
#
# Warning: openssl@1.1 1.1.1j is already installed and up-to-date.
# To reinstall 1.1.1j, run:
#   brew reinstall openssl@1.1
# Ray@Whites-MacBook-Pro ~ % brew install git
# ==> Downloading https://homebrew.bintray.com/bottles/git-2.30.2.big_sur.bottle.tar.gz
# ==> Downloading from https://d29vzk4ow07wi7.cloudfront.net/080017fabfad8c0047cbb2149b09a81e8a66f15253c58cd0f931f5dca7c4cb69?response-content-disposition=attachment%3Bfilename%3D%22git
# ######################################################################## 100.0%
# Error: git 2.30.0 is already installed
# To upgrade to 2.30.2, run:
#   brew upgrade git
# Ray@Whites-MacBook-Pro ~ % brew upgrade git
# ==> Upgrading 1 outdated package:
# git 2.30.0 -> 2.30.2
# ==> Upgrading git 2.30.0 -> 2.30.2
# ==> Downloading https://homebrew.bintray.com/bottles/git-2.30.2.big_sur.bottle.tar.gz
# Already downloaded: /Users/Ray/Library/Caches/Homebrew/downloads/6a802c1b3f70940248f087769f006ab3f8ac2bb4a3ce807c9ca02a9a10bc28ef--git-2.30.2.big_sur.bottle.tar.gz
# ==> Pouring git-2.30.2.big_sur.bottle.tar.gz
# ==> Caveats
# The Tcl/Tk GUIs (e.g. gitk, git-gui) are now in the `git-gui` formula.
#
# zsh completions and functions have been installed to:
#   /usr/local/share/zsh/site-functions
#
# Emacs Lisp files have been installed to:
#   /usr/local/share/emacs/site-lisp/git
# ==> Summary
# 🍺  /usr/local/Cellar/git/2.30.2: 1,501 files, 40.5MB
# Removing: /usr/local/Cellar/git/2.30.0... (1,486 files, 40.5MB)
# Ray@Whites-MacBook-Pro ~ % xed .
# xcode-select: error: tool 'xed' requires Xcode, but active developer directory '/Library/Developer/CommandLineTools' is a command line tools instance
# Ray@Whites-MacBook-Pro ~ % xcode-select
# xcode-select: error: no command option given
# Usage: xcode-select [options]
#
# Print or change the path to the active developer directory. This directory
# controls which tools are used for the Xcode command line tools (for example,
# xcodebuild) as well as the BSD development commands (such as cc and make).
#
# Options:
#   -h, --help                  print this help message and exit
#   -p, --print-path            print the path of the active developer directory
#   -s <path>, --switch <path>  set the path for the active developer directory
#   --install                   open a dialog for installation of the command line developer tools
#   -v, --version               print the xcode-select version
#   -r, --reset                 reset to the default command line tools path
# Ray@Whites-MacBook-Pro ~ % xcode-select --install
# xcode-select: note: install requested for command line developer tools
# Ray@Whites-MacBook-Pro ~ % sudo xcode-select --switch /Library/Developer/CommandLineTools
# Password:
# Ray@Whites-MacBook-Pro ~ % brew install git
# Updating Homebrew...
# Warning: git 2.30.2 is already installed and up-to-date.
# To reinstall 2.30.2, run:
#   brew reinstall git
# Ray@Whites-MacBook-Pro ~ % brew reinstall git
# ==> Downloading https://homebrew.bintray.com/bottles/git-2.30.2.big_sur.bottle.tar.gz
# Already downloaded: /Users/Ray/Library/Caches/Homebrew/downloads/6a802c1b3f70940248f087769f006ab3f8ac2bb4a3ce807c9ca02a9a10bc28ef--git-2.30.2.big_sur.bottle.tar.gz
# ==> Reinstalling git
# ==> Pouring git-2.30.2.big_sur.bottle.tar.gz
# ==> Caveats
# The Tcl/Tk GUIs (e.g. gitk, git-gui) are now in the `git-gui` formula.
#
# zsh completions and functions have been installed to:
#   /usr/local/share/zsh/site-functions
#
# Emacs Lisp files have been installed to:
#   /usr/local/share/emacs/site-lisp/git
# ==> Summary
# 🍺  /usr/local/Cellar/git/2.30.2: 1,501 files, 40.5MB
# Ray@Whites-MacBook-Pro ~ %

# fig, ax = plt.subplots(ncols=len(key_features), sharey=False, figsize=(20, 20))
# for feat in key_features:
#     final_data = correlated_df.abs()[key_features].dropna().drop(key_features[:6]).sort_values(feat)
#     pos = [list(final_data.index).index(val) for val in list(filtered_features)]
#
#     hmap = sns.heatmap(final_data, annot=False, ax=ax[key_features.index(feat)])
#     for p in pos:
#         hmap.add_patch(Rectangle((0, p), len(key_features), 1, edgecolor='blue', fill=False, lw=3))
#     hmap.set_title('Ordered by ' + feat + ' for ' + well)
#     plt.tight_layout()
# hmap.get_figure().savefig('WELL-{WELL}_TARGET-{TARGET}.pdf'.format(WELL=well, TARGET=TARGET), bbox_inches='tight')

# msk = np.random.rand(len(SOURCE)) < train_pct
#
# # Variable Selection (NOT Engineering)
# new_X = model_featsel.fit_transform(SOURCE[[c for c in SOURCE.columns if c != TARGET]],
#                                     SOURCE[TARGET])
# filtered_features = new_X.columns


# # >>> THIS IS THE REGRESSION PART
# # Length filtering, no column filtering
# # Filter training and testing sets to only include the selected features
# TRAIN = SOURCE[msk][filtered_features.union([TARGET])].reset_index(drop=True)
# TEST = SOURCE[~msk][filtered_features.union([TARGET])].reset_index(drop=True)
#
# X_TRAIN = TRAIN[[c for c in TRAIN.columns if c != TARGET]]
# Y_TRAIN = pd.DataFrame(TRAIN[TARGET])
#
# X_TEST = TEST[[c for c in TEST.columns if c != TARGET]]
# Y_TEST = pd.DataFrame(TEST[TARGET])
#
# feateng_X_TRAIN = model_feateng.fit_transform(X_TRAIN, Y_TRAIN)
# feating_X_TEST = model_feateng.transform(X_TEST)
#
# # model_feateng.score(X_TRAIN, Y_TRAIN)
# # model_feateng.new_feat_cols_
# # plt.scatter(model_feateng.predict(X_TEST), Y_TEST[TARGET], s=2)
# # model_feateng.score(X_TEST, Y_TEST)
#
# # performance(Y_TRAIN, model_feateng.predict(feateng_X_TRAIN))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MDS
# node_order = sorted(G_dupl.nodes())
# similarities = nx.adjacency_matrix(G_dupl, nodelist=node_order, weight='value').toarray()
#
# mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=1,
#                    dissimilarity="precomputed", n_jobs=1)
# pos = mds.fit(similarities).embedding_
# plt.scatter(pos[:, 0], pos[:, 1])


# plt.plot(SOURCE['I08'])
#
# plt.plot(SOURCE['I08'] / SOURCE['I08'].quantile(0.92))
# plt.plot(SOURCE['I30'] / SOURCE['I30'].quantile(0.92))
# plt.plot(SOURCE['I10'] / SOURCE['I10'].quantile(0.92))
# plt.plot(SOURCE['I43'] / SOURCE['I43'].quantile(0.92))
# plt.plot(SOURCE['I29'] / SOURCE['I29'].quantile(0.92))
# plt.plot(SOURCE['I56'] / SOURCE['I56'].quantile(0.92))
#
# plt.plot(SOURCE['I32'] / SOURCE['I32'].quantile(0.92))
# SOURCE.corr()
# plt.plot(SOURCE['I29'] / SOURCE['I29'].quantile(0.92))
#
# plt.plot(SOURCE['I10'] / SOURCE['I10'].quantile(0.92))
# plt.plot(SOURCE[TARG_duplET])

# _ = plt.hist(Y_TRAIN, bins=90)
# _ = plt.hist(model_feateng.predict(feateng_X_TRAIN), bins=90)

# Diverging: diverging_palette_Global
# https://seaborn.pydata.org/generated/seaborn.color_palette.html

# _ = correlation_matrix(SOURCE, 'Modeling.pdf', 'test', mask=False)


# dict(SOURCE.corr()['Heel_Pressure'].dropna().sort_values(ascending=False))

# model_metrics = performance(Y_TRAIN, model_feateng.predict(feateng_X_TRAIN))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#
# X, y = load_boston(True)
# pd.DataFrame(X)
#
# afreg = AutoFeatRegressor(verbose=1, feateng_steps=2)
# # fit autofeat on less data, otherwise ridge reg model_feateng with xval will overfit on new features
# X_train_tr = afreg.fit_transform(X[:480], y[:480])
# X_test_tr = afreg.transform(X[480:])
# print("autofeat new features:", len(afreg.new_feat_cols_))
# print("autofeat MSE on training data:", mean_squared_error(pd.DataFrame(y[:480]), afreg.predict(X_train_tr)))
# print("autofeat MSE on test data:", mean_squared_error(y[480:], afreg.predict(X_test_tr)))
# print("autofeat R^2 on training data:", r2_score(y[:480], afreg.predict(X_train_tr)))
# print("autofeat R^2 on test data:", r2_score(y[480:], afreg.predict(X_test_tr)))
#
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# es = ft.EntitySet(id='ipc_entityset')
#
# es = es.entity_from_dataframe(entity_id='FINALE', dataframe=FINALE,
#                               index='unique_id', time_index='Date')
#
# # Default primitives from featuretools
# default_agg_primitives = ft.list_primitives()[(ft.list_primitives()['type'] == 'aggregation') &
#                                               (ft.list_primitives()['valid_inputs'] == 'Numeric')
#                                               ]['name'].to_list()
# default_trans_primitives = [op for op in ft.list_primitives()[(ft.list_primitives()['type'] == 'transform') &
#                                                               (ft.list_primitives()['valid_inputs'] == 'Numeric')
#                                                               ]['name'].to_list()[:2]
#                             if op not in ['scalar_subtract_numeric_feature']]
#
# # DFS with specified primitives
# feature_names = ft.dfs(entityset=es, target_entity='FINALE',
#                        trans_primitives=default_trans_primitives,
#                        agg_primitives=None,
#                        max_depth=2,
#                        # n_jobs=-1,
#                        features_only=True)
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# # Create Entity Set for this project
# es = ft.EntitySet(id='ipc_entityset')
#
# FIBER_DATA.reset_index(inplace=True)
# DATA_INJECTION_STEAM.reset_index(inplace=True)
# DATA_INJECTION_PRESS.reset_index(inplace=True)
# DATA_PRODUCTION.reset_index(inplace=True)
# DATA_TEST.reset_index(inplace=True)
#
# # Create an entity from the client dataframe
# # This dataframe already has an index and a time index
# es = es.entity_from_dataframe(entity_id='DATA_INJECTION_STEAM', dataframe=DATA_INJECTION_STEAM.copy(),
#                               make_index=True, index='id_injector', time_index='Date')
# es = es.entity_from_dataframe(entity_id='DATA_INJECTION_PRESS', dataframe=DATA_INJECTION_PRESS.copy(),
#                               make_index=True, index='id_injector', time_index='Date')
#
# es = es.entity_from_dataframe(entity_id='DATA_PRODUCTION', dataframe=DATA_PRODUCTION.copy(),
#                               make_index=True, index='id_production', time_index='Date')
# es = es.entity_from_dataframe(entity_id='DATA_TEST', dataframe=DATA_TEST.copy(),
#                               make_index=True, index='id_production', time_index='Date')
# es = es.entity_from_dataframe(entity_id='FIBER_DATA', dataframe=FIBER_DATA.copy(),
#                               make_index=True, index='id_production', time_index='Date')
#
# rel_producer = ft.Relationship(es['DATA_PRODUCTION']['id_production'],
#                                es['DATA_TEST']['id_production'])
# es.add_relationship(rel_producer)
#
# # Add relationships for all columns between injection steam and pressure data
# injector_sensor_cols = [c for c in DATA_INJECTION_STEAM.columns.copy() if c not in ['Date']]
# injector_sensor_cols = ['CI06']
# for common_col in injector_sensor_cols:
#     rel_injector = ft.Relationship(es['DATA_INJECTION_STEAM']['id_injector'],
#                                    es['DATA_INJECTION_PRESS']['id_injector'])
#     es = es.add_relationship(rel_injector)
#
# es['DATA_INJECTION_STEAM']
#
# # EOF
# # EOF

# Fills all nan with small value (for feature selection)
# FINALE = FINALE.fillna(0.000000001).replace(np.nan, 0.000000001)

# for c in FINALE.columns:
#     print(c, sum(FINALE[c].isna()))

# Ultimate Goal
# FINALE = FINALE[FINALE['test_flag'] == True].reset_index(drop=True)
# TARGET = 'Oil'

# FINALE_FILTERED = FINALE[[c for c in FINALE.columns if c not in ['Date', 'unique_id', 'Pad', 'test_flag',
# '24_Fluid',  '24_Oil', '24_Water', 'Oil', 'Water',
# 'Gas', 'Fluid', 'Toe_Pressure', 'Pump_Speed',
# 'Casing_Pressure', 'Tubing_Pressure', 'Heel_Temp',
# 'Toe_Temp', 'Bin_1', 'Bin_2', 'Bin_3', 'Bin_4',
# 'Bin_5']]].replace(0.0, 0.0000001)
# FINALE_FILTERED = FINALE[[c for c in FINALE.columns if c not in ['Date', 'unique_id', 'Pad', 'test_flag',
#                                                                  '24_Fluid',  '24_Oil', '24_Water', 'Oil', 'Water',
#                                                                  'Gas', 'Fluid', 'Toe_Pressure', 'Pump_Speed',
#                                                                  'Casing_Pressure', 'Tubing_Pressure', 'Heel_Temp',
#                                                                  'Toe_Temp', 'Bin_1', 'Bin_2', 'Bin_3', 'Bin_4',
#                                                                  'Bin_5']]
#                          ].replace(0.0, 0.0000001)

# SOURCE = FINALE_FILTERED[FINALE_FILTERED['Well'] == well]
# SOURCE.drop(['Well'], axis=1, inplace=True)
# SOURCE.reset_index(drop=True, inplace=True)
# SOURCE = SOURCE.rolling(window=7).mean().fillna(method='bfill').fillna(method='ffill')

# model_feateng = AutoFeatRegressor(feateng_steps=2, verbose=3)
# model_featsel = FeatureSelector(verbose=6)
# train_pct = 0.8

# FINALE_FILTERED_ULTIMATE = FINALE[[c for c in FINALE.columns if c not in ['Date', 'unique_id', 'Pad', 'test_flag',
#                                         '24_Fluid',  '24_Oil', '24_Water', 'Water',
#                                         'Gas', 'Fluid', 'Toe_Pressure', 'Pump_Speed',
#                                         'Casing_Pressure', 'Tubing_Pressure', 'Heel_Temp',
#                                         'Toe_Temp', 'Bin_1', 'Bin_2', 'Heel_Pressure',
#                                         'Bin_3', 'Bin_4', 'Bin_5']]
# ].replace(0.0, 0.0000001)

# [OPTIONAL] FINALE PRE-PROCESSING
# finale_replace = {'date': 'Date',
#                   'producer_well': 'Well',
#                   'prod_casing_pressure': 'Casing_pressure',
#                   'pad': 'Pad',
#                   'prod_bhp_heel': 'Heel_Pressure',
#                   'prod_bhp_toe': 'Toe_Pressure',
#                   'hours_on_prod': 'Time_On'}
# _ = [finale_replace.update({'bin_{}'.format(i): 'Bin_{}'.format(i)}) for i in range(1, 6)]
# _ = [finale_replace.update({col: col}) for col in FINALE.columns if col not in finale_replace.keys()]
# FINALE.drop(['op_approved', 'eng_approved', 'uwi'], 1, inplace=True)
# FINALE.columns = FINALE.columns.to_series().map(finale_replace)

# prowell = unique_pro_wells[0]

# figure, ax = plt.subplots(nrows=len(unique_pro_wells), figsize=(10, 60))
# for prowell in unique_pro_wells:
#     # FINALE_pro.loc[FINALE_pro['PRO_Well'] == prowell,
#     #                'PRO_Oil'] = FINALE_pro[FINALE_pro['PRO_Well'] == prowell]['PRO_Oil'].interpolate()
#     ax[unique_pro_wells.index(prowell)].plot(FINALE[FINALE['PRO_Well'] == prowell]['PRO_Oil'])
#     # plt.tight_layout()

# plt.plot(FINALE_pro.loc[FINALE_pro['PRO_Well'] == prowell, 'PRO_Oil'])
# plt.plot(FINALE_pro['PRO_Oil'])

# FINALE_agg = FINALE_pro.groupby(by=['Date', 'Pad'], axis=0, sort=False, as_index=False).sum()
# FINALE_melted_pro = pd.melt(FINALE, id_vars=['Date'], value_vars=all_pro_data, var_name='metric', value_name='PRO_Pad')
# FINALE_melted_pro['PRO_Pad'] = FINALE_melted_pro['PRO_Pad'].apply(lambda x: PRO_PAD_KEYS.get(x))

# NOTE: Save outputs for reference (so you don't have to wait an hour every time)
# with open('Modeling Pickles/model_novarimps.pkl', 'wb') as f:
#     pickle.dump(model_novarimps, f)
# final_cumulative_varimps_pad.to_pickle('Modeling Pickles/final_cumulative_varimps.pkl')
# final_cumulative_varimps_pad.to_html('Modeling Reference Files/final_cumulative_varimps.html')

# NOTE: FOR REFERENCE
# Filtering of the variable importance summary
# FILT_final_cumulative_varimps = final_cumulative_varimps[~final_cumulative_varimps['model_name'
#                                                                                    ].str.contains('XGBoost')
#                                                          ].reset_index(drop=True).select_dtypes(float)
