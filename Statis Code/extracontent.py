# @Author: Shounak Ray <Ray>
# @Date:   20-Feb-2021 23:02:81:814  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: extracontent.py
# @Last modified by:   Ray
# @Last modified time: 11-Jun-2021 16:06:32:323  GMT-0600
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

# # Exclude any features encoded by default (H2O puts a `.` in the column name of these features)
# final_cumulative_varimps = final_cumulative_varimps.loc[
#     :, ~final_cumulative_varimps.columns.str.contains('.',regex=False)]

# for pwell in all_prod_wells:
#     # FINALE.loc[FINALE['PRO_Well'] == pwell, 'weights__' + cont_col] = adjusted_anomscores
#     if(hit_an_injector):
#         # Populate the remaining injector columns for all other wells with this one
#         # for pwell_sub in [c for c in all_prod_wells if c is not pwell]:
#         #     FINALE.loc[FINALE['PRO_Well'] == pwell_sub, 'weights__' + cont_col] = adjusted_anomscores
#         break
# lc_injectors = [k for k, v in col_reference.items() if 'I' in v][1:]

# INJECTOR_AGGREGATES = {}
# for propad, pad_candidates in candidates_by_prodpad.items():
#     # Select candidates (not all the wells)
#     local_candidates = pad_candidates.copy()
#     absence = []
#     for cand in local_candidates:
#         if cand not in all_injs:
#             print('> STATUS: Candidate {} removed, unavailable in initial data'.format(cand))
#             absence.append(cand)
#     local_candidates = [el for el in local_candidates if el not in absence]
#
#     FINALE_melted_inj = pd.melt(FINALE_inj, id_vars=['Date'], value_vars=local_candidates,
#                                 var_name='Injector', value_name='Steam')
#     FINALE_melted_inj['INJ_Pad'] = FINALE_melted_inj['Injector'].apply(lambda x: INJ_PAD_KEYS.get(x))
#     FINALE_melted_inj = FINALE_melted_inj[~FINALE_melted_inj['INJ_Pad'].isna()].reset_index(drop=True)
#     # To groupby injector pads, by=['Date', 'INJ_Pad']
#     FINALE_agg_inj = FINALE_melted_inj.groupby(by=['Date'], axis=0, sort=False, as_index=False).sum()
#     FINALE_agg_inj['PRO_Pad'] = propad
#     INJECTOR_AGGREGATES[propad] = FINALE_agg_inj
# INJECTOR_AGGREGATES = pd.concat(INJECTOR_AGGREGATES.values()).reset_index(drop=True)

# INJECTOR EDA PLOTTING
# if(plot_eda):
#     # FIGURE PLOTTING (INJECTION PAD-LEVEL STATISTICS)
#     unique_inj_pads = list(FINALE_agg_inj['INJ_Pad'].unique())
#     fig, ax = plt.subplots(nrows=len(unique_inj_pads), figsize=(15, 80))
#     for pad in unique_inj_pads:
#         subp = ax[unique_inj_pads.index(pad)]
#         temp = FINALE_agg_inj[FINALE_agg_inj['INJ_Pad'] == pad].sort_values('Date').reset_index(drop=True)
#         d_1 = list(temp['Date'])[0]
#         d_n = list(temp['Date'])[-1]
#         subp.plot(temp['Steam'], label='Injector ' + pad + '\n{} > {}'.format(d_1, d_n))
#         subp.legend()
#         plt.tight_layout()
#
#     plt.savefig('inj_pads_ts.png')


# # SYMBOL INITIALIZATION
# for c in df.select_dtypes(np.number).columns:
#     exec(f'{c} = sympy.Symbol("{c}")')
#
# # SQRT/EXP SYMPY COMPATIBILITY
# for i in range(len(all_new_fts)):
#     all_new_fts[i] = all_new_fts[i].replace('sqrt', 'sympy.sqrt').replace('exp', 'sympy.exp')
#
# _ = eval(all_new_fts[0])
#
# sympy.init_printing(quiet=True, use_latex=True)
#
# print(_, file=open('test.txt', 'wb'))
#
# all_new_fts[0]
#
# # RENDERING
# equ =
# sympy.preview(all_new_fts[0], viewer='file', filename='output.png')
# sympy.utilities.misc.find_executable('latex')

# # PAD LEVEL AGGREGATION FOR INJECTION AGGREGATION
# if(plot_eda):
#     # FIGURE PLOTTING (PRODUCTION PAD-LEVEL STATISTICS)
#     master_rows = len(unique_pro_pads)
#     master_cols = len(FINALE_agg_pro.select_dtypes(float).columns)
#     fig, ax = plt.subplots(nrows=master_rows, ncols=master_cols, figsize=(200, 50))
#     for pad in unique_pro_pads:
#         temp_pad = FINALE_agg_pro[FINALE_agg_pro['PRO_Pad'] == pad].sort_values('Date').reset_index(drop=True)
#         d_1 = list(temp_pad['Date'])[0]
#         d_n = list(temp_pad['Date'])[-1]
#         numcols = FINALE_agg_pro.select_dtypes(float).columns
#         for col in numcols:
#             temp = temp_pad[[col]]
#             temp = temp.interpolate('linear')
#             # if all(temp.isna()):
#             #     temp = temp.fillna(0)
#             subp = ax[unique_pro_pads.index(pad)][list(numcols).index(col)]
#             subp.plot(temp[col], label='Producer ' + pad + ', Metric ' +
#                       col + '\n{} > {}'.format(d_1, d_n))
#             subp.legend()
#             plt.tight_layout()
#         plt.tight_layout()
#
#     plt.savefig('pro_pads_cols_ts.png')

# _ = """
# ####################################
# ########  WELL-LEVEL MERGING #######
# ####################################
# """
# PRODUCER_AGGREGATES_PWELL = FINALE_agg_pro_pwell[FINALE_agg_pro_pwell['PRO_Well'].isin(available_pwells_transformed)]
# COMBINED_AGGREGATES_PWELL = pd.merge(PRODUCER_AGGREGATES_PWELL, INJECTOR_AGGREGATES_PWELL,
#                                      how='inner', on=['Date', 'PRO_Well'])
# COMBINED_AGGREGATES_PWELL, dropped_pwell = drop_singles(COMBINED_AGGREGATES_PWELL)
# COMBINED_AGGREGATES_PWELL.infer_objects().to_csv('Data/combined_ipc_aggregates_PWELL.csv')

# _ = """
# ####################################
# ##  AGGREGATION EDA – WELL LEVEL ###
# ####################################
# """
# def plot_aggregation_eda(df, resp_feature_1, resp_feature_2, wells_iterator, pad_val):
#     fig, ax = plt.subplots(figsize=(50, 25), nrows=2, ncols=2)
#     ax[0][0].set_title(f'Aggregation breakdown of {resp_feature_1}')
#     for pwell in wells_iterator:
#         _temp = df[df['PRO_Well'] == pwell][resp_feature_1].reset_index(drop=True)
#         _temp.plot(ax=ax[0][0], linewidth=0.9)
#     _temp = COMBINED_AGGREGATES[COMBINED_AGGREGATES['PRO_Pad'] == pad_val][resp_feature_1]
#     (_temp / 1).plot(ax=ax[0][0], c='black')
#
#     ax[0][1].set_title(f'Histogram of {resp_feature_1}')
#     ax[0][1].hist(_temp, bins=200)
#
#     ax[1][0].set_title(f'Aggregation breakdown of {resp_feature_2}')
#     for pwell in wells_iterator:
#         _temp = df[df['PRO_Well'] == pwell][resp_feature_2].reset_index(drop=True)
#         _temp.plot(ax=ax[1][0], linewidth=0.9)
#     _temp = COMBINED_AGGREGATES[COMBINED_AGGREGATES['PRO_Pad'] == pad_val][resp_feature_2]
#     (_temp / 1).plot(ax=ax[1][0], c='black')
#
#     ax[1][1].set_title(f'Histogram of {resp_feature_2}')
#     ax[1][1].hist(_temp, bins=200)
# plot_aggregation_eda(COMBINED_AGGREGATES, 'PRO_Adj_Alloc_Oil', 'weight',
#                      available_pwells_transformed[:7], 'A')
#
# os.system('say finished plotting aggregation')

# def visualize_anomalies(ft):
#     fig, ax = plt.subplots(figsize=(12, 7))
#     ft[ft['anomaly'] == 'Yes'][['date', 'selection']].plot(
#         x='date', y='selection', kind='scatter', c='red', s=3, ax=ax)
#     ft[ft['anomaly'] == 'No'][['date', 'selection']].plot(x='date', y='selection', kind='line', ax=ax)
#     plt.show()

# aggregated_metrics = aggregated_metrics.groupby(['Group', 'Run_Tag'],
#                                                 group_keys=False).apply(lambda x:
#                                                                         x.sort_values(['Rel_Val_RMSE', 'RMSE'],
#                                                                                       ascending=True))

# # benchmarks_combined['Run_Time'] = benchmarks_combined['Run_Time'].astype(float)
# benchmarks_combined[(benchmarks_combined['Math_Eng'] == True) &
#                     (benchmarks_combined['Weighted'] == True)].plot(x='Run_Time', y='Rel_Val_RMSE',
#                                                                     ax=ax,
#                                                                     kind='scatter', label='Eng + Weight',
#                                                                     c='blue')
# benchmarks_combined[(benchmarks_combined['Math_Eng'] == True) &
#                     (benchmarks_combined['Weighted'] == False)].plot(x='Run_Time', y='Rel_Val_RMSE',
#                                                                      ax=ax,
#                                                                      kind='scatter', label='Eng',
#                                                                      c='purple')
# benchmarks_combined[(benchmarks_combined['Math_Eng'] == False) &
#                     (benchmarks_combined['Weighted'] == True)].plot(x='Run_Time', y='Rel_Val_RMSE',
#                                                                     ax=ax,
#                                                                     kind='scatter', label='Weight',
#                                                                     c='red')
# benchmarks_combined[(benchmarks_combined['Math_Eng'] == False) &
#                     (benchmarks_combined['Weighted'] == False)].plot(x='Run_Time', y='Rel_Val_RMSE',
#                                                                      ax=ax,
#                                                                      kind='scatter', label='Naive',
#                                                                      c='green')
# _ = plt.legend(loc='upper left')

# fig, ax = plt.subplots(nrows=len(all_positions.keys()), ncols=2, figsize=(15, 100))
# for well in all_positions.keys():
#     group_df = all_positions[well]  # /all_positions[well].max()
#     axes_1 = ax[list(all_positions.keys()).index(well)][0]
#     axes_1.plot(group_df['UTM_Easting'], group_df['UTM_Northing'])
#     axes_1.set_xlabel('UTM Easting')
#     axes_1.set_ylabel('UTM Northing')
#     axes_1.set_title(well + ' UTM Coordinates')
#     # axes_1.set_ylim(1000000 + 3.963 * 10**6, 1000000 + 5.963 * 10**6)
#
#     axes_2 = ax[list(all_positions.keys()).index(well)][1]
#     axes_2.plot(group_df['Local_Easting'], group_df['Local_Northing'])
#     axes_2.set_xlabel('Local Easting')
#     axes_2.set_ylabel('Local Northing')
#     axes_2.set_title(well + ' Local Coordinates')
#     # axes_2.set_ylim(0, 225)
# plt.tight_layout()
# fig.suptitle('Coordinates Bounded by Provided Liner Depths XLSX File')
# plt.savefig('Modeling Reference Files/Candidate Selection Images/provided_coordinate_plots.png',
#             bbox_inches='tight')

# NOTE: IPC_MODULE S8_INJECTION
# def macro_constrain_allocations(constraints, suggestions, data):
#     current_summed = suggestions.groupby('Candidate_Injector')['Candidate_Units'].sum()
#     delta_tracker = {}
#     for injector, bounds in constraints.items():
#         """FORCE INSIDE CONSTRAINTS"""
#         group_df = suggestions[suggestions['Candidate_Injector'] == injector]
#         minima, maxima = bounds
#         # Check if total steam is greater than constraints
#         current_value = current_summed[injector]
#         delta = 0
#
#         exceeds = True if current_value > maxima else Falsee
#         # > If so, multiply individual values by p: {0 < p < 1} for maxima to be reached
#         # >> Over * p = Maxima --> p = Maxima/Over
#         if exceeds:
#             p = maxima / current_value
#             suggestions.loc[group_df.index, 'Candidate_Units'] = suggestions.loc[group_df.index, 'Candidate_Units'] * p
#             delta = -1 * sum(suggestions.loc[group_df.index, 'Candidate_Units'] * (1 - p))
#
#         below = True if current_value < minima else False
#         # > If so, multiply individual values by p: {1 < p < ∞} for minima to be reached
#         # >> Below * p = Minima --> p = Minima/Below
#         if below:
#             p = minima / current_value
#             suggestions.loc[group_df.index, 'Candidate_Units'] = suggestions.loc[group_df.index, 'Candidate_Units'] * p
#             delta = sum(suggestions.loc[group_df.index, 'Candidate_Units'] * (p - 1))
#
#         delta_tracker[injector] = delta
#         print(f'Delta for {injector}: {delta}, Original: {current_value} --> ({minima}, {maxima})')
#
#         """ZERO-SUM FULFILLMENT"""
#         # if delta != 0.0:
#         #     # For current injector delta, find all connecting producer wells
#         #     # For each producer well, find candidate proportions for this specific injector > Normalize
#         #     pwell_connections = group_df.set_index('PRO_Well')['Candidate_Proportion']
#         #     pwell_connections = pwell_connections / sum(pwell_connections)
#         #     # Distribute injector delta proportionally to each producer well
#         #     pwell_distribution = pwell_connections * delta
#         #     # For each producer well and its allocated "extra," find other candidates that aren't above maxima > Normalize
#         #     for pwell, extra in pwell_distribution.items():
#         #         pwell_frame = suggestions[suggestions['PRO_Well'] == pwell]
#         #         link = pwell_frame.set_index('Candidate_Injector')['Candidate_Units']
#         #         link = link.apply(lambda v: v if v <= constraints.get(v, (0, 100))[1] else None).dropna()
#         #         # Distribute producer-specific delta to these specific wells
#         #         # >> But, make sure to respect relative, injector proportions
#         #         micro_dist = pwell_frame[pwell_frame['Candidate_Injector'].isin(link.index)].set_index('Candidate_Injector')[
#         #             'Candidate_Proportion']
#         #         micro_dist = micro_dist / sum(micro_dist)
#         #         # >> Only distribute until a single maxima is reached
#         #         # print(micro_dist)
#         #         link = link + (micro_dist * extra)
#         #         # print(link)
#         #         # link = dict(link)
#         #         # inj_of_max = max(link, key=link.get)
#         #         # p_maximum = constraints.get(inj_of_max, (0, 100))[1] / link[inj_of_max]
#         #         # link = {k: v * p for k, v in link.items()}
#         #         for inj, new_units in link.items():
#         #             suggestions.loc[pwell_frame[pwell_frame['Candidate_Injector'] == inj].index,
#         #                             'Candidate_Units'] = new_units
#
#         # Move on to next injector and repeat...
#
#     final_vector = sum(delta_tracker.values())
#     print(f'SUMMED: {final_vector}')
#     current_sum = suggestions['Candidate_Units'].sum().sum()
#     available_sum = sum(data.values())
#     p = available_sum / current_sum
#     print(f'NOW: {current_sum}, LATER: {available_sum}')
#     suggestions['Candidate_Units'] *= p
#     final_sum = suggestions['Candidate_Units'].sum().sum()
#     print(f'FINAL: {final_sum}')
#     if delta > 0:   # Lots of reccomendations pushed up to minima
#         # Push everything down a little bit
#         pass
#     elif delta < 0:   # Lots of reccomendations pushed down to maxima
#         # Push everything up a little bit
#         pass
#
#     return suggestions

# NOTE: IPC_MODULE S8_INJECTION
# def produce_search_space(CANDIDATES, PI_DIST_MATRIX, II_DIST_MATRIX, RESOLUTION=RESOLUTION):
#     def optimal_injectors(isolates_PI, isolates_II):
#         return tuple(set([e for e in isolates_PI if e in isolates_II] + [e for e in isolates_II if e in isolates_PI]))
#
#     # 3 minutes and 15 seconds: RES = 0.025
#     # _ : RES = 0.01
#     search_space = {}
#     for thresh_PI in np.arange(0.0, 1 + RESOLUTION, RESOLUTION):
#         print(f'thresh_PI: {thresh_PI}')
#         search_space[thresh_PI] = {}
#         for thresh_II in np.arange(0.0, 1 + RESOLUTION, RESOLUTION):
#             impact_tracker_PI, isolates_PI = PI_imapcts(CANDIDATES, PI_DIST_MATRIX,
#                                                         CLOSENESS_THRESH_PI=thresh_PI)
#             impact_tracker_II, isolates_II = II_impacts(II_DIST_MATRIX,
#                                                         CLOSENESS_THRESH_II=thresh_II)
#             optimals = optimal_injectors(isolates_PI, isolates_II)
#             search_space[thresh_PI][thresh_II] = optimals
#     search_space_df = pd.DataFrame(search_space).reset_index().infer_objects()
#     _accessories.save_local_data_file(search_space_df, 'Data/S8 Files/threshold_search_space.csv')
#     print('SAVED')
#
#     return search_space_df
#
# def retrieve_search_space(min_bound=0.5, early=False):
#     search_space = _accessories.retrieve_local_data_file(
#         'Data/S8 Files/threshold_search_space.csv').drop('Unnamed: 0', 1)
#
#     if early:
#         return search_space
#
#     search_space_df = search_space.set_index('index').applymap(lambda x: len(x)).reset_index()
#     search_space_df = pd.melt(search_space_df, id_vars='index', value_vars=list(search_space_df)[1:]).infer_objects()
#     search_space_df.columns = ['thresh_PI', 'thresh_II', 'n_optimal']
#     search_space_df['thresh_II'] = search_space_df['thresh_II'].astype(float)
#     search_space_df['thresh_PI'] = search_space_df['thresh_PI'].astype(float)
#     search_space_df = search_space_df[(search_space_df['thresh_PI'] < min_bound) &
#                                       (search_space_df['thresh_II'] < min_bound)].reset_index(drop=True)
#
#     return search_space_df
#
# SEARCH_SPACE = produce_search_space(CANDIDATES, PI_DIST_MATRIX, II_DIST_MATRIX, RESOLUTION=0.001)
# plot_search_space(retrieve_search_space(min_bound=0.3, early=False), cmap=cm.turbo)

# NOTE: IPC_MODULE S8_INJECTION
# def plot_search_space(search_space_df, cmap=cm.turbo):
#     ax = Axes3D(plt.figure())
#     ax.plot_trisurf(search_space_df['thresh_PI'], search_space_df['thresh_II'], search_space_df['n_optimal'],
#                     cmap=cmap)
#     plt.title('Search Space When Finding Optimal Injectors')
#     ax.set_xlabel('thresh_PI')
#     ax.set_ylabel('thresh_II')
#     ax.set_zlabel('n_optimal')
#     plt.tight_layout()
#     plt.show()

# NOTE: IPC_MODULE S8_INJECTION
# def plot_relative_allocations(suggestions):
#     fig, ax = plt.subplots(figsize=(10, 28), nrows=len(suggestions['PRO_Well'].unique()))
#     for pwell, group_df in suggestions.groupby('PRO_Well'):
#         axis = ax[list(suggestions['PRO_Well'].unique()).index(pwell)]
#         axis.set_title(f'Production Well: {pwell}')
#         group_df.plot(x='Candidate_Injector', y='Candidate_Proportion', ax=axis, kind='bar')
#     plt.tight_layout()
#
# def plot_producer_delta(suggestions):
#     _temp = suggestions[['PRO_Well', 'Delta']].drop_duplicates().reset_index(drop=True).set_index('PRO_Well')
#     fig, ax = plt.subplots(figsize=(12, 8))
#     _temp.plot(kind='bar', ax=ax)
#     plt.title('Delta from original allocation to revised allocation')

# NOTE: IPC_MODULE S3_Weighting Weight EDA
# _ = """
# ####################################
# ###########  WEIGHT EDA ############
# ####################################
# """
# MAX = 150.0
# def plot_weights_eda(df, groupby_val, groupby_col, time_col='Date', weight_col='weight', col_thresh=None):
#     plt.figure(figsize=(30, 20))
#     _temp = df[df[groupby_col] == groupby_val].sort_values(time_col).reset_index(drop=True)
#     if(col_thresh is None):
#         iter_cols = _temp.columns
#     else:
#         iter_cols = _temp.columns[:col_thresh]
#     # Plot all features (normalized to 100 max)
#     for col in [c for c in iter_cols if c not in ['Date', 'PRO_Pad', 'PRO_Well', 'weight', 'PRO_Alloc_Oil']]:
#         __temp = _temp[['Date', col]].copy().fillna(_temp[col].mean())
#         __temp[col] = MAX * __temp[col] / (MAX if max(__temp[col]) is np.nan else max(__temp[col]))
#         # plt.hist(__temp[col], bins=100)
#         if(col in ['PRO_Adj_Alloc_Oil', 'Steam', 'PRO_Adj_Pump_Speed']):
#             lw = 0.75
#         else:
#             lw = 0.3
#         plt.plot(__temp[time_col], __temp[col], linewidth=lw, label=col)
#     plt.legend(loc='upper left', ncol=2)
#
#     # # Plot weight
#     plt.plot(_temp[time_col], _temp[weight_col])
#     plt.title(f'Weight Time Series for {groupby_col} = {groupby_val}')
#     plt.savefig(f'Manipulation Reference Files/Weight TS {groupby_col} = {groupby_val}.png')
# for pad in available_pads_transformed:
#     plot_weights_eda(COMBINED_AGGREGATES, groupby_val=pad, groupby_col='PRO_Pad',
#                      time_col='Date', weight_col='weight')
# for pwell in available_pwells_transformed:
#     plot_weights_eda(COMBINED_AGGREGATES_PWELL, groupby_val=pwell,
#                      groupby_col='PRO_Well', time_col='Date', weight_col='weight')
# os.system('say finished weight exploratory analysis')
