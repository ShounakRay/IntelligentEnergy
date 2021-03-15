# @Author: Shounak Ray <Ray>
# @Date:   20-Feb-2021 23:02:81:814  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: extracontent.py
# @Last modified by:   Ray
# @Last modified time: 15-Mar-2021 17:03:54:549  GMT-0600
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
