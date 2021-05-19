# @Author: Shounak Ray <Ray>
# @Date:   17-May-2021 10:05:57:572  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: h2o_prediction.py
# @Last modified by:   Ray
# @Last modified time: 18-May-2021 00:05:25:256  GMT-0600
# @License: [Private IP]

import ast
from typing import Final

import h2o
import numpy as np
import pandas as pd
from _references import _accessories

MAPPING = ast.literal_eval(open('mapping.txt', 'r').read().split('[Private IP]\n\n')[1])
INV_MAPPING = {v: k for k, v in MAPPING.items()}

# H2O Server Constants
IP_LINK: Final = 'localhost'                                  # Initializing the server on the local host (temporary)
SECURED: Final = True if(IP_LINK != 'localhost') else False   # https protocol doesn't work locally, should be True
PORT: Final = 54321                                           # Always specify the port that the server should use
SERVER_FORCE: Final = True                                    # Tries to init new server if existing connection fails


def h2o_model_prediction(model_path, new_data, tolerable_rmse, responder='PRO_Total_Fluid', just_predictions=False):
    with _accessories.suppress_stdout():
        h2o.init(https=SECURED,
                 ip=IP_LINK,
                 port=PORT,
                 start_h2o=SERVER_FORCE)

    # Convert the dataframe so that feature names are compatible with h2O models
    new_data.columns = [MAPPING.get(c) for c in new_data.columns if MAPPING.get(c) != '']
    new_data = new_data[[c for c in new_data.columns if c != None]]

    new_data['Date'] = pd.to_datetime(new_data['Date'])
    dates = new_data['Date'].copy()
    new_data = new_data.set_index('Date')

    # NOTE: Check if feature engineering is required (in which case, engineer new features!)
    with _accessories.suppress_stdout():
        model = h2o.load_model(model_path)
        input_features = model._model_json['output']['names']
        rigid_responder = model._model_json['response_column_name']
        original_features = [f for f in input_features if f in list(new_data)]
        # Ensure types
        wanted_types = {k: 'real' if v == float or v == int else 'enum'
                        for k, v in dict(new_data[original_features].dtypes).items()}
        new_data_H2F = h2o.H2OFrame(new_data[original_features], column_types=wanted_types)

    if len(input_features) > 30:    # There are always more the 30 engineered features in any trained H2O Model
        existing_variables = new_data.columns
        # Find the features that are engineered (don't exist in the original data features)
        repl_dict = {}
        for ft in original_features:
            conversions = {}
            for existing_var in existing_variables:
                if existing_var in ft:
                    # This means it was engineeÍred
                    # Substitute variable references with new_data['THE_ENGINEERED_FEATURE'],
                    #   store in replacement dict now
                    conversions[existing_var] = f"new_data['{existing_var}']"
                repl_dict[ft] = conversions
            for orig_name, sub in repl_dict[ft].items():
                string_to_execute = ft.replace(orig_name, sub)
            string_to_execute = f"new_data['{ft}'] = {string_to_execute}"
            exec(string_to_execute)

    predictions = model.predict(new_data_H2F).as_data_frame().infer_objects().rename(columns={'predict': 'predicted'})

    predictions['Date'] = pd.to_datetime(dates)
    predictions = predictions.set_index('Date')
    predictions['actual'] = new_data[rigid_responder].values
    predictions['algorithm_class'] = model._model_json['algo_full_name']
    predictions['algorithm_name'] = model._model_json['model_id']['name']
    predictions['accuracy'] = 1 - abs(predictions['actual'].values -
                                      predictions['predicted'].values) / predictions['actual']

    if just_predictions:
        return predictions

    val_rmse = np.sqrt(np.mean((new_data[rigid_responder] - predictions['predicted'])**2))
    rel_val_rmse = val_rmse - tolerable_rmse
    numeric_accuracies = predictions['accuracy'][(predictions['accuracy'] != -1 * np.inf) &
                                                 (predictions['accuracy'] != +1 * np.inf)]
    metrics = pd.DataFrame([[model._model_json['model_id']['name'], val_rmse, rel_val_rmse,
                             np.mean(numeric_accuracies)]],
                           columns=['algorithm_name', 'RMSE', 'Relative_RMSE', 'accuracy'])

    # h2o.remove_all()
    # h2o.cluster().shutdown()

    return predictions, metrics, model


# # NO Feature Engineering
# model_path = 'Modeling Reference Files/8882 – ENG: False, WEIGHT: False, TIME: 40/Models/GBM_grid__1_AutoML_20210428_051233_model_7'
# # Feature Engineering
# model_path = 'Modeling Reference Files/6177 – ENG: True, WEIGHT: False, TIME: 120/Models/GBM_grid__1_AutoML_20210428_135036_model_23'
# new_data = pd.read_csv('_data/S3 Files/combined_ipc_aggregates_ALL.csv')
# # TEMP: This conversion below is just for simulation
# new_data.columns = [INV_MAPPING.get(c) for c in new_data.columns if INV_MAPPING.get(c) != 'c']
# new_data = new_data[[c for c in new_data.columns if c != None]]
# tolerable_rmse = 133
# models_outputs, metric_outputs, model = h2o_model_prediction(model_path, new_data, tolerable_rmse)


# EOF

# EOF
