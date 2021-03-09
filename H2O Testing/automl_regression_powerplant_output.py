# @Author: Shounak Ray <Ray>
# @Date:   09-Mar-2021 11:03:25:250  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: automl_regression_powerplant_output.py
# @Last modified by:   Ray
# @Last modified time: 09-Mar-2021 12:03:39:393  GMT-0700
# @License: [Private IP]


import h2o
import pandas as pd
from h2o.automl import H2OAutoML

h2o.init()

# For the AutoML regression demo, we use the [Combined Cycle Power Plant]
#   (http://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant) dataset.
#   The goal here is to predict the energy output (in megawatts), given the temperature, ambient pressure,
#   relative humidity and exhaust vacuum values.  In this demo, you will use H2O's AutoML to outperform the
#   [state of the art results](https://www.sciencedirect.com/science/article/pii/S0142061514000908) on this task.

# Link of data
data_path = "https://github.com/h2oai/h2o-tutorials/raw/master/h2o-world-2017/automl/data/powerplant_output.csv"

# Load data into H2O
df = h2o.import_file(data_path)
df.describe()

y = "HourlyEnergyOutputMW"

splits = df.split_frame(ratios=[0.8], seed=1)
train = splits[0]
test = splits[1]

aml = H2OAutoML(max_runtime_secs=60, seed=1, project_name="powerplant_lb_frame")
aml.train(y=y, training_frame=train)

# For demonstration purposes, we will also execute a second AutoML run, this time providing the original, full
#   dataset, `df` (without passing a `leaderboard_frame`).  This is a more efficient use of our data since we can
#   use 100% of the data for training, rather than 80% like we did above.  This time our leaderboard will use
#   cross-validated metrics.
# *Note: Using an explicit `leaderboard_frame` for scoring may be useful in some cases, which is why the option is
#   available.*

aml2 = H2OAutoML(max_runtime_secs=60, seed=1, project_name="powerplant_full_data")
aml2.train(y=y, training_frame=df)


# *Note: We specify a `project_name` here for clarity.*
# ## Leaderboard
# Next, we will view the AutoML Leaderboard.  Since we specified a `leaderboard_frame` in the `H2OAutoML.train()`
#   method for scoring and ranking the models, the AutoML leaderboard uses the performance on this data to rank
#   the models.
# After viewing the `"powerplant_lb_frame"` AutoML project leaderboard, we compare that to the leaderboard for the
#   `"powerplant_full_data"` project.  We can see that the results are better when the full dataset is
#   used for training.
# A default performance metric for each machine learning task (binary classification, multiclass classification,
#   regression) is specified internally and the leaderboard will be sorted by that metric.
# In the case of regression, the default ranking metric is mean residual deviance.  In the future, the user will
#   be able to specify any of the H2O metrics so that different metrics can be used to generate rankings
#   on the leaderboard.

aml.leaderboard.head()


# Now we will view a snapshot of the top models.  Here we should see the two Stacked Ensembles at or near the top
#   of the leaderboard.  Stacked Ensembles can almost always outperform a single model.
aml2.leaderboard.head()


# This dataset comes from the [UCI Machine Learning Repository]
#   (http://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant) of machine learning datasets.
#   The data was used in a [publication](https://www.sciencedirect.com/science/article/pii/S0142061514000908)
#   in the *International Journal of Electrical Power & Energy Systems* in 2014.  In the paper, the authors achieved
#   a mean absolute error (MAE) of 2.818 and a Root Mean-Squared Error (RMSE) of 3.787 on their best model.
# So, with H2O's AutoML, we've already beaten the state-of-the-art in just 60 seconds of compute time!

# ## Predict Using Leader Model
# If you need to generate predictions on a test set, you can make predictions on the `"H2OAutoML"` object directly,
#   or on the leader model object.

pred = aml.predict(test)
pred.head()


# If needed, the standard `model_performance()` method can be applied to the AutoML leader model and a test set to
#   generate an H2O model performance object.

perf = aml.leader.model_performance(test)
