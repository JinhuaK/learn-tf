from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# read data from CSV files.
COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"

training_set = pd.read_csv("./02_boston/boston_train.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
test_set = pd.read_csv("./02_boston/boston_test.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv("./02_boston/boston_predict.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS)

# create a list of FeatureColumns for the input data.
feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

# instantiate a DNNRegressor for the neural network regression model
regressor = tf.estimator.DNNRegressor(feature_columns = feature_cols,
                                      hidden_units = [10, 10],
                                      model_dir = "./02_boston/boston_model")

# Building the input_fn
# shuffle: 配列をランダムに並び替える
def get_input_fn(data_set, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x = pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        y = pd.Series(data_set[LABEL].values),
        num_epochs = num_epochs,
        shuffle = shuffle)

regressor.train(input_fn = get_input_fn(training_set), steps = 5000)

# evaluating the model
ev = regressor.evaluate(input_fn = get_input_fn(test_set, num_epochs = 1, shuffle = False))

loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))

y = regressor.predict(input_fn = get_input_fn(prediction_set, num_epochs=1, shuffle = False))
# .predict() returns an iterator of dicts; convert to a list and print

#Predictions
predictions = list(p["predictions"] for p in itertools.islice(y, 6))
print("Predictions: {}".format(str(predictions)))
