from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

import numpy as np
import tensorflow as tf

#data Sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
    # If the training and test sets aren't stored locally, download them.
    if not os.path.exists(IRIS_TRAINING):
        raw = urlopen(IRIS_TRAINING_URL).read()
        with open(IRIS_TRAINING, "wb") as f:
            f.write(raw)

    if not os.path.exists(IRIS_TEST):
        raw = urlopen(IRIS_TEST_URL).read()
        with open(IRIS_TEST, "wb") as f:
            f.write(raw)

    # Load datasets.
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename = IRIS_TRAINING,
        target_dtype = np.int,
        features_dtype = np.float32)

    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename = IRIS_TEST,
        target_dtype = np.int,
        features_dtype = np.float32)

    # Specify that all features have real-value data
    #?(sepal width, sepal height, petal width, and petal height)
    feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns = feature_columns,
        hidden_units = [10, 20, 10],
        n_classes = 3,
        model_dir = "./iris_model")

    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": np.array(training_set.data)},
        y = np.array(training_set.target),
        num_epochs = None,
        shuffle = True)

    #train model.
    classifier.train(input_fn = train_input_fn, steps=2000)

    #define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": np.array(test_set.data)},
        y = np.array(test_set.target),
        num_epochs = 1,
        shuffle = False)

    #evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn = test_input_fn)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    #classify two new flower samples. ?
    new_samples = np.array(
        [[6.4, 3.2, 4.5, 1.5],
         [5.8, 3.1, 5.0, 1.7],
         [7.8, 4.1, 2.0, 3.7]], dtype = np.float32)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": new_samples},
        num_epochs = 1,
        shuffle = False)

    predictions = list(classifier.predict(input_fn = predict_input_fn))
    predicted_classes = [p["classes"] for p in predictions]

    print(
        "New samples, Class Predictions: {}\n"
        .format(predicted_classes))

if __name__ == "__main__":
    main()
