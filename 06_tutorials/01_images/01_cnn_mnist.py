from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# application logic start
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    #Input Layer
    # [batch_size, image_width, image_height, channels]
    # batch_size: Size of the subset of examples to use when performing GD during training?
    # image_width: Width of the example images.
    # image_height: Height of the example images.
    # channels: Number of color channels in the example images. For color images, the number of channels is 3 (red, green, blue).
    #           For monochrome images, there is just 1 channel (black).
    input_layer = tf.reshape(features["x"], 【-1， 28， 28， 1】)

    #Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 32,
        kernel_size = [5, 5],
        padding = "same",
        activation = tf.nn.relu)

    #Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2, 2], strides = 2)

    #converlutional layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 64,
        kernel_size = [5, 5],
        padding = "same",
        activation = tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2, 2], strides = 2)

    #Dense layer
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    dense = tf.laysers.dense(inputs = pool2_flat, units = 1024, activation = tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tr.estimator.ModeKeys.TRAIN)

    #logits layers
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        #Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.esitmator.EstimatorSpec(mode=mode, predictions=predictions)

    #calculate loss (for both train and eval modes)
    loss = tf.loses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    #configure the training Op (for train mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    #Add evaluation metrics (for Eval mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, preditions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops)


# application logic end

if __name__ = "__main__":
    tf.app.run()
