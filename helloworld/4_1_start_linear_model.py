import tensorflow as tf
sess = tf.Session()

W1 = tf.Variable([.3], dtype=tf.float32)
W2 = tf.Variable([.4], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
linear_model = W1*x1 + W2*x2 + b

# init is a handle to the TensorFlow sub-graph that initializes all the global variables.
# Until we call sess.run, the variables are uninitialized.
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x1: [1, 2, 3, 4], x2: [2, 3, 4, 5]}))
