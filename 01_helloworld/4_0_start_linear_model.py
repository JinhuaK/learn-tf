import tensorflow as tf
sess = tf.Session()

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

# init is a handle to the TensorFlow sub-graph that initializes all the global variables.
# Until we call sess.run, the variables are uninitialized.
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
