import tensorflow as tf
sess = tf.Session()

#model
W1 = tf.Variable([.3], dtype=tf.float32)
W2 = tf.Variable([.4], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
linear_model = W1*x1 + W2*x2 + b

# teacher data y
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# initializes
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(loss, {x1:[1, 2, 3, 4], x2:[2, 3, 4, 5], y:[2, 4, 6, 8]}))

# MANUALLY by reassigning the values of W and b to the perfect values of -1 and 1.
fixW1 = tf.assign(W1, [0.86666453])
fixW2 = tf.assign(W2, [1.13333416])
fixb = tf.assign(b, [-1.13333046])
sess.run([fixW1, fixW2, fixb])
print(sess.run(loss, {x1:[1, 2, 3, 4], x2:[2, 3, 4, 5], y:[2, 4, 6, 8]}))
