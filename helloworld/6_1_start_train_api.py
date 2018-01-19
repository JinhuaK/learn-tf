import tensorflow as tf

#Model parameters
W1 = tf.Variable([0.86], dtype=tf.float32)
W2 = tf.Variable([1.13], dtype=tf.float32)
b = tf.Variable([-1.13], dtype=tf.float32)

#Model input and output
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
linear_model = W1*x1 + W2*x2 + b
y = tf.placeholder(tf.float32)

#loss
loss = tf.reduce_sum(tf.square(linear_model - y))

# optimizer もともとは0.01, 0.02の方はもっとよかった
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#training data
x1_train = [1, 2, 3, 4]
x2_train = [2, 3, 4, 5]
y_train = [2, 4, 6, 8]

#initializer & training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
    sess.run(train, {x1: x1_train, x2: x2_train, y: y_train})

# evaluate training accuracy
curr_W1, curr_W2, curr_b, curr_loss = sess.run([W1, W2, b, loss], {x1: x1_train, x2: x2_train, y: y_train})

print("W1: %s W2: %s b: %s loss: %s"%(curr_W1, curr_W2, curr_b, curr_loss))
