import tensorflow as tf

x_data = [1.0, 2.0, 3.0]
y_data = [1.0, 2.0, 3.0]

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 1 and b 0, but Tensorflow will figure that out for us.
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis
hypothesis = W * X

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
a = tf.Variable(0.1)
descent = W - tf.mul(a, tf.reduce_mean(tf.mul((tf.mul(W, X) - Y), X)))
update = W.assign(descent)

# Before starting, initialize the variables, We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(100):
    sess.run(update, feed_dict={X:x_data, Y:y_data})
    print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))

