import tensorflow as tf

x_data = [ [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 2.0, 0.0, 4.0, 0.0],
        [1.0, 0.0, 3.0, 0.0, 5.0] ]
y_data = [1, 2, 3, 4, 5]

# Try to find values for W and b
W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))

# Our hypothesis
hypothesis = tf.matmul(W, x_data)

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Minimize
a = tf.Variable(0.1) # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Before starting, initialize the variables, We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(2001):
    sess.run(train)
    if(step % 20 == 0):
        print(step, sess.run(cost), sess.run(W))

# Learns best fit is W = [0 1 1]

