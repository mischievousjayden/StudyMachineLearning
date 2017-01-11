import tensorflow as tf
import numpy as np

xy = np.loadtxt("train.txt", unpack=True, dtype="float32")
x_data = xy[0:-1]
y_data = xy[-1]

print("x_data", x_data)
print("y_data", y_data)

# Try to find values for W and b
W = tf.Variable(tf.random_uniform([1, 3], -5.0, 5.0))

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

