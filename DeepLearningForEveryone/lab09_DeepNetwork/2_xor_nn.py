import tensorflow as tf
import numpy as np

xy = np.loadtxt("train.txt", unpack=True, dtype="float32")
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

num_features = 2
num_hidden = 2

W1 = tf.Variable(tf.random_uniform([num_features, num_hidden], -1.0, 1.0), name="Weight1")
W2 = tf.Variable(tf.random_uniform([num_hidden, 1], -1.0, 1.0), name="Weight2")

b1 = tf.Variable(tf.zeros([2]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")

# Our hypothesis
L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)

# Cost function
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

# Minimize
a = tf.Variable(0.01) # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Before starting, initialize the variables, We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
with tf.Session() as sess:
    sess.run(init)

    # Fit the line.
    for step in range(2000):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if(step % 200 == 0):
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W1), sess.run(W2), sess.run(b1), sess.run(b2))

    # Test model
    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data}))
    print("Accuracy:", accuracy.eval({X:x_data, Y:y_data}))

