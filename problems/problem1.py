# -*- coding: utf-8 -*-
"""
y = W * x + b
"""
import tensorflow as tf
import matplotlib.pyplot as plt


def init_session():
    sess = tf.InteractiveSession()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    return sess


learning_rate = 0.001
training_epochs = 600
display_steps = 50


# placeholders
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

x, y = 5.0, 10.0


# weigh
W = tf.Variable(0.0)

# Biases

# b = tf.constant(0.0)  # Problem 1, b constant
b = tf.Variable(0.0)  # Problem 2, b variable

y_output = tf.add(tf.multiply(W, X), b)  # y = W * x + b
error = tf.abs(tf.subtract(Y, y_output))  # difference between predicted and real


# Optimizer

optimizer = (tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
                     .minimize(error))


sess = init_session()

W_s = []
b_s = []


# Feeder

feeder = {X: x, Y: y}


# Training


for epoch in range(training_epochs):
    sess.run(optimizer, feed_dict=feeder)
    w = W.eval()
    b_ = b.eval()

    W_s.append(w)
    b_s.append(b_)

    # See results
    if ((epoch + 1) % display_steps) == 0:
        c = sess.run(error, feed_dict=feeder)
        print('Epoch :{}'.format(epoch + 1))
        print('W :{}'.format(w))
        print('B :{}'.format(b_))

print('--- Training end ---')
final_error = sess.run(error, feed_dict=feeder)
print('Final error: {}'.format(final_error))

y_predicted = y_output.eval(feed_dict=feeder)
print('Predicted y: {}'.format(y_predicted))


plt.plot(W_s, label='Weights')
plt.plot(b_s, label='Biases')
plt.legend()
plt.show()
