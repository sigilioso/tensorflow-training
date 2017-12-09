# -*- coding: utf-8 -*-
"""
XOR

f(0, 0) -> 0
f(0, 1) -> 1
f(1, 0) -> 1
f(1, 1) -> 0

Other references:
 - https://github.com/techdisrupt/XOR
 - https://medium.com/@claude.coulombe/the-revenge-of-perceptron-learning-xor-with-tensorflow-eb52cbdf6c60
""" # noqa
import time

import tensorflow as tf
import matplotlib.pyplot as plt
# --------------------------------------------------------------------------------
# Neural network configuration
# --------------------------------------------------------------------------------
"""
(0, 0) _____                ___ (0,)
            \              /
(0, 1) _____ (*) ___ __ (*)____ (1,)
                    /
(1, 0) _____ (*) ___\__ (*)____ (1,)
            /              \
(1, 1) ____/                \__ (0,)
"""

# Placeholders
# input
x_placeholder = tf.placeholder(tf.float32, shape=[4, 2])
# output
y_placeholder = tf.placeholder(tf.float32, shape=[4, 1])

# weights
w1 = tf.Variable(tf.random_uniform(shape=[2, 2], minval=-0.5, maxval=0.5))
w2 = tf.Variable(tf.random_uniform(shape=[2, 1], minval=-0.5, maxval=0.5))

# bias
b1 = tf.Variable(tf.zeros([2]))
b2 = tf.Variable(tf.zeros([1]))

# sigmoid to use values 0, 1
y1 = tf.sigmoid(tf.matmul(x_placeholder, w1) + b1)
y2 = tf.sigmoid(tf.matmul(y1, w2) + b2)

# Mean Squared Stimate (MSE)
error = tf.reduce_mean(tf.square(y2 - y_placeholder))
# Average Cross Entropy
# error = tf.reduce_mean(
#     ((y_placeholder * tf.log(y2)) + (1 - y_placeholder) * tf.log(1.0 - y2)) * -1
# )

# Variant of Gradient descent optimizer see:
# <https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer>
learning_rate = 0.002
train_step = tf.train.AdamOptimizer(learning_rate).minimize(error)
# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
# Inputs and outputs (data to train)
# --------------------------------------------------------------------------------
xor_x = [(0, 0), (0, 1), (1, 0), (1, 1)]
xor_y = [(0,), (1,), (1,), (0,)]
feeder = {x_placeholder: xor_x, y_placeholder: xor_y}
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Tensorflow session
# --------------------------------------------------------------------------------
session = tf.InteractiveSession()
session.run(tf.local_variables_initializer())
session.run(tf.global_variables_initializer())
# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
# Values to plot the training evolution
# --------------------------------------------------------------------------------
t_start = time.clock()
_y0, _y1, _y2, _y3 = [], [], [], []
errors = []
# --------------------------------------------------------------------------------


for epoch in range(10000):
    session.run(train_step, feeder)
    new_y = y2.eval(feeder)
    error1 = error.eval(feeder)

    if epoch % 10 == 0:
        _y0.append(new_y[0])
        _y1.append(new_y[1])
        _y2.append(new_y[2])
        _y3.append(new_y[3])
        errors.append(error1)

    if epoch % 1000 == 0:
        print('Epoch: {}'.format(epoch + 1))
        # print('W1 {}; W2 {}'.format(w1.eval(), w2.eval()))
        # print('bias1 {}; bias2 {}'.format(b1.eval(), b2.eval()))
        print('Output \n{}\n'.format(new_y))
        print('Error: {}'.format(error1))

    if error1 < 0.004:
        break

print('--------- THE END ----------')
print('Time: {}'.format(time.clock() - t_start))

plt.plot(_y0, label='y0')
plt.plot(_y1, label='y1')
plt.plot(_y2, label='y2')
plt.plot(_y3, label='y3')
plt.legend()
plt.show()

plt.plot(errors, label='Errors')
plt.legend()
plt.show()
