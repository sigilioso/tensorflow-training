# coding: utf-8
import os
import time
from contextlib import contextmanager
from types import SimpleNamespace

import yaml
import tensorflow as tf
import matplotlib.pyplot as plt

from data_handler import ImagesHandler, get_label_name

# See:
# - <https://www.tensorflow.org/tutorials/>
# - <https://www.tensorflow.org/get_started/get_started>
# - <https://www.tensorflow.org/get_started/mnist/beginners>
# - <https://www.tensorflow.org/get_started/mnist/pros>
# - <https://www.tensorflow.org/programmers_guide/saved_model>


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def plot_accuracy(data):
    accuracy_plot = plt.figure(0)
    plt.title('Accuracy history')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy (blue = train | green = test)')
    plt.plot(data['train'], 'b')
    plt.plot(data['test'], 'g')
    accuracy_plot.show()
    plt.show()


def get_elapsed_time(start_time, time_format='%Hh %Mm %Ss'):
    return time.strftime(time_format, time.gmtime(time.time() - start_time))


class NeuralNetwork():

    def __init__(self, config_path=None):
        self._load_config(config_path)
        self.data_handler = ImagesHandler(
            self.images_path,
            width=self.width,
            height=self.height
        )
        self.num_classes = self.data_handler.nclasses
        self.train_data = self.data_handler.train_batch_iter(self.batch_size)
        self.test_data = self.data_handler.test_data
        self.create_network_structure()

    def _load_config(self, config_path):
        if config_path is None:
            this_dir = os.path.dirname(os.path.realpath(__file__))
            config_path = os.path.join(this_dir, 'config.yml')
        with open(config_path) as f:
            config = yaml.load(f)

        data_config = config.get('data', {})
        self.width = data_config.get('width', 45)
        self.height = data_config.get('height', 45)
        self.num_classes = data_config.get('num_classes', 59)
        self.images_path = data_config.get('images_path')
        self.batch_size = data_config.get('batch_size', 300)

        network_config = config.get('network', {})
        self.n_first_layer = network_config.get('first_layer', 16)
        self.n_second_layer = network_config.get('second_layer', 32)
        self.n_third_layer = network_config.get('third_layer', 32)
        self.learning_rate = network_config.get('learning_rate', 0.001)
        self.save_to_file = network_config.get('save_to_file')
        self.restore_from_file = network_config.get('restore_from_file')

        training_config = config.get('training', {})
        self.training_iterations = training_config.get('iterations', 2000)
        self.keep_prob = training_config.get('keep_prob', 0.5)

    def create_network_structure(self):
        """
        Create neural network structure by creating:
        - self.ph.x, self.ph.y and self.ph.keep_prob placeholders
        - self.y_conv, self.train_step, self.correct_prediction, self.accuracy
        """
        # Init placeholders
        # See: <https://www.tensorflow.org/api_docs/python/tf/placeholder>
        #      <https://www.tensorflow.org/api_guides/python/reading_data#feeding>
        self.ph = SimpleNamespace()
        self.ph.x = tf.placeholder(tf.float32, shape=[None, self.width * self.height])
        self.ph.y = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.ph.keep_prob = tf.placeholder(tf.float32)
        # Reshape x to a specific tensor
        # See: <https://www.tensorflow.org/api_docs/python/tf/reshape>
        x_ = tf.reshape(self.ph.x, [-1, self.width, self.height, 1])

        # Network structure

        # See: <https://www.tensorflow.org/tutorials/layers>

        # First convolutional layer
        conv1 = tf.layers.conv2d(
            inputs=x_,
            filters=self.n_first_layer,
            kernel_size=[7, 7],
            padding='SAME'
        )
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        # Second convolutional layer
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=self.n_second_layer,
            kernel_size=[4, 4],
            padding='SAME'
        )
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        # Dropout
        pool2_drop = tf.nn.dropout(pool2, self.keep_prob)
        # Third 'dense' layer
        pool2_flatten = tf.reshape(
            pool2_drop,
            [-1, int(self.width / 4) * int(self.height / 4) * self.n_second_layer]
        )
        dense = tf.layers.dense(inputs=pool2_flatten, units=self.n_third_layer)
        dense_drop = tf.nn.dropout(dense, self.keep_prob)
        # Readout layer
        weights = weight_variable([self.n_third_layer, self.num_classes])
        bias = bias_variable([self.num_classes])

        self.y_conv = tf.matmul(dense_drop, weights) + bias

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.ph.y, logits=self.y_conv)
        )

        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.ph.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def save(self, sess):
        if self.save_to_file:
            saver = tf.train.Saver()
            saver.save(sess, self.save_to_file)
            print('\nMODEL SAVED TO {}\n'.format(self.save_to_file))

    def restore(self, sess):
        if self.restore_from_file:
            saver = tf.train.Saver()
            saver.restore(sess, self.restore_from_file)
            print('\nMODEL RESTORED FROM {}\n'.format(self.restore_from_file))

    @contextmanager
    def tf_session(self, restore=True, save=True):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            if restore:
                self.restore(sess)
            yield sess
            if save:
                self.save(sess)

    def train(self):
        start_time = time.time()

        accuracy_history = {'train': [], 'test': []}
        x_test, y_test = self.test_data
        with self.tf_session():

            for step in range(self.training_iterations):
                x_train, y_train = next(self.train_data)
                if step % 100 == 0:
                    train_accuracy = self.accuracy.eval(
                        feed_dict={self.ph.x: x_train, self.ph.y: y_train, self.ph.keep_prob: 1.0}
                    )
                    test_accuracy = self.accuracy.eval(
                        feed_dict={self.ph.x: x_test, self.ph.y: y_test, self.ph.keep_prob: 1.0}
                    )
                    elapsed_time = get_elapsed_time(start_time)
                    accuracy_history['train'].append(train_accuracy)
                    accuracy_history['test'].append(test_accuracy)
                    print(f'Step:              {step}',
                          f'Training accuracy: {train_accuracy}',
                          f'Test accuracy    : {test_accuracy}',
                          f'Elapsed time: {elapsed_time}',
                          sep='\n',
                          end='\n---\n')
                self.train_step.run(
                    feed_dict={self.ph.x: x_train, self.ph.y: y_train,
                               self.ph.keep_prob: self.keep_prob}
                )

            test_accuracy = self.accuracy.eval(
                feed_dict={self.ph.x: x_test, self.ph.y: y_test, self.ph.keep_prob: 1.0}
            )
            print('---')
            print('Final test accuracy {}'.format(test_accuracy))
            plot_accuracy(accuracy_history)

    def make_prediction(self, image_path):
        image = self.data_handler.process_image(image_path)
        with self.tf_session():
            y = self.y_conv.eval(feed_dict={self.ph.x: ImagesHandler.to_array([image])})
            label = tf.argmax(y, 1).eval()[0]
        return label


if __name__ == '__main__':
    nn = NeuralNetwork()
    # nn.train()
    image_path = './DITS-classification-data/classification test/6/2015-11-30 14-38-40.546.wmv_7667_0000_0577_0183_0077_0078.png'  # noqa
    label = nn.make_prediction(image_path)
    print('Predicted: {} - {}'.format(label, get_label_name(label)))
