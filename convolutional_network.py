

# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



class convolutional_neural_network():


    def __init__(self, image_size, image_channel, labels, hidden_node, kernel_size, kernel_depth, pool_size,
                 kernel_stride, pool_stride, learning_rate,regularization_rate, learning_rate_decay,
                 moving_average_decay, batch_size, steps, activation_function):

        assert len(kernel_size) == len(kernel_depth) == len(pool_size) == len(kernel_stride) == len(pool_stride)
        self.image_size = image_size
        self.image_channel = image_channel
        self.n_hidden = hidden_node
        self.n_output = labels

        self.kernel_size = kernel_size
        self.kernel_depth = kernel_depth
        self.kernel_depth.insert(0, image_channel)
        self.pool_size = pool_size
        self.kernel_stride = kernel_stride
        self.pool_stride = pool_stride

        self.learning_rate = learning_rate
        self.regularizer = regularization_rate
        self.lr_decay = learning_rate_decay
        self.moving_average = moving_average_decay
        self.batch_size = batch_size
        self.steps = steps
        self.activation_function = activation_function
        self.sess = tf.InteractiveSession()


    def get_parameters(self, shape):

        weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', shape[-1], initializer=tf.constant_initializer(0.1))

        if self.regularizer:

            regularization = tf.contrib.layers.l2_regularizer(self.regularizer)
            tf.add_to_collection('losses', regularization(weights))

        return weights, biases



    def inference(self, data):

        for i in range(len(self.kernel_size)):

            with tf.variable_scope('layer'+ str(2*i+1) + '-conv'):

                weights, biases = self.get_parameters([self.kernel_size[i], self.kernel_size[i],
                                                       self.kernel_depth[i],self.kernel_depth[i+1]])
                conv = tf.nn.conv2d(data, weights, [1,self.kernel_stride[i],self.kernel_stride[i],1], padding='SAME')
                activation = self.activation_function(tf.nn.bias_add(conv, biases))

            with tf.variable_scope('layer' + str(2*i+2) + '-pool'):

                data = tf.nn.max_pool(activation, [1,self.pool_size[i],self.pool_size[i],1],
                                      [1, self.pool_stride[i], self.pool_stride[i], 1], padding='SAME')

        data_shape = data.get_shape().as_list()
        nodes = data_shape[1] * data_shape[2] * data_shape[3]
        reshaped = tf.reshape(data, [-1, nodes])

        with tf.variable_scope('layer' + str(2*len(self.kernel_size)+1), '-fc'):

            weights, biases = self.get_parameters([nodes, self.n_hidden])

            layer1 = self.activation_function(tf.matmul(reshaped, weights) + biases)

        with tf.variable_scope('layer' + str(2*len(self.kernel_depth)+2), '-fc'):

            weights, biases = self.get_parameters([self.n_hidden, self.n_output])

            out = tf.matmul(layer1, weights) + biases


        return out



    def train(self, datasets):

        x = tf.placeholder(tf.float32, [None, self.image_size,
                                        self.image_size, self.image_channel], name='x-input')
        y = tf.placeholder(tf.float32, [None, self.n_output], name='y-input')

        out = self.inference(x)

        global_step = tf.Variable(0, trainable=False)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y, 1), logits=out)
        loss = tf.reduce_mean(cross_entropy)
        if self.regularizer:
            loss += tf.add_n(tf.get_collection('losses'))

        if self.lr_decay:
            self.learning_rate = tf.train.exponential_decay(self.learning_rate, global_step,
                                                            datasets.train.num_examples / self.batch_size,
                                                            self.lr_decay)
        train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss, global_step=global_step)

        if self.moving_average:
            variables_averages = tf.train.ExponentialMovingAverage(self.moving_average, global_step)
            variables_averages_op = variables_averages.apply(tf.trainable_variables())
            with tf.control_dependencies([train_op, variables_averages_op]):
                train_op = tf.no_op(name='train')

        corrct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(corrct_prediction, tf.float32))

        validation_reshaped = np.reshape(datasets.validation.images,
                                         (datasets.validation.num_examples, self.image_size, self.image_size, self.image_channel))
        validation_feed = {x: validation_reshaped, y: datasets.validation.labels}

        # test_reshaped = np.reshape(datasets.test.images,
        #                                  (datasets.test.num_examples, self.image_size, self.image_size,
        #                                   self.image_channel))
        # test_feed = {x: test_reshaped, y: datasets.test.labels}
        test_reshaped = np.reshape(datasets.test.images[0:5000],
                                   (5000, self.image_size, self.image_size,
                                    self.image_channel))
        test_feed = {x: test_reshaped, y: datasets.test.labels[0:5000]}

        self.sess.run(tf.global_variables_initializer())

        for i in range(self.steps):

            xs, ys = datasets.train.next_batch(self.batch_size)

            xs_reshaped = np.reshape(xs, (self.batch_size, self.image_size, self.image_size, self.image_channel))

            self.sess.run(train_op, feed_dict={x: xs_reshaped, y: ys})

            if i % 1000 == 0:
                loss_value, accuracy_val = self.sess.run([loss, accuracy], feed_dict=validation_feed)
                print('After %d training step(s), validation loss is %g, '
                      'validation accuracy = %g ' % (i, loss_value, accuracy_val))

        accuracy_test = self.sess.run(accuracy, feed_dict=test_feed)
        print('\n\nAfter all training steps, accuracy on test '
              'datasets is %g ' % accuracy_test)
        print('\n\n\nTraining Done!!!\n\n')



    def main(self):

        mnist = input_data.read_data_sets('./to/MNIST_data', one_hot=True)
        self.train(mnist)



if __name__ == '__main__':

    CNN = convolutional_neural_network(image_size=28, image_channel=1, labels=10, hidden_node=512,
                                       kernel_size=[5,3], kernel_depth=[32, 64], pool_size=[2, 2],
                                       kernel_stride=[1, 1], pool_stride=[2, 1], learning_rate=0.01,
                                       regularization_rate=0.001, learning_rate_decay=0.99, moving_average_decay=None,
                                       batch_size=100, steps=10000, activation_function=tf.nn.relu)
    CNN.main()







