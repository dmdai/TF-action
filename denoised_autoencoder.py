

import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt


MODEL_SAVE_PATH = './to/model/AGN_model'
MODEL_NAME = 'model.ckpt'


def xavier_init(fan_in, fan_out):

    low = -np.sqrt(6./(fan_in + fan_out))
    high = np.sqrt(6./(fan_in + fan_out))

    return tf.random_uniform([fan_in, fan_out], minval=low, maxval=high,
                             dtype=tf.float32)


class AdditiveGaussianNoiseAutoencoder():

    def __init__(self, input_node, hidden_node, learning_rate, regularization_rate,
                 scale, learning_rate_decay, exponential_moving_average, batch_size, activation_function, steps):

        self.n_input = input_node
        self.n_hidden = hidden_node
        self.learning_rate = learning_rate
        self.regularizer = regularization_rate
        self.scale = scale
        self.lr_decay = learning_rate_decay
        self.moving_average = exponential_moving_average
        self.steps = steps
        self.batch_size = batch_size
        self.activation_function = activation_function
        self.sess = tf.InteractiveSession()

    def inference(self, input_tensor):

        with tf.variable_scope('layer-extract'):

            weights = tf.Variable(xavier_init(self.n_input, self.n_hidden))
            biases = tf.get_variable('biases', self.n_hidden, initializer=tf.constant_initializer(0.1))

            if self.regularizer:
                regularizer = tf.contrib.layers.l2_regularizer(self.regularizer)
                tf.add_to_collection('losses', regularizer(weights))

            self.extractor = self.activation_function(tf.matmul(input_tensor+self.scale*tf.random_normal((self.n_input,)), weights) + biases)

        with tf.variable_scope('layer-reconstruction'):

            weights = tf.Variable(xavier_init(self.n_hidden, self.n_input))
            biases = tf.get_variable('biases', self.n_input , initializer=tf.constant_initializer(0.1))

            if self.regularizer:
                regularizer = tf.contrib.layers.l2_regularizer(self.regularizer)
                tf.add_to_collection('losses', regularizer(weights))

            self.reconstructor = tf.matmul(self.extractor, weights) + biases

        return self.reconstructor


    def train(self, dataset):

        self.x = tf.placeholder(tf.float32, [None, self.n_input], name='x-input')

        out = self.inference(self.x)

        global_step = tf.Variable(0, trainable=False)

        loss = tf.reduce_mean(tf.square(out-self.x))
        if self.regularizer:
            loss += tf.add_n(tf.get_collection('losses'))

        if self.lr_decay:
            self.learning_rate = tf.train.exponential_decay(self.learning_rate, global_step,
                                                   dataset.train.num_examples/self.batch_size, self.lr_decay, staircase=True)
        train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss, global_step=global_step)

        if self.moving_average:
            variables_averages = tf.train.ExponentialMovingAverage(self.moving_average, global_step)
            variables_averages_op = variables_averages.apply(tf.trainable_variables())
            with tf.control_dependencies([train_op, variables_averages_op]):
                train_op = tf.no_op('train')

        test_feed = {self.x:dataset.test.images}
        validation_feed = {self.x:dataset.validation.images}

        self.sess.run(tf.global_variables_initializer())

        # self.saver = tf.train.Saver()

        # plt.figure()

        for i in range(self.steps):

            xs, _ = dataset.train.next_batch(self.batch_size)

            self.sess.run(train_op, feed_dict={self.x:xs})

            if i % 1000 == 0:

                loss_val = self.sess.run(loss, feed_dict=validation_feed)
                print('After training %d step(s), loss on validation '
                          'set is %g. ' %(i, loss_val))

        #         plt.plot(i, loss_val, 'b-*', linewidth=3.0, markersize=6.0, markerfacecolor='g')
        #         plt.hold(True)
        # plt.show()

        # self.saver.save(self.sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))

        loss_test = self.sess.run(loss, feed_dict=test_feed)
        print('After training %d step(s), loss on test '
                      'set is %g. ' % (i, loss_test))



    def extract(self, input_tensor):

        return self.sess.run(self.extractor, feed_dict={self.x:input_tensor})



    def generate(self, hidden):

        if hidden is None:
            biases = tf.trainable_variables()[1]
            hidden = np.random.normal(size=biases)

        return self.sess.run(self.reconstructor, feed_dict={self.extractor:hidden})



    def reconstruct(self, input_tensor):

        return self.sess.run(self.reconstructor, feed_dict={self.x:input_tensor})



    def getparameters(self):

        weights = self.sess.run(tf.trainable_variables()[0])
        biases = self.sess.run(tf.trainable_variables()[1])

        return weights, biases





    def show_extract_image(self, input):

        input_extract = self.extract(input).reshape(20,20)

        figure = plt.figure()
        figure_original = figure.add_subplot(121)
        figure_original.imshow(input.reshape((28, 28)), cmap=plt.cm.gray)
        figure_original.axis('off')
        figure_original.set_title('Original Image')
        figure_extract = figure.add_subplot(122)
        figure_extract.imshow(input_extract, cmap=plt.cm.gray)
        figure_extract.axis('off')
        figure_extract.set_title('Extracted Feature')

        figure.suptitle('Show Original Image and Extracted Feature')
        figure.show()



    def show_reconstruct_image(self, input):

        input_reconstruct = self.reconstruct(input).reshape((28, 28))

        figure = plt.figure()
        figure_original = figure.add_subplot(121)
        figure_original.imshow(input.reshape((28,28)), cmap=plt.cm.gray)
        figure_original.axis('off')
        figure_original.set_title('Original Image')
        figure_reconstruct = figure.add_subplot(122)
        figure_reconstruct.imshow(input_reconstruct, cmap=plt.cm.gray)
        figure_reconstruct.axis('off')
        figure_reconstruct.set_title('Reconstructed Image')

        figure.suptitle('Show Original Image and Reconstructed Image')
        figure.show()




    def main(self, argv=None):

        mnist = input_data.read_data_sets('./to/MNIST_data', one_hot=True)
        self.train(mnist)

        # self.saver.restore(self.sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))

        for i in range(5):
            number = int(np.random.randint(1, 10000, 1))
        ## show extract/reconstruct images
            # self.show_extract_image(mnist.test.images[number, :].reshape(1, -1))
            self.show_reconstruct_image(mnist.test.images[number, :].reshape(1, -1))

        # stochastic_image = np.random.rand(1, 784)
        # self.show_reconstruct_image(stochastic_image)


        ## validate accuracy!!!
        # re = self.reconstruct(mnist.test.images)
        # print(self.sess.run(tf.reduce_mean(tf.square(re - mnist.test.images))))
        # print(self.parameters()[0].shape, self.parameters()[1].shape)
        #
        # ex = tf.nn.softplus(tf.matmul(mnist.test.images, self.parameterts()[0]) + self.parameters()[1])
        # ex1 = self.extract(mnist.test.images)
        # print(self.sess.run(tf.reduce_mean(tf.square(ex - ex1))))

        self.sess.close()


if __name__ == '__main__':

    AGN = AdditiveGaussianNoiseAutoencoder(input_node=784, hidden_node=400, learning_rate=0.1, regularization_rate=None,scale=0.01, learning_rate_decay=0.99, exponential_moving_average=None,
                                            batch_size=100, activation_function=tf.nn.relu, steps=3000)
    AGN.main()





