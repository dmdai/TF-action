

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt



def xavier_init(fan_in, fan_out):

    max = np.sqrt(6/(fan_in + fan_out))
    return  tf.random_uniform([fan_in, fan_out], maxval=max, minval=-max, dtype=tf.float32)


class NeuralNetwork():

    def __init__(self, input_node, hidden_node, labels, learning_rate, regularization_rate,
                 learning_rate_decay, moving_average_decay, batch_size, steps, activation_function):

        self.n_input = input_node
        self.n_hidden = hidden_node
        self.n_output = labels
        self.learning_rate = learning_rate
        self.regularizer = regularization_rate
        self.lr_decay = learning_rate_decay
        self.moving_average = moving_average_decay
        self.batch_size = batch_size
        self.steps = steps
        self.activation_function = activation_function
        self.sess = tf.InteractiveSession()



    def get_parameters(self, shape):

        weights = tf.Variable(xavier_init(shape[0], shape[1]))
        biases = tf.get_variable('biases', shape[1], initializer=tf.constant_initializer(0.1))

        if self.regularizer:

            regularization = tf.contrib.layers.l2_regularizer(self.regularizer)
            tf.add_to_collection('losses',regularization(weights))

        return weights, biases


    def inference(self, input_tensor):

        parameters = []

        with tf.variable_scope('layer1'):

            weights, biases = self.get_parameters([self.n_input, self.n_hidden])

            hidden_layer = self.activation_function(tf.matmul(input_tensor, weights) + biases)

            parameters += [weights, biases]

        with tf.variable_scope('layer2'):

            weights, biases = self.get_parameters([self.n_hidden, self.n_output])

            out = tf.matmul(hidden_layer, weights) + biases

        # return out
        return out, parameters



    def train(self, datasets):

        x = tf.placeholder(tf.float32, [None, self.n_input], name='x-input')
        y = tf.placeholder(tf.float32, [None, self.n_output], name='y-input')

        # out = self.inference(x)
        out, self.parameters = self.inference(x)

        global_step = tf.Variable(0, trainable=False)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y, 1), logits=out)
        loss = tf.reduce_mean(cross_entropy)
        if self.regularizer:
            loss += tf.add_n(tf.get_collection('losses'))

        if self.lr_decay:
            self.learning_rate = tf.train.exponential_decay(self.learning_rate, global_step,
                                                            datasets.train.num_examples/self.batch_size, self.lr_decay)
        train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss, global_step=global_step)

        if self.moving_average:
            variables_averages = tf.train.ExponentialMovingAverage(self.moving_average, global_step)
            variables_averages_op = variables_averages.apply(tf.trainable_variables())
            with tf.control_dependencies([train_op, variables_averages_op]):
                train_op = tf.no_op(name='train')

        corrct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(corrct_prediction, tf.float32))

        validation_feed = {x:datasets.validation.images, y:datasets.validation.labels}
        test_feed = {x:datasets.test.images, y:datasets.test.labels}

        self.sess.run(tf.global_variables_initializer())

        for i in range(self.steps):

            xs, ys = datasets.train.next_batch(self.batch_size)

            self.sess.run(train_op, feed_dict={x:xs, y:ys})

            if i % 1000 == 0:

                loss_value, accuracy_val = self.sess.run([loss, accuracy], feed_dict=validation_feed)
                print('After %d training step(s), validation loss is %g, '
                      'validation accuracy = %g ' %(i, loss_value, accuracy_val))

        accuracy_test = self.sess.run(accuracy, feed_dict=test_feed)
        print('\n\nAfter all training steps, accuracy on test '
                  'datasets is %g ' %accuracy_test)
        print('\n\n\nTraining Done!!!\n\n')



    def show_features(self, datasets):

        input = np.reshape(datasets.test.images[np.random.randint(1, 10000)], (1, 784))
        input_extract = self.activation_function(tf.matmul(input, self.parameters[0]) + self.parameters[1])
        input_extract = self.sess.run(input_extract).reshape(20, 20)

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


    def main(self):

        mnist = input_data.read_data_sets('./to/MNIST_data', one_hot=True)
        self.train(mnist)

        self.show_features(mnist)

        self.sess.close()



if __name__ == '__main__':

    NN = NeuralNetwork(input_node=784, hidden_node=400, labels=10, learning_rate=0.05,
                       regularization_rate=0.0001, learning_rate_decay=0.99, moving_average_decay=None,
                       batch_size=100, steps=50000, activation_function=tf.nn.relu)
    NN.main()


