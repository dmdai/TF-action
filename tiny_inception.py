
import cifar10, cifar10_input
import tensorflow as tf
slim = tf.contrib.slim


batch_size = 100
steps = 1000
data_dir = './to/cifar-10-batches-bin'
logdir = './to/model'





class TinyInceptionConvolutionalNeuralNetwork():


    def __init__(self, num_classes, is_training = True, dropout_prob = 0.8,
                 learning_rate = 0.01):
        self.num_classes = num_classes
        self.is_training = is_training
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate


    def tiny_inception_arg_scope(self, weight_decay = 0.00004, stddev = 0.1,
                        batch_norm_var_collection = 'moving_vars'):


        batch_norm_params = {
            'decay': 0.9997,
            'epsilon': 0.001,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'variables_collections': {
                'beta': None,
                'gamma': None,
                'moving_mean': [batch_norm_var_collection],
                'moving_variance': [batch_norm_var_collection],
            }
        }


        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope(
                [slim.conv2d],
                weights_initializer = tf.truncated_normal_initializer(stddev=stddev),
                activation_fn = tf.nn.relu,
                normalizer_fn = slim.batch_norm,
                normalizer_params = batch_norm_params) as sc:

                    return sc



    def tiny_inception_base(self, input_tensor, scope=None):

        # end_points = {}

        with tf.variable_scope(scope, 'Inception', [input_tensor]):

            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride = 1, padding = 'VALID'):
                net = slim.conv2d(input_tensor, 32, [3, 3], scope='Conv2d_1a_3x3')
                net = slim.conv2d(net, 32, [3, 3], scope='Conv2d_2a_3x3')
                net = slim.max_pool2d(net, [3, 3], scope='MaxPool_3a_3x3')
                net = slim.conv2d(net, 64, [1,1], scope='Conv2d_3b_1x1')
                net = slim.conv2d(net, 128, [2, 2], stride=2,  scope='Conv2d_4a_3x3')
                net = slim.max_pool2d(net, [2, 2], scope='MaxPool_5a_2x2')

            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride = 1, padding = 'SAME'):
                with tf.variable_scope('Mixed_5b'):

                    with tf.variable_scope('Branch_0'):

                        branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')

                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 =slim.conv2d(branch_1, 64, [5, 5],
                                              scope='Conv2d_0b_5x5')

                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 96, [1, 3],
                                               scope='Conv2d_0b_3x3')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 1],
                                               scope='Conv2d_0c_3x3')

                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 5],
                                               scope='Conv2d_0b_3x3')
                        branch_3 = slim.conv2d(branch_3, 96, [5, 1],
                                               scope='Conv2d_0c_3x3')


                    with tf.variable_scope('Branch_4'):
                        branch_4 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_1x1')
                        branch_4 =slim.conv2d(branch_4, 32, [1, 1],
                                              scope='Conv2d_0b_1x1')

                    net = tf.concat([branch_0, branch_1, branch_2, branch_3, branch_4], 3)

        return net




    def tiny_inception(self, input_tensor, reuse=None, scope='Inception'):

        with tf.variable_scope(scope, 'Inception', [input_tensor, self.num_classes],
                               reuse=reuse) as scope:

            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=self.is_training):

                net = self.tiny_inception_base(input_tensor, scope=scope)

        with tf.variable_scope('logits'):

            nodes = net.get_shape().as_list()

            net = slim.avg_pool2d(net, [nodes[1], nodes[2]], padding='VALID',
                                  scope='AvgPool_1a_8x8')
            net = slim.dropout(net, keep_prob=self.dropout_prob, scope='Dropout_1b')
            logits = slim.conv2d(net, self.num_classes, [1, 1], activation_fn=None,
                                 normalizer_fn=None, scope='Conv2d_1c_1x1')
            logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
            predictions = slim.softmax(logits, scope='Predictions')

        return logits, predictions





    def train_tiny_inception(self, input_tensor, labels, reuse=None, scope='Inception'):


        with tf.variable_scope(scope, 'Inception', [input_tensor, self.num_classes],
                               reuse=reuse) as scope:

            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=self.is_training):

                net = self.tiny_inception_base(input_tensor, scope=scope)

        with tf.variable_scope('logits'):

            nodes = net.get_shape().as_list()

            net = slim.avg_pool2d(net, [nodes[1], nodes[2]], padding='VALID',
                                  scope='AvgPool_1a_8x8')
            net = slim.dropout(net, keep_prob=self.dropout_prob, scope='Dropout_1b')
            logits = slim.conv2d(net, self.num_classes, [1, 1], activation_fn=None,
                                 normalizer_fn=None, scope='Conv2d_1c_1x1')
            logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
            # predictions = slim.softmax(logits, scope='Predictions')


        labels = tf.one_hot(labels, self.num_classes, 1, 0)
        loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
        loss += tf.add_n(tf.losses.get_regularization_losses())

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        train_op = slim.learning.create_train_op(loss, optimizer)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction), tf.float32)

        slim.learning.train(train_op, logdir, number_of_steps=steps)

















TNCNN = TinyInceptionConvolutionalNeuralNetwork(num_classes=10,
                is_training = False, dropout_prob = 0.8, learning_rate=0.01)

images_train, labels_train = cifar10_input.distorted_inputs(
    data_dir=data_dir, batch_size=batch_size)
with slim.arg_scope(TNCNN.tiny_inception_arg_scope()):
    TNCNN.train_tiny_inception(images_train, labels_train)


# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# pred = sess.run(predictions)
# sess.close()




# images_train, labels_train = cifar10_input.distorted_inputs(
#     data_dir=data_dir, batch_size=batch_size)

# images_test, labels_test = cifar10_input.inputs(eval_data=True,
#                                                 data_dir=data_dir, batch_size=batch_size)

































