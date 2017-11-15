import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



def add_layer(inputs, input_size, output_size, activation_function=None):

  weights = tf.Variable(tf.random_normal([input_size, output_size]))
  biases = tf.Variable(tf.random_normal([output_size]) + 0.1)

  Wx_plus_b = tf.matmul(inputs, weights) + biases

  if activation_function is None:
    output = Wx_plus_b
  else:
    output = activation_function(Wx_plus_b)

  return output



x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_data, y_data)
plt.ylim([-1, 1])
plt.ion()
plt.show()


xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 16, activation_function=tf.nn.relu)
prediction = add_layer(l1, 16, 1, activation_function=None)

loss = tf.reduce_mean(tf.square(prediction-ys))
train_step = tf.train.GradientDescentOptimizer(5e-2).minimize(loss)

with tf.device('/gpu:0'):

  with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(10000):

      sess.run(train_step, feed_dict={xs:x_data, ys:y_data})

      if i % 400 == 0:

        try:
          ax.lines.remove(lines[0])
        except Exception:
          pass

        prediction_values = sess.run(prediction, feed_dict={xs:x_data})

        lines = ax.plot(x_data, prediction_values, 'r-', lw=2.)
        ax.set_title('Epoches: ' + str(i))

        plt.pause(.5)
