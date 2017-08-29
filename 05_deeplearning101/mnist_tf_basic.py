"""
Burr Settles
Duolingo ML Dev Talk #5: Deep Learning 101

Train a simple 2-Layer neural network on MNIST data.
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

# download images + labels into mnist.train (10k) and mnist.test (60k)
mnist = mnist_data.read_data_sets("mnist_data", one_hot=True,
  reshape=False, validation_size=0)

# placeholder for input images (? x 28 x 28 x grayscale)
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# placeholder for output labels (? x 10-label "one-hot" vector)
T = tf.placeholder(tf.float32, [None, 10])
# placeholder for the learning rate
learning_rate = tf.placeholder(tf.float32)

# weights: initialize to small random values
HW = tf.Variable(tf.truncated_normal([28*28, 16], stddev=0.01))
OW = tf.Variable(tf.truncated_normal([16, 10], stddev=0.01))
# ReLU (hidden) biases: initialize to small positive values
Hb = tf.Variable(tf.ones([16])/20.)
# softmax (output) biases: initialize to zeros
Ob = tf.Variable(tf.zeros([10]))

# computation graph
X_ = tf.reshape(X, [-1, 28*28])           # flatten image data
HY = tf.nn.relu(tf.matmul(X_, HW) + Hb)   # ReLU activation
Ylogits = tf.matmul(HY, OW) + Ob
Y = tf.nn.softmax(Ylogits)                # softmax activation

# cross-entropy loss function
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=T)
loss = tf.reduce_sum(loss)

# optimization method
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# human-interpretable error metric :)
wrong = tf.not_equal(tf.argmax(Y, 1), tf.argmax(T, 1))
error = tf.reduce_mean(tf.cast(wrong, tf.float32))

# set up the TensorFlow session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# train + evaluate the model
n_steps = 10000
batch_size = 128
for step in range(n_steps+1):

  batch_X, batch_Y = mnist.train.next_batch(batch_size)
  sess.run(optimizer, {X: batch_X, T: batch_Y, learning_rate: 0.0001})

  # print diagnostic info every now and then...
  if step % 1000 == 0:
    trn_loss, trn_err = sess.run([loss, error], {X: batch_X, T: batch_Y})
    tst_err = sess.run(error, {X: mnist.test.images, T: mnist.test.labels})
    print '(%d) loss=%.4f train=%.4f test=%.4f' % (step, trn_loss, trn_err, tst_err)
