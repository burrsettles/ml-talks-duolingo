"""
Burr Settles
Duolingo ML Dev Talk #5: Deep Learning 101

Train a 2-Layer neural network on MNIST data.
Several config options + writes interpretable visualization images.
"""

import argparse
import math
import os

import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data as mnist_data


print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

argparser = argparse.ArgumentParser()
argparser.add_argument('-a', action="store", dest="act_function", default='relu', help="hidden unit activation functions (relu [default], sigmoid, tanh)")
argparser.add_argument('-b', action="store", dest="batch_size", default=128, help="number of training items per mini-batch [128]")
argparser.add_argument('-l', action="store", dest="learning_rate", type=float, default=0.001, help="learning rate [0.001]")
argparser.add_argument('-n', action="store", dest="n_hidden_units", type=int, default=16, help="number of hidden units [16]")
argparser.add_argument('-o', action="store", dest="optimizer", default='sgd', help="optimizer (sgd [default], adagrad, adam)")
argparser.add_argument('-p', action="store", dest="p_keep", type=float, default=1., help="probability of firing for dropout [1.]")

N_IMG_COLS = 8    # columns (for visualization stacking)
N_TEST_IMGS = 25  # test images to visualize


def prepare_for_image(array):
    mn = np.amin(array)
    mx = np.amax(array)
    array = np.multiply(np.subtract(array, mn), 256./(mx-mn))
    outimgs = []
    for h in range(array.shape[1]):
        outimgs.append(np.asarray(array[:,h]).reshape(28, 28))
    cols = []
    for r in range(0, array.shape[1], N_IMG_COLS):
        cols.append(np.vstack([a for a in outimgs[r:r+N_IMG_COLS]]))
    return np.hstack([a for a in cols])


if __name__ == '__main__':

    args = argparser.parse_args()

    # download images + labels into mnist.train (10k) and mnist.test (60k)
    mnist = mnist_data.read_data_sets("mnist_data", one_hot=True, reshape=False, validation_size=0)

    # create visualizations directory + raw images...
    viz_dir = 'mnist_viz_%d_%s_%s_%.2f' % (args.n_hidden_units, args.act_function, args.optimizer, args.p_keep)
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    for i in range(N_TEST_IMGS):
        inst = mnist.test.images[i]
        label_ = np.argmax(mnist.test.labels[i])
        test_img = Image.fromarray(np.multiply(inst.reshape(28, 28), 256.))
        test_img.convert('RGB').save('%s/test%d_%d.png' % (viz_dir, label_, i), 'PNG')
        # test_img.show()

    # print setup info
    print '-'*80
    print 'activation function: ', args.act_function
    print 'training batch size: ', args.batch_size
    print 'learning rate:       ', args.learning_rate
    print '# hidden units:      ', args.n_hidden_units
    print 'optimizer:           ', args.optimizer
    print 'dropout p_keep:      ', args.p_keep
    print 'writing visualizations to "%s"' % viz_dir
    print '-'*80

    # placeholder for input images (? x 28 x 28 x grayscale)
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    # placeholder for output labels (? x 10-label "one-hot" vector)
    T = tf.placeholder(tf.float32, [None, 10])
    # probability of keeping a node during dropout
    p_keep = tf.placeholder(tf.float32)
    # number of hidden units
    n_units = args.n_hidden_units

    # variables: weights + biases
    HW = tf.Variable(tf.truncated_normal([28*28, n_units], stddev=0.01))
    HB = tf.Variable(tf.zeros([n_units]))
    # for RELUs, hidden layer biases should actually be a small positive value
    if args.act_function == 'r':
        HB = tf.Variable(tf.ones([n_units])/10)
    OW = tf.Variable(tf.truncated_normal([n_units, 10], stddev=0.01))
    OB = tf.Variable(tf.zeros([10]))
    all_params = tf.concat([tf.reshape(HW, [-1]), tf.reshape(HB, [-1]), tf.reshape(OW, [-1]), tf.reshape(OB, [-1])], 0)

    # computation graph: activations + connections
    X_ = tf.reshape(X, [-1, 28*28])
    HWXB = tf.matmul(X_, HW) + HB
    if args.act_function == 'sigmoid':
        HY = tf.nn.sigmoid(HWXB)
    elif args.act_function == 'tanh':
        HY = tf.nn.tanh(HWXB)
    else:
        HY = tf.nn.relu(HWXB)
    HYd = tf.nn.dropout(HY, p_keep)

    # output layer activations + connections
    Ylogits = tf.matmul(HYd, OW) + OB
    Y = tf.nn.softmax(Ylogits)

    # cross-entropy loss function + L2-norm regularization
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=T)
    loss = loss + 0.01 * tf.nn.l2_loss(all_params)
    loss = tf.reduce_sum(loss)

    # optimizer
    if args.optimizer == 'adagrad':
        optimizer = tf.train.AdaGradOptimizer(args.learning_rate).minimize(loss)
    elif args.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)
    else:
        optimizer = tf.train.GradientDescentOptimizer(args.learning_rate).minimize(loss)

    # human-interpretable error metric :)
    wrong = tf.not_equal(tf.argmax(Y, 1), tf.argmax(T, 1))
    error = tf.reduce_mean(tf.cast(wrong, tf.float32))

    # initialize
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # run the model
    n_steps = 10 * args.batch_size * n_units
    exp_ct = 0
    logfile = open('%s/logfile.txt' % viz_dir, 'w')
    for step in range(n_steps):

        batch_X, batch_Y = mnist.train.next_batch(args.batch_size)
        sess.run(optimizer, {X: batch_X, T: batch_Y, p_keep: args.p_keep})

        # dump some diagnostic information every now and then
        if (step % 2**exp_ct == 0 or step == n_steps-1):
            exp_ct += 1

            trna, trnc = sess.run([error, loss], {X: batch_X, T: batch_Y, p_keep: 1.0})
            tsta, tstc, preds, actives = sess.run([error, loss, Y, HY], {X: mnist.test.images, T: mnist.test.labels, p_keep: 1.0})
            logfile.write('==== %d ================================\n' % step)
            logfile.write('%d lr=%.4f loss=%.4f train=%.4f test=%.4f\n' % (step, args.learning_rate, trnc, trna, tsta))
            print '%d lr=%.4f loss=%.4f train=%.4f test=%.4f' % (step, args.learning_rate, trnc, trna, tsta)

            # visualize out the weights matrix
            hidden_arr = tf.reshape(HW, [28*28, n_units])
            hidden_arr_img = Image.fromarray(prepare_for_image(np.asarray(sess.run(hidden_arr))))
            hidden_arr_img.convert('RGB').save('%s/%d_hidden_weights.png' % (viz_dir, step), 'PNG')
            # hidden_arr_img.show()

            output_arr = sess.run(tf.reshape(OW, [n_units, 10]))
            for label_ in range(10):
                h_o_arr = np.asarray(sess.run(tf.reshape(HW, [28*28, n_units])))
                t_weights = output_arr[:,label_]
                h_o_arr = np.multiply(h_o_arr, t_weights.reshape([-1, n_units]))
                h_o_arr_img = Image.fromarray(prepare_for_image(h_o_arr))
                h_o_arr_img.convert('RGB').save('%s/%d_output_weights.%d_%d.png' % (viz_dir, step, label_, i), 'PNG')
                # h_o_arr_img.show()

            # visualize predictions for the first few test images
            for i in range(N_TEST_IMGS):

                inst = mnist.test.images[i]
                t = mnist.test.labels[i]
                label_ = np.argmax(t)
                y = preds[i]
                label = np.argmax(y)
                t_weights = output_arr[:,label]
                a = actives[i]

                logfile.write('---- test item %d ----\n' % i)
                logfile.write('true:%d / pred:%d\n' % (label_, label))
                logfile.write('predictions:   %s\n' % str([round(x, 2) for x in y]))
                logfile.write('activations:   %s\n' % str([round(x, 2) for x in a]))
                logfile.write('output weights:%s\n' % str([round(x, 2) for x in t_weights]))

                active_arr = np.asarray(sess.run(tf.reshape(HW, [28*28, n_units])))
                active_arr = np.multiply(active_arr, a.reshape([-1, n_units]))
                active_arr = np.multiply(active_arr, t_weights.reshape([-1, n_units]))
                active_arr_img = Image.fromarray(prepare_for_image(active_arr))
                active_arr_img.convert('RGB').save('%s/%d_activation_weights.%d_%d_%d.png' % (viz_dir, step, label_, i, label), 'PNG')
                # active_arr_img.show()

    logfile.close()
