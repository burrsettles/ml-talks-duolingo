# Deep Learning 101 Labs

This directory contains neural network implementations for the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) database using the [TensorFlow](https://www.tensorflow.org/) Python API:

- `mnist_tf_basic.py`: basic 2-layer network with 16 ReLU hidden units
- `mnist_tf_fancy.py`: fancier 2-layer network with several command-line options (use `-h` to see options) and output visualizations


## Suggested exercises:

Here are several lab exercises designed to get you used to doing more with TensorFlow and deep learning in general.


### 1. Add more hidden layers.

Using `mnist_tf_fancy.py` as a starting point, modify the `-n` option to take a comma-delimited list of numbers that specify the layer structure. For example:

- `-n 0` trains a simple [softmax regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression) (i.e., no hidden layers, inputs are fully-connected with the output softmax layer)
- `-n 32` trains a 2-layer network with 32 hidden units (identical to current implementation)
- `-n 258,128,64,32` trains a 5-layer network with layers of size 258, 128, 63, and 32 hidden units respectively. The input layer connects to the first hidden layer (258 units) and the last hidden layer (32 units) connects to the output layer.

_Hint:_ try creating a list of `tf.Variable`s to hold the weights and biases of all the layers. :)


### 2. Train from a CSV data set.

Further modify the script to not use the hard-coded, built-in MNIST data set. Instead, add a CSV file command line argument where the first column is the output label, and all other columns are input features. If a CSV file has boolean (i.e., TRUE/FALSE) or cateogiral features (i.e., strings) you will have to figure out how to convert those to real values for the input tensor representation (e.g., `float32`).

Take note of the type of the first column (label): real numbers, binary classification, or multiclass. Be sure to use the appropriate loss function to minimize for each type.


### 3. Experiment with your own data set!

Create a CSV file of data relevant to your job at Duolingo that you think is interesting. That is, you want to predict some value (`T`) as a function of various input variables (`X`).

Experiment with different network architectures (layers, activation functions, optimizers, regularization, etc.). Feel free to share your results and I might present them in the next Deep Learning talk....