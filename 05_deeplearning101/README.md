# Deep Learning 101 Labs

This directory contains neural network implementations for the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) database using the [TensorFlow](https://www.tensorflow.org/) Python API:

- `mnist_tf_basic.py`: basic 2-layer network with 16 ReLU hidden units
- `mnist_tf_fancy.py`: fancier 2-layer network with command-line options to customize the network configuration (use `-h` to see what's available) as well as output visualizations


## Suggested exercises:

Here are several lab exercises designed to get you used to doing more with TensorFlow and deep learning in general. 


### 1. Add more hidden layers.

Using `mnist_tf_fancy.py` as a starting point, modify the `-n` option to take a period-delimited list of numbers that specify the layer structure. For example:

- `-n 0` trains a simple [softmax regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression) (i.e., no hidden layers, inputs are directly connected to the output softmax layer)
- `-n 32` trains a 2-layer network with 32 hidden units (identical to current implementation)
- `-n 258.128.64.32` trains a 5-layer network with hidden layers of size 258, 128, 63, and 32 units respectively. The inputs connect to the first hidden layer (258), and the last hidden layer (32) connects to the output layer.

_Hint:_ Try using a list of `tf.Variable` objects to hold the weights and biases of all the layers in one data structure. :)


### 2. Train from a data in a CSV file.

Further modify the script to not use the MNIST data set. Instead, add a command line argument for a CSV file where the first column is the output label, and all other columns are input features. If a CSV file has boolean (i.e., TRUE/FALSE) or categorical features (i.e., strings) you will have to figure out how to convert those to real number values for the tensor representation (e.g., `float32`).

_Hint:_ Boolean columns can simply be coded as `1.0`/`0.0` for TRUE/FALSE. Categorical columns (with _N_ possibilities) could be split up into _N_ boolean columns. For example, if `column1` can be `A`, `B`, or `C`, you might replace it with three new columns `column1_A`, `column1_B`, `column1_C`, and set the value to `1.0` for the appropriate column for the input tensor. This is the "one-hot vector" idea, and is related to coding with [indicator variables](https://en.wikipedia.org/wiki/Dummy_variable_(statistics)) in statistics.

Take note of the label type (first column): real numbers, binary, or non-binary classification. Be sure to use the appropriate loss function to minimize for the detected label type!


### 3. Experiment with your own data set!

Create a CSV file of data relevant to your Duolingo team or job that you think is interesting. That is, you want to predict some value (`T`, in the first column) as a function of various input variables (`X`, in the remaining columns).

Experiment with different network architectures (layers, activation functions, optimizers, regularization, etc.). Feel free to share your results! I might present them in the next Deep Learning talk....
