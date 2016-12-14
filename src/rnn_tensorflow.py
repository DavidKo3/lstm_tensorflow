'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN( X, weights, biases):
    
    # Prepare data shape to match 'rnn' function requirements
    # Current data input shape: (batch_size , n_steps, n_input)
    # Required shape : 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Permuting batch_size and n_steps
    X = tf.transpose(x, [1, 0 , 2]) # ( n_steps, batch_size , n_input)
    # Reshaping to (n_steps*batch_size , n_input)
    X = tf.reshape(x , [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    X = tf.split(0, n_steps, X)
   
   # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    
    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell,  x, dtype = tf.float32)
    
    
    
    
    