import tensorflow as tf
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from skimage import io, transform

#global variables
train_input_seq = [None]
train_output_seq = [None]
test_input_seq = [None]
test_output_seq = [None]


#parameters
batch = 1
epochs = 1000
frames = 10

#placeholders
#enc_in = tf.placeholder()
X = tf.placeholder(dtype = tf.float32)
Y = tf.placeholder(dtype = tf.float32)

c0 = tf.layers.conv2d(inputs = X,filers=64,kernel_size=3,activation = tf.nn.relu\
kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "d_conv0")

c1 = tf.layers.conv2d(inputs = c0,filers=64,kernel_size=3,activation = tf.nn.relu\
kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "d_conv1")

c2 = tf.layers.conv2d(inputs = c1,filers=64,kernel_size=3,activation = tf.nn.relu\
kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "d_conv2")

fc0 = tf.contrib.layers.fully_connected(c2, 1, activation_fn=None)

def load_data():
    global train_input_seq, train_output_seq, test_input_seq, test_output_seq

    data = np.load( '../data/mnist_test_seq.npy' )
    # ['clips', 'dims', 'input_raw_data']
    #(200K, 1, 64, 64) --> (10K, 20, 64, 64)-> number of sequences, frames/sequence, height, width
    data = np.reshape( data, [-1, 20, 64, 64, 1] )
    print("loading training data: data.shape", data.shape)
    train_input_seq = data[0:8000, 0:10]
    train_output_seq = data[0:8000, 10:]
    test_input_seq = data[8000:, 0:10]
    test_output_seq = data[8000:, 10:]

    return


#encoder
with tf.variable_scope("Encoder") as scope:
# def encoder():
    lstm = tf.contrib.rnn.LSTMcell(num_units = 512)
    state = lstm.zero_state(batch,"float")

    for f in range(frames):
        if f > 0:
            scope.reuse_variables()
        output, state = lstm(X[f], state)
    output_decoder = output
    state_decoder = state
    output_future = output
    state_future = state

#decoder
with tf.variable_scope("Decoder") as scope:
    decoder_outputs = []

    for f in range(frames):
        output_decoder,state_decoder = lstm(output_decoder, state_decoder)
        decoder_outputs.append(output_decoder)

with tf.variable_scope("FuturePredictor") as scope:
    future_outputs = []

    for f in range(frames):
        output_future, state_future = lstm(output_future, state_future)
        future_outputs.append(output_future)

#loss and optimization
with tf.variable_scope("Loss") as scope:

    loss = tf.reduced_sum(tf.square(decoder_outputs - Y))
    optim = tf.train.AdamOptimizer(lr).minimize(loss)

load_data()

#starting the session
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    summary_writer = tf.summary.FileWriter(_,graph=sess.graph)

    for e in range(epochs):

        prediction = sess.run([pred],feed_dict = {})
