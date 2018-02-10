import tensorflow as tf
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from skimage import io, transform

#parameters
batch = 1
epochs = 1000
frames = 10

#placeholders
enc_in = tf.placeholder()

def training():
    filename =

def read_data(filename):
    files = os.listdir(filename)
    inputs = []
    counter = 0
    random.shuffle(files)
    for f in files:
        inputs.append(transform.resize(io.imread(filename + '/' + f), (64,64,3), mode='constant'))
    inputs = np.asarray(imgs)

    return inputs

#encoder
with tf.variable_scope("Encoder") as scope:
# def encoder():
    lstm = tf.contrib.rnn.LSTMcell(num_units = 512)
    state = lstm.zero_state(batch,"float")

    for f in range(frames):
        if f > 0:
            scope.reuse_variables()
        output, state = lstm(enc_in[f], state)
        
#decoder
with tf.variable_scope("Decoder") as scope:
# def decoder():
    outputs = []
    zeros = np.zeros()#shape
    for f in range(frames):
        output, state = lstm(output, state)# state from encoder or new zero state?
        outputs.append(output)

with tf.variable_scope("FuturePredictor") as scope:
# def decoder():
    outputs = []
    zeros = np.zeros()#shape
    for f in range(frames):
        output, state = lstm(output, state)# state from encoder or new zero state?
        outputs.append(output)

#loss and optimization
with tf.variable_scope("Loss") as scope:

    loss = tf.reduced_sum(tf.square(outputs - Y))
    optim = tf.train.AdamOptimizer(lr).minimize(loss)



#starting the session
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    summary_writer = tf.summary.FileWriter(_,graph=sess.graph)

    for e in range(epochs):

        prediction = sess.run([pred],feed_dict = {})
