# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 12:51:18 2018

@author: Huy Pham
"""

import tensorflow as tf
import numpy as np
import pickle
#import random
#import math
import matplotlib.pyplot as plt
#from skimage import io, transform

tf.reset_default_graph()

#global variables
train_input_seq = [None]
train_output_seq = [None]
train_future_seq = [None]
test_input_seq = [None]
test_output_seq = [None]
test_future_seq = [None]
channel = 3

def load_data():
    global train_input_seq, train_output_seq, train_future_seq, test_input_seq, test_output_seq, test_future_seq

    data = np.load( 'datasets/ucf101_sample_train_patches.npy' )
    # ['clips', 'dims', 'input_raw_data']
    #(200K, 1, 64, 64) --> (10K, 20, 64, 64)-> number of sequences, frames/sequence, height, width
    
    data = np.swapaxes(data,2,3)
    data = np.swapaxes(data,3,4)
    print("loading training data: data.shape", np.shape(data))
    train_input_seq = data[0:8900, 0:10]
    train_output_seq = train_input_seq[0:, ::-1]
    train_future_seq = data[0:8900, 10:]

    test_input_seq = data[8900:, 0:10]
    test_output_seq = test_input_seq[0:, ::-1]
    test_future_seq = data[8900:,10:]
    
    return

load_data()

#parameters
batch_size = 64
epochs = 100
frames = 10
lr = 0.001

#placeholders
#enc_in = tf.placeholder()
X = tf.placeholder(tf.float32, [batch_size, frames, 32, 32, channel])
Y = tf.placeholder(tf.float32, [batch_size, frames, 32, 32, channel])
Y_future = tf.placeholder(tf.float32, [batch_size, frames, 32, 32, channel])

#
c0 = tf.reshape(X,[-1, 64, 64, channel])
c0 = tf.layers.conv2d(inputs = c0,filters=24,kernel_size=3,activation = tf.nn.relu, strides=(2,2), \
kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "d_conv0")

c1 = tf.layers.conv2d(inputs = c0,filters=24,kernel_size=3,activation = tf.nn.relu, strides=(2,2), \
kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "d_conv1")

c2 = tf.layers.conv2d(inputs = c1,filters=64,kernel_size=3,activation = tf.nn.relu, strides=(2,2),\
kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "d_conv2")

c3 = tf.layers.conv2d(inputs = c2,filters=64,kernel_size=3,activation = tf.nn.relu, strides=(2,2), \
kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "d_conv3")

c4 = tf.layers.conv2d(inputs = c3,filters=64,kernel_size=3,activation = tf.nn.relu, strides=(2,2),\
kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "d_conv4")

c5 = tf.layers.conv2d(inputs = c4,filters=64,kernel_size=3,activation = tf.nn.relu, strides=(2,2), \
kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "d_conv5")

def fully_connected(input, reuse=False):
    with tf.variable_scope("fc", reuse=reuse):
        out = tf.contrib.layers.fully_connected(input, 3072, activation_fn=None)
        out = tf.reshape(out, [batch_size, frames, -1])

    return out

inp = tf.reshape(c5, [batch_size, frames, -1])

#encoder
with tf.variable_scope("Encoder") as scope:
# def encoder():
    lstm = tf.contrib.rnn.LSTMCell(num_units = 1024)
    state = lstm.zero_state(batch_size,"float")
    datum = tf.split(inp, frames, axis = 1)

    for f in range(frames):
        if f > 0:
            scope.reuse_variables()
        output, state = lstm(tf.reshape(datum[f], [batch_size,-1]), state)
    output_decoder = output
    state_decoder = state
    output_future = output
    state_future = state

#decoder
zero_input = tf.zeros_like(tf.reshape( datum[0], [batch_size, -1] ) , "float" ) # generate a zero array using the shape of tmp

with tf.variable_scope("Decoder") as scope:
    decoder_outputs = []
    
    for f in range(frames):
        if f > 0:
            scope.reuse_variables()
        output_decoder, state_decoder = lstm(zero_input, state_decoder)
        decoder_outputs.append(output_decoder)
    decoder_outputs = tf.reshape(decoder_outputs,[batch_size*frames,-1])
decoder_outputs = fully_connected(decoder_outputs)

with tf.variable_scope("FuturePredictor") as scope:
    future_outputs = []
    
    for f in range(frames):
        if f > 0:
            scope.reuse_variables()
        output_future, state_future = lstm(zero_input, state_future)
        future_outputs.append(output_future)
    future_outputs = tf.reshape(future_outputs, [batch_size*frames, -1])
future_outputs = fully_connected(future_outputs, True)

#loss and optimization
#with tf.variable_scope("Loss") as scope:
target = tf.reshape(Y, [batch_size, frames, -1])
target_future = tf.reshape(Y_future, [batch_size, frames, -1])

loss = tf.reduce_mean(tf.reduce_sum(tf.square(target - decoder_outputs), axis=[1,2]))  #l2 loss
loss_future = tf.reduce_mean(tf.reduce_sum(tf.square(target_future - future_outputs),axis = [1,2]))
optim = tf.train.AdamOptimizer(lr).minimize(loss)
optim_future = tf.train.AdamOptimizer(lr).minimize(loss_future)

#starting the session
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    # summary_writer = tf.summary.FileWriter(_,graph=sess.graph)
    iteration = []
    avg_losses = []
    avg_losses_future = []
    img_pre = [None]
    img = [None]
    img_future = [None]

    for e in range(epochs):
        print(e)
        iteration.append(e)
        num_batches = int(len(train_input_seq)/batch_size)
        total_loss = 0
        total_loss_future = 0

        for i in range(num_batches):
            x_train = train_input_seq[i*batch_size:(i+1)*batch_size]

            y_train = train_output_seq[i*batch_size:(i+1)*batch_size]
            y_train_future = train_future_seq[i*batch_size:(i+1)*batch_size]
            batch_loss,batch_loss_future,train_optim,train_optim_future = sess.run([loss,loss_future,optim, optim_future],\
                feed_dict = {X: x_train, Y: y_train, Y_future: y_train_future})

            total_loss += batch_loss
            total_loss_future += batch_loss_future

        avg_losses.append(total_loss/num_batches)
        avg_losses_future.append(total_loss_future/num_batches)

    img_pre = test_input_seq[0:batch_size]
    tar = test_output_seq[0:batch_size]
    tar_future = test_future_seq[0:batch_size]
    img, img_future = sess.run([decoder_outputs, future_outputs], feed_dict = {X: img_pre})
    
print(iteration)
print(avg_losses)
print(avg_losses_future)

#plotting the loss
fig = plt.figure()
plt.plot(iteration, avg_losses)
plt.ylabel("losses")
plt.xlabel("epoch")
plt.show()
fig.savefig('ucf_decoder.png')

fig = plt.figure()
plt.plot(iteration, avg_losses_future)
plt.ylabel("losses")
plt.xlabel("epoch")
plt.show()
fig.savefig('ucf_decoder_future.png')

img_pre = np.reshape( img_pre, [batch_size, frames, 32, 32, channel] )
tar = np.reshape(tar, [batch_size, frames, 32, 32, channel])
tar_future = np.reshape(tar_future, [batch_size, frames, 32, 32, channel])
img = np.reshape( img, [batch_size, frames, 32, 32, channel])
img_future = np.reshape( img_future, [batch_size, frames, 32, 32, channel])

source = []
reverse = []
future = []
tar_reverse = []
tar_future_out = []
for t in range(frames):
    source.append(img_pre[1,t])
    tar_reverse.append(tar[1,t])
    tar_future_out.append(tar_future[1,t])
    reverse.append(img[1,t])
    future.append(img_future[1,t])

source = np.concatenate(source)
tar_reverse = np.concatenate(tar_reverse)
tar_future_out = np.concatenate(tar_future_out)
reverse = np.concatenate(reverse)
future = np.concatenate(future)

pic_reverse = np.rint(np.clip(reverse, 0, 255))
pic_future = np.rint(np.clip(reverse, 0, 255))

plt.imsave("ucf_Source.png", source)
plt.imsave("ucf_Target_Reverse.png", tar_reverse)
plt.imsave("ucf_Target_Future.png", tar_future_out)
plt.imsave("ucf_Reverse.png", pic_reverse.astype(np.uint8))
plt.imsave("ucf_Future.png", pic_future.astype(np.uint8))
