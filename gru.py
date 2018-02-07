import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell
# import tensorflow


class mygru(RNNCell):
    def __init__(self, num_unit):
        self.num_unit = num_unit

    @property
    def state_size(self):
        return self.num_unit

    @property
    def output_size(self):
        return self.num_unit

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            initializer = tf.contrib.layers.variance_scaling_initializer()
            vocab_size = inputs.get_shape().as_list()[1]
            W_z = tf.get_variable("W_z", [vocab_size, self.state_size], initializer=initializer)
            U_z = tf.get_variable("U_z", [self.state_size, self.state_size], initializer=initializer)
            B_z = tf.get_variable("B_z", [self.state_size], initializer=initializer)
            W_r = tf.get_variable("W_r", [vocab_size, self.state_size], initializer=initializer)
            U_r = tf.get_variable("U_r", [self.state_size, self.state_size], initializer=initializer)
            B_r = tf.get_variable("B_r", [self.state_size], initializer=initializer)
            W_h = tf.get_variable("W_h", [vocab_size, self.state_size], initializer=initializer)
            U_h = tf.get_variable("U_h", [self.state_size, self.state_size], initializer=initializer)
            B_h = tf.get_variable("B_h", [self.state_size], initializer=initializer)

            z = tf.sigmoid(tf.nn.bias_add(tf.matmul(inputs, W_z) + tf.matmul(state, U_z), B_z))
            r = tf.sigmoid(tf.nn.bias_add(tf.matmul(inputs, W_r) + tf.matmul(state, U_r), B_r))
            h_twiddle = tf.tanh(tf.nn.bias_add(tf.matmul(inputs, W_h) + tf.matmul(r * state, U_h), B_h))
            h = z * state + (1 - z) * h_twiddle

        return h, h
