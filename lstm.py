# import tensorflow
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.ops import array_ops

class mylstm(RNNCell):
    def __init__(self, num_unit):
        self.num_unit = num_unit

    @property
    def state_size(self):
        return 2 * self._num_units

    @property
    def output_size(self):
        return self.num_unit

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            initializer = tf.contrib.layers.variance_scaling_initializer()
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)
            input_size = inputs.get_shape().as_list()[1]
            
            W_if = tf.get_variable("W_if", [input_size, self.num_unit], initializer=initializer)
            B_if = tf.get_variable("B_if", [self.num_unit], initializer=initializer)
            W_hf = tf.get_variable("W_hf", [self.num_unit, self.num_unit], initializer=initializer)
            B_hf = tf.get_variable("B_hf", [self.num_unit], initializer=initializer)

            W_ii = tf.get_variable("W_ii", [input_size, self.num_unit], initializer=initializer)
            B_ii = tf.get_variable("B_ii", [self.num_unit], initializer=initializer)
            W_hi = tf.get_variable("W_hi", [self.num_unit, self.num_unit], initializer=initializer)
            B_hi = tf.get_variable("B_hi", [self.num_unit], initializer=initializer)
			
			W_ig = tf.get_variable("W_ig", [input_size, self.num_unit], initializer=initializer)
            B_ig = tf.get_variable("B_ig", [self.num_unit], initializer=initializer)
            W_hg = tf.get_variable("W_hg", [self.num_unit, self.num_unit], initializer=initializer)
            B_hg = tf.get_variable("B_hg", [self.num_unit], initializer=initializer)

			W_io = tf.get_variable("W_io", [input_size, self.num_unit], initializer=initializer)
            B_io = tf.get_variable("B_io", [self.num_unit], initializer=initializer)
            W_ho = tf.get_variable("W_ho", [self.num_unit, self.num_unit], initializer=initializer)
            B_ho = tf.get_variable("B_ho", [self.num_unit], initializer=initializer)
            
            f = tf.sigmoid(tf.nn.bias_add(tf.matmul(inputs, W_if), B_if) + tf.nn.bias_add(tf.matmul(h, W_hf), B_hf))
            i = tf.sigmoid(tf.nn.bias_add(tf.matmul(inputs, W_ii), B_ii) + tf.nn.bias_add(tf.matmul(h, W_hi), B_hi))
            g = tf.tanh(tf.nn.bias_add(tf.matmul(inputs, W_ig), B_ig) + tf.nn.bias_add(tf.matmul(h, W_hg), B_hg))
            o = tf.sigmoid(tf.nn.bias_add(tf.matmul(inputs, W_io), B_io) + tf.nn.bias_add(tf.matmul(h, W_ho), B_ho))
            
            c_prime = f * c + i * g
            h_prime	= o * tf.tanh(c_prime)

            new_state = array_ops.concat([c_prime, h_prime], 1)
        return h_prime, new_state
