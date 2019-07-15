import tensorflow as tf
import _pickle as pickle
import math
import numpy as np

class _Deterministic_NN():
    def __init__(self, input_dimensions, activation_unit, learning_rate, hidden_units, num_classes=1, num_dimensions=1):
        number_output_units = num_classes * num_dimensions
        self.activation_unit = activation_unit
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
        with tf.name_scope('inputs'):
            self.input_x = tf.placeholder(tf.float32, shape=(None, input_dimensions), name='input_x')
            self.input_y = tf.placeholder(tf.float32, shape=(None, number_output_units), name='input_y')
        with tf.name_scope('input_output_forward_pass_mapping'):
            # Defining the first hidden layer
            weights_layer_1 = tf.Variable(tf.truncated_normal([input_dimensions, hidden_units[0]], stddev=1.0 / math.sqrt(float(input_dimensions))))
            biases_layer_1 = tf.Variable(tf.zeros([hidden_units[0]]))
            hidden_output_1 = self.fetch_ACTIVATION_UNIT(tf.matmul(self.input_x, weights_layer_1) + biases_layer_1)
            # Defining the second hidden layer
            weights_layer_2 = tf.Variable(tf.truncated_normal([hidden_units[0], hidden_units[1]], stddev=1.0 / math.sqrt(float(hidden_units[0]))))
            biases_layer_2 = tf.Variable(tf.zeros([hidden_units[1]]))
            hidden_output_2 = self.fetch_ACTIVATION_UNIT(tf.matmul(hidden_output_1, weights_layer_2) + biases_layer_2)
            # Defining the third hidden layer
            weights_layer_3 = tf.Variable(tf.truncated_normal([hidden_units[1], hidden_units[2]], stddev=1.0 / math.sqrt(float(hidden_units[1]))))
            biases_layer_3 = tf.Variable(tf.zeros([hidden_units[2]]))
            hidden_output_3 = self.fetch_ACTIVATION_UNIT(tf.matmul(hidden_output_2, weights_layer_3) + biases_layer_3)
            # Mu Output
            weights_layer_output = tf.Variable(tf.truncated_normal([hidden_units[2], number_output_units], stddev=1.0 / math.sqrt(float(hidden_units[2]))))
            biases_layer_output = tf.Variable(tf.zeros(number_output_units))
            self.prediction = tf.add(tf.matmul(hidden_output_3, weights_layer_output), biases_layer_output, name='prediction')            
        with tf.name_scope('cost'):
            self.cost = tf.reduce_mean(tf.squared_difference(self.prediction, self.input_y), name='mean_squared_error')
        with tf.name_scope('optimization'):
            optimizer = tf.train.AdamOptimizer(learning_rate, name='adam_optimizer')
            self.training = optimizer.minimize(self.cost, global_step=self.global_step, name='training')            
        with tf.name_scope('summaries'):
            tf.summary.scalar(name='training_cost', tensor=self.cost)
            self.summary_op = tf.summary.merge_all()

        # import pdb;pdb.set_trace()
    def getTrainingCost(self):
        return self.cost

    def summarize(self):
        return self.summary_op

    def makeInference(self):
        return self.prediction

    def train(self):
        return self.training

    def fetch_ACTIVATION_UNIT(self, param):
        if self.activation_unit == 'RELU':
            return tf.nn.relu(param)
        elif self.activation_unit == 'SIGMOID':
            return tf.sigmoid(param)
        elif self.activation_unit == 'TANH':
            return tf.tanh(param)