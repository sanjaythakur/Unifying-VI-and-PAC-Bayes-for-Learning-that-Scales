import tensorflow as tf
import math
import os
from tqdm import tqdm
import copy
import sys
import _pickle as pickle

from Housekeeping import *
from _DNN import _Deterministic_NN
from Load_Controllers import Load_DNN

class Deterministic_NN():
    def __init__(self, input_dimensions, number_mini_batches, activation_unit, learning_rate, hidden_units, num_classes, num_dimensions):
        self.number_mini_batches = number_mini_batches
        self.epoch_start = 0
        self.mean_x, self.mean_y = 0., 0.
        self.deviation_x, self.deviation_y = 1., 1.
        self.DNN_graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config, graph=self.DNN_graph)
        with self.DNN_graph.as_default():
            self.DNN_Regressor = _Deterministic_NN(input_dimensions=input_dimensions, activation_unit=activation_unit, learning_rate=learning_rate, hidden_units=hidden_units, num_classes=num_classes, num_dimensions=num_dimensions)
            self.session.run(tf.global_variables_initializer())

    def train(self, train_x, train_y, epochs, configuration_identity):
        disposible_train_x, disposible_train_y = copy.deepcopy(train_x), copy.deepcopy(train_y)
        self.mean_x, self.deviation_x = get_mean_and_deviation(data=disposible_train_x)
        disposible_train_x = NORMALIZE(disposible_train_x, self.mean_x, self.deviation_x)
        self.mean_y, self.deviation_y = get_mean_and_deviation(data=disposible_train_y)
        disposible_train_y = NORMALIZE(disposible_train_y, self.mean_y, self.deviation_y)

        training_logs_directory = configuration_identity + 'training/'
        if not os.path.exists(training_logs_directory):
            os.makedirs(training_logs_directory)

        file_name_to_save_input_manipulation_data = training_logs_directory + 'input_manipulation_data.pkl'
        input_manipulation_data_to_store = {MEAN_KEY_X: self.mean_x, DEVIATION_KEY_X: self.deviation_x,
                                              MEAN_KEY_Y: self.mean_y, DEVIATION_KEY_Y: self.deviation_y}
        with open(file_name_to_save_input_manipulation_data, 'wb') as f:
            pickle.dump(input_manipulation_data_to_store, f)

        directory_to_save_tensorboard_data = training_logs_directory + TENSORBOARD_DIRECTORY
        saved_models_during_iterations_bbb = training_logs_directory + SAVED_MODELS_DURING_ITERATIONS_DIRECTORY
        saved_final_model_bbb = training_logs_directory + SAVED_FINAL_MODEL_DIRECTORY
        if not os.path.exists(directory_to_save_tensorboard_data):
            os.makedirs(directory_to_save_tensorboard_data)
        if not os.path.exists(saved_models_during_iterations_bbb):
            os.makedirs(saved_models_during_iterations_bbb)
        if not os.path.exists(saved_final_model_bbb):
            os.makedirs(saved_final_model_bbb)

        with self.DNN_graph.as_default():
            writer = tf.summary.FileWriter(directory_to_save_tensorboard_data, self.session.graph)
            saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=2)
            previous_minimum_loss = sys.float_info.max
            mini_batch_size = int(disposible_train_x.shape[0]/self.number_mini_batches)
            for epoch_iterator in tqdm(range(self.epoch_start, epochs)):
                disposible_train_x, disposible_train_y = randomize(disposible_train_x, disposible_train_y)
                ptr = 0
                for mini_batch_iterator in range(self.number_mini_batches):
                    x_batch = disposible_train_x[ptr:ptr+mini_batch_size, :]
                    y_batch = disposible_train_y[ptr:ptr+mini_batch_size, :]
                    _, loss, summary = self.session.run([self.DNN_Regressor.training, self.DNN_Regressor.cost, self.DNN_Regressor.summary_op], feed_dict={self.DNN_Regressor.input_x:x_batch, self.DNN_Regressor.input_y:y_batch})
                    if loss < previous_minimum_loss:
                        saver.save(self.session, saved_models_during_iterations_bbb + 'iteration', global_step=epoch_iterator, write_meta_graph=False)
                        previous_minimum_loss = loss
                    ptr += mini_batch_size
                    writer.add_summary(summary, global_step=tf.train.global_step(self.session, self.DNN_Regressor.global_step))     
            writer.close()
            saver.save(self.session, saved_final_model_bbb + 'final', write_state=False)

    def predict(self, data_x):
        disposible_x = copy.deepcopy(data_x)
        with self.DNN_graph.as_default():
            prediction = self.session.run(self.DNN_Regressor.prediction, feed_dict={self.DNN_Regressor.X_input:NORMALIZE(disposible_x, self.mean_x, self.deviation_x)})
        prediction = REVERSE_NORMALIZE(prediction, self.mean_y, self.deviation_y)
        return prediction, 0., prediction, prediction

    def close(self):
        #with self.BBB_Regressor_graph.as_default():
        self.session.close()


class LoadDNNRegressor():
    def __init__(self, controller_identity):
        self.DNN_Regressor_graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config, graph=self.DNN_Regressor_graph)
        with self.DNN_Regressor_graph.as_default():
            self.DNN_Regressor=Load_DNN(controller_identity=controller_identity, session=self.session)

    def predict(self, data_x):
        disposible_data_x = copy.deepcopy(data_x)
        with self.DNN_Regressor_graph.as_default():
            prediction = self.session.run(self.DNN_Regressor.pred, feed_dict={self.DNN_Regressor.input_x:NORMALIZE(disposible_data_x, self.DNN_Regressor.mean_x, self.DNN_Regressor.deviation_x)})
        prediction = REVERSE_NORMALIZE(prediction, self.DNN_Regressor.mean_y, self.DNN_Regressor.deviation_y)
        return prediction, 0., prediction, prediction

    def close(self):
        self.DNN_Regressor.sess.close()