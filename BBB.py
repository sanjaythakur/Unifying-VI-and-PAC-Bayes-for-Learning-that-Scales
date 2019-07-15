import tensorflow as tf
import math
import os
from tqdm import tqdm
import _pickle as pickle
import copy

from Housekeeping import *
from Load_Controllers import *

from _BBBNNRegression import _BBBNNRegression as BNN

class BBBRegression():

    def __init__(self, dataset_type, input_dimensions, regularizer, number_mini_batches, number_output_units, activation_unit, learning_rate, hidden_units, number_samples_variance_reduction, precision_alpha, weights_prior_mean_1, weights_prior_mean_2, weights_prior_deviation_1, weights_prior_deviation_2, mixture_pie, rho_mean, extra_likelihood_emphasis, PB_N, num_classes, num_dimensions, ss):
        self.number_mini_batches = number_mini_batches
        self.N_BOUND=(PB_N*1000)
        self.epoch_start = 0
        self.mean_x, self.mean_y = 0., 0.
        self.deviation_x, self.deviation_y = 1., 1.
        self.BBB_Regressor_graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config, graph=self.BBB_Regressor_graph)
        with self.BBB_Regressor_graph.as_default():
            self.BBB_Regressor = BNN(dataset_type=dataset_type, input_dimensions=input_dimensions, regularizer=regularizer, number_mini_batches=number_mini_batches,
                                     number_output_units=number_output_units, activation_unit=activation_unit, learning_rate=learning_rate,
                                     hidden_units=hidden_units, number_samples_variance_reduction=number_samples_variance_reduction, precision_alpha=precision_alpha,
                                     weights_prior_mean_1=weights_prior_mean_1, weights_prior_mean_2=weights_prior_mean_2, weights_prior_deviation_1=weights_prior_deviation_1,
                                     weights_prior_deviation_2=weights_prior_deviation_2, mixture_pie=mixture_pie, rho_mean=rho_mean, extra_likelihood_emphasis=extra_likelihood_emphasis,
                                     num_classes=num_classes, num_dimensions=num_dimensions, ss=ss)
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

        ELBOs, PB_bounds = [], []

        with self.BBB_Regressor_graph.as_default():
            #with tf.Session(config=config) as sess:
            #self.session.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(directory_to_save_tensorboard_data, self.session.graph)
            saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=2)
            previous_minimum_loss = sys.float_info.max
            mini_batch_size = int(disposible_train_x.shape[0]/self.number_mini_batches)
            for epoch_iterator in tqdm(range(self.epoch_start, epochs)):
                disposible_train_x, disposible_train_y = randomize(disposible_train_x, disposible_train_y)
                ptr = 0
                epoch_elbo, epoch_bound = 0., 0.
                for mini_batch_iterator in range(self.number_mini_batches):
                    x_batch = disposible_train_x[ptr:ptr+mini_batch_size, :]
                    y_batch = disposible_train_y[ptr:ptr+mini_batch_size, :]
                    _, loss, summary, elbo, bound = self.session.run([self.BBB_Regressor.train(), self.BBB_Regressor.getMeanSquaredError(), self.BBB_Regressor.summarize(), self.BBB_Regressor.ELBO, self.BBB_Regressor.pac_bayes_bound], feed_dict={self.BBB_Regressor.input_x:x_batch, self.BBB_Regressor.input_y:y_batch, self.BBB_Regressor.N_BOUND: self.N_BOUND})
                    self.session.run(self.BBB_Regressor.update_mini_batch_index())
                    if loss < previous_minimum_loss:
                        saver.save(self.session, saved_models_during_iterations_bbb + 'iteration', global_step=epoch_iterator, write_meta_graph=False)
                        previous_minimum_loss = loss
                    epoch_elbo += elbo
                    epoch_bound += bound 
                    ptr += mini_batch_size
                    writer.add_summary(summary, global_step=tf.train.global_step(self.session, self.BBB_Regressor.global_step))
                ELBOs.append(epoch_elbo)
                PB_bounds.append(epoch_bound)
                #if epoch_iterator % 2 == 0:
                #    print(BLUE('Training progress: ' + str(epoch_iterator) + '/' + str(epochs)))     
            writer.close()
            saver.save(self.session, saved_final_model_bbb + 'final', write_state=False)
        elbo_n_bound_file = configuration_identity + 'elbo_n_bound_convergence.pkl'
        data_to_store = {ELBO_CONVERGENCE_KEY:ELBOs, PAC_BAYES_BOUND_CONVERGENCE_KEY:PB_bounds}
        with open(elbo_n_bound_file, 'wb') as f:
            pickle.dump(data_to_store, f)

    def predict(self, data_x):
        disposible_data_x = copy.deepcopy(data_x)
        with self.BBB_Regressor_graph.as_default():
            mean_prediction, deviation_prediction, max_prediction, min_prediction = self.session.run([self.BBB_Regressor.mean_of_output_forward_pass, self.BBB_Regressor.deviation_of_output_forward_pass, self.BBB_Regressor.maximum_of_output_forward_pass, self.BBB_Regressor.minimum_of_output_forward_pass], feed_dict={self.BBB_Regressor.train_x:NORMALIZE(disposible_data_x, self.mean_x, self.deviation_x)})
        mean_prediction = REVERSE_NORMALIZE(mean_prediction, self.mean_y, self.deviation_y)
        max_prediction = REVERSE_NORMALIZE(max_prediction, self.mean_y, self.deviation_y)
        min_prediction = REVERSE_NORMALIZE(min_prediction, self.mean_y, self.deviation_y)
        deviation_prediction = deviation_prediction * self.deviation_y
        return mean_prediction, deviation_prediction, max_prediction, min_prediction

    def get_costs_n_errors(self, data_x):
        disposible_data_x = copy.deepcopy(data_x)
        with self.BBB_Regressor_graph.as_default():
            pr_cost, var_map_cost, ll_cost, cost, pred_err = self.session.run([self.BBB_Regressor.getCostforTraining(), self.BBB_Regressor.getMeanSquaredError()], feed_dict={self.BBB_Regressor.input_x:NORMALIZE(disposible_data_x, self.mean_x, self.deviation_x)})
        return pr_cost, var_map_cost, ll_cost, cost, pred_err

    def close(self):
        #with self.BBB_Regressor_graph.as_default():
        self.session.close()


class LoadBBBRegressor():
    def __init__(self, controller_identity):
        self.BBB_Regressor_graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config, graph=self.BBB_Regressor_graph)
        with self.BBB_Regressor_graph.as_default():
            self.BBB_Regressor=Load_BBB(controller_identity=controller_identity, session=self.session)
    
    def get_costs_n_errs(self, data_x, data_y, N_BOUND):
        disposible_data_x, disposible_data_y = copy.deepcopy(data_x), copy.deepcopy(data_y)
        with self.BBB_Regressor_graph.as_default():
            ll_cost, ELBO, PB_bound, cost, pred_err = self.session.run([self.BBB_Regressor.ll_cost, self.BBB_Regressor.ELBO, self.BBB_Regressor.pac_bayes_bound, self.BBB_Regressor.cost, self.BBB_Regressor.pred_err], feed_dict={self.BBB_Regressor.input_x:NORMALIZE(disposible_data_x, self.BBB_Regressor.mean_x, self.BBB_Regressor.deviation_x), self.BBB_Regressor.input_y: NORMALIZE(disposible_data_y, self.BBB_Regressor.mean_y, self.BBB_Regressor.deviation_y), self.BBB_Regressor.N_BOUND: N_BOUND})
            #pr_cost, var_map_cost, ll_cost, ELBO, pac_bayes_regularizer, cost, pred_err = self.session.run([self.BBB_Regressor.pr_cost, self.BBB_Regressor.var_MAP_cost, self.BBB_Regressor.ll_cost, self.BBB_Regressor.ELBO, self.BBB_Regressor.pac_bayes_regularizer, self.BBB_Regressor.cost, self.BBB_Regressor.pred_err], feed_dict={self.BBB_Regressor.input_x:NORMALIZE(disposible_data_x, self.BBB_Regressor.mean_x, self.BBB_Regressor.deviation_x), self.BBB_Regressor.input_y: NORMALIZE(disposible_data_y, self.BBB_Regressor.mean_y, self.BBB_Regressor.deviation_y)})
        return ll_cost, ELBO, PB_bound, cost, pred_err

    def get_pred_costs_errs(self, data_x, data_y, N_BOUND):
        disposible_data_x, disposible_data_y = copy.deepcopy(data_x), copy.deepcopy(data_y)
        with self.BBB_Regressor_graph.as_default():
            mean_prediction, deviation_prediction, max_prediction, min_prediction, ll_cost, ELBO, PB_bound, cost, pred_err = self.session.run([self.BBB_Regressor.mean_of_predictions, self.BBB_Regressor.deviation_of_predictions, self.BBB_Regressor.maximum_of_predictions, self.BBB_Regressor.minimum_of_predictions, self.BBB_Regressor.ll_cost, self.BBB_Regressor.ELBO, self.BBB_Regressor.pac_bayes_bound, self.BBB_Regressor.cost, self.BBB_Regressor.pred_err], feed_dict={self.BBB_Regressor.input_x:NORMALIZE(disposible_data_x, self.BBB_Regressor.mean_x, self.BBB_Regressor.deviation_x), self.BBB_Regressor.input_y: NORMALIZE(disposible_data_y, self.BBB_Regressor.mean_y, self.BBB_Regressor.deviation_y), self.BBB_Regressor.N_BOUND: N_BOUND})
        mean_prediction = REVERSE_NORMALIZE(mean_prediction, self.BBB_Regressor.mean_y, self.BBB_Regressor.deviation_y)
        max_prediction = REVERSE_NORMALIZE(max_prediction, self.BBB_Regressor.mean_y, self.BBB_Regressor.deviation_y)
        min_prediction = REVERSE_NORMALIZE(min_prediction, self.BBB_Regressor.mean_y, self.BBB_Regressor.deviation_y)
        deviation_prediction = deviation_prediction * self.BBB_Regressor.deviation_y
        return mean_prediction, deviation_prediction, max_prediction, min_prediction, ll_cost, ELBO, PB_bound, cost, pred_err

    def predict(self, data_x):
        disposible_data_x = copy.deepcopy(data_x)
        with self.BBB_Regressor_graph.as_default():
            mean_prediction, deviation_prediction, max_prediction, min_prediction = self.session.run([self.BBB_Regressor.mean_of_predictions, self.BBB_Regressor.deviation_of_predictions, self.BBB_Regressor.maximum_of_predictions, self.BBB_Regressor.minimum_of_predictions], feed_dict={self.BBB_Regressor.input_x:NORMALIZE(disposible_data_x, self.BBB_Regressor.mean_x, self.BBB_Regressor.deviation_x)})
        mean_prediction = REVERSE_NORMALIZE(mean_prediction, self.BBB_Regressor.mean_y, self.BBB_Regressor.deviation_y)
        max_prediction = REVERSE_NORMALIZE(max_prediction, self.BBB_Regressor.mean_y, self.BBB_Regressor.deviation_y)
        min_prediction = REVERSE_NORMALIZE(min_prediction, self.BBB_Regressor.mean_y, self.BBB_Regressor.deviation_y)
        deviation_prediction = deviation_prediction * self.BBB_Regressor.deviation_y
        return mean_prediction, deviation_prediction, max_prediction, min_prediction

    def close(self):
        self.BBB_Regressor.sess.close()