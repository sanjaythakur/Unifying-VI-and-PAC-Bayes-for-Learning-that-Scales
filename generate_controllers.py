import tensorflow as tf
import numpy as np
import argparse
import os
from Housekeeping import *
import math
from BBB import BBBRegression
from Deterministic_Neural_Network import Deterministic_NN
from Load_Controllers import Load_MuJoCo_Demonstrator

def run_experiment(experiment, controller_type, regularizer, epochs, number_mini_batches, activation_unit, learning_rate, hidden_units,
    number_samples_variance_reduction, precision_alpha, weights_prior_mean_1, weights_prior_mean_2, weights_prior_deviation_1,
    weights_prior_deviation_2, mixture_pie, rho_mean, extra_likelihood_emphasis_code, PB_N, ss):

    configuration_identifier = './logs/' + experiment + '/' + str(PB_N) + '/'
    if not os.path.exists(configuration_identifier):
        os.makedirs(configuration_identifier)
    extra_likelihood_emphasis = math.pow(10, extra_likelihood_emphasis_code)
    if experiment == 'sin':
        dataset_type = 'continuous'
        num_classes = 1
        num_dimensions = 1
        x_train, x_test = np.arange(-13, 5, 0.25).reshape(-1,1), np.arange(-25, 25., 0.25).reshape(-1,1)
        y_train, y_test = np.sin(x_train), np.sin(x_test)
    elif experiment == 'cos':
        dataset_type = 'continuous'
        num_classes = 1
        num_dimensions = 1
        x_train, x_test = np.arange(-13, 5, 0.25).reshape(-1,1), np.arange(-25, 25., 0.25).reshape(-1,1)
        y_train, y_test = np.cos(x_train), np.cos(x_test)
    elif experiment == 'MNIST':
        mnist = tf.keras.datasets.mnist
        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        #x_train, x_test = x_train / 255.0, x_test / 255.0
        dataset_type = 'categorical'
        num_classes = 10
        num_dimensions = 1
    else:
        demonstrator = Load_MuJoCo_Demonstrator(env_name=experiment)
        x_train, y_train = demonstrator.getStateActionPairs(configuration=configuration_identifier, window_size=1, number_rollouts=PB_N)
        dataset_type = 'continuous'
        num_classes = 1
        num_dimensions = y_train.shape[1]
    
    if controller_type == 'probabilistic':
        configuration_identifier = configuration_identifier + controller_type + '_' + str(extra_likelihood_emphasis_code) + '_' + regularizer + '/' + str(number_samples_variance_reduction) + '/' + str(ss) + '/'
        print(RED(configuration_identifier))
        controller = BBBRegression(dataset_type=dataset_type, input_dimensions=x_train.shape[1], regularizer=regularizer, number_mini_batches=number_mini_batches,
         number_output_units=y_train.shape[1], activation_unit=activation_unit, learning_rate=learning_rate, hidden_units=hidden_units, number_samples_variance_reduction=number_samples_variance_reduction,
          precision_alpha=precision_alpha, weights_prior_mean_1=weights_prior_mean_1, weights_prior_mean_2=weights_prior_mean_2, weights_prior_deviation_1=weights_prior_deviation_1,
           weights_prior_deviation_2=weights_prior_deviation_2, mixture_pie=mixture_pie, rho_mean=rho_mean, extra_likelihood_emphasis=extra_likelihood_emphasis,
            PB_N=PB_N, num_classes=num_classes, num_dimensions=num_dimensions, ss=ss)
    elif controller_type == 'deterministic':
        configuration_identifier = configuration_identifier + controller_type + '_' + regularizer + '/'
        print(RED(configuration_identifier))
        controller = Deterministic_NN(input_dimensions=x_train.shape[1], number_mini_batches=number_mini_batches, activation_unit=activation_unit, learning_rate=learning_rate, hidden_units=hidden_units, num_classes=num_classes, num_dimensions=num_dimensions)
    #controller.train_n_validate(train_x=x_train, train_y=y_train, test_x=x_test, test_y=y_test, epochs=epochs, configuration_identity=configuration_identifier)
    controller.train(train_x=x_train, train_y=y_train, epochs=epochs, configuration_identity=configuration_identifier)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str, help='Experiment', default='MuJoCo', choices=['sin', 'cos', 'MNIST', 'Ant', 'HalfCheetah', 'Hopper', 'Humanoid', 'InvertedDoublePendulum', 'InvertedPendulum', 'Reacher', 'Swimmer', 'Walker2d'])
    parser.add_argument('-c', '--controller_type', type=str, help='controller_type', choices=['probabilistic', 'deterministic'])
    parser.add_argument('-r', '--regularizer', type=str, help='Regularizer', default='Unregularized', choices=['L1', 'L2', 'PAC_Bayes', 'Unregularized'])
    parser.add_argument('-elec', '--extra_likelihood_emphasis_code', type=int, help='Extra Likelihood Emphasis Code', default=12, choices=[0, 6, 12])
    parser.add_argument('-MC', '--MC', type=int, help='Number of MC Samples', default=10, choices=[10, 25, 100])
    parser.add_argument('-pbn', '--pbn', type=int, help='PAC Bayes N', default=1, choices=[1, 10])
    parser.add_argument('-ss', '--ss', type=int, help='s square', default=3, choices=[0, 1, 2, 3])
    parser.add_argument('-epochs', '--epochs', type=int, help='Epochs', default=5000)
    args = parser.parse_args()
    run_experiment(experiment=args.experiment, controller_type=args.controller_type, regularizer=args.regularizer, epochs = args.epochs,
      number_mini_batches=20, activation_unit = 'RELU', learning_rate = 0.001, hidden_units = [90, 30, 10],
       number_samples_variance_reduction = args.MC, precision_alpha = 0.01, weights_prior_mean_1 = 0.,
        weights_prior_mean_2 = 0., weights_prior_deviation_1 = 0.4, weights_prior_deviation_2 = 0.4,
         mixture_pie = 0.7, rho_mean =-3., extra_likelihood_emphasis_code=args.extra_likelihood_emphasis_code, PB_N=args.pbn, ss=args.ss)