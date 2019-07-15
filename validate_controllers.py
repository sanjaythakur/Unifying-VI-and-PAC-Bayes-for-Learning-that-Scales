import tensorflow as tf
import _pickle as pickle
import numpy as np
import argparse
import os
import gym
from BBB import LoadBBBRegressor
from Load_Controllers import Load_MuJoCo_Demonstrator
from Housekeeping import *
from Deterministic_Neural_Network import LoadDNNRegressor

def validate_and_log_prediction_task(x_train, y_train, x_val, y_val, controller, configuration_identifier):
    y_hat, y_dev, y_max, y_min = controller.predict(data_x=x_val)
    train_pr_cost, train_var_map_cost, train_ll_cost, train_cost, train_pred_err = controller.get_costs_n_errors(data_x=x_train, data_y=y_train)
    val_pr_cost, val_var_map_cost, val_ll_cost, val_cost, val_pred_err = controller.get_costs_n_errors(data_x=x_val, data_y=y_val)

    file_to_save_results = configuration_identifier + 'validation_logs.pkl'
    data_to_save = {X_TRAIN_KEY: x_train, Y_TRAIN_KEY: y_train, X_TEST_KEY: x_val, Y_TEST_KEY: y_val,
                    Y_HAT_KEY: y_hat, Y_DEV_KEY: y_dev, Y_MAX_KEY: y_max, Y_MIN_KEY: y_min,
                     TRAIN_PRIOR_COST_KEY: train_pr_cost, TRAIN_VAR_MAP_COST_KEY: train_var_map_cost, 
                     TRAIN_LL_COST_KEY: train_ll_cost, TRAIN_COST_KEY: train_cost, TRAIN_PRED_ERR_KEY: train_pred_err,
                      VAL_PRIOR_COST_KEY: val_pr_cost, VAL_VAR_MAP_COST_KEY: val_var_map_cost,
                       VAL_LL_COST_KEY: val_ll_cost, VAL_COST_KEY: val_cost, VAL_PRED_ERR_KEY: val_pred_err}
    with open(file_to_save_results, 'wb') as f:
        pickle.dump(data_to_save, f)

def validate_DNN_n_log_MDP(experiment, num_rollouts, controller, configuration_identifier, regularizer, controller_type):
    configuration_identifier = configuration_identifier + controller_type + '_' + regularizer + '/'
    print(RED(configuration_identifier))
    all_states_across_rollouts = []
    all_controller_actions_across_rollouts = []
    all_demonstrator_actions_across_rollouts = []
    all_rewards_across_rollouts = []
    demonstrator_graph = tf.Graph()
    with demonstrator_graph.as_default():
        demonstrator_controller = Load_MuJoCo_Demonstrator(env_name=experiment)
    for rollout in range(num_rollouts):
        all_states = []
        all_controller_actions = []
        all_demonstrator_actions = []
        all_rewards = []
        env = gym.make(experiment + '-v1')
        observation = env.reset()
        done = False
        time_step = 0.
        while not done:
            observation = observation.astype(np.float32).reshape((1, -1))
            observation = np.append(observation, [[time_step]], axis=1)  # add time step feature
            all_states.append(observation[0])
            demonstrator_action = demonstrator_controller.getDemonstratorAction(state=observation)
            mean_action, dev_action, max_action, min_action = controller.predict(data_x=observation)
            all_controller_actions.append(mean_action)
            all_demonstrator_actions.append(demonstrator_action[0])
            observation, reward, done, info = env.step(mean_action)
            all_rewards.append(reward)
            time_step += 1e-3   
        all_states_across_rollouts.append(all_states)
        all_controller_actions_across_rollouts.append(all_controller_actions)
        all_demonstrator_actions_across_rollouts.append(all_demonstrator_actions)
        all_rewards_across_rollouts.append(all_rewards)
    logs_across_rollouts = {STATES_KEY: all_states_across_rollouts, DEMONSTRATOR_ACTION_KEY: all_demonstrator_actions_across_rollouts,
                            CONTROLLER_ACTION_KEY: all_controller_actions_across_rollouts, REWARDS_KEY: all_rewards_across_rollouts}
    file_to_save_results = configuration_identifier + 'validation_logs.pkl'
    with open(file_to_save_results, 'wb') as f:
        pickle.dump(logs_across_rollouts, f)

def validate_and_log_MDP(experiment, num_rollouts, controller, configuration_identifier, regularizer, controller_type, extra_likelihood_emphasis_code, number_samples_variance_reduction, PB_N, ss):
    # Generating and saving logs for the training data
    demonstrations_file = configuration_identifier + 'training_data.pkl'
    with open(demonstrations_file, 'rb') as f:
        demonstration_data = pickle.load(f)
    all_states = demonstration_data[X_TRAIN_KEY]
    all_actions = demonstration_data[Y_TRAIN_KEY]
    all_rewards = demonstration_data[REWARDS_KEY]
    tr_ll_cost, tr_ELBO, tr_PB_bound, tr_cost, tr_pred_err = controller.get_costs_n_errs(data_x=all_states, data_y=all_actions, N_BOUND=all_states.shape[0])
    # Validating the controller and saving generated logs
    configuration_identifier = configuration_identifier + controller_type + '_' + str(extra_likelihood_emphasis_code) + '_' + regularizer + '/' + str(number_samples_variance_reduction) + '/' + str(ss) + '/'
    print(RED(configuration_identifier))
    all_states_across_rollouts = []
    all_controller_actions_across_rollouts = []
    all_demonstrator_actions_across_rollouts = []
    all_rewards_across_rollouts = []
    all_ll_across_rollouts = []
    all_ELBO_across_rollouts = []
    #all_squashed_ELBO_across_rollouts = []
    all_PB_bounds_across_rollouts = []
    all_cost_across_rollouts = []
    all_pred_err_across_rollouts = []
    demonstrator_graph = tf.Graph()
    with demonstrator_graph.as_default():
        demonstrator_controller = Load_MuJoCo_Demonstrator(env_name=experiment)
    for rollout in range(num_rollouts):
        all_states = []
        all_controller_actions = []
        all_demonstrator_actions = []
        all_rewards = []
        #all_squashed_ELBO = []
        all_ll = []
        all_ELBO = []
        all_PB_bounds = []
        all_cost = []
        all_pred_err = []
        env = gym.make(experiment + '-v1')
        observation = env.reset()
        done = False
        time_step = 0.
        while not done:
            observation = observation.astype(np.float32).reshape((1, -1))
            observation = np.append(observation, [[time_step]], axis=1)  # add time step feature
            all_states.append(observation[0])
            #controller_action = controller.predict(data_x=observation)
            demonstrator_action = demonstrator_controller.getDemonstratorAction(state=observation)
            mean_action, dev_action, max_action, min_action, val_ll_cost, val_ELBO, val_PB_bound, val_cost, val_pred_err = controller.get_pred_costs_errs(data_x=observation, data_y=demonstrator_action, N_BOUND=1)
            all_controller_actions.append(mean_action)
            all_demonstrator_actions.append(demonstrator_action[0])
            observation, reward, done, info = env.step(mean_action)
            all_rewards.append(reward)
            all_ll.append(val_ll_cost)
            all_ELBO.append(val_ELBO)
            #all_squashed_ELBO.append(squashed_ELBO)
            all_PB_bounds.append(val_PB_bound)
            all_cost.append(val_cost)
            all_pred_err.append(val_pred_err)
            time_step += 1e-3   
        all_states_across_rollouts.append(all_states)
        all_controller_actions_across_rollouts.append(all_controller_actions)
        all_demonstrator_actions_across_rollouts.append(all_demonstrator_actions)
        all_rewards_across_rollouts.append(all_rewards)
        all_ll_across_rollouts.append(all_ll)
        all_ELBO_across_rollouts.append(all_ELBO)
        #all_squashed_ELBO_across_rollouts.append(all_squashed_ELBO)
        all_PB_bounds_across_rollouts.append(all_PB_bounds)
        all_cost_across_rollouts.append(all_cost)
        all_pred_err_across_rollouts.append(all_pred_err)
    logs_across_rollouts = {TRAIN_ELBO_KEY: tr_ELBO, TRAIN_PAC_BAYES_REGULARIZER_KEY: tr_PB_bound,
                            TRAIN_COST_KEY: tr_cost, TRAIN_PRED_ERR_KEY: tr_pred_err, TRAIN_LL_COST_KEY: tr_ll_cost,
                            STATES_KEY: all_states_across_rollouts, DEMONSTRATOR_ACTION_KEY: all_demonstrator_actions_across_rollouts,
                            CONTROLLER_ACTION_KEY: all_controller_actions_across_rollouts, REWARDS_KEY: all_rewards_across_rollouts,
                             VAL_LL_COST_KEY: all_ll_across_rollouts, VAL_ELBO_KEY:all_ELBO_across_rollouts,
                              VAL_PAC_BAYES_REGULARIZER_KEY:all_PB_bounds_across_rollouts,
                               VAL_COST_KEY: all_cost_across_rollouts, VAL_PRED_ERR_KEY: all_pred_err_across_rollouts}
    file_to_save_results = configuration_identifier + 'validation_logs.pkl'
    with open(file_to_save_results, 'wb') as f:
        pickle.dump(logs_across_rollouts, f)

def run_validation(experiment, controller_type, regularizer, extra_likelihood_emphasis_code, number_samples_variance_reduction, PB_N, ss):
    print(RED('Validation'))
    configuration_identifier = './logs/' + experiment + '/' + str(PB_N) + '/'
    if controller_type == 'probabilistic':
        #configuration_identifier = './logs/' + experiment + '/' + controller_type + '_' + str(extra_likelihood_emphasis_code) + '_' + regularizer + '/' + str(number_samples_variance_reduction) + '/' + str(PB_N) + '/' + str(ss) + '/'
        controller = LoadBBBRegressor(controller_identity= './logs/' + experiment + '/' + str(PB_N) + '/' + controller_type + '_' + str(extra_likelihood_emphasis_code) + '_' + regularizer + '/' + str(number_samples_variance_reduction) + '/' + str(ss) + '/')
        validate_and_log_MDP(experiment=experiment, num_rollouts=10, controller=controller, configuration_identifier=configuration_identifier, regularizer=regularizer, controller_type=controller_type,
         extra_likelihood_emphasis_code=extra_likelihood_emphasis_code, number_samples_variance_reduction=number_samples_variance_reduction, PB_N=PB_N, ss=ss)
    elif controller_type == 'deterministic':
        controller = LoadDNNRegressor(controller_identity=configuration_identifier + controller_type + '_' + regularizer + '/')
        validate_DNN_n_log_MDP(experiment=experiment, num_rollouts=10, controller=controller, configuration_identifier=configuration_identifier, regularizer=regularizer, controller_type=controller_type)
    else:
        if experiment == 'sin':
            dataset_type = 'continuous'
            num_classes = 1
            x_train, x_val = np.arange(-13, 5, 0.25).reshape(-1,1), np.arange(-25, 25., 0.25).reshape(-1,1)
            y_train, y_val = np.sin(x_train), np.sin(x_val)
            validate_and_log_prediction_task(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                                            controller=controller, configuration_identifier=configuration_identifier)
        elif experiment == 'cos':
            dataset_type = 'continuous'
            num_classes = 1
            x_train, x_val = np.arange(-13, 5, 0.25).reshape(-1,1), np.arange(-25, 25., 0.25).reshape(-1,1)
            y_train, y_val = np.cos(x_train), np.cos(x_val)
            validate_and_log_prediction_task(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                                            controller=controller, configuration_identifier=configuration_identifier)
        elif experiment == 'MNIST':
            mnist = tf.keras.datasets.mnist
            (x_train, y_train),(x_val, y_val) = mnist.load_data()
            #x_train, x_val = x_train / 255.0, x_val / 255.0
            dataset_type = 'categorical'
            num_classes = 10
            validate_and_log_prediction_task(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                                            controller=controller, configuration_identifier=configuration_identifier)   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str, help='Experiment', default='MuJoCo', choices=['sin', 'cos', 'MNIST', 'Ant', 'HalfCheetah', 'Hopper', 'Humanoid', 'InvertedDoublePendulum', 'InvertedPendulum', 'Reacher', 'Swimmer', 'Walker2d'])
    parser.add_argument('-c', '--controller_type', type=str, help='controller_type', choices=['probabilistic', 'deterministic'])
    parser.add_argument('-r', '--regularizer', type=str, help='Regularizer', default='Unregularized', choices=['L1', 'L2', 'PAC_Bayes', 'Unregularized'])
    parser.add_argument('-elec', '--extra_likelihood_emphasis_code', type=int, help='Extra Likelihood Emphasis Code', default=12, choices=[0, 6, 12])
    parser.add_argument('-MC', '--MC', type=int, help='Number of MC Samples', default=10, choices=[10, 25, 100])
    parser.add_argument('-pbn', '--pbn', type=int, help='PAC Bayes N', default=1, choices=[1, 10])
    parser.add_argument('-ss', '--ss', type=int, help='s square', default=3, choices=[0, 1, 2, 3])
    args = parser.parse_args()
    run_validation(experiment=args.experiment, controller_type=args.controller_type, regularizer=args.regularizer, extra_likelihood_emphasis_code=args.extra_likelihood_emphasis_code, number_samples_variance_reduction = args.MC, PB_N=args.pbn, ss=args.ss)