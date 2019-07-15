import argparse
#from tqdm import tqdm
import os
import gym
import _pickle as pickle
from Housekeeping import *
from Deterministic_Neural_Network import Deterministic_NN
from BBB import BBBRegressor
from Demonstrator import Load_Demonstrator
from Load_Controllers import Load_BBB, Load_DNN

def roll_in(env_name, controller, demonstrator, alpha=0., K=1):
    all_states, all_learner_actions, all_rewards, all_learner_deviations, all_learner_max_actions, all_learner_min_actions, all_demonstrator_actions, = [], [], [], [], [], [], []
    demonstrator_interjections = 0
    for kth_roll_in in range(K):
        env = gym.make(env_name)
        observation = env.reset()
        finish = False
        time_step = 0.0
        observation = np.append(observation, time_step) # add time step feature
        while not finish:
            all_states.append(observation)
            learner_action, learner_deviation, learner_max_prediction, learner_min_prediction = controller.predict(data_x=observation.reshape(1,-1))
            demonstrator_action = demonstrator.getDemonstratorAction(state=observation.reshape(1,-1))
            tau = np.random.rand()
            if tau < alpha:
                action_to_execute = demonstrator_action
                demonstrator_interjections += 1
            else:
                action_to_execute = learner_action
            observation, reward, finish, info = env.step(action_to_execute)
            all_learner_actions.append(learner_action[0])
            all_demonstrator_actions.append(demonstrator_action[0])
            all_rewards.append(reward)
            all_learner_deviations.append(learner_deviation)
            all_learner_max_actions.append(learner_max_prediction[0])
            all_learner_min_actions.append(learner_min_prediction[0])
            time_step += 1e-3
            observation = np.append(observation, time_step)
    return np.array(all_states), np.array(all_learner_actions), np.array(all_rewards), np.array(all_learner_deviations), np.array(all_learner_max_actions), np.array(all_learner_min_actions), np.array(all_demonstrator_actions), demonstrator_interjections

def run_probablistic_aggrevated(env_name, learner, experiment, window_size, number_initial_demonstrations, epochs, number_mini_batches, activation_unit, learning_rate, hidden_units,
                                number_samples_variance_reduction, precision_alpha, weights_prior_mean_1, weights_prior_mean_2, weights_prior_deviation_1,
                                 weights_prior_deviation_2, mixture_pie, rho_mean, extra_likelihood_emphasis):
    configuration_identifier = LOGS_DIRECTORY + learner + '/' + experiment + '/'
    demonstrator = Load_Demonstrator(env_name=env_name)
    pretraining_x, pretraining_y, drift_per_time_step = demonstrator.getStateActionPairs(window_size=window_size, number_rollouts=number_initial_demonstrations)
    if not os.path.exists(configuration_identifier):
        os.makedirs(configuration_identifier)
    if learner == 'BBB':
        print(GREEN('Creating BBB learner'))
        controller = BBBRegressor(drift_per_time_step=drift_per_time_step, temporal_windows_x_size=pretraining_x.shape[1], window_size=window_size, number_mini_batches=number_mini_batches,
         number_output_units=pretraining_y.shape[1], activation_unit=activation_unit, learning_rate=learning_rate, hidden_units=hidden_units, number_samples_variance_reduction=number_samples_variance_reduction,
          precision_alpha=precision_alpha, weights_prior_mean_1=weights_prior_mean_1, weights_prior_mean_2=weights_prior_mean_2, weights_prior_deviation_1=weights_prior_deviation_1,
           weights_prior_deviation_2=weights_prior_deviation_2, mixture_pie=mixture_pie, rho_mean=rho_mean, extra_likelihood_emphasis=extra_likelihood_emphasis)
    elif learner == 'DNN':
        print(GREEN('Creating DNN learner'))
        controller = Deterministic_NN(drift_per_time_step=drift_per_time_step, temporal_windows_x_size=pretraining_x.shape[1], window_size=window_size, number_mini_batches=number_mini_batches,
         number_output_units=pretraining_y.shape[1], activation_unit=activation_unit, learning_rate=learning_rate, hidden_units=hidden_units)
    
    if os.path.exists(configuration_identifier):
        print(GREEN('Reloading existing pretrained controller'))
        
    else:
        print(GREEN('Pretraining the learner'))
        controller.train(data_x=pretraining_x, data_y=pretraining_y, epochs=epochs, configuration_identity=configuration_identifier+'pretraining/')
    if experiment=='Sanity_Check':
        print(GREEN('Running sanity check experiment'))
        for iteration in range(SIMULATION_ITERATIONS):
            print(GREEN('Iteration number is ' + str(iteration)))
            iteration_identifier = configuration_identifier + 'simulation_iterator_' + str(iteration) + '/'
            if not os.path.exists(iteration_identifier):
                os.makedirs(iteration_identifier)
            all_states, all_learner_actions, all_rewards, all_learner_deviations, all_learner_max_actions, all_learner_min_actions, all_demonstrator_actions, demonstrator_interjections = roll_in(env_name, controller, demonstrator)
            print(BLUE(str(all_rewards)))
            q_values = demonstrator.getDemonstratorCostToGo(next_state=all_states[1:, :], reward=all_rewards[:-1])
            data_to_store = {STATES_LOG_KEY: all_states, LEARNER_CONTROL_MEANS_LOG_KEY: all_learner_actions, REWARDS_LOG_KEY: all_rewards, LEARNER_CONTROL_DEVIATIONS_LOG_KEY: all_learner_deviations,
                             LEARNER_CONTROL_MAXIMUMS_LOG_KEY: all_learner_max_actions, LEARNER_CONTROL_MINIMUMS_LOG_KEY: all_learner_min_actions, DEMONSTRATOR_Q_VALUES_LOG_KEY: q_values,
                              DEMONSTRATOR_CONTROLS_LOG_KEY: all_demonstrator_actions, NUMBER_INTERJECTIONS_KEY: demonstrator_interjections}
            file_to_store_data = iteration_identifier + 'logs.pkl'
            with open(file_to_store_data, 'wb') as f:
                pickle.dump(data_to_store, f)
            controller.train_pg(data_x=all_states[:-1], data_y=all_learner_actions[:-1], data_q=q_values, epochs=epochs, configuration_identity=iteration_identifier)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-en', '--env_name', type=str, help='MuJoCo domain', choices=['HalfCheetah-v1', 'Swimmer-v1'])
    parser.add_argument('-l', '--learner', type=str, help='Learner', choices=['BBB', 'DNN'])
    parser.add_argument('-e', '--experiment', type=str, help='Type of Experiment', choices=['Sanity_Check', 'Data_Efficiency', 'Catastrophic_Forgetting', 'Suboptimal_Demonstrations', 'Safety'])
    parser.add_argument('-ws', '--window_size', type=int, help='Number of time-steps in a moving window', default=1)
    parser.add_argument('-nid', '--number_initial_demonstrations', type=int, help='Number of initial demonstrations', default=1)

    args = parser.parse_args()

    run_probablistic_aggrevated(env_name=args.env_name, learner=args.learner, experiment=args.experiment, window_size=args.window_size, number_initial_demonstrations=args.number_initial_demonstrations, epochs = 10001,
     number_mini_batches = 20, activation_unit = 'RELU', learning_rate = 0.001, hidden_units= [90, 30, 10], number_samples_variance_reduction = 25, precision_alpha = 0.01,
       weights_prior_mean_1 = 0., weights_prior_mean_2 = 0., weights_prior_deviation_1 = 0.4, weights_prior_deviation_2 = 0.4, mixture_pie = 0.7, rho_mean = -3.,
        extra_likelihood_emphasis = 10000000000000000.)