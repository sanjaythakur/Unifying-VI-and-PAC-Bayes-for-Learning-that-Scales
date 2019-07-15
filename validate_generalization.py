import tensorflow as tf 

from gym.multiple_tasks import get_task_on_MUJOCO_environment
import _pickle as pickle
import argparse
from BBB import LoadBBBRegressor
from Housekeeping import *
from Deterministic_Neural_Network import LoadDNNRegressor

def validate_generalization(experiment, num_rollouts, controller, configuration_identifier, regularizer, number_samples_variance_reduction, PB_N, ss):
    #configuration_identifier = configuration_identifier
    rewards_across_tasks = []
    NUM_TASKS = 2
    for val_env in range(NUM_TASKS):
        env_rewards = []
        for rollout in range(num_rollouts):
            env = get_task_on_MUJOCO_environment(env_name=experiment, task_identity=str(val_env))
            observation = env.reset()
            done = False
            time_step = 0.
            episodic_reward = 0.
            while not done:
                observation = observation.astype(np.float32).reshape((1, -1))
                observation = np.append(observation, [[time_step]], axis=1)  # add time step feature
                mean_action, dev_action, max_action, min_action = controller.predict(data_x=observation)
                observation, reward, done, info = env.step(mean_action)
                episodic_reward += reward               
                time_step += 1e-3   
            env_rewards.append(episodic_reward)
        rewards_across_tasks.append(env_rewards)
    file_name = configuration_identifier + 'generalization_across_tasks.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(rewards_across_tasks, f)

def run_validation(experiment, controller_type, regularizer, extra_likelihood_emphasis_code, number_samples_variance_reduction, PB_N, ss):
    if controller_type == 'probabilistic':
        configuration_identifier = './logs/' + experiment + '/' + str(PB_N) + '/' + controller_type + '_' + str(extra_likelihood_emphasis_code) + '_' + regularizer + '/' + str(number_samples_variance_reduction) + '/' + str(ss) + '/'
        controller = LoadBBBRegressor(controller_identity=configuration_identifier)
    elif controller_type == 'deterministic':
        configuration_identifier = './logs/' + experiment + '/' + controller_type + '_' + regularizer + '/'
        controller = LoadDNNRegressor(controller_identity=configuration_identifier)
    print(RED(configuration_identifier))
    validate_generalization(experiment=experiment, num_rollouts=10, controller=controller, configuration_identifier=configuration_identifier, regularizer=regularizer, number_samples_variance_reduction=number_samples_variance_reduction, PB_N=PB_N, ss=ss) 

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