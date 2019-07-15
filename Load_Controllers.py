import tensorflow as tf
import _pickle as pickle, sys
import gym
import os

import sys
from Housekeeping import *

class Load_BBB():
    def __init__(self, controller_identity, session):
        graph = tf.get_default_graph()
        meta_information_directory_copycat = controller_identity + 'training/' + SAVED_FINAL_MODEL_DIRECTORY
        best_model_directory_copycat = controller_identity + 'training/' + SAVED_MODELS_DURING_ITERATIONS_DIRECTORY
        imported_meta = tf.train.import_meta_graph(meta_information_directory_copycat + 'final.meta')
        imported_meta.restore(session, tf.train.latest_checkpoint(best_model_directory_copycat))
        self.input_x = graph.get_tensor_by_name('inputs/input_x:0')
        self.input_y = graph.get_tensor_by_name('inputs/input_y:0')
        self.N_BOUND = graph.get_tensor_by_name('inputs/N_BOUND:0')
        self.mean_of_predictions = graph.get_tensor_by_name('final_outputs/pred_mean:0')
        self.deviation_of_predictions = graph.get_tensor_by_name('final_outputs/pred_sigma:0')
        self.maximum_of_predictions = graph.get_tensor_by_name('final_outputs/pred_max:0')
        self.minimum_of_predictions = graph.get_tensor_by_name('final_outputs/pred_min:0')
        self.pred_err = graph.get_tensor_by_name('error/pred_err:0')
        #self.pr_cost = graph.get_tensor_by_name('cost/prior_cost:0')
        #self.var_MAP_cost = graph.get_tensor_by_name('cost/var_MAP_cost:0')
        self.ll_cost = graph.get_tensor_by_name('cost/likelihood_cost:0')
        self.ELBO = graph.get_tensor_by_name('cost/ELBO:0')
        #self.squashed_ELBO = graph.get_tensor_by_name('cost/squashed_ELBO:0')
        #self.squashed_ELBO = graph.get_tensor_by_name('cost/squashed_ELBO:0')
        self.pac_bayes_bound = graph.get_tensor_by_name('cost/pac_bayes_bound:0')
        self.cost = graph.get_tensor_by_name('cost/cost:0')
        self.getMetaData(controller_identity)

    def getMetaData(self, controller_identity):
        relevant_file_name = controller_identity + 'training/' + 'input_manipulation_data.pkl'
        with open(relevant_file_name, 'rb') as f:
            stored_meta_data = pickle.load(f)
        self.mean_x = stored_meta_data[MEAN_KEY_X]
        self.deviation_x = stored_meta_data[DEVIATION_KEY_X]
        self.mean_y = stored_meta_data[MEAN_KEY_Y]
        self.deviation_y = stored_meta_data[DEVIATION_KEY_Y]


class Load_MuJoCo_Demonstrator():
    def __init__(self, env_name):
        self.env_name=env_name 
        self.getScaleAndOffset()
        policy_graph = tf.Graph()
        demonstrator_policy = './MuJoCo_Demonstrator/saved_models/' + self.env_name + '/policy/'
        self.sess_policy = tf.Session(graph=policy_graph)
        with policy_graph.as_default():
            imported_meta = tf.train.import_meta_graph(demonstrator_policy + 'final.meta')
            imported_meta.restore(self.sess_policy, tf.train.latest_checkpoint(demonstrator_policy))
            self.scaled_observation_node_policy = policy_graph.get_tensor_by_name('obs:0')
            self.output_action_node = policy_graph.get_tensor_by_name('output_action:0')
        valfunc_graph = tf.Graph()
        demonstrator_valfunc = './MuJoCo_Demonstrator/saved_models/' + self.env_name + '/valfunc/'
        self.sess_valfunc = tf.Session(graph=valfunc_graph)
        with valfunc_graph.as_default():
            imported_meta = tf.train.import_meta_graph(demonstrator_valfunc + 'final.meta')
            imported_meta.restore(self.sess_valfunc, tf.train.latest_checkpoint(demonstrator_valfunc))
            self.scaled_observation_node_valfunc = valfunc_graph.get_tensor_by_name('obs_valfunc:0')
            self.output_value_node = valfunc_graph.get_tensor_by_name('output_valfunc:0')

    def getScaleAndOffset(self, ):
        file_name = './MuJoCo_Demonstrator/saved_models/' + self.env_name + '/scale_and_offset.pkl'
        with open(file_name, "rb") as f:
            data_stored = pickle.load(f)
        self.scale = data_stored[SCALE_KEY]
        self.offset = data_stored[OFFSET_KEY]

    def getStateActionPairs(self, configuration, window_size=1, number_rollouts=1):
        file_name = configuration + 'training_data.pkl'
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                loaded_demonstrations = pickle.load(f)
            all_states = loaded_demonstrations[X_TRAIN_KEY]
            all_actions = loaded_demonstrations[Y_TRAIN_KEY]
            #all_rewards = loaded_demonstrations[REWARDS_KEY]
        else:
            all_states = []
            all_actions = []
            all_rewards = []
            for _ in range(number_rollouts):
                env = gym.make(self.env_name + '-v1')
                observation = env.reset()
                done = False
                #total_reward = 0.
                time_step = 0.
                while not done:
                    observation = observation.astype(np.float32).reshape((1, -1))
                    observation = np.append(observation, [[time_step]], axis=1)  # add time step feature
                    all_states.append(observation[0])
                    action = self.sess_policy.run(self.output_action_node, feed_dict={self.scaled_observation_node_policy: (observation - self.offset) * self.scale})
                    all_actions.append(action[0])
                    #value = self.sess_valfunc.run(self.output_value_node, feed_dict={self.scaled_observation_node_valfunc: (observation - self.offset) * self.scale})
                    #print(value)
                    observation, reward, done, info = env.step(action)
                    all_rewards.append(reward)
                    #total_reward += reward
                    time_step += 1e-3
                #print(BLUE(str(total_reward)))
            all_states, all_actions, all_rewards = np.array(all_states), np.array(all_actions), np.array(all_rewards)
            demonstrations_to_save = {X_TRAIN_KEY:all_states, Y_TRAIN_KEY: all_actions, REWARDS_KEY: all_rewards}
            with open(file_name, 'wb') as f:
                pickle.dump(demonstrations_to_save, f)
        return all_states, all_actions

    def getDemonstratorCostToGo(self, next_state, reward):
        #next_state = np.append(next_state, time_step, axis=1)
        value = self.sess_valfunc.run(self.output_value_node, feed_dict={self.scaled_observation_node_valfunc: (next_state - self.offset) * self.scale})
        return reward + .995*value

    def getDemonstratorAction(self, state):
        #state = np.append(state, time_step, axis=1)
        action = self.sess_policy.run(self.output_action_node, feed_dict={self.scaled_observation_node_policy: (state - self.offset) * self.scale})
        return action


class Load_DNN():
    def __init__(self, controller_identity, session):
        graph = tf.get_default_graph()
        meta_information_directory_copycat = controller_identity + 'training/' + SAVED_FINAL_MODEL_DIRECTORY
        best_model_directory_copycat = controller_identity + 'training/' + SAVED_MODELS_DURING_ITERATIONS_DIRECTORY
        imported_meta = tf.train.import_meta_graph(meta_information_directory_copycat + 'final.meta')
        imported_meta.restore(session, tf.train.latest_checkpoint(best_model_directory_copycat))
        self.input_x = graph.get_tensor_by_name('inputs/input_x:0')
        self.input_y = graph.get_tensor_by_name('inputs/input_y:0')
        self.pred = graph.get_tensor_by_name('input_output_forward_pass_mapping/prediction:0')
        self.cost = graph.get_tensor_by_name('cost/mean_squared_error:0')
        self.getMetaData(controller_identity)

    def getMetaData(self, controller_identity):
        relevant_file_name = controller_identity + 'training/' + 'input_manipulation_data.pkl'
        with open(relevant_file_name, 'rb') as f:
            stored_meta_data = pickle.load(f)
        self.mean_x = stored_meta_data[MEAN_KEY_X]
        self.deviation_x = stored_meta_data[DEVIATION_KEY_X]
        self.mean_y = stored_meta_data[MEAN_KEY_Y]
        self.deviation_y = stored_meta_data[DEVIATION_KEY_Y]