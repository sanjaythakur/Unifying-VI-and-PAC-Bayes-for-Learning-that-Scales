import numpy as np

#Defining colors for highlighting important aspects
GREEN = lambda x: '\x1b[32m{}\x1b[0m'.format(x)
BLUE = lambda x: '\x1b[34m{}\x1b[0m'.format(x)
RED = lambda x: '\x1b[31m{}\x1b[0m'.format(x)

INPUT_MANIPULATION_DIRECTORY = 'input_manipulation_directory/'
TENSORBOARD_DIRECTORY = 'tensorboard_directory/'
SAVED_DEMONSTRATOR_MODELS_DIRECTORY = 'saved_demonstrator_models/'
SAVED_MODELS_DURING_ITERATIONS_DIRECTORY = 'saved_models_during_iterations/'
SAVED_FINAL_MODEL_DIRECTORY = 'saved_final_model/'
LOGS_DIRECTORY = 'logs/'

NORMALIZE = lambda data, mean, deviation: np.divide(np.subtract(data, mean), deviation)
REVERSE_NORMALIZE = lambda data, mean, deviation: np.add((data * deviation), mean)

MEAN_KEY_X = 'mean_key_x'
DEVIATION_KEY_X = 'deviation_key_x'
MEAN_KEY_Y = 'mean_key_y'
DEVIATION_KEY_Y = 'deviation_key_y'

SCALE_KEY = 'SCALE'
OFFSET_KEY = 'OFFSET'

ELBO_CONVERGENCE_KEY = 'elbo_convergence_key'
PAC_BAYES_BOUND_CONVERGENCE_KEY = 'pac_bayes_bound_convergence_key'

X_TRAIN_KEY = 'x_train_key'
Y_TRAIN_KEY = 'y_train_key'
X_TEST_KEY = 'x_test_key'
Y_TEST_KEY = 'y_test_key'
Y_HAT_KEY = 'y_hat_key'
Y_DEV_KEY = 'y_dev_key'
Y_MAX_KEY = 'y_max_key'
Y_MIN_KEY = 'y_min_key'
TRAIN_PRIOR_COST_KEY = 'train_prior_cost_key'
TRAIN_VAR_MAP_COST_KEY = 'train_var_map_cost_key'
TRAIN_LL_COST_KEY = 'train_ll_cost_key'
TRAIN_ELBO_KEY = 'train_ELBO_key'
#TRAIN_SQUASHED_ELBO_KEY = 'train_squashed_ELBO_key'
TRAIN_PAC_BAYES_REGULARIZER_KEY = 'train_pac_bayes_regularizer_key'
TRAIN_COST_KEY = 'train_cost_key'
TRAIN_PRED_ERR_KEY = 'train_pred_err_key'
VAL_PRIOR_COST_KEY = 'val_prior_cost_key'
VAL_VAR_MAP_COST_KEY = 'val_var_map_cost_key'
VAL_LL_COST_KEY = 'val_ll_cost_key'
VAL_ELBO_KEY = 'val_ELBO_key'
#VAL_SQUASHED_ELBO_KEY = 'val_squashed_ELBO_key'
VAL_PAC_BAYES_REGULARIZER_KEY = 'val_pac_bayes_regularizer_key'
VAL_COST_KEY = 'val_cost_key'
VAL_PRED_ERR_KEY = 'val_pred_err_key'

STATES_KEY = 'states_key'
DEMONSTRATOR_ACTION_KEY = 'demonstrator_action_key'
CONTROLLER_ACTION_KEY = 'controller_action_key'
REWARDS_KEY = 'rewards_key'

def get_mean_and_deviation(data):
  mean_data = np.mean(data, axis = 0)
  deviation_data = np.std(data, axis = 0)
  for feature_index in range(deviation_data.shape[0]):
    if deviation_data[feature_index] == 0.:
      if mean_data[feature_index] == 0.:
        # This means all the values are 0.
        deviation_data[feature_index] = 1.
      else:
        # This means all the values are equal but not equal to 0.
        deviation_data[feature_index] = mean_data[feature_index]
  return mean_data, deviation_data

def str_to_bool(s):
  if s == 'True':
       return True
  elif s == 'False':
       return False

def randomize(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b