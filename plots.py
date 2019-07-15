import _pickle as pickle
from Housekeeping import *
import matplotlib.pyplot as plt
import math
from scipy.stats.stats import pearsonr
from IPython.display import HTML, display

class Tag:
    def __init__(self, name, value):
        self.name = name
        self.value = value
    def __repr__(self):
        return "<%s>%s</%s>"%(self.name, str(self.value), self.name)
        
class Linear:
    def __init__(self, data):
        self.data = list(data)
    def __repr__(self):
        return ''.join(list(map(str, self.data)))

class Table:
    def __init__(self, cols):
        self.header = list(iter(cols))
        self.len = len(self.header)
        self.contents = []
        self.compiled = False
    def add(self, data):
        if(len(data) != self.len):
            raise AssertionError
        self.contents.append(data)
        self.compiled = False
    
    def __repr__(self):
        if self.compiled:
            return self.compiledHTML
        header = Tag("tr", Linear(list(map(lambda x:Tag('th', x), self.header))))
        contents = []
        for c in self.contents:
            contents += [ Tag("tr", Linear(list(map(lambda x:Tag('td', x), c))))]
        
        self.compiledHTML = str(Tag('Table', Linear((header, Linear(contents)))))
        self.compiled = True
        return self.compiledHTML
    
    def display_notebook(self):
        display(HTML(str(self)))


def plot_prediction_tasks(experiment, controller_type, regularizer, extra_likelihood_emphasis_code, cost_type):
    configuration_identifier = dataset + '_' + controller_type + '_' + regularizer + '_' + str(extra_likelihood_emphasis_code) + '_' + cost_type + '/'

    # Load 
    # 1. y_hat, y_dev, y_max, y_min
    # 2. train_pr_cost, train_var_map_cost, train_ll_cost, train_cost, train_pred_err
    # 3. val_pr_cost, val_var_map_cost, val_ll_cost, val_cost, val_pred_err

    plt.plot(np.array(x_val).flatten(), np.array(y_val).flatten(), label='True')
    plt.plot(np.array(x_val).flatten(), np.array(y_hat).flatten(), label='Prediction', color='black')
    plt.fill_between(np.array(x_val).flatten(), np.array(y_hat).flatten(), np.array(y_hat).flatten()+np.array(y_dev).flatten(), facecolor='green', alpha=0.4, label='+1 sigma')
    plt.fill_between(np.array(x_val).flatten(), np.array(y_hat).flatten(), np.array(y_hat).flatten()-np.array(y_dev).flatten(), facecolor='green', alpha=0.4, label='-1 sigma')
    plt.fill_between(np.array(x_val).flatten(), np.array(y_hat).flatten(), np.array(y_hat).flatten()+np.array(y_max).flatten(), facecolor='green', alpha=0.2, label='Maximum')
    plt.fill_between(np.array(x_val).flatten(), np.array(y_hat).flatten(), np.array(y_hat).flatten()-np.array(y_min).flatten(), facecolor='green', alpha=0.2, label='Minimum')
    plt.legend()

    x = Table(('Dataset', 'Prior Cost', 'Variational MAP cost', '-ve Log Likelihood Cost', 'Cost', 'Predictive Error'))
    x.add(('Train', train_pr_cost, train_var_map_cost, train_ll_cost, train_cost, train_pred_err))
    x.add(('Test', val_pr_cost, val_var_map_cost, val_ll_cost, val_cost, val_pred_err))
    x.display_notebook()

def plot_MDP(experiment, controller_type, extra_likelihood_emphasis_code):
    all_regularizers = ['PAC_Bayes', 'Unregularized']
    x = Table(('Experiment', 'Tr nll', 'Tr pen in bound', 'Val nll', 'Tr ELBO', 'Tr Bound', 'Tr. cost', 'Tr pred err', 'Tr R',  'Test pred err', 'Test R', 'Test ELBO', 'Test bound', 'Test cost', 'pred err 2'))
    for regularizer in all_regularizers:
        train_logs_file = './logs/' + experiment + '_' + controller_type + '_' + str(extra_likelihood_emphasis_code) + '/' + 'training_data.pkl'
        with open(train_logs_file, 'rb') as f:
            train_data = pickle.load(f)
        train_R = np.sum(train_data[REWARDS_KEY])/NUM_DEM_TRAIN
        val_logs_file = './logs/' + experiment + '_' + controller_type + '_' + str(extra_likelihood_emphasis_code) + '/' + regularizer + '/validation_logs.pkl'
        with open(val_logs_file, 'rb') as f:
            loaded_data = pickle.load(f)
        train_pred_err = loaded_data[TRAIN_PRED_ERR_KEY]
        train_ll = loaded_data[TRAIN_LL_COST_KEY]
        val_ll = np.mean(loaded_data[VAL_LL_COST_KEY])
        train_ELBO = loaded_data[TRAIN_ELBO_KEY]
        train_bound = loaded_data[TRAIN_PAC_BAYES_REGULARIZER_KEY]
        if regularizer == 'PAC_Bayes':
            tr_penalty = train_bound + (math.pow(10, int(extra_likelihood_emphasis_code)) * train_ll)
        else:
            tr_penalty = 0.
        train_cost = loaded_data[TRAIN_COST_KEY]
        demonstrator_action_across_rollouts = loaded_data[DEMONSTRATOR_ACTION_KEY]
        controller_action_across_rollouts = loaded_data[CONTROLLER_ACTION_KEY]
        rewards_across_rollouts = loaded_data[REWARDS_KEY]
        all_ELBO_across_rollouts = loaded_data[VAL_ELBO_KEY]
        #all_squashed_ELBO_across_rollouts = loaded_data[VAL_SQUASHED_ELBO_KEY]
        all_pac_bayes_regularizer_across_rollouts = loaded_data[VAL_PAC_BAYES_REGULARIZER_KEY]
        all_cost_across_rollouts = loaded_data[VAL_COST_KEY]
        all_pred_err_across_rollouts = loaded_data[VAL_PRED_ERR_KEY]
        ### Average mean squared predictive error across rollouts ###
        average_pred_err = 0.
        for dem_rollout, con_rollout in zip(demonstrator_action_across_rollouts, controller_action_across_rollouts):
            pred_err = 0.
            for ts in range(len(dem_rollout)):
                pred_err += np.mean((np.subtract(dem_rollout[ts], con_rollout[ts]))**2)
            average_pred_err += pred_err
        average_pred_err = average_pred_err/len(demonstrator_action_across_rollouts)
        ### Rewards obtained across rollouts ###
        episodic_rewards = np.mean([sum(rollout) for rollout in rewards_across_rollouts])
        ELBOs = np.mean([sum(rollout) for rollout in all_ELBO_across_rollouts])
        #squashed_ELBOs = np.mean([sum(rollout) for rollout in all_squashed_ELBO_across_rollouts])
        val_bound = np.mean([sum(rollout) for rollout in all_pac_bayes_regularizer_across_rollouts])
        costs = np.mean([sum(rollout) for rollout in all_cost_across_rollouts])
        pred_errs_2 = np.mean([sum(rollout) for rollout in all_pred_err_across_rollouts])
        x.add((regularizer+' '+experiment, -train_ll, tr_penalty, -val_ll, train_ELBO, train_bound, train_cost, train_pred_err, train_R, average_pred_err, episodic_rewards, ELBOs, val_bound, costs, pred_errs_2))
    x.display_notebook()

def plot_ELBO_n_bound_progression(experiment, controller_type, extra_likelihood_emphasis_code, MC, PB_N, ss):
    file_name = './logs/' + experiment + '/' + str(PB_N) + '/' + controller_type + '_' + str(extra_likelihood_emphasis_code) + '_PAC_Bayes/' + str(MC) + '/' + str(ss) + '/elbo_n_bound_convergence.pkl'
    print(RED(file_name))
    with open(file_name, 'rb') as f:
        progression_data = pickle.load(f)
    ELBOs = [elbo/(math.pow(10, int(extra_likelihood_emphasis_code))) for elbo in progression_data[ELBO_CONVERGENCE_KEY]]
    PB_bounds = [bound/(math.pow(10, int(extra_likelihood_emphasis_code))) for bound in progression_data[PAC_BAYES_BOUND_CONVERGENCE_KEY]]
    #ELBOs = progression_data[ELBO_CONVERGENCE_KEY]
    #PB_bounds = progression_data[PAC_BAYES_BOUND_CONVERGENCE_KEY]

    #print(ELBOs[0])
    #print(PB_bounds[0])
    
    plt.plot(np.arange(0, len(PB_bounds), 1), PB_bounds, label='PAC Bayes Bound')
    #plt.ylim(bottom=0., top=top_lim)
    plt.xlabel('Epochs', fontweight='bold')
    plt.ylabel('Numerical Value', fontweight='bold')
    #plt.title('Positive Correlation between ELBO and PAC-Bayes bound', fontsize=10)
    plt.title('Configuration '+str(extra_likelihood_emphasis_code), fontsize=15)
    plt.legend()
    plt.legend()
    plt.show()

    plt.plot(np.arange(0, len(ELBOs), 1), ELBOs, label='ELBO')
    #plt.ylim(bottom=0., top=top_lim)
    plt.xlabel('Epochs', fontweight='bold')
    plt.ylabel('Numerical Value', fontweight='bold')
    #plt.title('Positive Correlation between ELBO and PAC-Bayes bound', fontsize=10)
    plt.title('Configuration '+str(extra_likelihood_emphasis_code), fontsize=15)
    plt.legend()
    plt.legend()
    plt.show()

    plt.plot(PB_bounds, ELBOs)
    #plt.ylim(bottom=0., top=top_lim)
    plt.xlabel('Bounds', fontweight='bold')
    plt.ylabel('ELBOs', fontweight='bold')
    #plt.title('Positive Correlation between ELBO and PAC-Bayes bound', fontsize=10)
    #plt.title('Configuration '+str(extra_likelihood_emphasis_code), fontsize=15)
    plt.legend()
    plt.legend()
    plt.show()

def generalization_risk_check(experiments, controller_type, emphasis_codes, MC, PB_N, ss):
    for experiment in experiments:
        print(BLUE(experiment))
        x = Table(['Metric/Emphasis Code'] + [str(emph_code) for emph_code in emphasis_codes])
        nll_test = ['Test nll']
        PB_bound = ['Bound']
        for emph_code in emphasis_codes:
            val_logs_file = './logs/' + experiment + '/' + str(PB_N) + '/' + controller_type + '_' + str(emph_code) + '_PAC_Bayes/' + str(MC) + '/' + str(ss) + '/validation_logs.pkl'
            with open(val_logs_file, 'rb') as f:
                val_data = pickle.load(f)
            #print(RED(val_data[VAL_LL_COST_KEY][0]))
            if experiment == 'Hopper' or experiment == 'Humanoid' or experiment == 'Walker2d' or experiment == 'Ant':
                nll = np.mean([np.mean(row) for row in val_data[VAL_LL_COST_KEY]])
                nll_test.append(-np.mean(nll))
            else:
                nll_test.append(-np.mean(val_data[VAL_LL_COST_KEY])*(math.pow(10, int(emph_code))))
            PB_bound.append(val_data[TRAIN_PAC_BAYES_REGULARIZER_KEY])
            #print(val_data[TRAIN_LL_COST_KEY])
        x.add(nll_test)
        x.add(PB_bound)
        x.display_notebook()
        print()

def get_ELBO_bound_correlation(experiments, controller_type, emphasis_codes, MC, PB_N, ss):
    #emphasis_codes = [0, 4, 8, 12]
    for experiment in experiments:
        print(BLUE(experiment))
        train_logs_file = './logs/' + experiment + '/' + str(PB_N) + '/training_data.pkl'
        with open(train_logs_file, 'rb') as f:
            train_data = pickle.load(f)
        print(RED('Average demonstrator reward is ' + str(np.sum(train_data[REWARDS_KEY])/NUM_DEM_TRAIN)))
        x = Table(['Metric/Emphasis Code'] + [str(emph_code) for emph_code in emphasis_codes])
        pearson_correlations = ['Pearson Correlation Coefficient']
        P_values = ['P Values']
        episodic_rewards = ['Episodic R']
        for emph_code in emphasis_codes:
            file_name = './logs/' + experiment + '/' + str(PB_N) + '/' + controller_type + '_' + str(emph_code) + '_PAC_Bayes/' + str(MC) + '/' + str(ss) + '/elbo_n_bound_convergence.pkl'
            with open(file_name, 'rb') as f:
                progression_data = pickle.load(f)
            ELBOs = progression_data[ELBO_CONVERGENCE_KEY]
            PB_bounds = progression_data[PAC_BAYES_BOUND_CONVERGENCE_KEY]
            pearson_corr, p_value = pearsonr(x=ELBOs, y=PB_bounds)
            pearson_correlations.append(str(pearson_corr))
            P_values.append(str(p_value))
            val_logs_file = './logs/' + experiment + '/' + str(PB_N) + '/' + controller_type + '_' + str(emph_code) + '_PAC_Bayes/' + str(MC) + '/' + str(ss) + '/validation_logs.pkl'
            with open(val_logs_file, 'rb') as f:
                val_data = pickle.load(f)
            episodic_rewards.append(str(np.mean([sum(rollout) for rollout in val_data[REWARDS_KEY]])))
        #print(pearson_correlations)
        #print(P_values)
        x.add(pearson_correlations)
        x.add(P_values)
        x.add(episodic_rewards)
        x.display_notebook()
        print()

def introspect_generalization(experiment, controller_type, emphasis_codes, MC, PB_N, ss):
    train_logs_file = './logs/' + experiment + '/' + str(PB_N) + '/training_data.pkl'
    with open(train_logs_file, 'rb') as f:
        train_data = pickle.load(f)
    det_logs_file = './logs/' + experiment + '/' + str(PB_N) + '/deterministic_Unregularized/generalization_across_tasks.pkl'
    with open(det_logs_file, 'rb') as f:
        det_data = pickle.load(f)
    print(RED('Average demonstrator reward is ' + str(np.sum(train_data[REWARDS_KEY])/NUM_DEM_TRAIN)))
    for emph_code in emphasis_codes:
        print(BLUE('Emphasis code is ' + str(emph_code)))
        x = Table(('Controller/Task Code', '0', '1'))
        rewards_row = ['PAC_Bayes']
        #for task_identity in range(10):
        file_name = './logs/' + experiment + '/' + str(PB_N) + '/' + controller_type + '_' + str(emph_code) + '_PAC_Bayes/' + str(MC) + '/' + str(ss) + '/generalization_across_tasks.pkl'
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        rewards_row = rewards_row + [str(np.mean(task_rewards)) for task_rewards in data]
        x.add(rewards_row)
        x.add(['Deterministic'] + [str(np.mean(task_rewards)) for task_rewards in det_data])
        x.display_notebook()
        print()