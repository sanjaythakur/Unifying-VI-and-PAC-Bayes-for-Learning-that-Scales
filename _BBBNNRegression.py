import tensorflow as tf
import math

class _BBBNNRegression():

    def __init__(self, dataset_type, input_dimensions, regularizer, number_mini_batches, number_output_units, activation_unit, learning_rate,
     hidden_units, number_samples_variance_reduction, precision_alpha, weights_prior_mean_1, weights_prior_mean_2,
      weights_prior_deviation_1, weights_prior_deviation_2, mixture_pie, rho_mean, extra_likelihood_emphasis, delta=0.1,
       num_classes=1, num_dimensions=1, ss=0):
        number_output_units = num_classes * num_dimensions
        self.dataset_type = dataset_type
        self.num_classes = num_classes
        self.regularizer = regularizer
        self.number_mini_batches = tf.constant(number_mini_batches, dtype=tf.int64)
        self.activation_unit = activation_unit
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.number_samples_variance_reduction = number_samples_variance_reduction
        self.precision_alpha = precision_alpha
        self.weights_prior_mean_1 = weights_prior_mean_1
        self.weights_prior_mean_2 = weights_prior_mean_2
        self.weights_prior_deviation_1 = weights_prior_deviation_1
        self.weights_prior_deviation_2 = weights_prior_deviation_2
        self.mixture_pie = mixture_pie
        self.rho_mean = rho_mean
        self.extra_likelihood_emphasis = extra_likelihood_emphasis

        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
        self.mini_batch_index = tf.Variable(1, dtype=tf.int64, trainable=False, name='mini_batch_index')

        #self.all_prior_cost = 0
        #self.all_variational_MAP_cost = 0
        self.all_likelihood_cost = 0

        output_forward_pass_1 = None
        output_forward_pass_2 = None
        output_forward_pass_3 = None
        output_forward_pass_4 = None
        output_forward_pass_5 = None
        output_forward_pass = None

        # Mixture Prior
        #ds = tf.contrib.distributions
        #self.WEIGHTS_PRIOR = ds.Mixture(cat=ds.Categorical(probs=[self.mixture_pie, 1.- self.mixture_pie]),
        #                                components=[ds.Normal(loc=self.weights_prior_mean_1, scale=self.weights_prior_deviation_1), ds.Normal(loc=self.weights_prior_mean_2, scale=self.weights_prior_deviation_2)],
        #                                name='WEIGHTS_MIXTURE_PRIOR')

        self.WEIGHTS_PRIOR = tf.distributions.Normal(loc=0., scale=1., name='WEIGHTS_PRIOR')

        with tf.name_scope('inputs'):
            self.input_x = tf.placeholder(tf.float32, shape=(None, input_dimensions), name='input_x')
            self.input_y = tf.placeholder(tf.float32, shape=(None, number_output_units), name='input_y')
            self.N_BOUND = tf.placeholder(tf.float32, shape=(), name='N_BOUND')            

        with tf.name_scope('input_output_forward_pass_mapping'):
            with tf.name_scope('between_input_and_first_hidden_layer'):
                self.sampled_weights_1, self.sampled_biases_1, self.mu_weights_1, self.rho_weights_1, self.mu_biases_1, self.rho_biases_1 = self.get_weights_and_phi(shape=(input_dimensions, self.hidden_units[0]))
                #self.all_variational_MAP_cost = self.all_variational_MAP_cost + variational_MAP_cost
                #self.all_prior_cost = self.all_prior_cost + prior_cost
                for variance_reductor_iterator in range(self.number_samples_variance_reduction):
                    if output_forward_pass_1 == None:
                        output_forward_pass_1 = self.fetch_ACTIVATION_UNIT(tf.matmul(self.input_x, self.sampled_weights_1[variance_reductor_iterator]) + self.sampled_biases_1[variance_reductor_iterator])[None]
                    else:
                        output_forward_pass_1 = tf.concat([output_forward_pass_1, self.fetch_ACTIVATION_UNIT(tf.matmul(self.input_x, self.sampled_weights_1[variance_reductor_iterator]) + self.sampled_biases_1[variance_reductor_iterator])[None]], 0)            
            
            with tf.name_scope('between_hidden_layers'):
                self.sampled_weights_2, self.sampled_biases_2, self.mu_weights_2, self.rho_weights_2, self.mu_biases_2, self.rho_biases_2 = self.get_weights_and_phi(shape=(self.hidden_units[0], self.hidden_units[1]))
                #self.all_variational_MAP_cost = self.all_variational_MAP_cost + variational_MAP_cost
                #self.all_prior_cost = self.all_prior_cost + prior_cost
                for variance_reductor_iterator in range(self.number_samples_variance_reduction):
                    if output_forward_pass_2 == None:
                        output_forward_pass_2 = self.fetch_ACTIVATION_UNIT(tf.matmul(output_forward_pass_1[variance_reductor_iterator], self.sampled_weights_2[variance_reductor_iterator]) + self.sampled_biases_2[variance_reductor_iterator])[None]
                    else:
                        output_forward_pass_2 = tf.concat([output_forward_pass_2, self.fetch_ACTIVATION_UNIT(tf.matmul(output_forward_pass_1[variance_reductor_iterator], self.sampled_weights_2[variance_reductor_iterator]) + self.sampled_biases_2[variance_reductor_iterator])[None]], 0)
                self.sampled_weights_3, self.sampled_biases_3, self.mu_weights_3, self.rho_weights_3, self.mu_biases_3, self.rho_biases_3 = self.get_weights_and_phi(shape=(self.hidden_units[1], self.hidden_units[2]))
                #self.all_variational_MAP_cost = self.all_variational_MAP_cost + variational_MAP_cost
                #self.all_prior_cost = self.all_prior_cost + prior_cost
                for variance_reductor_iterator in range(self.number_samples_variance_reduction):
                    if output_forward_pass_3 == None:
                        output_forward_pass_3 = self.fetch_ACTIVATION_UNIT(tf.matmul(output_forward_pass_2[variance_reductor_iterator], self.sampled_weights_3[variance_reductor_iterator]) + self.sampled_biases_3[variance_reductor_iterator])[None]
                    else:
                        output_forward_pass_3 = tf.concat([output_forward_pass_3, self.fetch_ACTIVATION_UNIT(tf.matmul(output_forward_pass_2[variance_reductor_iterator], self.sampled_weights_3[variance_reductor_iterator]) + self.sampled_biases_3[variance_reductor_iterator])[None]], 0)

            with tf.name_scope('between_last_hidden_and_output_layer'):
                self.sampled_weights_4, self.sampled_biases_4, self.mu_weights_4, self.rho_weights_4, self.mu_biases_4, self.rho_biases_4 = self.get_weights_and_phi(shape=(self.hidden_units[-1], number_output_units))
                #self.all_variational_MAP_cost = self.all_variational_MAP_cost + variational_MAP_cost
                #self.all_prior_cost = self.all_prior_cost + prior_cost
                for variance_reductor_iterator in range(self.number_samples_variance_reduction):
                    if dataset_type == 'categorical':
                        if output_forward_pass == None:
                            output_forward_pass = tf.nn.softmax(tf.matmul(output_forward_pass_3[variance_reductor_iterator], self.sampled_weights_4[variance_reductor_iterator]) + self.sampled_biases_4[variance_reductor_iterator])[None]
                        else:
                            output_forward_pass = tf.concat([output_forward_pass, (tf.matmul(output_forward_pass_3[variance_reductor_iterator], self.sampled_weights_4[variance_reductor_iterator]) + self.sampled_biases_4[variance_reductor_iterator])[None]], 0)
                        model_distribution = tf.distributions.Categorical(probs=output_forward_pass)
                        self.all_likelihood_cost = self.all_likelihood_cost + tf.reduce_sum(model_distribution.log_prob(self.input_y))
                    elif dataset_type == 'continuous':
                        if output_forward_pass == None:
                            output_forward_pass = (tf.matmul(output_forward_pass_3[variance_reductor_iterator], self.sampled_weights_4[variance_reductor_iterator]) + self.sampled_biases_4[variance_reductor_iterator])[None]
                        else:
                            output_forward_pass = tf.concat([output_forward_pass, (tf.matmul(output_forward_pass_3[variance_reductor_iterator], self.sampled_weights_4[variance_reductor_iterator]) + self.sampled_biases_4[variance_reductor_iterator])[None]], 0)
                        model_distribution = tf.distributions.Normal(loc=output_forward_pass[variance_reductor_iterator], scale=(1.0/tf.sqrt(self.precision_alpha)))
                        self.all_likelihood_cost = self.all_likelihood_cost + tf.reduce_sum(model_distribution.log_prob(self.input_y))

        with tf.name_scope('final_outputs'):
            if dataset_type == 'categorical':
                '''
                predicted_classes = tf.argmax(output_forward_pass, axis=1)
                self.mean_of_output_forward_pass = tf.reduce_max(tf.bincount())
                self.deviation_of_output_forward_pass = 
                self.maximum_of_output_forward_pass = tf.reduce_max(predicted_classes, axis=0, name='prediction_maximum')
                self.minimum_of_output_forward_pass = tf.reduce_min(predicted_classes, axis=0, name='prediction_minimum')
                '''
                pass
                ## This is to be fixed ##
            elif dataset_type == 'continuous':
                mean_of_output_forward_pass_temporary, variance_of_output_forward_pass = tf.nn.moments(output_forward_pass, 0, name='pred_mean_n_var')
                self.mean_of_output_forward_pass = tf.identity(mean_of_output_forward_pass_temporary, name='pred_mean')
                self.deviation_of_output_forward_pass = tf.sqrt(variance_of_output_forward_pass, name='pred_sigma')
                self.maximum_of_output_forward_pass = tf.reduce_max(output_forward_pass, axis=0, name='pred_max')
                self.minimum_of_output_forward_pass = tf.reduce_min(output_forward_pass, axis=0, name='pred_min')

        with tf.name_scope('cost'):
            self.intercost_minibatch_weight_pie = (tf.pow(2., tf.to_float(self.number_mini_batches - self.mini_batch_index)))/(tf.pow(2., tf.to_float(self.number_mini_batches)) - 1)
            #self.complexity_cost = self.all_variational_MAP_cost - self.all_prior_cost
            #self.var_MAP_cost = tf.identity(self.all_variational_MAP_cost, name='var_MAP_cost')
            #self.pr_cost = tf.identity(self.all_prior_cost, name='prior_cost')
            #self.ll_cost = tf.multiply(self.extra_likelihood_emphasis, self.all_likelihood_cost, name='likelihood_cost')
            self.all_likelihood_cost = tf.divide(self.all_likelihood_cost, tf.cast(self.number_samples_variance_reduction, tf.float32), name='likelihood_cost')
            self.all_prior_cost = tf.divide(tf.reduce_sum(self.WEIGHTS_PRIOR.log_prob(self.sampled_weights_1)) + tf.reduce_sum(self.WEIGHTS_PRIOR.log_prob(self.sampled_biases_1)) + tf.reduce_sum(self.WEIGHTS_PRIOR.log_prob(self.sampled_weights_2)) + tf.reduce_sum(self.WEIGHTS_PRIOR.log_prob(self.sampled_biases_2)) + tf.reduce_sum(self.WEIGHTS_PRIOR.log_prob(self.sampled_weights_3)) + tf.reduce_sum(self.WEIGHTS_PRIOR.log_prob(self.sampled_biases_3)) + tf.reduce_sum(self.WEIGHTS_PRIOR.log_prob(self.sampled_weights_4)) + tf.reduce_sum(self.WEIGHTS_PRIOR.log_prob(self.sampled_biases_4)), tf.cast(self.number_samples_variance_reduction, tf.float32))
            self.weight_vector = tf.concat([tf.reshape(self.sampled_weights_1, [-1]), tf.reshape(self.sampled_biases_1, [-1]), tf.reshape(self.sampled_weights_2, [-1]), tf.reshape(self.sampled_biases_2, [-1]), tf.reshape(self.sampled_weights_3, [-1]), tf.reshape(self.sampled_biases_3, [-1]), tf.reshape(self.sampled_weights_4, [-1]), tf.reshape(self.sampled_biases_4, [-1])], axis=0)
            self.mu_vector = tf.concat([tf.reshape(self.mu_weights_1, [-1]), tf.reshape(self.mu_biases_1, [-1]), tf.reshape(self.mu_weights_2, [-1]), tf.reshape(self.mu_biases_2, [-1]), tf.reshape(self.mu_weights_3, [-1]), tf.reshape(self.mu_biases_3, [-1]), tf.reshape(self.mu_weights_4, [-1]), tf.reshape(self.mu_biases_4, [-1])], axis=0)
            self.rho_vector = tf.concat([tf.reshape(self.rho_weights_1, [-1]), tf.reshape(self.rho_biases_1, [-1]), tf.reshape(self.rho_weights_2, [-1]), tf.reshape(self.rho_biases_2, [-1]), tf.reshape(self.rho_weights_3, [-1]), tf.reshape(self.rho_biases_3, [-1]), tf.reshape(self.rho_weights_4, [-1]), tf.reshape(self.rho_biases_4, [-1])], axis=0)
            self.variational_distribution = tf.distributions.Normal(loc=self.mu_vector, scale=tf.log(1 + tf.exp(self.rho_vector)))
            self.all_variational_MAP_cost = tf.divide(tf.reduce_sum(self.variational_distribution.log_prob(self.weight_vector)), tf.cast(self.number_samples_variance_reduction, tf.float32))

            self.ELBO = tf.subtract((self.intercost_minibatch_weight_pie * (self.all_variational_MAP_cost - self.all_prior_cost)), (self.extra_likelihood_emphasis * self.all_likelihood_cost), name='ELBO')
            if self.regularizer == 'PAC_Bayes':
                if ss == 0:
                    s_square = self.extra_likelihood_emphasis
                elif ss == 1:
                    s_square = (1/self.precision_alpha)
                elif ss == 2:
                    s_square = (self.extra_likelihood_emphasis/self.precision_alpha)
                else:
                    s_square = (1/self.precision_alpha) + self.extra_likelihood_emphasis
                self.pac_bayes_bound = tf.subtract((((self.all_variational_MAP_cost - self.all_prior_cost + tf.log(1/delta))/self.N_BOUND) + (s_square/2)), ((self.extra_likelihood_emphasis * self.all_likelihood_cost)/self.N_BOUND), name='pac_bayes_bound')
                #self.pac_bayes_bound = tf.add((-self.extra_likelihood_emphasis * self.all_likelihood_cost), tf.add(((tf.reduce_sum(tf.distributions.kl_divergence(self.variational_distribution, self.WEIGHTS_PRIOR)) + tf.log(1/delta))/N_BOUND), ((self.extra_likelihood_emphasis*self.precision_alpha)/2)), name='pac_bayes_bound')
                self.cost = tf.identity(self.ELBO, name='cost')
            else:
                self.pac_bayes_bound = tf.constant(0., name='pac_bayes_bound')
                self.cost = tf.multiply(-self.extra_likelihood_emphasis, self.all_likelihood_cost, name='cost')

        with tf.name_scope('error'):
            if dataset_type == 'continuous':
                self.pred_err = tf.reduce_mean(tf.squared_difference(self.mean_of_output_forward_pass, self.input_y), name='pred_err')
            elif dataset_type == 'categorical':
                pass
                ##################################
                ##################################
                ##################################
                ### Need to finish this up later ####

                #self.pred_train_err = tf.equal(tf.argmax())
                #self.pred_val_err = 

        with tf.name_scope('optimization'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate, name='adam_optimizer')
            self.training = optimizer.minimize(self.cost, global_step=self.global_step, name='training')

        with tf.name_scope('summaries'):
            #tf.summary.scalar(name='pred_err_log', tensor=self.pred_err)
            #tf.summary.scalar(name='deviation_of_output_forward_pass_log', tensor=tf.reduce_mean(self.deviation_of_output_forward_pass))
            #tf.summary.scalar(name='prior_cost_log', tensor=self.all_prior_cost)
            #tf.summary.scalar(name='variational_MAP_cost_log', tensor=self.all_variational_MAP_cost)
            #tf.summary.scalar(name='likelihood_cost_log', tensor=self.all_likelihood_cost)
            #tf.summary.scalar(name='complexity_cost_log', tensor=self.complexity_cost)
            tf.summary.scalar(name='cost_log', tensor=self.cost)
            self.summary_op = tf.summary.merge_all()

        with tf.name_scope('mini_batch_index_update'):
            self.mini_batch_index_update = tf.assign(ref=self.mini_batch_index, value=((self.mini_batch_index % self.number_mini_batches) + 1), name='mini_batch_index_update')


    def fetch_ACTIVATION_UNIT(self, param):
        if self.activation_unit == 'RELU':
            return tf.nn.relu(param)
        elif self.activation_unit == 'SIGMOID':
            return tf.sigmoid(param)
        elif self.activation_unit == 'TANH':
            return tf.tanh(param)


    def update_mini_batch_index(self):
        return self.mini_batch_index_update


    def getCostforTraining(self):
        return self.all_prior_cost.eval(), self.all_variational_MAP_cost.eval() , self.all_likelihood_cost.eval(), self.cost.eval()
        #return self.cost.eval()


    def summarize(self):
        return self.summary_op


    def makeInference(self):
        return self.mean_of_output_forward_pass, self.deviation_of_output_forward_pass, self.maximum_of_output_forward_pass, self.minimum_of_output_forward_pass


    def train(self):
        return self.training


    def getMeanSquaredError(self):
        return self.pred_err


    def get_weights_and_phi(self, shape):

        mu_weights = tf.Variable(tf.zeros(shape=[self.number_samples_variance_reduction, shape[0], shape[1]], dtype=tf.float32), name='mu_weights')
        rho_weights = tf.Variable(tf.truncated_normal(shape=[self.number_samples_variance_reduction, shape[0], shape[1]], mean=self.rho_mean, stddev = 1.0/(math.sqrt(float(shape[0]))), dtype=tf.float32), name='rho_weights')

        mu_bias = tf.Variable(tf.zeros(shape=[self.number_samples_variance_reduction, shape[1]], dtype=tf.float32), name='mu_bias')
        rho_bias = tf.Variable(tf.truncated_normal(shape=[self.number_samples_variance_reduction, shape[1]], mean=self.rho_mean, stddev = 1., dtype=tf.float32), name='rho_bias')

        sampled_weights = mu_weights + ( tf.log(1 + tf.exp( rho_weights)) * tf.random_normal(shape=[self.number_samples_variance_reduction, shape[0], shape[1]], mean=0., stddev=1.0, name='weights_randomizer'))
        sampled_biases = mu_bias + ( tf.log(1 + tf.exp(rho_bias)) * tf.random_normal(shape=[self.number_samples_variance_reduction, shape[1]], mean=0., stddev= 1.0, name='bias_randomizer'))

        #sampled_weights_for_prediction = mu_weights + ( tf.log(1 + tf.exp(rho_weights)) * tf.random_normal(shape=[self.number_samples_variance_reduction, shape[0], shape[1]], mean=0., stddev=1.0, name='weights_randomizer_for_prediction'))
        #sampled_biases_for_prediction = mu_bias + ( tf.log(1 + tf.exp(rho_bias)) * tf.random_normal(shape=[self.number_samples_variance_reduction, shape[1]], mean=0., stddev= 1.0, name='bias_randomizer_for_prediction'))

        #variational_distribution_weights = tf.distributions.Normal(loc=mu_weights, scale=tf.log(1 + tf.exp(rho_weights)))
        #variational_distribution_bias = tf.distributions.Normal(loc=mu_bias, scale=tf.log(1 + tf.exp(rho_bias)))

        #all_variational_MAP_cost = tf.reduce_sum(variational_distribution_weights.log_prob(sampled_weights)) + tf.reduce_sum(variational_distribution_bias.log_prob(sampled_biases))

        #all_prior_cost = tf.reduce_sum(self.WEIGHTS_PRIOR.log_prob(sampled_weights)) + tf.reduce_sum(self.WEIGHTS_PRIOR.log_prob(sampled_biases))
        
        #return sampled_weights, sampled_biases, all_variational_MAP_cost, all_prior_cost, mu_weights, rho_weights, mu_bias, rho_bias, sampled_weights_for_prediction, sampled_biases_for_prediction
        return sampled_weights, sampled_biases, mu_weights, rho_weights, mu_bias, rho_bias