# Python 3
import numpy as np 
import tensorflow as tf
import pickle
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
import inspect
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

class TSO:
	def __init__(
		self, 
		env,
		des_state,
		epsilon_min = 0.01,
		epsilon_decay = 0.995,
		trials = 1000,
		batch_size = 0,
		min_batch_size = 32,
		max_batch_size = 256,
		batch_size_divisor = 4,
		initial_fit_epochs = 25,
		re_fit_epochs = 1,
		evaluation_epochs = 10,
		gradient_opt_threshold = 2,
		tuning_threshold = 3,
		initial_random_iterations = 5,
		max_num_models = 3,
		network_optimization_calls = 100,
		num_layers = 4,
		num_neurons_per_layer = 425,
		learning_rate = 0.001,
		print_progress=True):
		
		
		'''
		Reinforcement Learning parameters
		env = Reinforcement Learning Environment (input = action, output = state, reward, done, {}) --> used to get the shapes
		
		DQN Parameter:
		epsilon = probablity of exploring a new action rather than predicting an action --> epsilon_min < epsilon < 1
		epsilon_min = minimum epsilon (if 0 no exploration at a specific progress)
		epsilon_decay = value to decrease epsilon in each time step --> epsilon = epsilon_decay * epsilon: 0 < epsilon_decay < 1
		'''
		self._env = env
		self._epsilon = 1.0
		self._epsilon_min = epsilon_min
		self._epsilon_decay = epsilon_decay
		
		
		'''
		Numpy array of the desired state
		'''
		self._des_state = des_state
		
		'''
		Each time step step is stored in a buffer
		buffer_capacity = number of values stored during evaluation
		buffer_counter = index of the current evaluation in the buffer
		
		Storage of the values of each time step:
		state_buffer, action_buffer, reward_buffer, next_state_buffer
		'''
		self._buffer_capacity = trials
		self._buffer_counter = 0
		self._state_buffer = np.zeros((self._buffer_capacity, self._env.observation_space.shape[1]))
		self._action_buffer = np.zeros((self._buffer_capacity, self._env.action_space.shape[1]))
		self._reward_buffer = np.zeros((self._buffer_capacity, 1))
		
		'''
		Desired State: The state the agent should achive with its action taken
		'''
		
		self._norm_des_state = np.array(())
		
		'''
		batch size = seperates the dataset into batches. After each batch the gradient of the neural network is updated
		- if batch_size > 0: then the batch_size is a static value (common values: 32, 64, 128)
		- if batch_size = 0: the batch size is dynamically calculated dependng on the buffer counter: 
			batch_size = min(max(min_batch_size, int(buffer_counter / batch_size_divisor)), max_batch_size)
		'''
		# Num of tuples to train on.
		self._batch_size = batch_size
		self._min_batch_size = min_batch_size
		self._max_batch_size = max_batch_size
		self._batch_size_divisor = batch_size_divisor
		self._dyn_batch_size = False
		if self._batch_size == 0:
			self._dyn_batch_size = True
		
		'''
		the epochs describe how often model is trained on the dataset
		- initial_fit_epochs = the epochs the neural network model is trained on after creation		
		- re_fit_epochs = the epochs the neural network model is trained on if a re-fit is desired (gradient_opt_threshold)
		- evaluation_epochs = the epochs the neural network model is trained on during model evaluation (hyperparameter tuning and model selection)
		'''
		self._initial_fit_epochs = initial_fit_epochs
		self._re_fit_epochs = re_fit_epochs
		self._evaluation_epochs = evaluation_epochs
		
		'''
		fail_count = counts up if the neural network model predicts an action which is not correct
		if fail_count > gradient_opt_threshold
			- the weights of the neural network are optimized with the newly generated state action pairs
		if fail_count > tuning_threshold
			- with scikit learn bayessian optimization gp_minimize is applied to generate a new neural network is
			- the learning rate, the number of layers and the number of neurons of each layer are determined depending on the existing data
		max_num_models = defines how often a new neural network model through hyperparameter tuning is generated
		initial_random_iterations = defines how many random iterations should at least be performed before creating the first neural network model
		'''
		self._action_from_model = False
		self._fail_count = 0
		self._gradient_opt_threshold = gradient_opt_threshold
		self._tuning_threshold = tuning_threshold
		self._max_num_models = max_num_models
		self._initial_random_iterations = initial_random_iterations
		
		'''
		random_actions and predicted_actions are lists where the index is stored depending on the generated action was random or predicted
		'''
		self._random_actions = []
		self._predicted_actions = []
		
		'''
		models = list where the hyperparameters of the tuning are stored
		num_models = the current amount of models created during optimization
		model = neural network model for predictting actions
		'''
		self._models = []
		self._num_models = 0
		self._model = None
		
		self._neural_network_config = [num_layers, num_neurons_per_layer]
		self._learning_rate = learning_rate
		
		'''
		intialize the class to compile and fit the neural network
		'''
		
		self._nno = neural_network_operations()
		
		
		'''
		neural_network_config = list to create the neural network model [num_layers, num_neurons]
		learning_rate = learning rate of the neural network model

		if max_num_models is zero the hyperparameters from neural_network_config and learning rate are taken instead of optimizing the network several times
		'''
		if self._max_num_models == 0:
			self._model = self._nno.create_network(
				self._env.action_space.shape[1],
				self._env.observation_space.shape[1],
				num_layers,
				num_neurons_per_layer
			)
		
		'''
		Print progress in console
		'''
		
		self._print_progress = print_progress
		
		'''
		initialization of the hyperparameter tuning class
		'''
		
		self._network_optimization_calls = network_optimization_calls
		self._optimization = model_optimization(self._neural_network_config, self._learning_rate, self._nno, network_optimization_calls=self._network_optimization_calls)
		
		
		'''
		initialize list of completed trials
		'''
		
		self._done_list = []
		
		'''
		Exceptions
		'''
		
		self._starting_index = self._buffer_counter
		self._trials = self._buffer_capacity
		
		if type(self._des_state).__module__ == np.__name__ or type(self._des_state) is np.ndarray:
			if self._des_state.shape[0] > 1:
				raise ValueError('The shape of the des_state should (1, ' + str(self._env.observation_space.shape[1]) + ')')
		else:
			raise ValueError('The format of the des_state should be a numpy array')
	
	'''
	Create the action
	'''
	def act(self):
		
		# determine the new batch size dynamic if desired
		if self._dyn_batch_size:
			self._batch_size = min(max(self._min_batch_size, int(self._buffer_counter / self._batch_size_divisor)), self._max_batch_size)
		
		# sample a random action
		action = self._env.action_space.sample()
		
		# PREDICT ACTION FROM MODEL: if epsilon is less than the a random number and the first initial actions are taken select the
		self._rand = np.random.random()
		if self._buffer_counter > self._initial_random_iterations and self._rand > self._epsilon:
			self._action_from_model = True
			# scale the existing states and the desired state on the same range
			states, norm_des_state = self.scale_state()
			actions = self._action_buffer[0:self._buffer_counter,:]
			
			# reset the normed desired state only if the model will be updated
			if self._norm_des_state.size == 0 or self._model == None or self._fail_count > self._gradient_opt_threshold or self._fail_count > self._tuning_threshold:
				self._norm_des_state = norm_des_state
			
			
			# CREATE NEW MODEL: if no model exists or the existing model did not perform well the last iterations ("tuning_threshold")
			# create a new model if the maximum amount of models is not reached
			if (self._model == None or self._fail_count > self._tuning_threshold) and self._num_models < self._max_num_models:
				# set the model counters zero
				self._fail_count = 0
				
				# Find the hyperparameters through bayessian optimization
				num_layers, num_neurons, learning_rate = self._optimization.bayessian_tune(states, actions, self._batch_size, self._evaluation_epochs)
				
				# Store the hyperparameters
				self._models.append([num_layers, num_neurons, learning_rate])
				self._num_models = len(self._models)
				
				
				# if there is more than one model compare the performance of the existing ones
				if self._num_models > 1:
					num_layers, num_neurons, learning_rate = self.compare_models(states, actions)
				
				self._neural_network_config = [num_layers, num_neurons]
				self._learning_rate = learning_rate
				
				# create the model with the best model performance so far
				
				self._model = self._nno.create_network(
					self._env.action_space.shape[1],
					self._env.observation_space.shape[1],
					num_layers,
					num_neurons
				)
				
				
				history, self._model = self._nno.fit_model(self._model, states, actions, epochs=self._initial_fit_epochs, batch_size=self._batch_size, learning_rate=self._learning_rate)
			
			# RE-FIT THE MODEL: if the model exists but did not perform well the last iterations ("gradient_opt_threshold")
			# adjust the weights of the model
			elif self._fail_count > self._gradient_opt_threshold:
				
				# if the number of models created exceeds the maximum set the model counters to zero 
				if self._num_models >= self._max_num_models:
					self._fail_count = 0
				
				history, self._model = self._nno.fit_model(self._model, states, actions, epochs=self._re_fit_epochs, batch_size=self._batch_size, learning_rate=self._learning_rate)
			# predict the acton from the model
			action = self._model.predict(self._norm_des_state)
			self._predicted_actions.append(self._buffer_counter)
		else:
			self._action_from_model = False
			self._random_actions.append(self._buffer_counter)
		
		# clip the action if necessary
		legal_action = np.clip(action, self._env.action_space.low[0], self._env.action_space.high[0])
		
		return legal_action
	
	'''
	Function to scle the state and the desired state to the same range
	'''
	def scale_state(self):
		state_scale = MinMaxScaler( feature_range=(0, 1) )
		
		# create one numpy array containing the states collected so far and the desired state
		states_fit = np.vstack((self._state_buffer[0:self._buffer_counter,:], self._des_state))
		# scale the new numpy array
		states_raw = state_scale.fit_transform( states_fit )
		# get the scaled collected states
		states = states_raw[0:self._buffer_counter,:]
		# get the desired scaled state
		norm_des_state = np.array(states_raw[-1,:], ndmin=2)
		
		return states, norm_des_state
	
	'''
	Function to compare the exisiting models if there is more than one
	'''
	def compare_models(self, states, actions):
		# initialize the minimum loss so far and the final model
		min_loss = 1
		fin_model = None
		
		# loop the existing neural network models
		for m in self._models:
			# create the model
			
			n_model = self._nno.create_network(
				self._env.action_space.shape[1],
				self._env.observation_space.shape[1],
				m[0],
				m[1]
			)
			
			# train it 
			
			history, self._model = self._nno.fit_model(n_model, states, actions, epochs=self._evaluation_epochs, batch_size=self._batch_size, learning_rate=m[2])
			
			# get the loss
			current_loss = history['loss'][-1] + history['val_loss'][-1]
			
			# check if the loss is lower than the lowest so far
			if current_loss <= min_loss:
				min_loss = current_loss
				fin_model = m
		
		# return the hyperparameters of the best model so far
		num_layers = fin_model[0]
		num_neurons = fin_model[1]
		learning_rate = fin_model[2]
		
		return num_layers, num_neurons, learning_rate
	
	'''
	Save values of the current time step into memory and count the buffer_counter up
	'''
	def remember(self, state, action, reward, done):
		# decrase epsilon: exploration / exploitation behavior
		self._epsilon *= self._epsilon_decay
		self._epsilon = max(self._epsilon_min, self._epsilon)
		
		self._state_buffer[self._buffer_counter,:] = state
		self._action_buffer[self._buffer_counter,:] = action
		self._reward_buffer[self._buffer_counter,:] = reward
		self._buffer_counter += 1
		
	def step(self, trial):
		self._env.reset()
		action = self.act()
		
		self._env._trial = trial
		
		state, reward, done, _ = self._env.step(action)
		
		if reward != -4:
			if done:
				self._done_list.append(trial)
			else:
				if self._action_from_model:
					self._fail_count += 1
			
			self.remember(state, action, reward, done)
			# self._model.save_weights(model_path)
		
		random_hits = list(set(self._done_list).intersection(self._random_actions))
		predicted_hits = list(set(self._done_list).intersection(self._predicted_actions))
		random_hits.sort()
		predicted_hits.sort()
		
		if self._print_progress:
			print("Trial: " + str(trial) + ", current Reward: " + str(reward))
			print("Best Reward: " + str(max(self._reward_buffer[0:trial+1,:])[0]) + " at Iteration: " + str(self._reward_buffer[0:trial+1,:].tolist().index(max(self._reward_buffer[0:trial+1,:]))))
			print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
			print("Erfolgreiche Versuche: " + str(len(self._done_list)))
			print(" Predicted:" + str(len(predicted_hits)))
			print(" Random: " + str(len(random_hits)))
			print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
			for name in self._env._opt_crit.dtype.names:
				print(name + ': ' + str(self._env._results[name][0]) + ' / ' + str(self._env._opt_crit[name]))
			print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
			if ((trial + 1) % 1000) == 0:
				print(predicted_hits)
			
		return done

class model_optimization():
	def __init__(self, neural_network_config, learning_rate, nno, network_optimization_calls = 100):
		dim_learning_rate = Real(low=1e-5, high=1e-1, prior='log-uniform', name='learning_rate')
		dim_layers = Integer(low=1, high=10, name='num_layers')
		dim_nodes = Integer(low=1, high=512, name='num_neurons')

		self._dimensions = [dim_layers,
			dim_nodes,
			dim_learning_rate]
		
		self._default_parameters = [neural_network_config[0], neural_network_config[1], learning_rate]
		self._network_optimization_calls = network_optimization_calls
		self._nno = nno

	def bayessian_tune(self, states, actions, batch_size, epochs):
		
		@use_named_args(dimensions=self._dimensions)
		def fitness(num_layers, num_neurons,
				learning_rate
			):
			
			model = self._nno.create_network(
				actions.shape[1],
				states.shape[1],
				num_layers,
				num_neurons
			)
			
			history, self._model = self._nno.fit_model(model, states, actions, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
			
			loss = history['loss'][-1] + history['val_loss'][-1]
			
			if np.isnan(loss):
				loss = 1
			
			return loss
		
		
		search_result = gp_minimize(func=fitness,
			dimensions=self._dimensions,
			acq_func='EI', # Expected Improvement.
			n_calls=self._network_optimization_calls,
			x0=self._default_parameters)
		
		num_layers = search_result.x[0]
		num_neurons = search_result.x[1]
		learning_rate = search_result.x[2]

		return num_layers, num_neurons, learning_rate
		
class neural_network_operations():
	def __init__(self, monitor='val_loss', min_delta=0, patience=2):
		self._monitor = monitor
		self._min_delta = min_delta
		self._patience = patience
	
	def create_network(self, output_shape, input_shape, num_layers, num_neurons):
	
		model = tf.keras.Sequential()
		kernel_initializer = tf.keras.initializers.RandomUniform()
		model.add(tf.keras.Input(shape=input_shape))
		
		
		for s in range(num_layers):
			model.add(tf.keras.layers.Dense(num_neurons, 
				activation="relu", 
				dtype=tf.float32))
		
		model.add(tf.keras.layers.Dense(output_shape, kernel_initializer=kernel_initializer, 
			dtype=tf.float32))
		
		return model
	
	
	# @tf.function
	def fit_model(self, model, input, output, epochs=100, batch_size=32, learning_rate=1e-3, earlystopping = True):
		
		train_size = int(input.shape[0] * 0.8)	
		
		full_dataset = tf.data.Dataset.from_tensor_slices((input, output))
		full_dataset = full_dataset.shuffle(input.shape[0])
		
		train_dataset = full_dataset.take(train_size)
		train_dataset = train_dataset.batch(batch_size)
		
		val_dataset = full_dataset.skip(train_size)
		val_dataset = val_dataset.batch(batch_size)
		
		optimizer = tf.keras.optimizers.Adam(learning_rate)
		
		loss_history = list()
		val_loss_history = list()
		monitor_value = list
		no_improvement_count = 0
		for epoch in range(epochs):
			train_loss = 0
			step = 0
			for x_train_batch, y_train_batch in train_dataset:
				with tf.GradientTape() as tape:
					prediction = model(x_train_batch, training=True)
					loss = tf.keras.losses.mean_squared_error(y_train_batch, prediction)
					train_loss += float(np.mean(loss.numpy()))
				
				grad = tape.gradient(loss, model.trainable_variables)
				optimizer.apply_gradients(zip(grad, model.trainable_variables))
				step += 1
			
			loss_history.append(train_loss / step)
			
			val_loss = 0
			step = 0
			for x_val, y_val in val_dataset:
				prediction = model(x_val, training=False)
				loss = tf.keras.losses.mean_squared_error(y_val, prediction)
				try:
					val_loss += float(np.mean(loss.numpy()))
				except:
					pass
				step += 1
			val_loss_history.append(val_loss / step)
			
			if epoch > 1 and earlystopping:
				if self._monitor == 'val_loss':
					monitor_value = val_loss_history
				elif self._monitor == 'loss':
					monitor_value = loss_history
				improvement = abs(monitor_value[-2] - monitor_value[-1])
								
				if improvement <= self._min_delta:
					no_improvement_count +=1
				if no_improvement_count == self._patience:
					break
		history = {}
		history['loss'] = loss_history
		history['val_loss'] = val_loss_history
		
		return history, model