#! usr/bin/python
import numpy as np
import tensorflow as tf
import dnn_gorder
from sklearn.metrics import log_loss
import os
from tensorflow.contrib.layers import fully_connected
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import deque
tf.logging.set_verbosity(tf.logging.INFO)
gpu_options = tf.GPUOptions(allow_growth=True)

def model_fn(n_classes, n_hidden, learning_rate): 
	n_inputs = 1
	n_outputs = 1
	# build the neural network
	initializer = tf.contrib.layers.variance_scaling_initializer()
	X = tf.placeholder(tf.float32, shape=[n_classes, n_inputs])  # X: [n_classes, 1]-dim
	hidden = fully_connected(X, n_hidden, activation_fn=tf.nn.elu,
		weights_initializer=initializer)
	logits = fully_connected(hidden, n_outputs, activation_fn=None,
		weights_initializer=initializer)
	outputs = tf.nn.sigmoid(logits)

	# select a random action based on the estimated probabilities
	p_inc_and_dec = tf.concat(axis=1, values=[outputs, 1 - outputs])
	action = tf.multinomial(tf.log(p_inc_and_dec), num_samples=1)

	y = tf.ones(shape = [n_classes, n_inputs]) - tf.to_float(action)
	cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
		labels=y, logits=logits)
	optimizer = tf.train.RMSPropOptimizer(learning_rate)

	grads_and_vars = optimizer.compute_gradients(cross_entropy)
	gradients = [grad for grad, variable in grads_and_vars]

	gradient_placeholders = []
	grads_and_vars_feed = []
	for grad, variable in grads_and_vars:
		gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
		gradient_placeholders.append(gradient_placeholder)
		grads_and_vars_feed.append((gradient_placeholder, variable))
	# the operator to apply updated gradient
	training_op = optimizer.apply_gradients(grads_and_vars_feed)

	return p_inc_and_dec, action, gradients, training_op, gradient_placeholders, X

def tuning_sampling_by_percentage(p, action_val, tuning_rate, noise_coef):
	# 0 is increase and 1 is decrease
	inc_and_dec = np.where(action_val < 0.5, 1, -1)
	#noise = np.multiply(np.random.normal(loc=0, scale=0.01, size=p.shape), p)
	# inpose the action and a noise on P
	p = p + np.multiply(inc_and_dec * tuning_rate, p)  # + noise
	# make each element no negative
	p = np.where(p < 0, 0, p) 
	# L1 normalization
	p = p / np.sum(p)
	return p

def tuning_sampling_by_truncate(p, action_val, tuning_rate, noise_coef):
	# 0 is increase and 1 is decrease
	pre_action_val = action_val
	pre_p = p
	#print("p.shape", p.shape)
	tuning_rate = float(1.0 /(p.shape[0]*5))
	inc_and_dec = np.where(action_val < 0.5, 1, -1)
	# inpose the action on P
	p = p + inc_and_dec * tuning_rate
	# make each element no negative
	p = np.where(p < 0, 0, p) 
	# L1 normalization
	p = p / np.sum(p)
	cross_ent = - np.sum( np.multiply(pre_action_val, np.log(action_val + 1e-9)))
	return p, cross_ent

def discount_rewards(rewards, discount_rate):
	discounted_rewards = np.empty(len(rewards))
	cumulative_rewards = 0
	for step in reversed(range(len(rewards))):
		cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
		discounted_rewards[step] = cumulative_rewards
	return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
	all_discounted_rewards = [discount_rewards(rewards, discount_rate)
		for rewards in all_rewards]
	flat_rewards = np.concatenate(all_discounted_rewards)
	reward_mean = flat_rewards.mean()
	reward_std = flat_rewards.std()
	return [(discounted_rewards - reward_mean)/reward_std
		for discounted_rewards in all_discounted_rewards]

def main(args):
  input_folder = args.input_data_folder # input data for regression/classification
  model_dir = args.model_dir # regression/classification model dir
  verbose = args.verbose 

  # classifier learning params
  bs = args.batch_size # regression/classification batch size
  steps = args.steps # number of regression/classification steps per train
  lr = args.learning_rate # regression/classification learning rate
  set_size = args.w - 1 # set size of deep set
  n_eval_data = args.n_eval_data # number of eval data to generate
  n_hidden = args.n_hidden # number of hidden units in regression/classifica

  # RL learning params
  p_initial = args.p_initial # choose of the initial p 
  n_iterations = args.n_iterations # number of RL training steps
  n_trains = args.n_trains # number of regression/classification trains per RL updates
  save_iterations = args.save_iterations 
  tuning_rate = args.tuning_rate # P tuning rate
  noise_coef = args.noise_coef # the coefficient on the standard normal noise of P
  learning_rate = args.rl_learning_rate # RL learning rate
  rl_n_hidden = args.rl_n_hidden # number of hidden units in RL model
  gamma = args.gamma # discount rate of reward
  
  # Data pre-processing 
  train_data, train_labels, n_classes, _, train_norm = dnn_gorder.build_dataset(input_folder)
  print ("training data shape: ")
  print (train_data.shape, train_labels.shape, train_norm.shape)
  # Build the evaluation data and input_fn
  eval_input_fn, initial_p = dnn_gorder.build_eval_input_fn(n_eval_data, n_classes, set_size, 
  	train_data, train_labels, train_norm)
  
  if p_initial is "NORM" : # choose the norm as initial p
  	initial_p = train_norm.astype(np.float32) / np.sum(train_norm).astype(np.float32)
  else : # uniform initial p
  	initial_p = np.zeros(shape=(n_classes, 1)) + 1.0 / n_classes
  	initial_p = initial_p / np.sum(initial_p)
  print ("initial p " , initial_p.shape, np.max(initial_p), np.min(initial_p))
  print ("max prob of initial p: ", np.max(initial_p))

  
  if verbose:
      print("Real Class: %d"%(n_classes))
  # Build DeepSet classifier
  classifier = tf.estimator.Estimator(model_fn=dnn_gorder.model_fn,
                                      params={"n_classes": n_classes, "learning_rate": lr, 
                                      "n_hidden": n_hidden},
                                      model_dir=model_dir,
                                      config=tf.estimator.RunConfig(session_config=tf.ConfigProto(gpu_options=gpu_options)))
  if verbose:
  	print("RL begin...")
  # Build the RL model
  p_inc_and_dec, action, gradients, training_op, gradient_placeholders, X = model_fn(
  	n_classes= n_classes, n_hidden = rl_n_hidden, learning_rate = learning_rate)
  init = tf.global_variables_initializer()
  saver = tf.train.Saver()
  print_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
  with tf.Session() as sess:
	init.run()
	# rewards in past iterations for computing baseline
	reward_record = deque(maxlen = 100)

	for iteration in range(n_iterations):
		all_rewards = []
		all_gradients = []
		# get the base line from past rewards
		baseline = np.mean(list(reward_record))

		p = initial_p
		for _ in range(n_trains):
			# feed current state p to RL model, get possible "best" action and gradients	
			p_inc_and_dec_val, action_val, gradients_val = sess.run(
				[p_inc_and_dec, action, gradients],
				feed_dict={X: p.reshape(n_classes, 1)}) 
			# tune the p value
			#print("p_inc_and_dec:", p_inc_and_dec_val.shape, p_inc_and_dec_val)
			#print("action:", action_val.shape, action_val)
			p, cross_ent = tuning_sampling_by_truncate(p, action_val, tuning_rate, noise_coef)
			if verbose:
				# print the top 10 maximum p
				p2 = p.reshape((p.shape[0]))
				idx = (-p2).argsort()[:10]
				print ("top tuned p: ", p2[idx])
				print ("cross entropy: ", cross_ent)

			# train the classifier
			classifier.train(input_fn=lambda:dnn_gorder.train_input_fn(
				train_data, train_labels, train_norm, 
				p.reshape(n_classes), # inject the tuned p to classifier
				set_size = set_size, shuffle=True, batch_size=bs), steps=steps
			)

			# evalate the classifier, get the rewards(negative of the RMSE)
			eval_results = classifier.evaluate(input_fn=eval_input_fn)
			reward = - eval_results["RMSE"]
			if verbose:
				print("reward values: ", reward)
			# cache the reward
			reward_record.append(reward)
			all_rewards.append(reward)
			all_gradients.append(gradients_val)

		# skip top interations to get baseline
		if iteration < 1: 
			continue
		
		feed_dict = {}
		all_rewards = discount_rewards(all_rewards, gamma)
		for var_index, gradient_placeholder in enumerate(gradient_placeholders):
			# multiply the gradients by the action scores, and compute the mean
			mean_gradients = -np.mean(
				[(reward - baseline) * (all_gradients[train_index][var_index])
					for train_index, reward in enumerate(all_rewards)], axis = 0)
			#print("mean_gradients: ", mean_gradients)
			feed_dict[gradient_placeholder] = mean_gradients
		# execute one step policy gradient
		if verbose:
			print("The %d -th iteration policy gradient ..."%(iteration))
		sess.run(training_op, feed_dict= feed_dict)
		# RL model checkpoint
		if iteration % save_iterations == 0:
			saver.save(sess, "./my_policy_net_pg.ckpt")
		if iteration in print_list:
			print("iteration: %d"%(iteration))
			dnn_gorder.gen_order_union(args)
			dnn_gorder.gen_order_sep(args)


if __name__ == "__main__":
  parser = ArgumentParser("dnn", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
  #Required 
  parser.add_argument("--input_data_folder", default="../../data/wv", help="Input data folder")
  parser.add_argument("--model_dir", default="models_tmp")
  #Learning Parameters
  parser.add_argument("--batch_size", default=128, type=int) # regression/classification batch size
  parser.add_argument("--steps", default=1000, type=int) # number of regression/classification steps per train
  parser.add_argument("--learning_rate", default = 0.001, type=float) # regression/classification learning rate
  parser.add_argument("--w", default = 5, type = int) # set size of deep set
  parser.add_argument("--n_eval_data", default = 100, type = int) # number of eval data to generate
  parser.add_argument("--n_hidden", default = 256, type = int) # number of hidden units in regression/classification model

  # RL parameters
  parser.add_argument("--rl_learning_rate", default = 0.001, type=float) # RL learning rate
  parser.add_argument("--n_iterations", default = 200, type=int) # number of RL training steps
  parser.add_argument("--n_trains", default = 10, type=int) # number of regression/classification trains per RL updates
  parser.add_argument("--p_initial", default = "NORM") #  the option of initial p value 1) NORM 2)UNIFORM
  parser.add_argument("--tuning_rate", default = 0.2, type=float) # the percentage of p increase/decrease 
  parser.add_argument("--noise_coef", default = 1e-8, type=float) # the coefficient on the standard normal noise of P
  parser.add_argument("--save_iterations", default = 100, type=int) # number of RL steps per checkpoint
  parser.add_argument("--rl_n_hidden", default = 256, type=int) # number of hidden units in RL model
  parser.add_argument("--gamma", default = 0.9, type=float) # discount rate of reward
  #Others
  parser.add_argument("--verbose", default=False, type=bool)

  args = parser.parse_args() 
  print(args)
  #main(args)
  dnn_gorder.gen_order_union(args)
  dnn_gorder.gen_order_sep(args)
