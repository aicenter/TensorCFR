#!/usr/bin/env python3
import tensorflow as tf

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.commons.constants import DEFAULT_TOTAL_STEPS, DEFAULT_AVERAGING_DELAY
from src.domains.FlattenedDomain import FlattenedDomain
from src.nn.NNMockUp import NNMockUp
from src.utils.tf_utils import get_default_config_proto


def get_sorted_permutation():
	return [2, 1, 0]


class TensorCFR_NN(TensorCFRFixedTrunkStrategies):
	def __init__(self, domain: FlattenedDomain, neural_net=None, nn_input_permutation=None, trunk_depth=0):
		"""
		Constructor for an instance of TensorCFR algorithm with given parameters (as a TensorFlow computation graph).

		:param domain: The domain of the game (as an instance of class `FlattenedDomain`). TensorCFR (the CFR+ algorithm)
		 will be launched for this game.
		:param trunk_depth: The number of levels of the trunk where the strategies are kept fixed. It should be an integer
			 between `0` to `self.domain.levels`. It defaults to `0` (no trunk).
		"""
		super().__init__(domain, trunk_depth)
		self.neural_net = neural_net if neural_net is not None else NNMockUp()
		self.nn_input_permutation = nn_input_permutation if nn_input_permutation is not None else get_sorted_permutation()
		self.session = tf.Session(config=get_default_config_proto())
		self.construct_computation_graph()
		with tf.variable_scope("initialization"):
			setup_messages, feed_dictionary = self.set_up_feed_dictionary(method="by-domain")
			print(setup_messages)
		self.session.run(tf.global_variables_initializer(), feed_dict=feed_dictionary)

	def construct_computation_graph(self):
		self.cfr_step_op = self.do_cfr_step()

	def cfr_strategies_after_fixed_trunk(self, total_steps=DEFAULT_TOTAL_STEPS, delay=DEFAULT_AVERAGING_DELAY,
	                                     storing_strategies=False, profiling=False, register_strategies_on_step=list()):
		# a list of returned average strategies
		# the parameter `register_strategies_on_step` is used to determine which strategy export
		return_average_strategies = list()

		# if the `register_strategies_on_step` list is empty, register just the last iteration
		if len(register_strategies_on_step) == 0:
			register_strategies_on_step.append(total_steps - 1)

		self.cfr_parameters = {
			"total_steps"    : total_steps,
			"averaging_delay": delay,
			"trunk_depth"    : self.trunk_depth,
		}
		self.set_up_cfr_parameters(delay, total_steps)
		self.set_log_directory()
		with tf.summary.FileWriter(self.log_directory, tf.get_default_graph()):
			for step in range(total_steps):
				self.session.run(self.cfr_step_op)

				if step in register_strategies_on_step:
					return_average_strategies.append({
						"step": step,
						"average_strategy": [self.session.run(x) for x in self.average_infoset_strategies]
					})
		return return_average_strategies

	def predict_equilibrial_values(self, input_reaches):
		permutate_op = tf.contrib.distributions.bijectors.Permute(permutation=self.nn_input_permutation)

		permuted_input = tf.expand_dims(
			permutate_op.forward(input_reaches),    # permute input reach probabilities
			axis=0,                                 # simulate batch size of 1 for prediction
			name="expanded_permuted_input"
		)

		np_permuted_input = self.session.run(permuted_input)

		# use neural net to predict equilibrium values
		predicted_equilibrial_values = self.neural_net.predict(np_permuted_input)

		# permute back the expected values
		permuted_predictions = permutate_op.inverse(predicted_equilibrial_values)
		return tf.identity(permuted_predictions, name="permuted_predictions")
