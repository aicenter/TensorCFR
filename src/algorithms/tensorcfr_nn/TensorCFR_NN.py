#!/usr/bin/env python3
import tensorflow as tf

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.domains.FlattenedDomain import FlattenedDomain
from src.nn.NNMockUp import NNMockUp


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
		self.nn_input_permutation = nn_input_permutation if neural_net is not None else get_sorted_permutation()

	def predict_equilibrial_values(self, input_reaches):
		permutate_op = tf.contrib.distributions.bijectors.Permute(permutation=self.nn_input_permutation)

		# permute input reach probabilities
		permuted_input = permutate_op.forward(input_reaches)

		# use neural net to predict equilibrium values
		predicted_equilibrial_values = self.neural_net.predict(permuted_input)

		# permute back the expected values
		permuted_predictions = permutate_op.inverse(predicted_equilibrial_values)
		return permuted_predictions
