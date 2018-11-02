#!/usr/bin/env python3
import tensorflow as tf

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.domains.FlattenedDomain import FlattenedDomain
from src.domains.available_domains import get_domain_by_name


def get_sorted_permutation():
	return [2, 1, 0]


class NeuralNetMockUp:
	def predict(self, input_tensor):
		return [5, 5, 6]


class TensorCFR_NN(TensorCFRFixedTrunkStrategies):
	def __init__(self, domain: FlattenedDomain, trunk_depth=0):
		"""
		Constructor for an instance of TensorCFR algorithm with given parameters (as a TensorFlow computation graph).

		:param domain: The domain of the game (as an instance of class `FlattenedDomain`). TensorCFR (the CFR+ algorithm)
		 will be launched for this game.
		:param trunk_depth: The number of levels of the trunk where the strategies are kept fixed. It should be an integer
			 between `0` to `self.domain.levels`. It defaults to `0` (no trunk).
		"""
		super().__init__(domain, trunk_depth)

	def predict_equilibrium_values(self, input_tensor, neural_net, permutation_tensor):
		permutation = tf.contrib.distributions.bijectors.Permute(permutation=permutation_tensor)

		# permute input reach probabilities
		tensor_permutation = permutation.forward(input_tensor)

		# use neural net to predict equilibrium values
		predicted_equilibrium_values = neural_net.predict(tensor_permutation)

		# permute back the expected values
		tensor_inverse_permutation = permutation.inverse(predicted_equilibrium_values)
		return tensor_inverse_permutation


if __name__ == '__main__':
	domain = get_domain_by_name("II-GS3_gambit_flattened")
	nn = NeuralNetMockUp()
	tensorcfr = TensorCFR_NN(
		domain,
		trunk_depth=4
	)

	permutation = get_sorted_permutation()
	equilibrium_values = tensorcfr.predict_equilibrium_values([-1., 0., 1.], nn, permutation)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print(sess.run(equilibrium_values))
