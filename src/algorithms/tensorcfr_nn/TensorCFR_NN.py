#!/usr/bin/env python3
import tensorflow as tf

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.domains.available_domains import get_domain_by_name


def get_sorted_permutation():
	return [2, 1, 0]


class NeuralNetMockUp:
	def predict(self, input_tensor):
		return [5, 5, 6]


class TensorCFR_NN(TensorCFRFixedTrunkStrategies):
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

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print(sess.run(tensorcfr.predict_equilibrium_values([-1., 0., 1.], nn, get_sorted_permutation())))
