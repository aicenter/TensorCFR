#!/usr/bin/env python3
import tensorflow as tf


class NeuralNetMockUp:
	def predict(self, input_tensor):
		return [5, 5, 6]


class TensorCFR_NN:  # TODO pridat TensorCFRFixedTrunkStrategies
	def get_sorted_permutation(self):
		return [2, 1, 0]

	def predict_equilibrium_values(self, input_tensor, neural_net, permutation_tensor=None):
		if permutation_tensor is None:
			permutation_tensor = self.get_sorted_permutation()

		permutation = tf.contrib.distributions.bijectors.Permute(permutation=permutation_tensor)

		# permute input reach probabilities
		tensor_permutation = permutation.forward(input_tensor)

		# use neural net to predict equilibrium values
		predicted_equilibrium_values = neural_net.predict(tensor_permutation)

		# permute back the expected values
		tensor_inverse_permutation = permutation.inverse(predicted_equilibrium_values)
		return tensor_inverse_permutation


if __name__ == '__main__':
	nn = NeuralNetMockUp()
	tensorcfr_nn = TensorCFR_NN()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print(sess.run(tensorcfr_nn.predict_equilibrium_values([-1., 0., 1.], nn)))
