#!/usr/bin/env python3
import tensorflow as tf

from src.algorithms.tensorcfr_nn.TensorCFR_NN import TensorCFR_NN
from src.domains.available_domains import get_domain_by_name
from src.nn.features.goofspiel.IIGS3.sorting_permutation_by_public_states import get_permutation_by_public_states
from src.utils.tf_utils import print_tensors

if __name__ == '__main__':
	domain_ = get_domain_by_name("II-GS3_gambit_flattened")
	nn_input_permutation = get_permutation_by_public_states()
	tensorcfr = TensorCFR_NN(
		domain_,
		nn_input_permutation=nn_input_permutation,
		trunk_depth=7
	)

	# input_reaches = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 96, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
	#                              24, 25, 26, 27, 28, 29, 30, 31, 69, 33, 34, 35])
	input_reaches = tf.range(len(nn_input_permutation), name="input_reaches")
	equilibrium_values = tensorcfr.predict_equilibrial_values(input_reaches)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print_tensors(sess, [input_reaches, equilibrium_values])
