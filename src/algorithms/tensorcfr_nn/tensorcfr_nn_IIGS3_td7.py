#!/usr/bin/env python3
import tensorflow as tf

from src.algorithms.tensorcfr_nn.TensorCFR_NN import TensorCFR_NN
from src.domains.available_domains import get_domain_by_name
from src.nn.features.goofspiel.IIGS3.sorting_permutation_by_public_states import get_permutation_by_public_states

if __name__ == '__main__':
	domain_ = get_domain_by_name("II-GS3_gambit_flattened")
	tensorcfr = TensorCFR_NN(
		domain_,
		trunk_depth=7
	)

	permutation = get_permutation_by_public_states()
	input_reaches = tf.constant([-1., 0., 2.])
	equilibrium_values = tensorcfr.predict_equilibrial_values(input_reaches)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print(sess.run(equilibrium_values))
