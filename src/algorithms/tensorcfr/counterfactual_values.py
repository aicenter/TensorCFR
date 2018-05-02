#!/usr/bin/env python3

import tensorflow as tf

from src.algorithms.tensorcfr.TensorCFR import TensorCFR
from src.domains.domain01.Domain01 import domain01
from src.domains.matching_pennies.MatchingPennies import matching_pennies
from src.utils.tensor_utils import print_tensors


if __name__ == '__main__':
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for tensorcfr in [TensorCFR(domain01), TensorCFR(matching_pennies)]:
			print(">>>>>>>>>> {} <<<<<<<<<<".format(tensorcfr.domain.domain_name))
			nodal_reach_probabilities = tensorcfr.get_nodal_reach_probabilities()
			expected_values = tensorcfr.get_expected_values()
			cf_values_nodes = tensorcfr.get_nodal_cf_values()
			infoset_cf_values, infoset_cf_values_per_actions = tensorcfr.get_infoset_cf_values()
			for i in range(tensorcfr.domain.levels):
				print("########## Level {} ##########".format(i))
				print_tensors(sess, [
					nodal_reach_probabilities[i],
					expected_values[i],
					cf_values_nodes[i]
				])
				if i < tensorcfr.domain.levels - 1:
					print_tensors(sess, [
						tensorcfr.domain.current_infoset_strategies[i],
						infoset_cf_values_per_actions[i],
						infoset_cf_values[i],
					])
