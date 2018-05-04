#!/usr/bin/env python3

import tensorflow as tf

from src.algorithms.tensorcfr.TensorCFR import TensorCFR
from src.domains.available_domains import get_domain_by_name
from src.utils.tensor_utils import print_tensors

if __name__ == '__main__':
	domain01 = get_domain_by_name("domain01")
	matching_pennies = get_domain_by_name("matching_pennies")
	for tensorcfr in [TensorCFR(domain01), TensorCFR(matching_pennies)]:
		print(">>>>>>>>>> {} <<<<<<<<<<".format(tensorcfr.domain.domain_name))
		infoset_cf_values, infoset_cf_values_per_actions = tensorcfr.get_infoset_cf_values()
		regrets = tensorcfr.get_regrets()
		update_regrets_ops = tensorcfr.update_positive_cumulative_regrets()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for level in range(tensorcfr.domain.acting_depth):
				print("########## Level {} ##########".format(level))
				print_tensors(sess, [infoset_cf_values_per_actions[level], infoset_cf_values[level], regrets[level]])
				print("___________________________________\n")
				# TODO create a unit out of the following `print_tensors()`
				print_tensors(sess, [
					tensorcfr.domain.infoset_acting_players[level],
					tensorcfr.domain.current_updating_player,
					tensorcfr.domain.positive_cumulative_regrets[level],
					tensorcfr.domain.positive_cumulative_regrets[level],
					update_regrets_ops[level],
					tensorcfr.domain.positive_cumulative_regrets[level],
					tensorcfr.domain.positive_cumulative_regrets[level],
					update_regrets_ops[level],
					tensorcfr.domain.positive_cumulative_regrets[level],
					tensorcfr.domain.positive_cumulative_regrets[level]
				])
