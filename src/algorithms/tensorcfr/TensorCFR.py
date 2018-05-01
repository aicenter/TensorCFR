#!/usr/bin/env python3

import tensorflow as tf


class TensorCFR:
	def __init__(self):
		pass


if __name__ == '__main__':
	import src.domains.domain01.domain01_as_numpy_values as d1
	domain01 = Domain(
			domain_name="domain01",
			actions_per_levels=d1.actions_per_levels,
			node_to_infoset=d1.node_to_infoset,
			node_types=d1.node_types,
			utilities=d1.utilities,
			infoset_acting_players=d1.infoset_acting_players,
			initial_infoset_strategies=d1.initial_infoset_strategies,
	)
	import src.domains.matching_pennies.matching_pennies_as_numpy_values as mp
	matching_pennies = Domain(
			domain_name="matching_pennies",
			actions_per_levels=mp.actions_per_levels,
			node_to_infoset=mp.node_to_infoset,
			node_types=mp.node_types,
			utilities=mp.utilities,
			infoset_acting_players=mp.infoset_acting_players,
			initial_infoset_strategies=mp.initial_infoset_strategies,
	)

	# tensorcfr_domain = TensorCFR()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		domain01.print_domain(sess)
		matching_pennies.print_domain(sess)
