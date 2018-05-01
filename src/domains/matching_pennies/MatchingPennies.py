#!/usr/bin/env python3

import tensorflow as tf

import src.domains.matching_pennies.matching_pennies_as_numpy_values as mp
from src.domains.Domain import Domain

matching_pennies = Domain(
		domain_name="matching_pennies",
		actions_per_levels=mp.actions_per_levels,
		node_to_infoset=mp.node_to_infoset,
		node_types=mp.node_types,
		utilities=mp.utilities,
		infoset_acting_players=mp.infoset_acting_players,
		initial_infoset_strategies=mp.initial_infoset_strategies,
)


if __name__ == '__main__':
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		matching_pennies.print_domain(sess)
