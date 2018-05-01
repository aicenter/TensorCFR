#!/usr/bin/env python3

import tensorflow as tf

import src.domains.domain01.domain01_as_numpy_values as d1
from src.domains.Domain import Domain

domain01 = Domain(
		domain_name="domain01",
		actions_per_levels=d1.actions_per_levels,
		node_to_infoset=d1.node_to_infoset,
		node_types=d1.node_types,
		utilities=d1.utilities,
		infoset_acting_players=d1.infoset_acting_players,
		initial_infoset_strategies=d1.initial_infoset_strategies,
)


if __name__ == '__main__':
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		domain01.print_domain(sess)
