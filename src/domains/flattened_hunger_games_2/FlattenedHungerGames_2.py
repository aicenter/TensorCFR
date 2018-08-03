#!/usr/bin/env python3

import tensorflow as tf

import src.domains.flattened_hunger_games_2.flattened_hunger_games_2_as_numpy_values as fhg_2
from src.domains.FlattenedDomain import FlattenedDomain


# flattened domain `hunger_games`: see `doc/hunger_games_via_drawing.png` and `doc/hunger_games_2/`

def get_flattened_domain_hunger_games_2():
	return FlattenedDomain(
		domain_name="flattened_hunger_games_2",
		domain_parameters={},
		action_counts=fhg_2.action_counts,
		node_to_infoset=fhg_2.node_to_infoset,
		utilities=fhg_2.utilities,
		infoset_acting_players=fhg_2.infoset_acting_players,
		initial_infoset_strategies=fhg_2.initial_infoset_strategies,
	)


if __name__ == '__main__':
	flattened_hunger_games_2 = get_flattened_domain_hunger_games_2()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		flattened_hunger_games_2.print_domain(sess)
