#!/usr/bin/env python3

import tensorflow as tf

import src.domains.flattened_hunger_games.flattened_hunger_games_as_numpy_values as fhg
from src.domains.FlattenedDomain import FlattenedDomain


# flattened domain `hunger_games`: see `doc/hunger_games_via_drawing.png` and `doc/hunger_games/`

def get_flattened_domain_hunger_games():
	return FlattenedDomain(
		domain_name="flattened_hunger_games",
		domain_parameters={},
		action_counts=fhg.action_counts,
		node_to_infoset=fhg.node_to_infoset,
		utilities=fhg.utilities,
		infoset_acting_players=fhg.infoset_acting_players,
		initial_infoset_strategies=fhg.initial_infoset_strategies,
	)


if __name__ == '__main__':
	flattened_hunger_games = get_flattened_domain_hunger_games()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		flattened_hunger_games.print_domain(sess)
