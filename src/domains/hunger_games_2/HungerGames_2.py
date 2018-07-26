#!/usr/bin/env python3

import tensorflow as tf

import src.domains.hunger_games_2.hunger_games_2_as_numpy_values as hg2
from src.domains.Domain import Domain


# domain `hunger_games_2`: see doc/hunger_games_via_drawing.png


def get_domain_hunger_games_2():
	return Domain(
			domain_name="hunger_games",
			actions_per_levels=hg.actions_per_levels,
			node_to_infoset=hg.node_to_infoset,
			node_types=hg.node_types,
			utilities=hg.utilities,
			infoset_acting_players=hg.infoset_acting_players,
			initial_infoset_strategies=hg.initial_infoset_strategies,
	)


if __name__ == '__main__':
	hunger_games = get_domain_hunger_games()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		hunger_games.print_domain(sess)
