#!/usr/bin/env python3

import tensorflow as tf

from src.commons.constants import PLAYER1, PLAYER2
from src.domains.Domain import Domain


class TensorCFR:
	def __init__(self, domain: Domain):
		self.domain = domain

	@staticmethod
	def get_the_other_player_of(tensor_variable_of_player):
		with tf.variable_scope("get_the_other_player"):
			return tf.where(
					condition=tf.equal(tensor_variable_of_player, PLAYER1),
					x=PLAYER2,
					y=PLAYER1,
					name="get_the_other_player"
			)

	def swap_players(self):
		with tf.variable_scope("swap_players"):
			with tf.variable_scope("new_updating_player"):
				assign_new_updating_player = tf.assign(
						ref=self.domain.current_updating_player,
						value=TensorCFR.get_the_other_player_of(self.domain.current_updating_player),
						name="assign_new_updating_player",
				)
			with tf.variable_scope("new_opponent"):
				assign_opponent = tf.assign(
						ref=self.domain.current_opponent,
						value=TensorCFR.get_the_other_player_of(self.domain.current_opponent),
						name="assign_new_opponent",
				)
			return tf.tuple(
					[
						assign_new_updating_player,
						assign_opponent,
					],
					name="swap",
			)


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

	tensorcfr_domain01 = TensorCFR(domain01)
	tensorcfr_matching_pennies = TensorCFR(matching_pennies)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		tensorcfr_domain01.domain.print_domain(sess)
		tensorcfr_matching_pennies.domain.print_domain(sess)
