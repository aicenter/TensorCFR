#!/usr/bin/env python3

import tensorflow as tf

from src.commons.constants import PLAYER1, PLAYER2
from src.domains.Domain import Domain
from src.utils.distribute_strategies_to_nodes import distribute_strategies_to_nodes
from src.utils.tensor_utils import print_tensors


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

	def get_node_strategies(self):
		with tf.variable_scope("node_strategies"):
			return [
				distribute_strategies_to_nodes(
						self.domain.current_infoset_strategies[level],
						self.domain.node_to_infoset[level],
						name="node_strategies_lvl{}".format(level)
				) for level in range(self.domain.acting_depth)
			]

	def get_node_cf_strategies(self, updating_player=None):
		if updating_player is None:
			updating_player = self.domain.current_updating_player
		with tf.variable_scope("node_cf_strategies"):
			# TODO generate node_cf_strategies_* with tf.where on node_strategies
			return [
				distribute_strategies_to_nodes(
						self.domain.current_infoset_strategies[level],
						self.domain.node_to_infoset[level],
						updating_player=updating_player,
						acting_players=self.domain.infoset_acting_players[level],
						name="node_cf_strategies_lvl{}".format(level)
				) for level in range(self.domain.acting_depth)
			]

	def show_strategies(self, session):
		node_strategies = self.get_node_strategies()
		node_cf_strategies = self.get_node_cf_strategies()
		for level in range(self.domain.acting_depth):
			print("########## Level {} ##########".format(level))
			print_tensors(session, [
				self.domain.node_to_infoset[level],
				self.domain.current_infoset_strategies[level],
				node_strategies[level],
				self.domain.infoset_acting_players[level],
				node_cf_strategies[level],
			])


if __name__ == '__main__':
	from src.domains.domain01.Domain01 import domain01
	from src.domains.matching_pennies.MatchingPennies import matching_pennies
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for tensorcfr in [TensorCFR(domain01), TensorCFR(matching_pennies)]:
			tensorcfr.domain.print_domain(sess)
