#!/usr/bin/env python3

import tensorflow as tf

from src.commons.constants import CHANCE_PLAYER, PLAYER1, PLAYER2, DEFAULT_AVERAGING_DELAY, INT_DTYPE
from src.utils.tensor_utils import print_tensors


class Domain:
	def __init__(self, domain_name, actions_per_levels, node_to_infoset, node_types, utilities, infoset_acting_players,
	             initial_infoset_strategies, reach_probability_of_root_node=None):
		self.domain_name = domain_name
		with tf.variable_scope(self.domain_name) as self.domain_scope:
			# tensors on tree dimensions
			self.actions_per_levels = actions_per_levels    # maximum number of actions per each level
			self.levels = len(self.actions_per_levels) + 1  # accounting for 0th level
			self.acting_depth = len(self.actions_per_levels)
			self.shape = [self.actions_per_levels[:i] for i in range(self.levels)]

			# tensors on tree definition
			self.node_to_infoset = [
				tf.get_variable("node_to_infoset_lvl{}".format(level), initializer=node_to_infoset[level])
				for level in range(self.acting_depth)
			]
			self.node_types = [
				tf.get_variable("node_types_lvl{}".format(level), initializer=node_types[level])
				for level in range(self.levels)
			]
			self.utilities = [
				tf.get_variable("utilities_lvl{}".format(level), initializer=utilities[level])
				for level in range(self.levels)
			]
			self.infoset_acting_players = [
				tf.get_variable("infoset_acting_players_lvl{}".format(level), initializer=infoset_acting_players[level])
				for level in range(self.acting_depth)
			]

			# tensors on strategies
			if reach_probability_of_root_node is None:
				self.reach_probability_of_root_node = tf.get_variable("reach_probability_of_root_node", initializer=1.0)
			else:
				self.reach_probability_of_root_node = reach_probability_of_root_node
			self.initial_infoset_strategies = [
				tf.placeholder_with_default(
						input=initial_infoset_strategies[level],
						shape=[len(infoset_acting_players[level]), actions_per_levels[level]],
						name="initial_infoset_strategies_lvl{}".format(level)
				)
				for level in range(self.acting_depth)
			]
			self.current_infoset_strategies = [
				tf.get_variable(
						"current_infoset_strategies_lvl{}".format(level),
						initializer=initial_infoset_strategies[level],
				) for level in range(self.acting_depth)
			]
			self.positive_cumulative_regrets = [
				tf.get_variable(
						"positive_cumulative_regrets_lvl{}".format(level),
						initializer=tf.zeros_like(
								self.current_infoset_strategies[level]
						),
				) for level in range(self.acting_depth)
			]
			self.cumulative_infoset_strategies = [    # used for the final average strategy
				tf.get_variable(
						name="cumulative_infoset_strategies_lvl{}".format(level),
						initializer=tf.zeros_like(
								self.current_infoset_strategies[level]
						),
				)
				for level in range(self.acting_depth)
			]

			# tensors on CFR+ iterations
			self.cfr_step = tf.get_variable(     # counter of CFR+ iterations
					"cfr_step",
					initializer=1,
					dtype=tf.int32,
			)
			self.averaging_delay = tf.get_variable(  # https://arxiv.org/pdf/1407.5042.pdf (Figure 2)
					name="averaging_delay",
					initializer=DEFAULT_AVERAGING_DELAY,
					dtype=tf.int32,
			)

			# tensors on players
			self.current_updating_player = tf.get_variable(
					"current_updating_player",
					initializer=PLAYER1,
					dtype=INT_DTYPE,
			)
			self.current_opponent = tf.get_variable(
					"current_opponent",
					initializer=PLAYER2,
					dtype=INT_DTYPE,
			)
			self.player_owning_the_utilities = tf.constant(
					PLAYER1,  # `utilities[]` are defined from this player's point of view
					name="player_owning_the_utilities"
			)
			self.signum_of_current_player = tf.where(
					condition=tf.equal(self.current_updating_player, self.player_owning_the_utilities),
					x=1.0,
					y=-1.0,  # Opponent's utilities in zero-sum games = (-utilities) of `player_owning_the_utilities`
					name="signum_of_current_player",
			)
			self.infosets_of_non_chance_player = [
				tf.reshape(
						tf.not_equal(infoset_acting_players[level], CHANCE_PLAYER),
						shape=[self.current_infoset_strategies[level].shape[0]],
						name="infosets_of_acting_player_lvl{}".format(level)
				) for level in range(self.acting_depth)
			]

	def get_infoset_acting_players(self):
		return self.infoset_acting_players

	def print_domain(self, session):
		for level in range(self.levels):
			print("########## Level {} ##########".format(level))
			if level == 0:
				print_tensors(sess, [self.reach_probability_of_root_node])
			print_tensors(sess, [self.node_types[level], self.utilities[level]])
			if level != range(self.levels)[-1]:
				print_tensors(sess, [
					self.node_to_infoset[level],
					self.infoset_acting_players[level],
					self.initial_infoset_strategies[level],
					self.current_infoset_strategies[level],
					self.positive_cumulative_regrets[level],
					self.cumulative_infoset_strategies[level],
				])
		print("########## Misc ##########")
		print_tensors(session, [
			self.cfr_step,
			self.averaging_delay,
			self.current_updating_player,
			self.current_opponent,
			self.signum_of_current_player,
			self.player_owning_the_utilities,
		])


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
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		domain01.print_domain(session=sess)
