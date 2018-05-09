#!/usr/bin/env python3

import tensorflow as tf

from src.commons.constants import CHANCE_PLAYER, PLAYER1, PLAYER2, DEFAULT_AVERAGING_DELAY, INT_DTYPE, FLOAT_DTYPE
from src.utils.tensor_utils import print_tensors
from src.utils.gambit_efg_loader import GambitEFGLoader


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
			with tf.variable_scope("node_to_infoset", reuse=tf.AUTO_REUSE):
				self.node_to_infoset = [
					tf.get_variable("node_to_infoset_lvl{}".format(level), initializer=node_to_infoset[level])
					for level in range(self.acting_depth)
				]
			with tf.variable_scope("nodal_types", reuse=tf.AUTO_REUSE):
				self.node_types = [
					tf.get_variable("node_types_lvl{}".format(level), initializer=tf.cast(node_types[level], dtype=INT_DTYPE))
					for level in range(self.levels)
				]
			with tf.variable_scope("utilities", reuse=tf.AUTO_REUSE):
				self.utilities = [
					tf.get_variable(
							"utilities_lvl{}".format(level),
							initializer=tf.cast(utilities[level], dtype=FLOAT_DTYPE),
					)
					for level in range(self.levels)
				]
			with tf.variable_scope("acting_players", reuse=tf.AUTO_REUSE):
				self.infoset_acting_players = [
					tf.get_variable("infoset_acting_players_lvl{}".format(level), initializer=infoset_acting_players[level])
					for level in range(self.acting_depth)
				]

			# tensors on strategies
			if reach_probability_of_root_node is None:
				self.reach_probability_of_root_node = tf.get_variable(
						"reach_probability_of_root_node",
						initializer=tf.cast(1.0, dtype=FLOAT_DTYPE),
				)
			else:
				self.reach_probability_of_root_node = tf.cast(reach_probability_of_root_node, dtype=FLOAT_DTYPE)
			with tf.variable_scope("initial_strategies", reuse=tf.AUTO_REUSE):
				self.initial_infoset_strategies = [
					tf.placeholder_with_default(
							input=tf.cast(initial_infoset_strategies[level], dtype=FLOAT_DTYPE),
							shape=[len(infoset_acting_players[level]), actions_per_levels[level]],
							name="initial_infoset_strategies_lvl{}".format(level),
					)
					for level in range(self.acting_depth)
				]
			with tf.variable_scope("current_strategies", reuse=tf.AUTO_REUSE):
				self.current_infoset_strategies = [
					tf.get_variable(
							"current_infoset_strategies_lvl{}".format(level),
							initializer=self.initial_infoset_strategies[level],
					) for level in range(self.acting_depth)
				]
			with tf.variable_scope("cumulative_regrets", reuse=tf.AUTO_REUSE):
				self.positive_cumulative_regrets = [
					tf.get_variable(
							"positive_cumulative_regrets_lvl{}".format(level),
							initializer=tf.zeros_like(
									self.current_infoset_strategies[level]
							),
							dtype=FLOAT_DTYPE,
					) for level in range(self.acting_depth)
				]
			with tf.variable_scope("cumulative_strategies", reuse=tf.AUTO_REUSE):
				self.cumulative_infoset_strategies = [    # used for the final average strategy
					tf.get_variable(
							name="cumulative_infoset_strategies_lvl{}".format(level),
							initializer=tf.zeros_like(
									self.current_infoset_strategies[level]
							),
							dtype=FLOAT_DTYPE,
					)
					for level in range(self.acting_depth)
				]

			# tensors on CFR+ iterations
			with tf.variable_scope("iterations", reuse=tf.AUTO_REUSE):
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
			with tf.variable_scope("players", reuse=tf.AUTO_REUSE):
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
						x=tf.cast(1.0, dtype=FLOAT_DTYPE),
						# Opponent's utilities in zero-sum games = (-utilities) of `player_owning_the_utilities`
						y=tf.cast(-1.0, dtype=FLOAT_DTYPE),
						name="signum_of_current_player",
				)
			with tf.variable_scope("nonchance_infosets", reuse=tf.AUTO_REUSE):
				self.infosets_of_non_chance_player = [
					tf.get_variable(
							name="infosets_of_acting_player_lvl{}".format(level),
							initializer=tf.reshape(
									tf.not_equal(infoset_acting_players[level], CHANCE_PLAYER),
									shape=[self.current_infoset_strategies[level].shape[0]],
							)
					) for level in range(self.acting_depth)
				]

	@classmethod
	def init_from_gambit_file(cls, path_to_gambitfile, domain_name="from_gambit"):
		domain_numpy_tensors = GambitEFGLoader(path_to_gambitfile)
		return cls(
			domain_name,
			domain_numpy_tensors.actions_per_levels,
			domain_numpy_tensors.node_to_infoset,
			domain_numpy_tensors.node_types,
			domain_numpy_tensors.utilities,
			domain_numpy_tensors.infoset_acting_players,
			domain_numpy_tensors.initial_infoset_strategies
		)

	def get_infoset_acting_players(self):
		return self.infoset_acting_players

	def print_misc_variables(self, session):
		print("########## Misc ##########")
		print_tensors(session, [
			self.cfr_step,
			self.averaging_delay,
			self.current_updating_player,
			self.current_opponent,
			self.signum_of_current_player,
			self.player_owning_the_utilities,
		])

	def print_domain(self, session):
		print(">>>>>>>>>> {} <<<<<<<<<<".format(self.domain_name))
		for level in range(self.levels):
			print("########## Level {} ##########".format(level))
			if level == 0:
				print_tensors(session, [self.reach_probability_of_root_node])
			print_tensors(session, [self.node_types[level], self.utilities[level]])
			if level != range(self.levels)[-1]:
				print_tensors(session, [
					self.node_to_infoset[level],
					self.infoset_acting_players[level],
					self.initial_infoset_strategies[level],
					self.current_infoset_strategies[level],
					self.positive_cumulative_regrets[level],
					self.cumulative_infoset_strategies[level],
				])
		self.print_misc_variables(session)


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
	import os
	domain01_efg = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'doc', 'domain01_via_gambit.efg')
	domain01_gambit = Domain.init_from_gambit_file(domain01_efg)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		domain01.print_domain(sess)
		matching_pennies.print_domain(sess)
		domain01_gambit.print_domain(sess)
