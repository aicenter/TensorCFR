#!/usr/bin/env python3
from pprint import pprint

import numpy as np
import tensorflow as tf

from src.commons.constants import CHANCE_PLAYER, PLAYER1, PLAYER2, DEFAULT_AVERAGING_DELAY, INT_DTYPE, FLOAT_DTYPE, \
	REACH_PROBABILITY_OF_ROOT, PROJECT_ROOT, RANDOM_SEED
from src.utils.cfr_utils import get_parents_from_action_counts
from src.utils.gambit_flattened_domains.loader import GambitLoader
from src.utils.tensor_utils import print_tensors, normalize


class FlattenedDomain:
	def __init__(self, domain_name, action_counts, node_to_infoset, utilities, infoset_acting_players,
	             initial_infoset_strategies, reach_probability_of_root_node=None):
		self.domain_name = domain_name
		with tf.variable_scope(self.domain_name, reuse=tf.AUTO_REUSE) as self.domain_scope:
			# tensors on tree dimensions
			self.action_counts = action_counts    # count of (nodal) actions at each levels
			self.levels = len(self.action_counts)
			self.acting_depth = len(self.action_counts) - 1   # the last level has only terminal nodes
			self.max_actions_per_levels = [
				np.amax(self.action_counts[level])
				for level in range(self.acting_depth)
			]    # maximum number of actions per each level
			self.parents = get_parents_from_action_counts(self.action_counts)

			# tensors on tree definition
			self.node_to_infoset = [
				tf.get_variable(
						"node_to_infoset_lvl{}".format(level),
						initializer=tf.cast(node_to_infoset[level], dtype=INT_DTYPE),
						dtype=INT_DTYPE,
				)
				for level in range(self.acting_depth)
			]
			self.mask_of_inner_nodes = [
				tf.greater(
					action_count,
					0,
					name="mask_of_inner_nodes_lvl{}".format(level)
				)
				for level, action_count in enumerate(self.action_counts)
			]
			self.inner_node_to_infoset = self.mask_out_values_in_terminal_nodes(
				self.node_to_infoset,
				name="node_to_infoset"
			)
			self.utilities = [
				tf.get_variable(
						"utilities_lvl{}".format(level),
						initializer=tf.cast(utilities[level], dtype=FLOAT_DTYPE),
				)
				for level in range(self.levels)
			]
			self.infoset_acting_players = [
				tf.get_variable(
						"infoset_acting_players_lvl{}".format(level),
						initializer=tf.cast(infoset_acting_players[level], dtype=INT_DTYPE),
						dtype=INT_DTYPE,
				)
				for level in range(self.acting_depth)
			]
			action_counts_of_inner_nodes = self.mask_out_values_in_terminal_nodes(
				self.action_counts,
				name="action_counts"
			)
			self.infoset_action_counts = [
				tf.scatter_nd_update(
					ref=tf.Variable(
						tf.zeros_like(
							self.infoset_acting_players[level]
						)
					),
					indices=tf.expand_dims(
						self.inner_node_to_infoset[level],
						axis=-1
					),
					updates=action_counts_of_inner_nodes[level],
					name="infoset_action_counts_lvl{}".format(level),
				)
				for level in range(len(self.infoset_acting_players))
			]

			# tensors on strategies
			if reach_probability_of_root_node is None:
				self.reach_probability_of_root_node = tf.get_variable(
						"reach_probability_of_root_node",
						initializer=tf.cast(REACH_PROBABILITY_OF_ROOT, dtype=FLOAT_DTYPE),
				)
			else:
				self.reach_probability_of_root_node = tf.cast(reach_probability_of_root_node, dtype=FLOAT_DTYPE)
			self.initial_infoset_strategies = [
				tf.placeholder_with_default(
						input=tf.cast(initial_infoset_strategies[level], dtype=FLOAT_DTYPE),
						shape=[len(infoset_acting_players[level]), self.max_actions_per_levels[level]],
						name="initial_infoset_strategies_lvl{}".format(level),
				)
				for level in range(self.acting_depth)
			]
			self.current_infoset_strategies = [
				tf.get_variable(
						"current_infoset_strategies_lvl{}".format(level),
						initializer=self.initial_infoset_strategies[level],
				) for level in range(self.acting_depth)
			]
			self.positive_cumulative_regrets = [
				tf.get_variable(
						"positive_cumulative_regrets_lvl{}".format(level),
						initializer=tf.zeros_like(
								self.current_infoset_strategies[level]
						),
						dtype=FLOAT_DTYPE,
				) for level in range(self.acting_depth)
			]
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
					x=tf.cast(1.0, dtype=FLOAT_DTYPE),
					# Opponent's utilities in zero-sum games = (-utilities) of `player_owning_the_utilities`
					y=tf.cast(-1.0, dtype=FLOAT_DTYPE),
					name="signum_of_current_player",
			)
			self.infosets_of_non_chance_player = [
				tf.reshape(
						tf.not_equal(infoset_acting_players[level], CHANCE_PLAYER),
						shape=[self.current_infoset_strategies[level].shape[0]],
						name="infosets_of_non_chance_player_lvl{}".format(level)
				) for level in range(self.acting_depth)
			]

	def mask_out_values_in_terminal_nodes(self, list_of_1D_tensors, name):
		return [
			tf.boolean_mask(
				tensor,
				mask=self.mask_of_inner_nodes[level],
				name="{}_masked_out_in_terminal_nodes_lvl{}".format(name, level)
			)
			for level, tensor in enumerate(list_of_1D_tensors)
		]

	@classmethod
	def init_from_gambit_file(cls, path_to_gambitfile, domain_name="from_gambit"):
		domain_numpy_tensors = GambitLoader(path_to_gambitfile)
		return cls(
			domain_name,
			domain_numpy_tensors.number_of_nodes_actions,
			domain_numpy_tensors.node_to_infoset,
			domain_numpy_tensors.utilities,
			domain_numpy_tensors.infoset_acting_players,
			domain_numpy_tensors.initial_infoset_strategies
		)

	def get_infoset_acting_players(self):
		return self.infoset_acting_players

	def generate_random_strategies(self, seed=None):   # TODO generate up to trunk
		random_weights = [
			tf.random_uniform(
				shape=tf.shape(strategy),
				seed=seed,
				name="random_weights_lvl{}".format(level)
			)
			for level, strategy in enumerate(self.initial_infoset_strategies)
		]
		mask_of_valid_actions = [
			tf.logical_and(
				tf.sequence_mask(
					action_count
				),
				tf.expand_dims(
					self.infosets_of_non_chance_player[level],
					axis=-1
				),
				name="mask_of_valid_actions_lvl{}".format(level)
			)
			for level, action_count in enumerate(self.infoset_action_counts)
		]
		normalized_random_weights = [
			normalize(
				tf.where(
					condition=mask,
					x=random_weights[level],
					y=tf.zeros_like(random_weights[level]),
					name="masked_out_random_weights_lvl{}".format(level)
				),
				name="normalized_random_weights_lvl{}".format(level)
			)
			for level, mask in enumerate(mask_of_valid_actions)
		]
		tf_random_strategies = [
			tf.where(
				condition=mask_non_chance_players,
				x=normalized_random_weights[level],
				y=self.initial_infoset_strategies[level],    # re-use strategy of the chance player
				name="random_strategies_lvl{}".format(level)
			)
			for level, mask_non_chance_players in enumerate(self.infosets_of_non_chance_player)
		]
		with tf.Session(
			# config=tf.ConfigProto(device_count={'GPU': 0})  # uncomment to run on CPU
		) as tmp_session:
			tmp_session.run(tf.global_variables_initializer())
			np_random_strategies = [
				tmp_session.run(tf_op)
				for tf_op in tf_random_strategies
			]
		return np_random_strategies

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
		print("action_counts:")
		pprint(self.action_counts, indent=1, width=80)
		print("max_actions_per_level:")
		pprint(self.max_actions_per_levels, indent=1, width=4)
		print("parents:")
		print_tensors(session, self.parents)
		print("levels: ", self.levels)
		print("acting_depth: ", self.acting_depth)

	def print_domain(self, session):
		print(">>>>>>>>>> {} <<<<<<<<<<".format(self.domain_name))
		for level in range(self.levels):
			print("########## Level {} ##########".format(level))
			if level == 0:
				print_tensors(session, [self.reach_probability_of_root_node])
			print_tensors(session, [self.utilities[level]])
			if level != range(self.levels)[-1]:
				print_tensors(session, [
					self.node_to_infoset[level],
					self.infoset_acting_players[level],
					self.infosets_of_non_chance_player[level],
					self.infoset_action_counts[level],
					self.initial_infoset_strategies[level],
					self.current_infoset_strategies[level],
					self.positive_cumulative_regrets[level],
					self.cumulative_infoset_strategies[level]
				])
		self.print_misc_variables(session)


if __name__ == '__main__':
	import os
	import src.domains.flattened_hunger_games.flattened_hunger_games_as_numpy_values as fhg
	flattened_hunger_games = FlattenedDomain(
			domain_name="flattened_hunger_games",
			action_counts=fhg.action_counts,
			node_to_infoset=fhg.node_to_infoset,
			utilities=fhg.utilities,
			infoset_acting_players=fhg.infoset_acting_players,
			initial_infoset_strategies=fhg.initial_infoset_strategies,
	)
	handcoded_domains = [
		flattened_hunger_games
	]

	domain01_efg = os.path.join(
		PROJECT_ROOT,
		'doc',
		'domain01_via_gambit.efg'
	)
	hunger_games_efg = os.path.join(
		PROJECT_ROOT,
		'doc',
		'hunger_games',
		'hunger_games_via_gambit.efg'
	)
	hunger_games_2_efg = os.path.join(
		PROJECT_ROOT,
		'doc',
		'hunger_games_2',
		'hunger_games_via_gambit.efg'
	)
	efg_files = [
		domain01_efg,
		hunger_games_efg,
		hunger_games_2_efg
	]
	domain_names = [
		"domain01_gambit",
		"hunger_games_gambit",
		"hunger_games_2_gambit",
	]
	domains_from_gambit = [
		FlattenedDomain.init_from_gambit_file(efg_file, domain_name)
		for efg_file, domain_name in zip(efg_files, domain_names)
	]

	domains = handcoded_domains + domains_from_gambit
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for domain in domains:
			domain.print_domain(sess)
			for i in range(2):
				print("Random strategies #{}:".format(i + 1))
				strategies = domain.generate_random_strategies(seed=RANDOM_SEED + i)
				for level, strategy in enumerate(strategies):
					print("Level {}".format(level))
					pprint(strategy.tolist(), indent=1, width=140)
