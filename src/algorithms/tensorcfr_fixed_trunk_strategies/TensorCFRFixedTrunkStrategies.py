#!/usr/bin/env python3
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from src.commons.constants import PLAYER1, PLAYER2, DEFAULT_TOTAL_STEPS, FLOAT_DTYPE, \
	DEFAULT_AVERAGING_DELAY, INT_DTYPE, DEFAULT_DATASET_SIZE, ALL_PLAYERS
from src.domains.FlattenedDomain import FlattenedDomain
from src.domains.available_domains import get_domain_by_name
from src.utils.cfr_utils import flatten_strategies_via_action_counts, get_action_and_infoset_values, \
	distribute_strategies_to_inner_nodes
from src.utils.other_utils import get_current_timestamp
from src.utils.tensor_utils import print_tensors, expanded_multiply, scatter_nd_sum, masked_assign, normalize


class TensorCFRFixedTrunkStrategies:
	def __init__(self, domain: FlattenedDomain, trunk_depth=0):
		"""
		Constructor for an instance of TensorCFR algorithm with given parameters (as a TensorFlow computation graph).

		:param domain: The domain of the game (as an instance of class `FlattenedDomain`). TensorCFR (the CFR+ algorithm)
		 will be launched for this game.
		:param trunk_depth: The number of levels of the trunk where the strategies are kept fixed. It should be an integer
			 between `0` to `self.domain.levels`. It defaults to `0` (no trunk).
		"""
		self.domain = domain
		self.session = None

		with tf.variable_scope("increment_step"):
			self.increment_cfr_step = tf.assign_add(
				ref=self.domain.cfr_step,
				value=1,
				name="increment_cfr_step"
			)
		self.average_infoset_strategies = None
		self.set_average_infoset_strategies()

		self.summary_writer = None
		self.log_directory = None
		self.trunk_depth = trunk_depth
		self.boundary_level = self.trunk_depth
		last_level_with_infosets = self.domain.acting_depth - 1
		assert 0 <= self.boundary_level <= last_level_with_infosets, \
			"Invalid boundary_level == {}: make sure that 0 <= trunk_depth <= {}.".format(
				self.boundary_level,
				last_level_with_infosets
			)
		self.trunk_depth_infoset_cfvs = None
		self.trunk_depth_nodal_expected_values = None
		self.cfr_parameters = {}
		self.data_id = None

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
					value=TensorCFRFixedTrunkStrategies.get_the_other_player_of(self.domain.current_updating_player),
					name="assign_new_updating_player",
				)
			with tf.variable_scope("new_opponent"):
				assign_opponent = tf.assign(
					ref=self.domain.current_opponent,
					value=TensorCFRFixedTrunkStrategies.get_the_other_player_of(self.domain.current_opponent),
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
			node_strategies = [
				distribute_strategies_to_inner_nodes(
					self.domain.current_infoset_strategies[level],
					self.domain.node_to_infoset[level],
					self.domain.mask_of_inner_nodes[level],
					name="node_strategies_lvl{}".format(level)
				) for level in range(self.domain.acting_depth)
			]
			flattened_node_strategies = flatten_strategies_via_action_counts(
				node_strategies,
				self.domain.action_counts
			)
			return flattened_node_strategies

	def get_node_cf_strategies(self, for_player=None):
		if for_player is None:
			for_player = self.domain.current_updating_player
		with tf.variable_scope("node_cf_strategies"):
			# TODO generate node_cf_strategies_* with tf.where on node_strategies
			node_cf_strategies = [
				distribute_strategies_to_inner_nodes(
					self.domain.current_infoset_strategies[level],
					self.domain.node_to_infoset[level],
					self.domain.mask_of_inner_nodes[level],
					for_player=for_player,
					acting_players=self.domain.infoset_acting_players[level],
					name="node_cf_strategies_lvl{}".format(level)
				) for level in range(self.domain.acting_depth)
			]
			flattened_node_cf_strategies = flatten_strategies_via_action_counts(
				node_cf_strategies,
				self.domain.action_counts,
				basename="nodal_cf_strategies"
			)
			return flattened_node_cf_strategies

	def show_strategies(self):
		node_strategies = self.get_node_strategies()
		node_cf_strategies = self.get_node_cf_strategies()
		for level in range(self.domain.acting_depth):
			print("########## Level {} ##########".format(level))
			print_tensors(self.session, [
				self.domain.node_to_infoset[level],
				self.domain.current_infoset_strategies[level],
				node_strategies[level],
				self.domain.infoset_acting_players[level],
				node_cf_strategies[level],
			])

	def get_expected_values(self, for_player=None):
		"""
		Compute expected values of nodes using the bottom-up tree traversal.

		:param for_player: The player for which the expected values are computed. These values are usually computed for the
		 updating player when counterfactual values are computed. Therefore, by default the expected values are computed for
		 the `current_updating_player`, i.e. multiplied with `signum` of `signum_of_current_player`.
		:return: The expected values of nodes based on `current_infoset_strategies`.
		"""
		if for_player is None:
			signum = self.domain.signum_of_current_player
			player_name = "current_player"
		else:
			signum = tf.where(
				condition=tf.equal(for_player, self.domain.player_owning_the_utilities),
				x=tf.cast(1.0, dtype=FLOAT_DTYPE),
				# Opponent's utilities in zero-sum games = (-utilities) of `player_owning_the_utilities`
				y=tf.cast(-1.0, dtype=FLOAT_DTYPE),
				name="signum_of_player_{}".format(for_player),
			)
			player_name = "player{}".format(for_player)

		node_strategies = self.get_node_strategies()
		with tf.variable_scope("expected_values"):
			expected_values = [None] * self.domain.levels
			with tf.variable_scope("level{}".format(self.domain.levels - 1)):
				expected_values[self.domain.levels - 1] = tf.multiply(
					signum,
					self.domain.utilities[self.domain.levels - 1],
					name="expected_values_lvl{}_for_{}".format(self.domain.levels - 1, player_name)
				)
			for level in reversed(range(self.domain.levels - 1)):
				with tf.variable_scope("level{}".format(level)):
					weighted_sum_of_values = tf.segment_sum(
						data=node_strategies[level + 1] * expected_values[level + 1],
						segment_ids=self.domain.parents[level + 1],
						name="weighted_sum_of_values_lvl{}".format(level),
					)
					scatter_copy_indices = tf.expand_dims(
						tf.cumsum(
							tf.ones_like(weighted_sum_of_values, dtype=INT_DTYPE),
							exclusive=True,
						),
						axis=-1,
						name="scatter_copy_indices_lvl{}".format(level)
					)
					extended_weighted_sum = tf.scatter_nd(
						indices=scatter_copy_indices,
						updates=weighted_sum_of_values,
						shape=self.domain.utilities[level].shape,
						name="extended_weighted_sum_lvl{}".format(level)
					)
					expected_values[level] = tf.where(
						condition=self.domain.mask_of_inner_nodes[level],
						x=extended_weighted_sum,
						y=signum * tf.reshape(
							self.domain.utilities[level],
							shape=[self.domain.utilities[level].shape[-1]],
						),
						name="expected_values_lvl{}_for_{}".format(level, player_name)
					)
		return expected_values

	def show_expected_values(self):
		self.domain.print_misc_variables(session=self.session)
		node_strategies = self.get_node_strategies()
		expected_values_for_current_player = self.get_expected_values()
		expected_values_for_player1 = self.get_expected_values(for_player=PLAYER1)
		expected_values_for_player2 = self.get_expected_values(for_player=PLAYER2)
		for level in reversed(range(self.domain.levels)):
			print("########## Level {} ##########".format(level))
			if level < len(node_strategies):
				print_tensors(self.session, [node_strategies[level]])
			print_tensors(self.session, [
				tf.multiply(
					self.domain.signum_of_current_player,
					self.domain.utilities[level],
					name="signum_utilities_lvl{}".format(level)
				),
				expected_values_for_current_player[level],
				expected_values_for_player1[level],
				expected_values_for_player2[level],
			])

	def get_nodal_reach_probabilities(self, for_player=None):
		"""
		Compute reach probabilities of nodes using the top-down tree traversal.

		:param for_player: The player for which the reach probabilities are computed. These probabilities are usually
		 computed for the updating player when counterfactual values are computed. Therefore, `for_player` is set to
			`current_updating_player` by default.
		:return: The reach probabilities of nodes based on `current_infoset_strategies`.
		"""
		nodal_strategies = self.get_node_cf_strategies(for_player=for_player)
		if for_player == ALL_PLAYERS:
			nodal_strategies = self.get_node_strategies()
			player_name = "all_players"
		elif for_player in [PLAYER1, PLAYER2]:
			player_name = "player{}".format(for_player)
		else:
			player_name = "current_player"

		with tf.variable_scope("nodal_reach_probabilities_for_{}".format(player_name)):
			nodal_reach_probabilities = [None] * self.domain.levels
			nodal_reach_probabilities[0] = tf.expand_dims(
				self.domain.reach_probability_of_root_node,
				axis=0
			)
			for level in range(1, self.domain.levels):
				nodal_reach_probabilities[level] = tf.multiply(
					tf.gather(
						nodal_reach_probabilities[level - 1],
						indices=self.domain.parents[level],
						name="children_reach_probabilities_lvl{}".format(level)
					),
					nodal_strategies[level],
					name="nodal_reach_probabilities_lvl{}".format(level)
				)
			return nodal_reach_probabilities

	def get_infoset_reach_probabilities(self, for_player=None):
		"""
		:param for_player: The player for which the reach probabilities are computed. These probabilities are usually
		 computed for the opponent when his strategies are cumulated. Therefore, `for_player` is set to `current_opponent`
		 by default.
		:return: The reach probabilities of information sets based on `current_infoset_strategies`.
		"""
		if for_player is None:
			for_player = self.domain.current_opponent
		nodal_reach_probabilities = self.get_nodal_reach_probabilities(for_player)
		with tf.variable_scope("infoset_reach_probabilities"):
			infoset_reach_probabilities = [None] * self.domain.levels
			with tf.variable_scope("level0"):
				infoset_reach_probabilities[0] = tf.identity(
					nodal_reach_probabilities[0],
					name="infoset_reach_probabilities_lvl0"
				)
			inner_nodal_reach_probabilities = self.domain.mask_out_values_in_terminal_nodes(
				nodal_reach_probabilities,
				name="nodal_reach_probabilities"
			)
			for level in range(1, self.domain.levels - 1):
				with tf.variable_scope("level{}".format(level)):
					scatter_nd_sum_indices = tf.expand_dims(
						self.domain.inner_node_to_infoset[level],
						axis=-1,
						name="expanded_node_to_infoset_lvl{}".format(level)
					)
					scatter_nd_sum_shape = self.domain.infoset_acting_players[level].shape
					infoset_reach_probabilities[level] = scatter_nd_sum(
						indices=scatter_nd_sum_indices,
						updates=inner_nodal_reach_probabilities[level],
						shape=scatter_nd_sum_shape,
						name="infoset_reach_probabilities_lvl{}".format(level)
					)
		return infoset_reach_probabilities

	def show_reach_probabilities(self, session):
		node_cf_strategies = self.get_node_cf_strategies()
		nodal_reach_probabilities = {}
		for player in [PLAYER1, PLAYER2, ALL_PLAYERS]:
			nodal_reach_probabilities[player] = self.get_nodal_reach_probabilities(for_player=player)
		infoset_reach_probabilities = self.get_infoset_reach_probabilities()
		for level in range(self.domain.levels):
			print("########## Level {} ##########".format(level))
			for player in [PLAYER1, PLAYER2, ALL_PLAYERS]:
				print_tensors(session, [nodal_reach_probabilities[player][level]])
			print("___________________________________\n")
			if level < self.domain.levels - 1:
				print_tensors(session, [
					self.domain.node_to_infoset[level],
					infoset_reach_probabilities[level],
					self.domain.current_infoset_strategies[level],
					node_cf_strategies[level],
				])

	def get_nodal_cf_values(self, for_player=None):  # TODO verify and write a unittest
		"""
		Compute counterfactual values of nodes by (tensor-)multiplying reach probabilities and expected values.

		:param for_player: The player for which the counterfactual values are computed. These values are usually
		 computed for the updating player. Therefore, `for_player` is set to `current_updating_player` by default.
		:return: The counterfactual values of nodes based on `current_infoset_strategies`.
		"""
		expected_values = self.get_expected_values(for_player=for_player)
		reach_probabilities = self.get_nodal_reach_probabilities(for_player=for_player)
		with tf.variable_scope("nodal_counterfactual_values"):
			return [
				tf.multiply(
					reach_probabilities[level],
					expected_values[level],
					name="nodal_cf_value_lvl{}".format(level)
				) for level in range(self.domain.levels)
			]

	def get_infoset_cf_values(self, for_player=None):  # TODO verify and write a unittest
		"""
		Compute infoset(-action) counterfactual values by summing relevant counterfactual values of nodes.

		:param for_player: The player for which the counterfactual values are computed. These values are usually
		 computed for the updating player. Therefore, `for_player` is set to `current_updating_player` by default.
		:return: The infoset(-action) counterfactual values based on `current_infoset_strategies`.
		"""
		if for_player is None:
			player_name = "current_player"
		else:
			player_name = "player{}".format(for_player)
		nodal_cf_values = self.get_nodal_cf_values(for_player=for_player)
		infoset_actions_cf_values, infoset_cf_values = [], []
		for level in range(self.domain.acting_depth):
			infoset_action_cf_value, infoset_cf_value = get_action_and_infoset_values(
				values_in_children=nodal_cf_values[level + 1],
				action_counts=self.domain.action_counts[level],
				parental_node_to_infoset=self.domain.inner_node_to_infoset[level],
				infoset_strategy=self.domain.current_infoset_strategies[level],
				name="cf_values_lvl{}_for_{}".format(level, player_name)
			)
			infoset_cf_values.append(infoset_cf_value)
			infoset_actions_cf_values.append(infoset_action_cf_value)
		return infoset_actions_cf_values, infoset_cf_values

	def get_infoset_mask_non_imaginary_children(self):  # TODO unittest
		with tf.variable_scope("infoset_mask_non_imaginary_children"):
			infoset_mask_non_imaginary_children = [None] * self.domain.acting_depth
			for level, infoset_action_count in enumerate(self.domain.infoset_action_counts):
				with tf.variable_scope("level{}".format(level)):
					infoset_mask_non_imaginary_children[level] = tf.sequence_mask(
						lengths=infoset_action_count,
						name="infoset_mask_non_imaginary_children_lvl{}".format(level)
					)
		return infoset_mask_non_imaginary_children

	def get_infoset_uniform_strategies(self):  # TODO unittest
		with tf.variable_scope("infoset_uniform_strategies"):
			infoset_mask_non_imaginary_children = self.get_infoset_mask_non_imaginary_children()
			infoset_uniform_strategies = [None] * self.domain.acting_depth
			for level, infoset_mask_of_1_level in enumerate(infoset_mask_non_imaginary_children):
				with tf.variable_scope("level{}".format(level)):
					infoset_mask_non_imaginary_children_float_dtype = tf.cast(
						infoset_mask_of_1_level,
						dtype=FLOAT_DTYPE,
					)
					# Note: An all-0's row cannot be normalized. This is caused when an infoset has only imaginary children. As of
					#       now, an all-0's row is kept without normalizing.
					count_of_actions = tf.reduce_sum(
						infoset_mask_non_imaginary_children_float_dtype,
						axis=-1,
						keepdims=True,
						name="count_of_actions_lvl{}".format(level),
					)
					infosets_with_no_actions = tf.squeeze(
						tf.equal(count_of_actions, 0.0),
						name="rows_summing_to_zero_lvl{}".format(level)
					)
					reciprocals = tf.where(
						condition=infosets_with_no_actions,
						x=infoset_mask_non_imaginary_children_float_dtype,
						y=infoset_mask_non_imaginary_children_float_dtype / count_of_actions,
						name="normalize_where_nonzero_sum_lvl{}".format(level),
					)
					infoset_uniform_strategies[level] = tf.where(
						condition=self.domain.infosets_of_non_chance_player[level],
						x=reciprocals,
						y=self.domain.current_infoset_strategies[level],
						name="infoset_uniform_strategies_lvl{}".format(level),
					)
		return infoset_uniform_strategies

	def get_regrets(self):  # TODO verify and write a unittest
		infoset_action_cf_values, infoset_cf_values = self.get_infoset_cf_values()
		infoset_mask_non_imaginary_children = self.get_infoset_mask_non_imaginary_children()
		with tf.variable_scope("regrets"):
			regrets = [None] * self.domain.acting_depth
			for level in range(self.domain.acting_depth):
				with tf.variable_scope("level{}".format(level)):
					regrets[level] = tf.where(
						condition=infoset_mask_non_imaginary_children[level],
						x=infoset_action_cf_values[level] - infoset_cf_values[level],
						y=tf.zeros_like(
							infoset_action_cf_values[level],
							name="zero_regrets_of_imaginary_children_lvl{}".format(level),
						),
						name="regrets_lvl{}".format(level),
					)
		return regrets

	def update_positive_cumulative_regrets(self, regrets=None):  # TODO verify and write a unittest
		if regrets is None:
			regrets = self.get_regrets()
		with tf.variable_scope("update_cumulative_regrets"):
			update_regrets_ops = [None] * self.domain.acting_depth
			for level in range(self.domain.acting_depth):
				with tf.variable_scope("level{}".format(level)):
					# TODO optimize by: pre-define `infosets_of_player1` and `infosets_of_player2` (in domain definitions) and
					#  switch
					infosets_of_updating_player = tf.reshape(
						tf.equal(self.domain.infoset_acting_players[level], self.domain.current_updating_player),
						shape=[self.domain.positive_cumulative_regrets[level].shape[0]],
						name="infosets_of_updating_player_lvl{}".format(level),
					)
					# TODO implement and use `masked_assign_add` here
					update_regrets_ops[level] = masked_assign(
						ref=self.domain.positive_cumulative_regrets[level],
						mask=infosets_of_updating_player,
						value=tf.maximum(
							tf.cast(0.0, dtype=FLOAT_DTYPE),
							self.domain.positive_cumulative_regrets[level] + regrets[level]
						),
						name="update_regrets_lvl{}".format(level)
					)
			return update_regrets_ops

	def get_strategy_matched_to_regrets(self):  # TODO unittest
		update_regrets = self.update_positive_cumulative_regrets()
		infoset_uniform_strategies = self.get_infoset_uniform_strategies()
		with tf.control_dependencies(update_regrets):
			with tf.variable_scope("strategies_matched_to_regrets"):
				strategies_matched_to_regrets = [None] * (self.domain.levels - 1)
				for level in range(self.domain.acting_depth):
					with tf.variable_scope("level{}".format(level)):
						sums_of_regrets = tf.reduce_sum(
							self.domain.positive_cumulative_regrets[level].read_value(),
							axis=-1,
							keepdims=True,
							name="sums_of_regrets_lvl{}".format(level)
						)
						normalized_regrets = tf.divide(
							self.domain.positive_cumulative_regrets[level].read_value(),
							sums_of_regrets,
							name="normalized_regrets_lvl{}".format(level)
						)
						zero_sum_rows = tf.squeeze(
							tf.equal(sums_of_regrets, 0),
							name="zero_sum_rows_lvl{}".format(level)
						)
						# Note: An all-0's row cannot be normalized. Thus, when PCRegrets sum to 0, a uniform strategy is used
						#  instead.
						# TODO verify uniform strategy is created (mix of both tf.where branches)
						strategies_matched_to_regrets[level] = tf.where(
							condition=zero_sum_rows,
							x=infoset_uniform_strategies[level],
							y=normalized_regrets,
							name="strategies_matched_to_regrets_lvl{}".format(level)
						)
				return strategies_matched_to_regrets

	def update_strategy_of_updating_player(self, acting_player=None):  # TODO unittest
		"""
		Update for the strategy for the given `acting_player`.

		Take into account the `self.trunk_depth`, i.e., the strategies at levels `0`, `1`, ... `trunk_depth - 1` are kept
		 intact (fixed) during the CFR iterations.

		Args:
			:param acting_player: A variable. An index of the player whose strategies are to be updated.

		Returns:
			A corresponding TensorFlow operation (from the computation graph).
		"""
		if acting_player is None:
			acting_player = self.domain.current_updating_player
		infoset_strategies_matched_to_regrets = self.get_strategy_matched_to_regrets()
		infoset_acting_players = self.domain.get_infoset_acting_players()
		ops_update_infoset_strategies = [None] * self.domain.acting_depth
		with tf.variable_scope("update_strategy_of_updating_player"):
			for level in range(self.trunk_depth, self.domain.acting_depth):
				with tf.variable_scope("level{}".format(level)):
					infosets_of_acting_player = tf.reshape(
						# `tf.reshape` to force "shape of 2D tensor" == [number of infosets, 1]
						tf.equal(infoset_acting_players[level], acting_player),
						shape=[self.domain.current_infoset_strategies[level].shape[0]],
						name="infosets_of_updating_player_lvl{}".format(level)
					)
					ops_update_infoset_strategies[level] = masked_assign(
						ref=self.domain.current_infoset_strategies[level],
						mask=infosets_of_acting_player,
						value=infoset_strategies_matched_to_regrets[level],
						name="op_update_infoset_strategies_lvl{}".format(level)
					)
			return ops_update_infoset_strategies[self.trunk_depth:]

	def get_weighted_averaging_factor(self, delay=None):  # see https://arxiv.org/pdf/1407.5042.pdf (Section 2)
		if delay is None:
			delay = self.domain.averaging_delay
		with tf.variable_scope("weighted_averaging_factor"):
			if delay is None:  # when `delay` is None, no weighted averaging is used
				return tf.constant(
					1.0,
					dtype=FLOAT_DTYPE,
					name="weighted_averaging_factor"
				)
			else:
				return tf.cast(
					tf.maximum(self.domain.cfr_step - delay, 0),
					dtype=FLOAT_DTYPE,
					name="weighted_averaging_factor",
				)

	def cumulate_strategy_of_opponent(self, opponent=None):  # TODO unittest
		if opponent is None:
			opponent = self.domain.current_opponent
		infoset_acting_players = self.domain.get_infoset_acting_players()
		infoset_reach_probabilities = self.get_infoset_reach_probabilities()
		with tf.variable_scope("cumulate_strategy_of_opponent"):
			cumulate_infoset_strategies_ops = [None] * self.domain.acting_depth
			for level in range(self.domain.acting_depth):
				with tf.variable_scope("level{}".format(level)):
					infosets_of_opponent = tf.reshape(  # `tf.reshape` to force "shape of 2D tensor" == [number of infosets, 1]
						tf.equal(infoset_acting_players[level], opponent),
						shape=[self.domain.current_infoset_strategies[level].shape[0]],
						name="infosets_of_opponent_lvl{}".format(level)
					)
					averaging_factor = self.get_weighted_averaging_factor()
					cumulate_infoset_strategies_ops[level] = masked_assign(
						# TODO implement and use `masked_assign_add` here
						ref=self.domain.cumulative_infoset_strategies[level],
						mask=infosets_of_opponent,
						value=self.domain.cumulative_infoset_strategies[level] + averaging_factor * expanded_multiply(
							expandable_tensor=infoset_reach_probabilities[level],
							expanded_tensor=self.domain.current_infoset_strategies[level],
						),
						name="op_cumulate_infoset_strategies_lvl{}".format(level)
					)
			return cumulate_infoset_strategies_ops

	def process_strategies(self, acting_player=None, opponent=None):
		if acting_player is None:
			acting_player = self.domain.current_updating_player
		if opponent is None:
			opponent = self.domain.current_opponent
		update_ops = self.update_strategy_of_updating_player(
			acting_player=acting_player
		)
		cumulate_ops = self.cumulate_strategy_of_opponent(opponent=opponent)
		return tf.tuple(update_ops + cumulate_ops, name="process_strategies")

	def set_average_infoset_strategies(self):
		# TODO Do not normalize over imaginary nodes. <- Do we need to solve this? Or is it already ok (cf. `bottomup-*.py`)
		with tf.variable_scope("average_strategies"):
			self.average_infoset_strategies = [None] * self.domain.acting_depth
			norm_of_strategies = [None] * self.domain.acting_depth
			infosets_with_nonzero_norm = [None] * self.domain.acting_depth
			for level in range(self.domain.acting_depth):
				with tf.variable_scope("level{}".format(level)):
					norm_of_strategies[level] = tf.reduce_sum(
						self.domain.cumulative_infoset_strategies[level],
						axis=-1,
						keepdims=True,
						name="norm_of_strategies_lvl{}".format(level),
					)
					infosets_with_nonzero_norm[level] = tf.squeeze(
						tf.not_equal(norm_of_strategies[level], 0.0),
						name="infosets_with_nonzero_norm_lvl{}".format(level)
					)
					self.average_infoset_strategies[level] = tf.where(
						condition=tf.logical_and(
							self.domain.infosets_of_non_chance_player[level],
							infosets_with_nonzero_norm[level],
							name="non_chance_infosets_with_nonzero_norm_lvl{}".format(level)
						),
						x=tf.cast(
							normalize(self.domain.cumulative_infoset_strategies[level]),
							dtype=FLOAT_DTYPE,
						),
						y=self.domain.current_infoset_strategies[level],
						name="average_infoset_strategies_lvl{}".format(level),
					)

	def do_cfr_step(self):
		ops_process_strategies = self.process_strategies()
		with tf.control_dependencies(ops_process_strategies):
			ops_swap_players = self.swap_players()
			op_inc_step = self.increment_cfr_step
		return tf.tuple(
			ops_process_strategies + ops_swap_players + [op_inc_step],
			name="cfr_step"
		)

	def assign_avg_strategies_to_current_strategies(self):  # TODO unittest
		"""
		Assign average strategies to current strategies.

		Returns:
			A corresponding TensorFlow operation (from the computation graph).
		"""
		with tf.variable_scope("assign_avg_strategies_to_current_strategies"):
			ops_assign_strategies = [None] * self.domain.acting_depth
			for level, avg_strategy in enumerate(self.average_infoset_strategies):
				with tf.variable_scope("level{}".format(level)):
					ops_assign_strategies[level] = masked_assign(
						ref=self.domain.current_infoset_strategies[level],
						mask=self.domain.infosets_of_non_chance_player[level],
						value=avg_strategy,
						name="assign_avg_strategies_to_current_strategies_lvl{}".format(level)
					)
		return ops_assign_strategies

	def combine_infoset_values_based_on_owners(self, tensor_of_player1, tensor_of_player2, level,
	                                           name="infoset_cf_values"):
		"""
		Combine `tensor_of_player1` and `tensor_of_player2` that correspond to some infoset-related values at level `level`.

		Use values of `tensor_of_player1` for `PLAYER1`'s infosets, values of `tensor_of_player2` for `PLAYER2`'s infosets,
		 `np.nan` otherwise.

		:param tensor_of_player1: A tensor of infoset values for `PLAYER1`.
		:param tensor_of_player2: A tensor of infoset values for `PLAYER2`.
		:param level: The tree level for which `tensor_of_player1` and `tensor_of_player2` are defined.
		:param name: A string used for naming tensors.

		Returns:
			A corresponding TensorFlow operation (from the computation graph).
		"""
		mask_of_acting_players = {}
		for player in [PLAYER1, PLAYER2]:
			mask_of_acting_players[player] = tf.equal(
				self.domain.infoset_acting_players[level],
				player,
				name="mask_of_acting_player{}_lvl{}".format(player, level)
			)
		default_values_at_chance_infosets = tf.fill(
			dims=tf.shape(tensor_of_player1),
			value=np.nan,
			name="{}_at_chance_infosets".format(name)
		)
		return tf.where(
			condition=mask_of_acting_players[PLAYER1],
			x=tensor_of_player1,
			y=tf.where(
				condition=mask_of_acting_players[PLAYER2],
				x=tensor_of_player2,
				y=default_values_at_chance_infosets
			),
			name="{}_lvl{}_based_on_owners".format(name, level)
		)

	def combine_inner_nodal_values_based_on_owners(self, tensor_of_player1, tensor_of_player2, level,
	                                               name="inner_nodal_values"):
		"""
		Combine `tensor_of_player1` and `tensor_of_player2` that correspond to some inner-nodal-related values at `level`.

		Use values of `tensor_of_player1` for `PLAYER1`'s nodes, values of `tensor_of_player2` for `PLAYER2`'s nodes,
		 `np.nan` in case of chance nodes (terminal nodes are considered to be excluded from `tensor_of_player*`).

		:param tensor_of_player1: A tensor of nodal values for `PLAYER1`.
		:param tensor_of_player2: A tensor of nodal values for `PLAYER2`.
		:param level: The tree level for which `tensor_of_player1` and `tensor_of_player2` are defined.
		:param name: A string used for naming tensors.

		Returns:
			A corresponding TensorFlow operation (from the computation graph).
		"""
		inner_nodal_mask_of_acting_players = {}
		for player in [PLAYER1, PLAYER2]:
			inner_nodal_mask_of_acting_players[player] = tf.equal(
				self.domain.inner_nodal_acting_players[level],
				player,
				name="nodal_mask_of_acting_player{}_lvl{}".format(player, level)
			)
		default_values_at_chance_nodes = tf.fill(
			dims=tf.shape(tensor_of_player1),
			value=np.nan,
			name="{}_at_chance_nodes".format(name)
		)
		return tf.where(
			condition=inner_nodal_mask_of_acting_players[PLAYER1],
			x=tensor_of_player1,
			y=tf.where(
				condition=inner_nodal_mask_of_acting_players[PLAYER2],
				x=tensor_of_player2,
				y=default_values_at_chance_nodes
			),
			name="{}_lvl{}_based_on_owners".format(name, level)
		)

	def get_infoset_ranges_at_trunk_depth(self):  # TODO unittest
		"""
		Get infoset reach probabilities at the bottom of the trunk (at `self.boundary_level`).

		Returns:
			A corresponding TensorFlow operation (from the computation graph).
		"""
		ranges = {}
		if self.trunk_depth > 0:
			ranges = {}
			for player in [PLAYER1, PLAYER2]:
				opponent = PLAYER2 if player == PLAYER1 else PLAYER1
				range_of_the_player = self.get_infoset_reach_probabilities(for_player=opponent)
				ranges[player] = tf.expand_dims(
					range_of_the_player[self.boundary_level],
					axis=-1
				)

			ranges["combined_players"] = self.combine_infoset_values_based_on_owners(
				tensor_of_player1=ranges[PLAYER1],
				tensor_of_player2=ranges[PLAYER2],
				level=self.boundary_level,
				name="ranges"
			)
		return ranges["combined_players"]

	def get_infoset_cfvs_at_trunk_depth(self):  # TODO unittest
		"""
		Get infoset counterfactual values at the bottom of the trunk.

		Returns:
			A corresponding TensorFlow operation (from the computation graph).
		"""
		if self.trunk_depth_infoset_cfvs is None and self.trunk_depth > 0:
			self.trunk_depth_infoset_cfvs = {}
			for player in [PLAYER1, PLAYER2]:
				_, infoset_cf_values = self.get_infoset_cf_values(for_player=player)
				self.trunk_depth_infoset_cfvs[player] = infoset_cf_values[self.boundary_level]

			self.trunk_depth_infoset_cfvs["combined_players"] = self.combine_infoset_values_based_on_owners(
				tensor_of_player1=self.trunk_depth_infoset_cfvs[PLAYER1],
				tensor_of_player2=self.trunk_depth_infoset_cfvs[PLAYER2],
				level=self.boundary_level
			)
		return self.trunk_depth_infoset_cfvs["combined_players"]

	def get_nodal_reaches_at_trunk_depth(self):  # TODO unittest
		"""
		Get inner nodal reach probabilities of all players at the bottom of the trunk (at `self.boundary_level`).

		Returns:
			A corresponding TensorFlow operation (from the computation graph).
		"""
		if self.trunk_depth > 0:
			inner_nodal_reaches_for_all_players = self.domain.mask_out_values_in_terminal_nodes(
				self.get_nodal_reach_probabilities(for_player=ALL_PLAYERS),
				name="nodal_reaches"
			)
			return inner_nodal_reaches_for_all_players[self.boundary_level]
		else:
			raise ValueError('Trunk depth {} has to be positive to get nodal reaches.'.format(self.trunk_depth))

	def get_nodal_expected_values_at_trunk_depth(self):  # TODO unittest
		"""
		Get nodal expected values at the bottom of the trunk (at `self.boundary_level`).

		Returns:
			A corresponding TensorFlow operation (from the computation graph).
		"""
		if self.trunk_depth_nodal_expected_values is not None \
			and self.trunk_depth_nodal_expected_values["combined_players"] is not None:
			return self.trunk_depth_nodal_expected_values["combined_players"]
		elif self.trunk_depth > 0:
			self.trunk_depth_nodal_expected_values = {}
			for player in [PLAYER1, PLAYER2]:
				expected_values = self.get_expected_values(for_player=player)
				inner_nodal_expected_values = self.domain.mask_out_values_in_terminal_nodes(
					expected_values,
					name="expected_values"
				)
				self.trunk_depth_nodal_expected_values[player] = inner_nodal_expected_values[self.boundary_level]

			self.trunk_depth_nodal_expected_values["combined_players"] = self.combine_inner_nodal_values_based_on_owners(
				tensor_of_player1=self.trunk_depth_nodal_expected_values[PLAYER1],
				tensor_of_player2=self.trunk_depth_nodal_expected_values[PLAYER2],
				level=self.boundary_level
			)
			return self.trunk_depth_nodal_expected_values["combined_players"]
		else:
			raise ValueError('Trunk depth {} has to be positive to get nodal values.'.format(self.trunk_depth))

	def get_trunk_info_to_store(self):
		if self.trunk_depth <= 0:
			return None

		trunk_depth_ranges = self.get_infoset_ranges_at_trunk_depth()
		trunk_depth_infoset_cfvs = self.get_infoset_cfvs_at_trunk_depth()
		count_of_infosets = tf.cast(
			tf.shape(trunk_depth_ranges)[0],
			dtype=FLOAT_DTYPE
		)
		trunk_depth_infoset_indices = tf.expand_dims(
			tf.range(
				count_of_infosets
			),
			axis=-1,
			name="infoset_indices_lvl{}".format(self.boundary_level)
		)
		data_id_column = self.data_id * tf.ones_like(trunk_depth_infoset_indices)
		concat_trunk_info_tensors = tf.concat(
			[
				data_id_column,
				trunk_depth_infoset_indices,
				trunk_depth_ranges,
				trunk_depth_infoset_cfvs
			],
			axis=-1,
			name="concat_trunk_info_tensors_lvl{}".format(self.boundary_level)
		)
		masked_out_trunk_info_tensors = tf.boolean_mask(
			concat_trunk_info_tensors,
			mask=self.domain.infosets_of_non_chance_player[self.boundary_level],
			name="masked_out_trunk_info_tensors_lvl{}".format(self.boundary_level)
		)
		return masked_out_trunk_info_tensors

	def get_trunk_info_of_nodes(self):
		if self.trunk_depth <= 0:
			return None

		inner_node_to_infoset = tf.expand_dims(
			tf.cast(
				self.domain.inner_node_to_infoset[self.boundary_level],
				dtype=FLOAT_DTYPE
			),
			axis=-1,
			name="node_to_infoset_lvl{}".format(self.boundary_level)
		)
		data_id_column = self.data_id * tf.ones_like(inner_node_to_infoset)
		nodal_enumerations = [
			tf.range(
				len(action_counts_in_a_level),
				dtype=FLOAT_DTYPE,
				name="nodal_enumeration_lvl{}".format(level)
			)
			for level, action_counts_in_a_level in enumerate(self.domain.action_counts)
		]
		inner_nodal_enumerations = self.domain.mask_out_values_in_terminal_nodes(
			nodal_enumerations,
			name="nodal_enumeration"
		)
		inner_nodal_indices = tf.expand_dims(
			inner_nodal_enumerations[self.boundary_level],
			axis=-1,
			name="inner_nodal_indices_lvl{}".format(self.boundary_level)
		)
		inner_nodal_reaches_for_all_players = tf.expand_dims(
			self.get_nodal_reaches_at_trunk_depth(),
			axis=-1,
			name="inner_nodal_reaches_for_all_players_lvl{}".format(self.boundary_level)
		)
		inner_nodal_expected_values = tf.expand_dims(
			self.get_nodal_expected_values_at_trunk_depth(),
			axis=-1,
			name="inner_nodal_expected_values_lvl{}".format(self.boundary_level)
		)

		concat_trunk_info_tensors = tf.concat(
			[
				data_id_column,
				inner_nodal_indices,
				inner_node_to_infoset,
				inner_nodal_reaches_for_all_players,
				inner_nodal_expected_values
			],
			axis=-1,
			name="concat_trunk_info_tensors_lvl{}".format(self.boundary_level)
		)
		return concat_trunk_info_tensors

	def set_up_feed_dictionary(self, method="by-domain", initial_strategy_values=None, seed=None):
		if method == "by-domain":
			# TODO: @janrudolf Fix here
			# if self.domain.initial_infoset_strategies has nans use uniform methods
			return "Initializing strategies via domain definitions...\n", {}  # default value of `initial_infoset_strategies`
		elif method == "custom":
			if initial_strategy_values is None:
				raise ValueError('No "initial_strategy_values" given.')
			if len(initial_strategy_values) != len(self.domain.initial_infoset_strategies):
				raise ValueError(
					'Mismatched "len(initial_strategy_values) == {}" and "len(initial_infoset_strategies) == {}".'.format(
						len(initial_strategy_values), len(self.domain.initial_infoset_strategies)
					)
				)
			# TODO @mathemage Fix problem:
			#  `custom` initializing does not work in `test.algorithms.tensorcfr_fixed_trunk_strategies`
			print("initial_strategy_values")
			print(initial_strategy_values)
			return "Initializing strategies to custom values defined by user...\n", {
				self.domain.initial_infoset_strategies[level]: initial_strategy_values[level]
				for level in range(self.domain.acting_depth)
			}
		elif method == "random":
			np_random_strategies = self.domain.generate_random_strategies(
				seed=seed,
				trunk_depth=self.trunk_depth,
			)
			return "Initializing strategies to random distributions...\n", {
				self.domain.initial_infoset_strategies[level]: np_random_strategies[level]
				for level in range(self.domain.acting_depth)
			}
		else:
			raise ValueError('Undefined method "{}" for set_up_feed_dictionary().'.format(method))

	def get_basename_from_cfr_parameters(self):
		basename_from_cfr_parameters = "{}-{}-{}".format(
			self.domain.domain_name,
			datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
			",".join(
				("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
				 for key, value in sorted(self.cfr_parameters.items()))).replace("/", "-")
		)
		return basename_from_cfr_parameters

	def set_log_directory(self):
		self.log_directory = "logs/" + self.get_basename_from_cfr_parameters()
		if not os.path.exists(self.log_directory):
			os.mkdir(self.log_directory)

	def store_final_average_strategies(self):
		print_tensors(self.session, self.average_infoset_strategies)
		print("Storing average strategies to '{}'...".format(self.log_directory))
		for level in range(len(self.average_infoset_strategies)):
			np.savetxt(
				'{}/average_infoset_strategies_lvl{}.csv'.format(self.log_directory, level),
				self.session.run(self.average_infoset_strategies[level]),
				delimiter=',',
			)

	def store_trunk_info_of_infosets(self, dataset_basename, dataset_directory=""):
		self.session.run(self.assign_avg_strategies_to_current_strategies())

		if not os.path.exists(dataset_directory):
			os.mkdir(dataset_directory)
		csv_filename = '{}/infoset_dataset_{}.csv'.format(dataset_directory, dataset_basename)
		print("[data_id #{} @{}] Generating dataset at the trunk-boundary and storing to '{}'...".format(
			self.data_id,
			get_current_timestamp(),
			csv_filename
		))

		csv_file = open(csv_filename, 'ab')  # binary mode for appending
		print_tensors(self.session, [self.get_trunk_info_to_store()]),
		np.savetxt(
			csv_file,
			self.session.run(self.get_trunk_info_to_store()),
			fmt="%7d,\t %7d,\t %.4f,\t %+.4f",
			header="data_id,\t IS_id,\t range,\t CFV" if self.data_id == 0 else "",
		)

	def store_trunk_info_of_nodes(self, dataset_basename, dataset_directory=""):
		self.session.run(self.assign_avg_strategies_to_current_strategies())

		if not os.path.exists(dataset_directory):
			os.mkdir(dataset_directory)
		csv_filename = '{}/nodal_dataset_{}.csv'.format(dataset_directory, dataset_basename)
		print("[data_id #{} @{}] Generating dataset at the trunk-boundary and storing to '{}'...".format(
			self.data_id,
			get_current_timestamp(),
			csv_filename
		))

		csv_file = open(csv_filename, 'ab')  # binary mode for appending
		trunk_info_of_nodes = self.get_trunk_info_of_nodes()
		print_tensors(self.session, [trunk_info_of_nodes]),
		np.savetxt(
			csv_file,
			self.session.run(trunk_info_of_nodes),
			fmt="%7d,\t %7d,\t %7d,\t %+.6f,\t %+.6f",
			header="data_id,\t nodal_index,\t node_to_infoset,\t nodal_reach,\t nodal_expected_value" if self.data_id == 0
			else "",
		)

	def cfr_strategies_after_fixed_trunk(self, total_steps=DEFAULT_TOTAL_STEPS, delay=DEFAULT_AVERAGING_DELAY,
	                                     storing_strategies=False, profiling=False):
		self.cfr_parameters = {
			"total_steps"    : total_steps,
			"averaging_delay": delay,
			"trunk_depth"    : self.trunk_depth,
		}
		self.set_log_directory()
		if profiling:
			self.log_directory += "-profiling"
		with tf.variable_scope("initialization"):
			setup_messages, feed_dictionary = self.set_up_feed_dictionary(method="by-domain")
			print(setup_messages)

		cfr_step_op = self.do_cfr_step()

		with tf.Session(
			# config=tf.ConfigProto(device_count={'GPU': 0})  # uncomment to run on CPU
		) as self.session:
			self.session.run(tf.global_variables_initializer(), feed_dict=feed_dictionary)
			with tf.summary.FileWriter(self.log_directory, tf.get_default_graph()) as writer:
				for step in range(total_steps):
					"""
					Profiler gives the Model report with total compute time and memory consumption.
					- Add CUDA libs to LD_LIBRARY_PATH: https://github.com/tensorflow/tensorflow/issues/8830
					- For `cmd` see:
					https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/python_api.md#time-and-memory
					"""
					if profiling:
						run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
						metadata = tf.RunMetadata()
						self.session.run(cfr_step_op, options=run_options, run_metadata=metadata)
						tf.profiler.profile(
							self.session.graph,
							run_meta=metadata,
							# cmd='op',
							cmd='scope',
							options=tf.profiler.ProfileOptionBuilder.time_and_memory()
						)
						writer.add_run_metadata(
							metadata,
							"step{}".format(step)
						)  # save metadata about time and memory for tensorboard
					else:
						self.session.run(cfr_step_op)

				if storing_strategies:
					self.store_final_average_strategies()

	def generate_dataset_at_trunk_depth(self, total_steps=DEFAULT_TOTAL_STEPS, delay=DEFAULT_AVERAGING_DELAY,
	                                    dataset_for_nodes=True, dataset_size=DEFAULT_DATASET_SIZE, dataset_directory="",
	                                    seed=None):
		self.cfr_parameters = {
			"total_steps"    : total_steps,
			"averaging_delay": delay,
			"trunk_depth"    : self.trunk_depth,
		}
		basename_from_cfr_parameters = self.get_basename_from_cfr_parameters()
		cfr_step_op = self.do_cfr_step()

		for self.data_id in range(dataset_size):
			# TODO place the for-loop inside the with-block below in order to keep a single session
			if seed is not None:
				seed_of_iteration = seed + self.data_id
			else:
				seed_of_iteration = None
			with tf.variable_scope("initialization"):
				setup_messages, feed_dictionary = self.set_up_feed_dictionary(
					method="random",
					seed=seed_of_iteration
				)
				print("[data_id #{} @{}] {}".format(self.data_id, get_current_timestamp(), setup_messages))

			with tf.Session(
				# config=tf.ConfigProto(device_count={'GPU': 0})  # uncomment to run on CPU
			) as self.session:
				self.session.run(tf.global_variables_initializer(), feed_dict=feed_dictionary)
				for _ in range(total_steps):
					# TODO replace for-loop with `tf.while_loop`: https://www.tensorflow.org/api_docs/python/tf/while_loop
					self.session.run(cfr_step_op)
				if self.trunk_depth > 0:
					if dataset_for_nodes:
						self.store_trunk_info_of_nodes(
							dataset_basename=basename_from_cfr_parameters,
							dataset_directory=dataset_directory
						)
					else:
						self.store_trunk_info_of_infosets(
							dataset_basename=basename_from_cfr_parameters,
							dataset_directory=dataset_directory
						)

	def randomize_strategies(self, seed):  # TODO unittest
		"""
		Reset infoset strategies to random ones.

		:param seed: The seed used to test random strategies

		Returns:
			A corresponding TensorFlow operation (from the computation graph).
		"""
		tf_uniform_strategies = self.get_infoset_uniform_strategies()
		with tf.variable_scope("randomize_strategies"):
			tf_random_strategies = self.domain.get_tf_random_strategies(
				seed=seed,
				trunk_depth=self.trunk_depth
			)
			ops_randomize_strategies = [
				tf.assign(
					current_strategies_per_level,
					value=tf_random_strategies[level] if level in range(self.trunk_depth)
					else tf_uniform_strategies[level]
				)
				for level, current_strategies_per_level in enumerate(self.domain.current_infoset_strategies)
			]
		return ops_randomize_strategies

	def generate_dataset_single_session(self, total_steps=DEFAULT_TOTAL_STEPS, delay=DEFAULT_AVERAGING_DELAY,
	                                    dataset_for_nodes=True, dataset_size=DEFAULT_DATASET_SIZE, dataset_directory="",
	                                    seed=None):
		self.cfr_parameters = {
			"total_steps"    : total_steps,
			"averaging_delay": delay,
			"trunk_depth"    : self.trunk_depth,
		}
		basename_from_cfr_parameters = self.get_basename_from_cfr_parameters()
		cfr_step_op = self.do_cfr_step()

		with tf.Session(
			# config=tf.ConfigProto(device_count={'GPU': 0})  # uncomment to run on CPU
		) as self.session:
			for self.data_id in range(dataset_size):
				self.session.run(tf.global_variables_initializer())
				print("[data_id #{} @{}]".format(self.data_id, get_current_timestamp()))
				if seed is not None:
					seed_of_iteration = seed + self.data_id
				else:
					seed_of_iteration = None

				self.session.run(
					self.randomize_strategies(seed=seed_of_iteration)
				)

				for _ in range(total_steps):
					# TODO replace for-loop with `tf.while_loop`: https://www.tensorflow.org/api_docs/python/tf/while_loop
					self.session.run(cfr_step_op)
				if self.trunk_depth > 0:
					if dataset_for_nodes:
						self.store_trunk_info_of_nodes(
							dataset_basename=basename_from_cfr_parameters,
							dataset_directory=dataset_directory
						)
					else:
						self.store_trunk_info_of_infosets(
							dataset_basename=basename_from_cfr_parameters,
							dataset_directory=dataset_directory
						)

	def generate_dataset_tf_while_loop(self, total_steps=DEFAULT_TOTAL_STEPS, delay=DEFAULT_AVERAGING_DELAY,
	                                   dataset_for_nodes=True, dataset_size=DEFAULT_DATASET_SIZE, dataset_directory="",
	                                   seed=None):
		self.cfr_parameters = {
			"total_steps"    : total_steps,
			"averaging_delay": delay,
			"trunk_depth"    : self.trunk_depth,
		}
		basename_from_cfr_parameters = self.get_basename_from_cfr_parameters()

		def condition(i, cfr_step_op):
			return tf.less(i, total_steps)

		def body(i, cfr_step_op):
			i = tf.add(i, 1)
			return [i, cfr_step_op]

		i = tf.constant(0)
		all_cfr_steps = tf.while_loop(
			cond=condition,
			body=body,
			loop_vars=[i, self.do_cfr_step()],
			parallel_iterations=1,
			back_prop=False,
		)

		with tf.Session(
			# config=tf.ConfigProto(device_count={'GPU': 0})  # uncomment to run on CPU
		) as self.session:
			for self.data_id in range(dataset_size):
				self.session.run(tf.global_variables_initializer())
				print("[data_id #{} @{}]".format(self.data_id, get_current_timestamp()))
				if seed is not None:
					seed_of_iteration = seed + self.data_id
				else:
					seed_of_iteration = None

				self.session.run(
					self.randomize_strategies(seed=seed_of_iteration)
				)
				self.session.run(all_cfr_steps)

				if self.trunk_depth > 0:
					if dataset_for_nodes:
						self.store_trunk_info_of_nodes(
							dataset_basename=basename_from_cfr_parameters,
							dataset_directory=dataset_directory
						)
					else:
						self.store_trunk_info_of_infosets(
							dataset_basename=basename_from_cfr_parameters,
							dataset_directory=dataset_directory
						)


if __name__ == '__main__':
	# domain_ = get_domain_by_name("flattened_hunger_games")
	# domain_ = get_domain_by_name("flattened_hunger_games_2")
	# domain_ = get_domain_by_name("flattened_domain01_via_gambit")
	# domain_ = get_domain_by_name("II-GS2_gambit_flattened")
	domain_ = get_domain_by_name("II-GS3_gambit_flattened")
	# domain_ = get_domain_by_name("IIGS5_gambit_flattened")
	# domain_ = get_domain_by_name("IIGS6_gambit_flattened")

	tensorcfr_instance = TensorCFRFixedTrunkStrategies(
		domain_,
		trunk_depth=4
	)
	tensorcfr_instance.cfr_strategies_after_fixed_trunk(
		# total_steps=10,
		storing_strategies=True,
		# profiling=True,
		# delay=0
	)
