#!/usr/bin/env python3
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from src.commons.constants import PLAYER1, PLAYER2, TERMINAL_NODE, IMAGINARY_NODE, DEFAULT_TOTAL_STEPS, FLOAT_DTYPE, \
	DEFAULT_AVERAGING_DELAY, INT_DTYPE
from src.domains.FlattenedDomain import FlattenedDomain
from src.domains.available_domains import get_domain_by_name
from src.utils.cfr_utils import distribute_strategies_to_nodes, flatten_strategies_via_action_counts, \
	get_action_and_infoset_values
from src.utils.tensor_utils import print_tensors, expanded_multiply, scatter_nd_sum, masked_assign, normalize


class TensorCFRFlattenedDomains:
	def __init__(self, domain: FlattenedDomain):
		self.domain = domain
		with tf.variable_scope("increment_step"):
			self.increment_cfr_step = tf.assign_add(
					ref=self.domain.cfr_step,
					value=1,
					name="increment_cfr_step"
			)
		self.summary_writer = None

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
						value=TensorCFRFlattenedDomains.get_the_other_player_of(self.domain.current_updating_player),
						name="assign_new_updating_player",
				)
			with tf.variable_scope("new_opponent"):
				assign_opponent = tf.assign(
						ref=self.domain.current_opponent,
						value=TensorCFRFlattenedDomains.get_the_other_player_of(self.domain.current_opponent),
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
				distribute_strategies_to_nodes(
						self.domain.current_infoset_strategies[level],
						self.domain.node_to_infoset[level],
						name="node_strategies_lvl{}".format(level)
				) for level in range(self.domain.acting_depth)
			]
			flattened_node_strategies = flatten_strategies_via_action_counts(node_strategies, self.domain.action_counts)
			return flattened_node_strategies

	def get_node_cf_strategies(self, updating_player=None):
		if updating_player is None:
			updating_player = self.domain.current_updating_player
		with tf.variable_scope("node_cf_strategies"):
			# TODO generate node_cf_strategies_* with tf.where on node_strategies
			node_cf_strategies = [
				distribute_strategies_to_nodes(
						self.domain.current_infoset_strategies[level],
						self.domain.node_to_infoset[level],
						updating_player=updating_player,
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

	def get_expected_values(self):
		node_strategies = self.get_node_strategies()
		with tf.variable_scope("expected_values"):
			expected_values = [None] * self.domain.levels
			with tf.variable_scope("level{}".format(self.domain.levels - 1)):
				expected_values[self.domain.levels - 1] = tf.multiply(
						self.domain.signum_of_current_player,
						self.domain.utilities[self.domain.levels - 1],
						name="expected_values_lvl{}".format(self.domain.levels - 1),
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
							shape=self.domain.node_types[level].shape,
							name="extended_weighted_sum_lvl{}".format(level)
					)
					expected_values[level] = tf.where(
							condition=tf.equal(self.domain.node_types[level], TERMINAL_NODE),
							x=self.domain.signum_of_current_player * tf.reshape(
									self.domain.utilities[level],
									shape=[self.domain.utilities[level].shape[-1]],
							),
							y=extended_weighted_sum,
							name="expected_values_lvl{}".format(level)
					)
		return expected_values

	def show_expected_values(self, session):
		self.domain.print_misc_variables(session=session)
		node_strategies = self.get_node_strategies()
		expected_values = self.get_expected_values()
		for level in reversed(range(self.domain.levels)):
			print("########## Level {} ##########".format(level))
			if level < len(node_strategies):
				print_tensors(session, [node_strategies[level]])
			print_tensors(session, [
				tf.multiply(
						self.domain.signum_of_current_player,
						self.domain.utilities[level],
						name="signum_utilities_lvl{}".format(level)
				),
				expected_values[level]
			])

	def get_nodal_reach_probabilities(self, for_player=None):
		"""
		Compute reach probabilities of nodes using the top-down tree traversal.

		:param for_player: The player for which the reach probabilities are computed. These probabilities are usually
		 computed for the updating player when counterfactual values are computed. Therefore, `for_player` is set to
			`current_updating_player` by default.
		:return: The reach probabilities of nodes based on `current_infoset_strategies`.
		"""
		if for_player is None:
			for_player = self.domain.current_updating_player
		node_cf_strategies = self.get_node_cf_strategies(updating_player=for_player)
		with tf.variable_scope("nodal_reach_probabilities"):
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
						node_cf_strategies[level],
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
				infoset_reach_probabilities[0] = tf.identity(nodal_reach_probabilities[0],
				                                             name="infoset_reach_probabilities_lvl0")
			for level in range(1, self.domain.levels - 1):
				with tf.variable_scope("level{}".format(level)):
					scatter_nd_sum_indices = tf.expand_dims(
							self.domain.node_to_infoset[level],
							axis=-1,
							name="expanded_node_to_infoset_lvl{}".format(level))
					scatter_nd_sum_updates = nodal_reach_probabilities[level]
					scatter_nd_sum_shape = self.domain.infoset_acting_players[level].shape
					infoset_reach_probabilities[level] = scatter_nd_sum(
							indices=scatter_nd_sum_indices,
							updates=scatter_nd_sum_updates,
							shape=scatter_nd_sum_shape,
							name="infoset_reach_probabilities_lvl{}".format(level)
					)
		return infoset_reach_probabilities

	def show_reach_probabilities(self, session):
		node_cf_strategies = self.get_node_cf_strategies()
		nodal_reach_probabilities = self.get_nodal_reach_probabilities()
		# infoset_reach_probabilities = self.get_infoset_reach_probabilities()
		for level in range(self.domain.levels):
			print("########## Level {} ##########".format(level))
			print_tensors(session, [nodal_reach_probabilities[level]])
			if level < self.domain.levels - 1:
				print_tensors(session, [
					self.domain.node_to_infoset[level],
					# infoset_reach_probabilities[level],
					self.domain.current_infoset_strategies[level],
					node_cf_strategies[level],
				])

	def get_nodal_cf_values(self):  # TODO verify and write a unittest
		expected_values = self.get_expected_values()
		reach_probabilities = self.get_nodal_reach_probabilities()
		with tf.variable_scope("nodal_counterfactual_values"):
			return [
				tf.multiply(
						reach_probabilities[level],
						expected_values[level],
						name="nodal_cf_value_lvl{}".format(level)
				) for level in range(self.domain.levels)
			]

	def get_infoset_action_cf_values(self):  # TODO verify and write a unittest
		node_cf_values = self.get_nodal_cf_values()
		with tf.variable_scope("infoset_action_cf_values"):
			cf_values_infoset_actions = [None] * (self.domain.levels - 1)
			cf_values_infoset_actions[0] = tf.expand_dims(
					node_cf_values[1],
					axis=0,
					name="infoset_action_cf_values_lvl0"
			)
			for level in range(1, self.domain.levels - 1):  # TODO replace for-loop with parallel_map on TensorArray?
				cf_values_infoset_actions[level] = scatter_nd_sum(
						indices=tf.expand_dims(self.domain.node_to_infoset[level], axis=-1),
						updates=node_cf_values[level + 1],
						shape=self.domain.current_infoset_strategies[level].shape,
						name="infoset_action_cf_values_lvl{}".format(level),
				)
			return cf_values_infoset_actions

	def get_infoset_cf_values(self):  # TODO verify and write a unittest
		nodal_cf_values = self.get_nodal_cf_values()
		infoset_actions_cf_values, infoset_cf_values = [], []
		for level in range(self.domain.acting_depth):
			infoset_action_cf_value, infoset_cf_value = get_action_and_infoset_values(
					values_in_children=nodal_cf_values[level + 1],
					action_counts=self.domain.action_counts[level],
					parental_node_to_infoset=self.domain.node_to_infoset[level],
					infoset_strategy=self.domain.current_infoset_strategies[level],
					name="cf_values_lvl{}".format(level)
			)
			infoset_cf_values.append(infoset_cf_value)
			infoset_actions_cf_values.append(infoset_action_cf_value)
		return infoset_actions_cf_values, infoset_cf_values

	# TODO update implementation here
	def get_infoset_children_mask_of_imaginary_actions(self):  # TODO unittest
		with tf.variable_scope("infoset_children_types"):
			infoset_children_types = [None] * (self.domain.levels - 1)
			for level in range(self.domain.levels - 1):
				with tf.variable_scope("level{}".format(level)):
					if level == 0:
						infoset_children_types[0] = tf.expand_dims(
								self.domain.node_types[1],
								axis=0,
								name="infoset_children_types_lvl0"
						)
					else:
						infoset_children_types[level] = tf.scatter_nd_update(
								ref=tf.Variable(
										tf.zeros_like(
												self.domain.current_infoset_strategies[level],
												dtype=self.domain.node_types[level + 1].dtype
										)
								),
								indices=tf.expand_dims(self.domain.node_to_infoset[level], axis=-1),
								updates=self.domain.node_types[level + 1],
								name="infoset_children_types_lvl{}".format(level)
						)
			return infoset_children_types

	def get_infoset_uniform_strategies(self):  # TODO unittest
		with tf.variable_scope("infoset_uniform_strategies"):
			infoset_children_types = self.get_infoset_children_mask_of_imaginary_actions()
			infoset_uniform_strategies = [None] * (self.domain.levels - 1)
			for level in range(self.domain.acting_depth):
				with tf.variable_scope("level{}".format(level)):
					infoset_uniform_strategies[level] = tf.cast(
							tf.not_equal(infoset_children_types[level], IMAGINARY_NODE),
							dtype=FLOAT_DTYPE,
					)
					# Note: An all-0's row cannot be normalized. This is caused when an infoset has only imaginary children. As of
					#       now, an all-0's row is kept without normalizing.
					count_of_actions = tf.reduce_sum(
							infoset_uniform_strategies[level],
							axis=-1,
							keepdims=True,
							name="count_of_actions_lvl{}".format(level),
					)
					infosets_with_no_actions = tf.squeeze(
							tf.equal(count_of_actions, 0.0),
							name="rows_summing_to_zero_lvl{}".format(level)
					)
					infoset_uniform_strategies[level] = tf.where(
							condition=infosets_with_no_actions,
							x=infoset_uniform_strategies[level],
							y=tf.divide(
									infoset_uniform_strategies[level],
									count_of_actions,
							),
							name="normalize_where_nonzero_sum_lvl{}".format(level),
					)
					infoset_uniform_strategies[level] = tf.where(
							condition=self.domain.infosets_of_non_chance_player[level],
							x=infoset_uniform_strategies[level],
							y=self.domain.current_infoset_strategies[level],
							name="infoset_uniform_strategies_lvl{}".format(level),
					)
		return infoset_uniform_strategies

	def get_regrets(self):  # TODO verify and write a unittest
		infoset_action_cf_values, infoset_cf_values = self.get_infoset_cf_values()
		infoset_children_types = self.get_infoset_children_mask_of_imaginary_actions()
		with tf.variable_scope("regrets"):
			regrets = [None] * self.domain.acting_depth
			for level in range(self.domain.acting_depth):
				with tf.variable_scope("level{}".format(level)):
					regrets[level] = tf.where(
							condition=tf.equal(
									infoset_children_types[level],
									IMAGINARY_NODE,
									name="non_imaginary_children_lvl{}".format(level)
							),
							x=tf.zeros_like(
									infoset_action_cf_values[level],
									name="zero_regrets_of_imaginary_children_lvl{}".format(level),
							),
							y=infoset_action_cf_values[level] - infoset_cf_values[level],
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
		if acting_player is None:
			acting_player = self.domain.current_updating_player
		infoset_strategies_matched_to_regrets = self.get_strategy_matched_to_regrets()
		infoset_acting_players = self.domain.get_infoset_acting_players()
		ops_update_infoset_strategies = [None] * self.domain.acting_depth
		with tf.variable_scope("update_strategy_of_updating_player"):
			for level in range(self.domain.acting_depth):
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
			return ops_update_infoset_strategies

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
		update_ops = self.update_strategy_of_updating_player(acting_player=acting_player)
		cumulate_ops = self.cumulate_strategy_of_opponent(opponent=opponent)
		return tf.tuple(update_ops + cumulate_ops, name="process_strategies")

	def get_average_infoset_strategies(self):
		# TODO Do not normalize over imaginary nodes. <- Do we need to solve this? Or is it already ok (cf. `bottomup-*.py`)
		with tf.variable_scope("average_strategies"):
			average_infoset_strategies = [None] * self.domain.acting_depth
			norm_of_strategies = [None] * self.domain.acting_depth
			infosets_with_nonzero_norm = [None] * self.domain.acting_depth
			for level in range(self.domain.acting_depth):
				# TODO add variable scope `level{}`
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
					average_infoset_strategies[level] = tf.where(
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
		return average_infoset_strategies

	def do_cfr_step(self):
		ops_process_strategies = self.process_strategies()
		with tf.control_dependencies(ops_process_strategies):
			ops_swap_players = self.swap_players()
			op_inc_step = self.increment_cfr_step
		return tf.tuple(
				ops_process_strategies + ops_swap_players + [op_inc_step],
				name="cfr_step"
		)


def set_up_feed_dictionary(tensorcfr_instance, method="by-domain", initial_strategy_values=None):
	if method == "by-domain":
		# TODO: @janrudolf Fix here
		# if tensorcfr_instance.domain.initial_infoset_strategies has nans use uniform methods
		return "Initializing strategies via domain definitions...\n", {}  # default value of `initial_infoset_strategies`
	elif method == "uniform":
		with tf.variable_scope("initialize_strategies"):
			uniform_strategies_tensors = tensorcfr_instance.get_infoset_uniform_strategies()
			with tf.Session() as temp_sess:
				temp_sess.run(tf.global_variables_initializer())
				uniform_strategy_arrays = temp_sess.run(uniform_strategies_tensors)
			return "Initializing to uniform strategies...\n", {
				tensorcfr_instance.domain.initial_infoset_strategies[level]: uniform_strategy_arrays[level]
				for level in range(tensorcfr_instance.domain.acting_depth)
			}
	elif method == "custom":
		if initial_strategy_values is None:
			raise ValueError('No "initial_strategy_values" given.')
		if len(initial_strategy_values) != len(tensorcfr_instance.domain.initial_infoset_strategies):
			raise ValueError(
					'Mismatched "len(initial_strategy_values) == {}" and "len(initial_infoset_strategies) == {}".'.format(
							len(initial_strategy_values), len(tensorcfr_instance.domain.initial_infoset_strategies)
					)
			)
		return "Initializing strategies to custom values defined by user...\n", {
			tensorcfr_instance.domain.initial_infoset_strategies[level]: initial_strategy_values[level]
			for level in range(tensorcfr_instance.domain.acting_depth)
		}
	else:
		raise ValueError('Undefined method "{}" for set_up_feed_dictionary().'.format(method))


def get_log_dir_path(tensorcfr_instance, hyperparameters):
	log_dir_path = "logs/{}-{}-{}".format(
			tensorcfr_instance.domain.domain_name,
			datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
			",".join(
					("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
					 for key, value in sorted(hyperparameters.items()))).replace("/", "-")
	)
	if not os.path.exists("logs"):
		os.mkdir("logs")
	return log_dir_path


def set_up_cfr(tensorcfr_instance):
	# TODO extract these lines to a UnitTest
	# setup_messages, feed_dictionary = set_up_feed_dictionary(tensorcfr_instance)
	# setup_messages, feed_dictionary = set_up_feed_dictionary(tensorcfr_instance, method="by-domain")
	setup_messages, feed_dictionary = set_up_feed_dictionary(tensorcfr_instance, method="uniform")
	# setup_messages, feed_dictionary = set_up_feed_dictionary(tensorcfr_instance, method="custom")
	# #  should raise ValueError
	# setup_messages, feed_dictionary = set_up_feed_dictionary(tensorcfr_instance,
	# 		method="custom",
	# 		initial_strategy_values=[
	# 			[[1.0, 0.0]],
	# 		],
	# )  # should raise ValueError
	# setup_messages, feed_dictionary = set_up_feed_dictionary(tensorcfr_instance,
	# 		method="custom",
	# 		initial_strategy_values=[   # on domain `matching_pennies`
	# 			[[1.0, 0.0]],
	# 			[[1.0, 0.0]],
	# 		]
	# )
	# setup_messages, feed_dictionary = set_up_feed_dictionary(tensorcfr_instance, method="invalid")
	# #  should raise ValueError
	return feed_dictionary, setup_messages


def log_before_all_steps(tensorcfr_instance, session, setup_messages, total_steps, averaging_delay):
	print("TensorCFRFlattenedDomains\n")
	print(setup_messages)
	print_tensors(session, tensorcfr_instance.domain.current_infoset_strategies)
	print("Running {} CFR+ iterations, averaging_delay == {}...\n".format(total_steps, averaging_delay))


def log_before_every_step(tensorcfr_instance, session, infoset_cf_values, infoset_action_cf_values,
                          nodal_cf_values, expected_values, reach_probabilities, regrets):
	print("########## CFR+ step #{} ##########".format(tensorcfr_instance.domain.cfr_step.eval()))
	print_tensors(session, reach_probabilities)
	print("___________________________________\n")
	print_tensors(session, expected_values)
	print("___________________________________\n")
	print_tensors(session, nodal_cf_values)
	print("___________________________________\n")
	print_tensors(session, infoset_action_cf_values)
	print("___________________________________\n")
	print_tensors(session, infoset_cf_values)
	print("___________________________________\n")
	print_tensors(session, regrets)
	print("___________________________________\n")
	print_tensors(session, infoset_action_cf_values)
	print("___________________________________\n")
	print_tensors(session, infoset_cf_values)
	print("___________________________________\n")
	print_tensors(session, regrets)
	print("___________________________________\n")
	print_tensors(session, tensorcfr_instance.domain.positive_cumulative_regrets)
	print("___________________________________\n")
	print_tensors(session, regrets)
	print("___________________________________\n")


def log_after_every_step(tensorcfr_instance, session, strategies_matched_to_regrets):
	print_tensors(session, tensorcfr_instance.domain.positive_cumulative_regrets)
	print("___________________________________\n")
	print_tensors(session, strategies_matched_to_regrets)
	print("___________________________________\n")
	print_tensors(session, tensorcfr_instance.domain.current_infoset_strategies)


def log_after_all_steps(tensorcfr_instance, session, average_infoset_strategies, log_dir_path):
	print("###################################\n")
	print_tensors(session, tensorcfr_instance.domain.cumulative_infoset_strategies)
	print("___________________________________\n")
	print_tensors(session, average_infoset_strategies)

	print("Storing average strategies to '{}'...".format(log_dir_path))

	for level in range(len(average_infoset_strategies)):
		np.savetxt(
				'{}/average_infoset_strategies_level_{}.csv'.format(log_dir_path, level),
				session.run(average_infoset_strategies[level]),
				delimiter=',',
		)


def run_cfr(tensorcfr_instance: TensorCFRFlattenedDomains, total_steps=DEFAULT_TOTAL_STEPS, quiet=False,
            delay=DEFAULT_AVERAGING_DELAY, profiling=False):
	with tf.variable_scope("initialization"):
		feed_dictionary, setup_messages = set_up_cfr(tensorcfr_instance)
		assign_averaging_delay_op = tf.assign(
				ref=tensorcfr_instance.domain.averaging_delay,
				value=delay,
				name="assign_averaging_delay"
		)
	cfr_step_op = tensorcfr_instance.do_cfr_step()

	# tensors to log if quiet is False
	reach_probabilities = tensorcfr_instance.get_nodal_reach_probabilities() if not quiet else None
	expected_values = tensorcfr_instance.get_expected_values() if not quiet else None
	nodal_cf_values = tensorcfr_instance.get_nodal_cf_values() if not quiet else None
	infoset_cf_values, infoset_action_cf_values = tensorcfr_instance.get_infoset_cf_values() if not quiet \
		else (None, None)
	regrets = tensorcfr_instance.get_regrets() if not quiet else None
	strategies_matched_to_regrets = tensorcfr_instance.get_strategy_matched_to_regrets() if not quiet else None
	average_infoset_strategies = tensorcfr_instance.get_average_infoset_strategies()

	with tf.Session() as session:
		session.run(tf.global_variables_initializer(), feed_dict=feed_dictionary)
		hyperparameters = {
			"total_steps"    : total_steps,
			"averaging_delay": delay,
		}
		log_dir_path = get_log_dir_path(tensorcfr_instance, hyperparameters)
		if profiling:
			log_dir_path += "-profiling"

		assigned_averaging_delay = session.run(assign_averaging_delay_op)
		if quiet is False:
			log_before_all_steps(tensorcfr_instance, session, setup_messages, total_steps, assigned_averaging_delay)

		with tf.summary.FileWriter(log_dir_path, tf.get_default_graph()) as writer:
			for i in range(total_steps):
				if quiet is False:
					log_before_every_step(tensorcfr_instance, session, infoset_cf_values, infoset_action_cf_values,
					                      nodal_cf_values, expected_values, reach_probabilities, regrets)

				"""
				Profiler gives the Model report with total compute time and memory consumption.
				- Add CUDA libs to LD_LIBRARY_PATH: https://github.com/tensorflow/tensorflow/issues/8830
				- For `cmd` see:
				https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/python_api.md#time-and-memory
				"""
				if profiling:
					run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
					metadata = tf.RunMetadata()
					session.run(cfr_step_op, options=run_options, run_metadata=metadata)
					tf.profiler.profile(
							session.graph,
							run_meta=metadata,
							# cmd='op',
							cmd='scope',
							options=tf.profiler.ProfileOptionBuilder.time_and_memory()
					)
					writer.add_run_metadata(metadata, "step{}".format(i))  # save metadata about time and memory for tensorboard
				else:
					session.run(cfr_step_op)

				if quiet is False:
					log_after_every_step(tensorcfr_instance, session, strategies_matched_to_regrets)
			log_after_all_steps(tensorcfr_instance, session, average_infoset_strategies, log_dir_path)


if __name__ == '__main__':
	# domain = get_domain_by_name("domain01")
	# domain = get_domain_by_name("matching_pennies")
	# domain = get_domain_by_name("invalid domain name test")
	domain = get_domain_by_name("flattened_hunger_games")
	tensorcfr = TensorCFRFlattenedDomains(domain)

	# infoset_action_cf_values_, infoset_cf_values_ = tensorcfr.get_infoset_cf_values()
	# alternating_cf_values = [
	# 	value
	# 	for pair_of_values in zip(infoset_action_cf_values_, infoset_cf_values_)
	# 	for value in pair_of_values
	# ]
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# tensorcfr.domain.print_domain(sess)
		# tensorcfr.show_strategies(sess)
		# tensorcfr.show_expected_values(sess)
		# tensorcfr.show_reach_probabilities(sess)
		# sess.run(tensorcfr.swap_players())
		# tensorcfr.show_reach_probabilities(sess)
		# print_tensors(sess, tensorcfr.get_nodal_cf_values())
		# print_tensors(sess, alternating_cf_values)
		# sess.run(tensorcfr.swap_players())
		# print_tensors(sess, alternating_cf_values)
		print_tensors(sess, tensorcfr.get_regrets())

	# run_cfr(
	# 		# total_steps=10,
	# 		tensorcfr_instance=tensorcfr,
	# 		quiet=True,
	# 		# profiling=True,
	# 		# delay=0
	# )
