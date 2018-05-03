#!/usr/bin/env python3

import tensorflow as tf

from src.commons.constants import PLAYER1, PLAYER2, TERMINAL_NODE, IMAGINARY_NODE
from src.domains.Domain import Domain
from src.utils.distribute_strategies_to_nodes import distribute_strategies_to_nodes
from src.utils.tensor_utils import print_tensors, expanded_multiply, scatter_nd_sum, masked_assign, normalize


class TensorCFR:
	def __init__(self, domain: Domain):
		self.domain = domain
		with tf.variable_scope("increment_step"):
			self.increment_cfr_step = tf.assign_add(
					ref=self.domain.cfr_step,
					value=1,
					name="increment_cfr_step"
			)

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

	def get_expected_values(self):
		node_strategies = self.get_node_strategies()
		with tf.variable_scope("expected_values"):
			expected_values = [None] * self.domain.levels
			expected_values[self.domain.levels - 1] = tf.multiply(
					self.domain.signum_of_current_player,
					self.domain.utilities[self.domain.levels - 1],
					name="expected_values_lvl{}".format(self.domain.levels - 1)
			)
			for level in reversed(range(self.domain.levels - 1)):
				weighted_sum_of_values = tf.reduce_sum(
						input_tensor=node_strategies[level] * expected_values[level + 1],
						axis=-1,
						name="weighted_sum_of_values_lvl{}".format(level))
				expected_values[level] = tf.where(
						condition=tf.equal(self.domain.node_types[level], TERMINAL_NODE),
						x=self.domain.signum_of_current_player * self.domain.utilities[level],
						y=weighted_sum_of_values,
						name="expected_values_lvl{}".format(level))
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
		:param for_player: The player for which the reach probabilities are computed. These probabilities are usually computed
		 for the updating player when counterfactual values are computed. Therefore, `for_player` is set to
			`current_updating_player` by default.
		:return: The reach probabilities of nodes based on `current_infoset_strategies`.
		"""
		if for_player is None:
			for_player = self.domain.current_updating_player
		# TODO continue here
		node_cf_strategies = self.get_node_cf_strategies(updating_player=for_player)
		with tf.variable_scope("nodal_reach_probabilities"):
			nodal_reach_probabilities = [None] * self.domain.levels
			nodal_reach_probabilities[0] = self.domain.reach_probability_of_root_node
			for level in range(1, self.domain.levels):
				nodal_reach_probabilities[level] = expanded_multiply(
						expandable_tensor=nodal_reach_probabilities[level - 1],
						expanded_tensor=node_cf_strategies[level - 1],
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
		infoset_reach_probabilities = self.get_infoset_reach_probabilities()
		for level in range(self.domain.levels):
			print("########## Level {} ##########".format(level))
			print_tensors(session, [nodal_reach_probabilities[level]])
			if level < self.domain.levels - 1:
				print_tensors(session, [
					self.domain.node_to_infoset[level],
					infoset_reach_probabilities[level],
					self.domain.current_infoset_strategies[level],
					node_cf_strategies[level],
				])

	def get_nodal_cf_values(self):  # TODO verify and write a unittest
		expected_values = self.get_expected_values()
		reach_probabilities = self.get_nodal_reach_probabilities()
		with tf.variable_scope("nodal_counterfactual_values"):
			return [tf.multiply(reach_probabilities[level], expected_values[level],
			                    name="nodal_cf_value_lvl{}".format(level)) for level in range(self.domain.levels)]

	def get_infoset_cf_values_per_actions(self):  # TODO verify and write a unittest
		node_cf_values = self.get_nodal_cf_values()
		with tf.variable_scope("infoset_cf_values_per_actions"):
			cf_values_infoset_actions = [None] * (self.domain.levels - 1)
			cf_values_infoset_actions[0] = tf.expand_dims(
					node_cf_values[1],
					axis=0,
					name="infoset_cf_values_per_actions_lvl0"
			)
			for level in range(1, self.domain.levels - 1):  # TODO replace for-loop with parallel_map on TensorArray?
				cf_values_infoset_actions[level] = scatter_nd_sum(
						indices=tf.expand_dims(self.domain.node_to_infoset[level], axis=-1),
						updates=node_cf_values[level + 1],
						shape=self.domain.current_infoset_strategies[level].shape,
						name="infoset_cf_values_per_actions_lvl{}".format(level),
				)
			return cf_values_infoset_actions

	def get_infoset_cf_values(self):  # TODO verify and write a unittest
		infoset_cf_values_per_actions = self.get_infoset_cf_values_per_actions()
		with tf.variable_scope("infoset_cf_values"):
			infoset_cf_values = [
				tf.reduce_sum(
						self.domain.current_infoset_strategies[level] * infoset_cf_values_per_actions[level],
						axis=-1,
						keepdims=True,
						name="infoset_cf_values_lvl{}".format(level),
						) for level in range(self.domain.levels - 1)
			]
		return infoset_cf_values, infoset_cf_values_per_actions

	def get_infoset_children_types(self):  # TODO unittest
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
			infoset_children_types = self.get_infoset_children_types()
			infoset_uniform_strategies = [None] * (self.domain.levels - 1)
			for level in range(self.domain.acting_depth):
				with tf.variable_scope("level{}".format(level)):
					infoset_uniform_strategies[level] = tf.to_float(tf.not_equal(infoset_children_types[level], IMAGINARY_NODE))
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
		infoset_cf_values, infoset_cf_values_per_actions = self.get_infoset_cf_values()
		infoset_children_types = self.get_infoset_children_types()
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
									infoset_cf_values_per_actions[level],
									name="zero_regrets_of_imaginary_children_lvl{}".format(level),
							),
							y=infoset_cf_values_per_actions[level] - infoset_cf_values[level],
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
					# TODO optimize by: pre-define `infosets_of_player1` and `infosets_of_player2` (in domain definitions) and switch
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
								0.0,
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
						# Note: An all-0's row cannot be normalized. Thus, when PCRegrets sum to 0, a uniform strategy is used instead.
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
					infosets_of_acting_player = tf.reshape(  # `tf.reshape` to force "shape of 2D tensor" == [number of infosets, 1]
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
			if delay is None:   # when `delay` is None, no weighted averaging is used
				return tf.constant(
						1.0,
						name="weighted_averaging_factor"
				)
			else:
				return tf.to_float(
						tf.maximum(self.domain.cfr_step - delay, 0),
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
							x=normalize(self.domain.cumulative_infoset_strategies[level]),
							y=self.domain.current_infoset_strategies[level],
							name="average_infoset_strategies_lvl{}".format(level)
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


if __name__ == '__main__':
	from src.domains.domain01.Domain01 import domain01
	from src.domains.matching_pennies.MatchingPennies import matching_pennies

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for tensorcfr in [TensorCFR(domain01), TensorCFR(matching_pennies)]:
			tensorcfr.domain.print_domain(sess)
