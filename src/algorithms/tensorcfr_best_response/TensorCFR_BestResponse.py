#!/usr/bin/env python3

import tensorflow as tf

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.commons.constants import DEFAULT_TOTAL_STEPS, DEFAULT_AVERAGING_DELAY
from src.domains import FlattenedDomain
from src.utils.tf_utils import get_default_config_proto, masked_assign


class TensorCFR_BestResponse(TensorCFRFixedTrunkStrategies):
	def __init__(self, best_responder, trunk_strategies, domain: FlattenedDomain, trunk_depth=0):
		super().__init__(domain, trunk_depth)
		self.trunk_strategies = trunk_strategies
		self.best_responder = best_responder
		self.average_strategies_over_steps = None
		self.final_br_value = None
		self.construct_ops()

	def update_strategy_of_updating_player(self, acting_player=None):
		"""
		Update for the strategy for the given `acting_player`.

		Keep trunk strategies fixed if he is not `best_responder`.

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
			for level in range(self.domain.acting_depth):
				with tf.variable_scope("level{}".format(level)):
					infosets_of_acting_player = tf.reshape(
						# `tf.reshape` to force "shape of 2D tensor" == [number of infosets, 1]
						tf.equal(infoset_acting_players[level], acting_player),
						shape=[self.domain.current_infoset_strategies[level].shape[0]],
						name="infosets_of_updating_player_lvl{}".format(level)
					)
					if level < self.trunk_depth:
						infosets_of_best_responder = tf.reshape(
							# `tf.reshape` to force "shape of 2D tensor" == [number of infosets, 1]
							tf.equal(infoset_acting_players[level], self.best_responder),
							shape=[self.domain.current_infoset_strategies[level].shape[0]],
							name="infosets_of_best_responder_lvl{}".format(level)
						)
						infosets_to_update = tf.logical_and(
							infosets_of_acting_player,
							infosets_of_best_responder,
							name="infosets_to_update_lvl{}".format(level)
						)
					else:
						infosets_to_update = infosets_of_acting_player
					ops_update_infoset_strategies[level] = masked_assign(
						ref=self.domain.current_infoset_strategies[level],
						mask=infosets_to_update,
						value=infoset_strategies_matched_to_regrets[level],
						name="op_update_infoset_strategies_lvl{}".format(level)
					)
			return ops_update_infoset_strategies

	def construct_ops(self):
		self.cfr_step_op = self.do_cfr_step()
		self.set_initial_strategies = [
			tf.assign(
				current_strategies_per_level,
				value=self.trunk_strategies[level]
			)
			for level, current_strategies_per_level in enumerate(self.domain.current_infoset_strategies[:self.trunk_depth])
		]
		self.best_response_value_op = tf.identity(
			self.get_expected_values(for_player=self.best_responder)[0][0],
			name="best_response_value"
		)
		self.set_final_strategies = [
			tf.assign(
				self.domain.current_infoset_strategies[level],
				value=self.average_infoset_strategies[level],
				name="assign_final_average_strategies_lvl{}".format(level)
			)
			for level in range(self.domain.acting_depth)
		]

	def get_final_best_response_value(self, total_steps=DEFAULT_TOTAL_STEPS, delay=DEFAULT_AVERAGING_DELAY, verbose=False,
	                                  register_strategies_on_step=None):
		if register_strategies_on_step is None:
			register_strategies_on_step = [total_steps - 1]   # by default, register just the last iteration
		self.average_strategies_over_steps = list()             # reset the list

		with tf.variable_scope("initialization"):
			setup_messages, feed_dictionary = self.set_up_feed_dictionary(method="by-domain")
			print(setup_messages)

		with tf.Session(config=get_default_config_proto()) as self.session:
			self.session.run(tf.global_variables_initializer())
			self.session.run(self.set_initial_strategies)
			for step in range(total_steps):
				self.session.run(self.cfr_step_op)

				if step in register_strategies_on_step:
					self.average_strategies_over_steps.append({
						"step"            : step,
						"average_strategy": [self.session.run(strategy).tolist() for strategy in self.average_infoset_strategies]
					})

			self.session.run(self.set_final_strategies)
			self.final_br_value = self.session.run(self.best_response_value_op)
			if verbose:
				self.log_after_all_steps()
		return self.final_br_value
