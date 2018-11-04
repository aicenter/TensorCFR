#!/usr/bin/env python3

import tensorflow as tf

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.commons.constants import DEFAULT_TOTAL_STEPS, DEFAULT_AVERAGING_DELAY
from src.domains import FlattenedDomain
from src.utils.tf_utils import get_default_config_proto, masked_assign


class TensorCFR_BestResponse(TensorCFRFixedTrunkStrategies):
	def __init__(self, best_responder, trunk_strategies, domain: FlattenedDomain, trunk_depth=0):
		super().__init__(domain, trunk_depth)
		self.best_response_values = []
		self.trunk_strategies = trunk_strategies
		self.best_responder = best_responder
		self.final_br_value = None

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

	# TODO rename to get_final_best_response_value
	# TODO profiling -> verbose
	# TODO remove storing_strategies
	# TODO register_strategies_on_step
	def get_best_response_value_via_cfr(self, total_steps=DEFAULT_TOTAL_STEPS, delay=DEFAULT_AVERAGING_DELAY,
	                                    storing_strategies=False, profiling=False, register_strategies_on_step=list()):
		# a list of returned average strategies
		# the parameter `register_strategies_on_step` is used to determine which strategy export
		return_average_strategies = list()

		# if the `register_strategies_on_step` list is empty, register just the last iteration
		if len(register_strategies_on_step) == 0:
			register_strategies_on_step.append(total_steps - 1)

		self.cfr_parameters = {
			"total_steps"    : total_steps,
			"averaging_delay": delay,
			"trunk_depth"    : self.trunk_depth,
		}
		self.set_up_cfr_parameters(delay, total_steps)
		self.set_log_directory()
		if profiling:
			self.log_directory += "-profiling"
		with tf.variable_scope("initialization"):
			setup_messages, feed_dictionary = self.set_up_feed_dictionary(method="by-domain")
			print(setup_messages)

		cfr_step_op = self.do_cfr_step()

		set_initial_strategies = [
			tf.assign(
				current_strategies_per_level,
				value=self.trunk_strategies[level]
			)
			for level, current_strategies_per_level in enumerate(self.domain.current_infoset_strategies[:self.trunk_depth])
		]
		best_response_value = tf.identity(
			self.get_expected_values(for_player=self.best_responder)[0][0],
			name="best_response_value"
		)
		set_final_strategies = [
			tf.assign(
				self.domain.current_infoset_strategies[level],
				value=self.average_infoset_strategies[level],
				name="assign_final_average_strategies_lvl{}".format(level)
			)
			for level in range(self.domain.acting_depth)
		]

		with tf.Session(config=get_default_config_proto()) as self.session:
			self.session.run(tf.global_variables_initializer())
			self.session.run(set_initial_strategies)
			with tf.summary.FileWriter(self.log_directory, tf.get_default_graph()):
				for step in range(total_steps):
					self.session.run(cfr_step_op)
					self.best_response_values.append(self.session.run(best_response_value))

					if step in register_strategies_on_step:
						# if the number of step `i` is in `register_strategies_on_step` then add the average strategy
						# self.set_average_infoset_strategies()
						return_average_strategies.append({
							"step"            : step,
							"average_strategy": [self.session.run(x) for x in self.average_infoset_strategies]
						})

					if storing_strategies:
						self.store_final_average_strategies()

				self.session.run(set_final_strategies)
				self.final_br_value = self.session.run(best_response_value)
				self.log_after_all_steps()
		return self.final_br_value
