#!/usr/bin/env python3

import tensorflow as tf

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.commons.constants import DEFAULT_TOTAL_STEPS, DEFAULT_AVERAGING_DELAY, PLAYER1
from src.domains import FlattenedDomain
from src.domains.available_domains import get_domain_by_name
from src.utils.tf_utils import get_default_config_proto, print_tensors, masked_assign


class TensorCFR_BestResponse(TensorCFRFixedTrunkStrategies):
	def __init__(self, best_responder, trunk_strategies, domain: FlattenedDomain, trunk_depth=0):
		super().__init__(domain, trunk_depth)
		self.trunk_strategies = trunk_strategies
		self.best_responder = best_responder

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
					ops_update_infoset_strategies[level] = masked_assign(
						ref=self.domain.current_infoset_strategies[level],
						mask=infosets_of_acting_player,  # TODO modify for BR: tf.and
						value=infoset_strategies_matched_to_regrets[level],
						name="op_update_infoset_strategies_lvl{}".format(level)
					)
			return ops_update_infoset_strategies

	# TODO refactor to make use of method overriding
	def cfr_strategies_after_fixed_trunk(self, total_steps=DEFAULT_TOTAL_STEPS, delay=DEFAULT_AVERAGING_DELAY,
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

		with tf.Session(config=get_default_config_proto()) as self.session:
			self.session.run(tf.global_variables_initializer())
			self.session.run(set_initial_strategies)
			for step in range(total_steps):
				print("CFR step #{}".format(step))
				self.session.run(cfr_step_op)
				print_tensors(self.session, self.domain.initial_infoset_strategies)
				print_tensors(self.session, self.domain.current_infoset_strategies)

				if step in register_strategies_on_step:
					# if the number of step `i` is in `register_strategies_on_step` then add the average strategy
					# self.set_average_infoset_strategies()
					return_average_strategies.append(
						{"step"            : step,
						 "average_strategy": [self.session.run(x) for x in self.average_infoset_strategies]})

				if storing_strategies:
					self.store_final_average_strategies()
			self.log_after_all_steps()
		return return_average_strategies


if __name__ == '__main__':
	# domain_ = get_domain_by_name("flattened_hunger_games")
	# domain_ = get_domain_by_name("flattened_hunger_games_2")
	# domain_ = get_domain_by_name("flattened_domain01_via_gambit")
	# domain_ = get_domain_by_name("II-GS2_gambit_flattened")
	domain_ = get_domain_by_name("II-GS3_gambit_flattened")
	# domain_ = get_domain_by_name("IIGS5_gambit_flattened")
	# domain_ = get_domain_by_name("IIGS6_gambit_flattened")

	trunk_strategies_gs3_td4 = [
		[  # infoset strategies at level 0
			[1.]
		],

		[  # infoset strategies at level 1
			[0.1, 0.9, 0.]
		],

		[  # infoset strategies at level 2
			[0.69, 0.01, 0.3]
		],

		[  # infoset strategies at level 3
			[1.],
			[1.],
			[1.],
			[1.],
			[1.],
			[1.],
			[1.],
			[1.],
			[1.]
		]
	]

	tensorcfr_instance = TensorCFR_BestResponse(
		best_responder=PLAYER1,
		trunk_strategies=trunk_strategies_gs3_td4,
		domain=domain_,
		trunk_depth=4
	)
	tensorcfr_instance.cfr_strategies_after_fixed_trunk(
		total_steps=10,
		# storing_strategies=True,
		# profiling=True,
		# delay=0
		register_strategies_on_step=[1, 500, 999]
	)
