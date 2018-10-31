#!/usr/bin/env python3

import tensorflow as tf

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.commons.constants import DEFAULT_TOTAL_STEPS, DEFAULT_AVERAGING_DELAY
from src.domains import FlattenedDomain
from src.domains.available_domains import get_domain_by_name
from src.utils.tf_utils import get_default_config_proto


class TensorCFR_BestResponse(TensorCFRFixedTrunkStrategies):
	def __init__(self, trunk_strategies, domain: FlattenedDomain, trunk_depth=0):
		super().__init__(domain, trunk_depth)
		self.trunk_strategies = trunk_strategies

	def set_up_feed_dictionary(self, method="trunk", initial_strategy_values=None, seed=None):
		if method == "trunk":
			raise NotImplementedError
		else:
			return super().set_up_feed_dictionary(method, initial_strategy_values, seed)

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

		with tf.Session(config=get_default_config_proto()) as self.session:
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

	tensorcfr_instance = TensorCFR_BestResponse(
		[],  # TODO
		domain_,
		trunk_depth=4
	)
	tensorcfr_instance.cfr_strategies_after_fixed_trunk(
		# total_steps=10,
		# storing_strategies=True,
		# profiling=True,
		# delay=0
		register_strategies_on_step=[1, 500, 999]
	)
