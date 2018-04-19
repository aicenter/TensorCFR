import re

import tensorflow as tf
import os
import datetime

from src.algorithms.tensorcfr_domain01.bottomup_expected_values import get_expected_values
from src.algorithms.tensorcfr_domain01.cfr_step import do_cfr_step
from src.algorithms.tensorcfr_domain01.counterfactual_values import get_nodal_cf_values, get_cf_values_infoset, \
	get_cf_values_infoset_actions
from src.algorithms.tensorcfr_domain01.regrets import get_regrets
from src.algorithms.tensorcfr_domain01.strategy_matched_to_regrets import get_strategy_matched_to_regrets
from src.algorithms.tensorcfr_domain01.topdown_reach_probabilities import get_nodal_reach_probabilities
from src.algorithms.tensorcfr_domain01.uniform_strategies import get_infoset_uniform_strategies
from src.algorithms.tensorcfr_domain01.update_strategies import get_average_infoset_strategies
from src.commons.constants import DEFAULT_TOTAL_STEPS, DEFAULT_TOTAL_STEPS_ON_SMALL_DOMAINS, DEFAULT_AVERAGING_DELAY
from src.domains.domain01.domain_definitions import cfr_step, current_infoset_strategies, \
	cumulative_infoset_strategies, positive_cumulative_regrets, initial_infoset_strategies, acting_depth, averaging_delay
from src.utils.tensor_utils import print_tensors


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png

def set_up_feed_dictionary(method="by-domain", initial_strategy_values=None):
	if method == "by-domain":
		return "Initializing strategies via domain definitions...\n", {}  # default value of `initial_infoset_strategies`
	elif method == "uniform":
		with tf.variable_scope("initialize_strategies"):
			uniform_strategies_tensors = get_infoset_uniform_strategies()
			with tf.Session() as temp_sess:
				temp_sess.run(tf.global_variables_initializer())
				uniform_strategy_arrays = temp_sess.run(uniform_strategies_tensors)
			return "Initializing to uniform strategies...\n", {
				initial_infoset_strategies[level]: uniform_strategy_arrays[level]
				for level in range(acting_depth)
			}
	elif method == "custom":
		if initial_strategy_values is None:
			raise ValueError('No "initial_strategy_values" given.')
		if len(initial_strategy_values) != len(initial_infoset_strategies):
			raise ValueError(
					'Mismatched "len(initial_strategy_values) == {}" and "len(initial_infoset_strategies) == {}".'.format(
							len(initial_strategy_values), len(initial_infoset_strategies)
					)
			)
		return "Initializing strategies to custom values defined by user...\n", {
			initial_infoset_strategies[level]: initial_strategy_values[level]
			for level in range(acting_depth)
		}
	else:
		raise ValueError('Undefined method "{}" for set_up_feed_dictionary().'.format(method))


def set_up_tensorboard(session, hyperparameters):
	log_dir = "logs/{}-{}-{}".format(
			"domain01",
			datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
			",".join(
					("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
					 for key, value in sorted(hyperparameters.items()))).replace("/", "-")
	)
	if not os.path.exists("logs"):
		os.mkdir("logs")
	with tf.variable_scope("tensorboard_operations"):
		summary_writer = tf.contrib.summary.create_file_writer(log_dir, flush_millis=10 * 1000)
		with summary_writer.as_default():
			tf.contrib.summary.initialize(session=session, graph=session.graph)


def set_up_cfr():
	# TODO extract these lines to a UnitTest
	# setup_messages, feed_dictionary = set_up_feed_dictionary()
	# setup_messages, feed_dictionary = set_up_feed_dictionary(method="by-domain")
	setup_messages, feed_dictionary = set_up_feed_dictionary(method="uniform")
	# setup_messages, feed_dictionary = set_up_feed_dictionary(method="custom")  # should raise ValueError
	# setup_messages, feed_dictionary = set_up_feed_dictionary(
	# 		method="custom",
	# 		initial_strategy_values=[
	# 			[[1.0, 0.0]],
	# 		],
	# )  # should raise ValueError
	# setup_messages, feed_dictionary = set_up_feed_dictionary(
	# 		method="custom",
	# 		initial_strategy_values=[
	# 			[[1.0, 0.0]],
	# 			[[1.0, 0.0]],
	# 		]
	# )
	# setup_messages, feed_dictionary = set_up_feed_dictionary(method="invalid")  # should raise ValueError
	return feed_dictionary, setup_messages


def log_before_all_steps(sess, setup_messages, total_steps, averaging_delay):
	print("TensorCFR\n")
	print(setup_messages)
	print_tensors(sess, current_infoset_strategies)
	print("Running {} CFR+ iterations, averaging_delay == {}...\n".format(total_steps, averaging_delay))


def log_before_every_step(sess, cf_values_infoset, cf_values_infoset_actions, cf_values_nodes, expected_values, reach_probabilities,
                          regrets):
	print("########## CFR+ step #{} ##########".format(cfr_step.eval()))
	print_tensors(sess, reach_probabilities)
	print("___________________________________\n")
	print_tensors(sess, expected_values)
	print("___________________________________\n")
	print_tensors(sess, cf_values_nodes)
	print("___________________________________\n")
	print_tensors(sess, cf_values_infoset_actions)
	print("___________________________________\n")
	print_tensors(sess, cf_values_infoset)
	print("___________________________________\n")
	print_tensors(sess, regrets)
	print("___________________________________\n")
	print_tensors(sess, cf_values_infoset_actions)
	print("___________________________________\n")
	print_tensors(sess, cf_values_infoset)
	print("___________________________________\n")
	print_tensors(sess, regrets)
	print("___________________________________\n")
	print_tensors(sess, positive_cumulative_regrets)
	print("___________________________________\n")
	print_tensors(sess, regrets)
	print("___________________________________\n")


def log_after_every_step(sess, strategies_matched_to_regrets):
	print_tensors(sess, positive_cumulative_regrets)
	print("___________________________________\n")
	print_tensors(sess, strategies_matched_to_regrets)
	print("___________________________________\n")
	print_tensors(sess, current_infoset_strategies)


def log_after_all_steps(average_infoset_strategies, sess):
	print("###################################\n")
	print_tensors(sess, cumulative_infoset_strategies)
	print("___________________________________\n")
	print_tensors(sess, average_infoset_strategies)


def run_cfr(total_steps=DEFAULT_TOTAL_STEPS, quiet=False, delay=DEFAULT_AVERAGING_DELAY):
	with tf.variable_scope("initialization"):
		feed_dictionary, setup_messages = set_up_cfr()
		assign_averaging_delay_op = tf.assign(ref=averaging_delay, value=delay, name="assign_averaging_delay")
	cfr_step_op = do_cfr_step()

	# tensors to log if quiet is False
	reach_probabilities = get_nodal_reach_probabilities() if not quiet else None
	expected_values = get_expected_values() if not quiet else None
	cf_values_nodes = get_nodal_cf_values() if not quiet else None
	cf_values_infoset = get_cf_values_infoset() if not quiet else None
	cf_values_infoset_actions = get_cf_values_infoset_actions() if not quiet else None
	regrets = get_regrets() if not quiet else None
	strategies_matched_to_regrets = get_strategy_matched_to_regrets() if not quiet else None
	average_infoset_strategies = get_average_infoset_strategies() if not quiet else None

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer(), feed_dict=feed_dictionary)
		hyperparameters = {
			"total_steps": total_steps,
			"averaging_delay": delay,
		}
		set_up_tensorboard(session=sess, hyperparameters=hyperparameters)
		assigned_averaging_delay = sess.run(assign_averaging_delay_op)
		if quiet is False:
			log_before_all_steps(sess, setup_messages, total_steps, assigned_averaging_delay)
		for _ in range(total_steps):
			if quiet is False:
				log_before_every_step(sess, cf_values_infoset, cf_values_infoset_actions, cf_values_nodes, expected_values,
				                      reach_probabilities, regrets)
			sess.run(cfr_step_op)
			if quiet is False:
				log_after_every_step(sess, strategies_matched_to_regrets)
		if quiet is False:
			log_after_all_steps(average_infoset_strategies, sess)


if __name__ == '__main__':
	# run_cfr(total_steps=10, delay=0)
	run_cfr(total_steps=10, delay=0, quiet=True)
	# run_cfr(total_steps=DEFAULT_TOTAL_STEPS_ON_SMALL_DOMAINS, delay=5)
	# run_cfr(quiet=True)
	# run_cfr(quiet=True, total_steps=10000)
