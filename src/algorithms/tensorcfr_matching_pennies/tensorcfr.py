import tensorflow as tf

from src.algorithms.tensorcfr_matching_pennies.uniform_strategies import assign_uniform_strategies_to_players, \
	get_infoset_uniform_strategies
from src.commons.constants import DEFAULT_TOTAL_STEPS
from src.algorithms.tensorcfr_matching_pennies.cfr_step import do_cfr_step
from src.domains.matching_pennies.domain_definitions import cfr_step, current_infoset_strategies, \
	cumulative_infoset_strategies, positive_cumulative_regrets, initial_infoset_strategies, acting_depth
from src.algorithms.tensorcfr_matching_pennies.strategy_matched_to_regrets import get_strategy_matched_to_regrets
from src.algorithms.tensorcfr_matching_pennies.update_strategies import get_average_infoset_strategies
from src.utils.tensor_utils import print_tensors


# game of matching pennies: see doc/matching_pennies_efg_illustration.jpg

def setup_feed_dictionary(method="by-domain"):
	if method == "by-domain":
		return "Initializing strategies via domain definitions...\n", {}  # default value of `initial_infoset_strategies`
	elif method == "uniform":
		uniform_strategies_tensors = get_infoset_uniform_strategies()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			uniform_strategy_arrays = sess.run(uniform_strategies_tensors)
		feed_dictionary = {
			initial_infoset_strategies[0]: uniform_strategy_arrays[0],
			initial_infoset_strategies[1]: uniform_strategy_arrays[1],
		}
		return "Initializing strategies to uniform ones...\n", feed_dictionary
	# elif method == "custom":
	# 	return "Initializing strategies to custom values defined by user...\n", get_infoset_uniform_strategies()
	else:
		raise ValueError('Undefined method "{}" for setup_feed_dictionary().'.format(method))


def run_cfr(total_steps=DEFAULT_TOTAL_STEPS):
	# strategy_initializer_message, feed_dictionary = setup_feed_dictionary()
	# strategy_initializer_message, feed_dictionary = setup_feed_dictionary(method="by-domain")
	strategy_initializer_message, feed_dictionary = setup_feed_dictionary(method="uniform")
	# strategy_initializer_message, feed_dictionary = setup_feed_dictionary(method="invalid name")

	cfr_step_op = do_cfr_step()
	strategies_matched_to_regrets = get_strategy_matched_to_regrets()
	average_infoset_strategies = get_average_infoset_strategies()
	with tf.Session() as sess:
		print("TensorCFR\n")

		sess.run(tf.global_variables_initializer(), feed_dict=feed_dictionary)
		print(strategy_initializer_message)
		print_tensors(sess, current_infoset_strategies)

		print("Running {} CFR+ iterations...\n".format(total_steps))
		for _ in range(total_steps):
			print("########## CFR+ step #{} ##########".format(cfr_step.eval()))
			sess.run(cfr_step_op)
			print_tensors(sess, positive_cumulative_regrets)
			print("___________________________________\n")
			print_tensors(sess, strategies_matched_to_regrets)
			print("___________________________________\n")
			print_tensors(sess, current_infoset_strategies)
		print("###################################\n")
		print_tensors(sess, cumulative_infoset_strategies)
		print("___________________________________\n")
		print_tensors(sess, average_infoset_strategies)


if __name__ == '__main__':
	# run_cfr(total_steps=DEFAULT_TOTAL_STEPS_ON_SMALL_DOMAINS)
	run_cfr()
