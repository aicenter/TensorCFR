import tensorflow as tf

from src.algorithms.tensorcfr_matching_pennies.uniform_strategies import assign_uniform_strategies_to_players
from src.commons.constants import DEFAULT_TOTAL_STEPS
from src.algorithms.tensorcfr_matching_pennies.cfr_step import do_cfr_step
from src.domains.matching_pennies.domain_definitions import cfr_step, current_infoset_strategies, \
	cumulative_infoset_strategies, positive_cumulative_regrets
from src.algorithms.tensorcfr_matching_pennies.strategy_matched_to_regrets import get_strategy_matched_to_regrets
from src.algorithms.tensorcfr_matching_pennies.update_strategies import get_average_infoset_strategies
from src.utils.tensor_utils import print_tensors


# game of matching pennies: see doc/matching_pennies_efg_illustration.jpg

def get_strategy_initializer():
	return "Initializing strategies to uniform ones...\n", assign_uniform_strategies_to_players()


def run_cfr(total_steps=DEFAULT_TOTAL_STEPS):
	strategy_initializer_message, strategy_initializer = get_strategy_initializer()
	cfr_step_op = do_cfr_step()
	strategies_matched_to_regrets = get_strategy_matched_to_regrets()
	average_infoset_strategies = get_average_infoset_strategies()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("TensorCFR\n")
		# print("Using user-defined strategies...\n")
		# print_tensors(sess, current_infoset_strategies)
		print(strategy_initializer_message)
		sess.run(strategy_initializer)
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
