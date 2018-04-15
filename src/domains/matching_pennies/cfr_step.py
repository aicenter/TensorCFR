import tensorflow as tf

from src.algorithms.tensorcfr.swap_players import swap_players
from src.commons.constants import DEFAULT_TOTAL_STEPS_ON_SMALL_DOMAINS, PLAYER1, PLAYER2
from src.domains.matching_pennies.domain_definitions import get_infoset_acting_players, cfr_step, \
	current_updating_player, current_opponent, cumulative_infoset_strategies, current_infoset_strategies, \
	positive_cumulative_regrets
from src.domains.matching_pennies.regrets import update_positive_cumulative_regrets
from src.domains.matching_pennies.update_strategies import update_strategy_of_acting_player, \
	cumulate_strategy_of_opponent
from src.utils.tensor_utils import print_tensors


# game of matching pennies: see doc/matching_pennies_efg_illustration.jpg

def process_strategies(acting_player=PLAYER1, opponent=PLAYER2):
	update_regrets_ops = update_positive_cumulative_regrets()
	update_ops = update_strategy_of_acting_player(acting_player=acting_player)
	cumulate_ops = cumulate_strategy_of_opponent(opponent=opponent)
	ops = [
		op
		for sublist in map(list, zip(update_regrets_ops, update_ops, cumulate_ops))
		for op in sublist
	]
	return ops


def increment_cfr_step():
	return tf.assign_add(
			ref=cfr_step,
			value=1,
			name="increment_cfr_step_op"
	)


def do_cfr_step():
	return tf.group(
			process_strategies(acting_player=current_updating_player, opponent=current_opponent)
			+ [swap_players(), increment_cfr_step()],
			name="cfr_step"
	)


if __name__ == '__main__':
	total_steps = DEFAULT_TOTAL_STEPS_ON_SMALL_DOMAINS
	infoset_acting_players_ = get_infoset_acting_players()
	process_strategies_ops = process_strategies()
	cfr_step_op = do_cfr_step()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("Running {} CFR+ iterations...\n".format(total_steps))
		for _ in range(total_steps):
			print("########## Start of CFR+ step {} ##########".format(cfr_step.eval()))
			print_tensors(sess, [current_updating_player, current_opponent])
			print("___________________________________\n")
			print_tensors(sess, positive_cumulative_regrets)
			print("___________________________________\n")
			print_tensors(sess, current_infoset_strategies)
			print("___________________________________\n")
			print_tensors(sess, cumulative_infoset_strategies)
			sess.run(cfr_step_op)
