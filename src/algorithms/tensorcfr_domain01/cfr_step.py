import tensorflow as tf

from src.algorithms.tensorcfr_domain01.regrets import update_positive_cumulative_regrets
from src.algorithms.tensorcfr_domain01.swap_players import swap_players
from src.algorithms.tensorcfr_domain01.update_strategies import process_strategies
from src.domains.domain01.domain_definitions import get_infoset_acting_players, cfr_step, \
	current_updating_player, current_opponent, cumulative_infoset_strategies, current_infoset_strategies, \
	positive_cumulative_regrets, averaging_delay
from src.utils.tensor_utils import print_tensors


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png


def increment_cfr_step():
	with tf.variable_scope("increment_step"):
		return tf.assign_add(
				ref=cfr_step,
				value=1,
				name="increment_cfr_step_op"
		)


def do_cfr_step():
	with tf.variable_scope("group_update_regrets"):
		ops_update_regrets = tf.group(
			update_positive_cumulative_regrets(),
			name="group_update_regrets",
		)
	ops_process_strategies = process_strategies()
	with tf.control_dependencies([ops_update_regrets, ops_process_strategies]):
		op_swap_players = swap_players()
		op_inc_step = increment_cfr_step()
	with tf.variable_scope("cfr_step"):
		return tf.group(
				[ops_update_regrets, ops_process_strategies, op_swap_players, op_inc_step],
				name="cfr_step"
		)


if __name__ == '__main__':
	total_steps = 4
	infoset_acting_players_ = get_infoset_acting_players()
	cfr_step_op = do_cfr_step()
	delay = 0
	with tf.variable_scope("initialization"):
		assign_averaging_delay_op = tf.assign(ref=averaging_delay, value=delay, name="assign_averaging_delay")
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(assign_averaging_delay_op)
		# set_up_tensorboard(
		# 		session=sess,
		# 		hyperparameters={
		# 			"total_steps": total_steps,
		# 			"averaging_delay": delay,
		# 		},
		# 		basename="cfr_step",
		# 		domain_name="domain01",
		# )
		print("Running {} CFR+ iterations...\n".format(total_steps))
		for _ in range(total_steps):
			print("########## Start of CFR+ step {} ##########".format(cfr_step.eval()))
			print_tensors(sess, [current_updating_player, current_opponent])
			print("___________________________________\n")
			print_tensors(sess, infoset_acting_players_)
			print("___________________________________\n")
			print_tensors(sess, positive_cumulative_regrets)
			print("___________________________________\n")
			print_tensors(sess, current_infoset_strategies)
			print("___________________________________\n")
			print_tensors(sess, cumulative_infoset_strategies)
			sess.run(cfr_step_op)
