import tensorflow as tf

from src.constants import PLAYER1, PLAYER2
from src.domains.domain01.domain01 import get_infoset_strategies, get_infoset_acting_players, cfr_step
from src.domains.domain01.update_strategies import update_strategy_of_acting_player, cumulate_strategy_of_opponent
from src.utils.tensor_utils import print_tensors


# custom-made game: doc/domain_01.png (https://gitlab.com/beyond-deepstack/TensorCFR/blob/master/doc/domain_01.png)

def process_strategies(acting_player=PLAYER1, opponent=PLAYER2):
	update_ops = update_strategy_of_acting_player(acting_player=acting_player)
	cumulate_ops = cumulate_strategy_of_opponent(opponent=opponent)
	ops = [
		op
		for sublist in map(list, zip(update_ops, cumulate_ops))
		for op in sublist
	]
	return ops


def do_cfr_step():
	increment_cfr_step_op = tf.assign_add(
			ref=cfr_step,
			value=1,
			name="increment_cfr_step_op"
	)
	cfr_ops = process_strategies(acting_player=PLAYER1, opponent=PLAYER2) + [increment_cfr_step_op]
	return tf.group(cfr_ops, name="cfr_step")


if __name__ == '__main__':
	infoset_strategies_ = get_infoset_strategies()
	infoset_acting_players_ = get_infoset_acting_players()
	process_strategies_ops = process_strategies()
	cfr_step_op = do_cfr_step()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("########## Process strategies ##########")
		print_tensors(sess, [cfr_step, cfr_step])
		print_tensors(sess, infoset_strategies_)
		sess.run(cfr_step_op)
		print_tensors(sess, [cfr_step, cfr_step])
		print_tensors(sess, infoset_strategies_)
		sess.run(cfr_step_op)
		print_tensors(sess, [cfr_step, cfr_step])
		print_tensors(sess, infoset_strategies_)
