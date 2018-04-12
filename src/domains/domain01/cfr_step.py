import tensorflow as tf

from src.domains.domain01.domain01 import get_infoset_strategies, get_infoset_acting_players, cfr_step, \
	current_updating_player, current_opponent
from src.domains.domain01.update_strategies import process_strategies
from src.utils.tensor_utils import print_tensors


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png

def increment_cfr_step():
	return tf.assign_add(
			ref=cfr_step,
			value=1,
			name="increment_cfr_step_op"
	)


def do_cfr_step():
	return tf.group(
			process_strategies(acting_player=PLAYER1, opponent=PLAYER2) + [increment_cfr_step()],
			name="cfr_step"
	)


if __name__ == '__main__':
	infoset_strategies_ = get_infoset_strategies()
	infoset_acting_players_ = get_infoset_acting_players()
	process_strategies_ops = process_strategies()
	cfr_step_op = do_cfr_step()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("########## Process strategies ##########")
		for _ in range(4):
			print("########## Start of CFR step {} ##########".format(cfr_step.eval()))
			print_tensors(sess, infoset_strategies_)
			sess.run(cfr_step_op)
