import tensorflow as tf

from src.constants import PLAYER1
from src.domains.domain01.domain01 import levels, get_infoset_strategies, get_infoset_acting_players, acting_depth
from src.domains.domain01.strategy_matched_to_regrets import get_strategy_matched_to_regrets
from src.utils.tensor_utils import print_tensors, masked_assign


# custom-made game: doc/domain_01.png (https://gitlab.com/beyond-deepstack/TensorCFR/blob/master/doc/domain_01.png)

def update_strategy_of_acting_player(acting_player):  # TODO unittest
	infoset_strategies_matched_to_regrets = get_strategy_matched_to_regrets()
	infoset_strategies = get_infoset_strategies()
	infoset_acting_players = get_infoset_acting_players()
	update_infoset_strategies_ops = [None] * acting_depth
	for level in range(acting_depth):
		infosets_belonging_to_acting_player = tf.reshape(tf.equal(infoset_acting_players[level], acting_player),
		                                                 shape=[infoset_strategies[level].shape[0]],
		                                                 name="infosets_of_acting_player_lvl{}".format(level))
		update_infoset_strategies_ops[level] = masked_assign(ref=infoset_strategies[level],
		                                                     mask=infosets_belonging_to_acting_player,
		                                                     value=infoset_strategies_matched_to_regrets[level],
		                                                     name="update_infoset_strategies_ops_lvl{}".format(level))
	return update_infoset_strategies_ops


def cumulate_strategy_of_opponent():  # TODO unittest
	raise NotImplementedError


if __name__ == '__main__':
	infoset_strategies_ = get_infoset_strategies()
	infoset_acting_players_ = get_infoset_acting_players()
	infoset_strategies_matched_to_regrets_ = get_strategy_matched_to_regrets()
	update_infoset_strategies = update_strategy_of_acting_player(acting_player=PLAYER1)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(levels - 1):
			print("########## Level {} ##########".format(i))
			print_tensors(sess, [
				infoset_strategies_[i],
				infoset_acting_players_[i],
				infoset_strategies_matched_to_regrets_[i],
				update_infoset_strategies[i],
				infoset_strategies_[i]
			])
