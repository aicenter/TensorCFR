import tensorflow as tf

from src.constants import PLAYER1
from src.domains.domain01.domain_01 import levels, get_IS_strategies, get_IS_acting_players
from src.domains.domain01.strategy_matched_to_regrets import get_strategy_matched_to_regrets
from src.utils.tensor_utils import print_tensors, masked_assign


# custom-made game: doc/domain_01.png (https://gitlab.com/beyond-deepstack/TensorCFR/blob/master/doc/domain_01.png)

def update_strategy_of_acting_player(acting_player):  # TODO unittest
	IS_strategies_matched_to_regrets = get_strategy_matched_to_regrets()
	IS_strategies = get_IS_strategies()
	IS_acting_players = get_IS_acting_players()
	acting_depth = len(IS_strategies)
	update_IS_strategies_ops = [None] * acting_depth
	for level in range(acting_depth):
		infosets_belonging_to_acting_player = tf.reshape(tf.equal(IS_acting_players[level], acting_player),
		                                                 shape=[IS_strategies[level].shape[0]],
		                                                 name="infosets_of_acting_player_lvl{}".format(level))
		update_IS_strategies_ops[level] = masked_assign(ref=IS_strategies[level], mask=infosets_belonging_to_acting_player,
		                                                value=IS_strategies_matched_to_regrets[level],
		                                                name="update_IS_strategies_ops_lvl{}".format(level))
	return update_IS_strategies_ops


def cumulate_strategy_of_opponent():  # TODO unittest
	raise NotImplementedError


if __name__ == '__main__':
	IS_strategies_ = get_IS_strategies()
	IS_acting_players_ = get_IS_acting_players()
	IS_strategies_matched_to_regrets_ = get_strategy_matched_to_regrets()
	update_IS_strategies = update_strategy_of_acting_player(acting_player=PLAYER1)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(levels - 1):
			print("########## Level {} ##########".format(i))
			print_tensors(sess, [
				IS_strategies_[i],
				IS_acting_players_[i],
				IS_strategies_matched_to_regrets_[i],
				update_IS_strategies[i],
				IS_strategies_[i]
			])
