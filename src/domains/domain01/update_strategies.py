import tensorflow as tf

from src.constants import PLAYER1, PLAYER2
from src.domains.domain01.domain01 import levels, get_infoset_strategies, get_infoset_acting_players, acting_depth, \
	cumulative_infoset_strategies
from src.domains.domain01.strategy_matched_to_regrets import get_strategy_matched_to_regrets
from src.domains.domain01.topdown_reach_probabilities import get_infoset_reach_probabilities
from src.utils.tensor_utils import print_tensors, masked_assign, expanded_multiply


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png

def update_strategy_of_acting_player(acting_player):  # TODO unittest
	infoset_strategies_matched_to_regrets = get_strategy_matched_to_regrets()
	infoset_strategies = get_infoset_strategies()
	infoset_acting_players = get_infoset_acting_players()
	update_infoset_strategies_ops = [None] * acting_depth
	for level in range(acting_depth):
		infosets_of_acting_player = tf.reshape(tf.equal(infoset_acting_players[level], acting_player),
		                                       shape=[infoset_strategies[level].shape[0]],
		                                       name="infosets_of_acting_player_lvl{}".format(level))
		update_infoset_strategies_ops[level] = masked_assign(ref=infoset_strategies[level],
		                                                     mask=infosets_of_acting_player,
		                                                     value=infoset_strategies_matched_to_regrets[level],
		                                                     name="update_infoset_strategies_ops_lvl{}".format(level))
	return update_infoset_strategies_ops


def cumulate_strategy_of_opponent(opponent):  # TODO unittest
	infoset_strategies = get_infoset_strategies()
	infoset_acting_players = get_infoset_acting_players()
	infoset_reach_probabilities = get_infoset_reach_probabilities()
	cumulate_infoset_strategies_ops = [None] * acting_depth
	for level in range(acting_depth):
		infosets_of_opponent = tf.reshape(
				tf.equal(infoset_acting_players[level], opponent),
				shape=[infoset_strategies[level].shape[0]],
				name="infosets_of_opponent_lvl{}".format(level)
		)
		cumulate_infoset_strategies_ops[level] = masked_assign(
				# TODO implement and use `masked_assign_add` here
				ref=cumulative_infoset_strategies[level],
				mask=infosets_of_opponent,
				value=cumulative_infoset_strategies[level] + expanded_multiply(
						expandable_tensor=infoset_reach_probabilities[level],
						expanded_tensor=infoset_strategies[level],
				),
				name="cumulate_infoset_strategies_ops_lvl{}".format(level)
		)
	return cumulate_infoset_strategies_ops


if __name__ == '__main__':
	infoset_strategies_ = get_infoset_strategies()
	infoset_acting_players_ = get_infoset_acting_players()
	infoset_strategies_matched_to_regrets_ = get_strategy_matched_to_regrets()
	infoset_reach_probabilities_ = get_infoset_reach_probabilities()
	update_infoset_strategies = update_strategy_of_acting_player(acting_player=PLAYER1)
	cumulate_infoset_strategies = cumulate_strategy_of_opponent(opponent=PLAYER2)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("########## Update strategy ##########")
		for i in range(levels - 1):
			print("########## Level {} ##########".format(i))
			print_tensors(sess, [
				infoset_strategies_[i],
				infoset_acting_players_[i],
				infoset_strategies_matched_to_regrets_[i],
				update_infoset_strategies[i],
				infoset_strategies_[i]
			])
		print("########## Cumulate strategy ##########")
		for i in range(levels - 1):
			print("########## Level {} ##########".format(i))
			print_tensors(sess, [
				# TODO add to the unittest of cumulate_infoset_strategies()
				infoset_acting_players_[i],
				infoset_reach_probabilities_[i],
				infoset_strategies_[i],
				cumulative_infoset_strategies[i],
				cumulative_infoset_strategies[i],
				cumulate_infoset_strategies[i],
				cumulative_infoset_strategies[i],
				cumulative_infoset_strategies[i],
				cumulate_infoset_strategies[i],
				cumulative_infoset_strategies[i],
				cumulative_infoset_strategies[i],
			])
