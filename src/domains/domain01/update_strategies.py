import tensorflow as tf

from src.constants import PLAYER1, PLAYER2
from src.domains.domain01.domain01 import levels, get_infoset_acting_players, acting_depth, \
	cumulative_infoset_strategies, averaging_delay, cfr_step, immediate_infoset_strategies
from src.domains.domain01.strategy_matched_to_regrets import get_strategy_matched_to_regrets
from src.domains.domain01.topdown_reach_probabilities import get_infoset_reach_probabilities
from src.utils.tensor_utils import print_tensors, masked_assign, expanded_multiply


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png

def update_strategy_of_acting_player(acting_player):  # TODO unittest
	infoset_strategies_matched_to_regrets = get_strategy_matched_to_regrets()
	infoset_acting_players = get_infoset_acting_players()
	update_infoset_strategies_ops = [None] * acting_depth
	for level in range(acting_depth):
		infosets_of_acting_player = tf.reshape(tf.equal(infoset_acting_players[level], acting_player),
		                                       shape=[immediate_infoset_strategies[level].shape[0]],
		                                       name="infosets_of_acting_player_lvl{}".format(level))
		update_infoset_strategies_ops[level] = masked_assign(ref=immediate_infoset_strategies[level],
		                                                     mask=infosets_of_acting_player,
		                                                     value=infoset_strategies_matched_to_regrets[level],
		                                                     name="op_update_infoset_strategies_lvl{}".format(level))
	return update_infoset_strategies_ops


def get_weighted_averaging_factor(delay=averaging_delay):  # see https://arxiv.org/pdf/1407.5042.pdf (Section 2)
	if delay is None:
		return tf.constant(1.0, name="weighted_averaging_factor")
	else:
		return tf.to_float(
				tf.maximum(cfr_step - delay, 0),
				name="weighted_averaging_factor",
		)


def cumulate_strategy_of_opponent(opponent):  # TODO unittest
	infoset_acting_players = get_infoset_acting_players()
	infoset_reach_probabilities = get_infoset_reach_probabilities()
	cumulate_infoset_strategies_ops = [None] * acting_depth
	for level in range(acting_depth):
		infosets_of_opponent = tf.reshape(
				tf.equal(infoset_acting_players[level], opponent),
				shape=[immediate_infoset_strategies[level].shape[0]],
				name="infosets_of_opponent_lvl{}".format(level)
		)
		averaging_factor = get_weighted_averaging_factor()
		cumulate_infoset_strategies_ops[level] = masked_assign(
				# TODO implement and use `masked_assign_add` here
				ref=cumulative_infoset_strategies[level],
				mask=infosets_of_opponent,
				value=cumulative_infoset_strategies[level] + averaging_factor * expanded_multiply(
						expandable_tensor=infoset_reach_probabilities[level],
						expanded_tensor=immediate_infoset_strategies[level],
				),
				name="op_cumulate_infoset_strategies_lvl{}".format(level)
		)
	return cumulate_infoset_strategies_ops


def process_strategies(acting_player=PLAYER1, opponent=PLAYER2):
	update_ops = update_strategy_of_acting_player(acting_player=acting_player)
	cumulate_ops = cumulate_strategy_of_opponent(opponent=opponent)
	ops = [
		op
		for sublist in map(list, zip(update_ops, cumulate_ops))
		for op in sublist
	]
	return ops


if __name__ == '__main__':
	infoset_acting_players_ = get_infoset_acting_players()
	infoset_strategies_matched_to_regrets_ = get_strategy_matched_to_regrets()
	infoset_reach_probabilities_ = get_infoset_reach_probabilities()
	update_infoset_strategies = update_strategy_of_acting_player(acting_player=PLAYER1)
	ops_cumulate_infoset_strategies = cumulate_strategy_of_opponent(opponent=PLAYER2)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("########## Update strategy ##########")
		for i in range(levels - 1):
			print("########## Level {} ##########".format(i))
			print_tensors(sess, [
				immediate_infoset_strategies[i],
				infoset_acting_players_[i],
				infoset_strategies_matched_to_regrets_[i],
				update_infoset_strategies[i],
				immediate_infoset_strategies[i]
			])
		print("########## Cumulate strategy ##########")
		for i in range(levels - 1):
			print("########## Level {} ##########".format(i))
			# TODO add to the unittest of ops_cumulate_infoset_strategies()
			print_tensors(sess, [
				infoset_acting_players_[i],
				infoset_reach_probabilities_[i],
				immediate_infoset_strategies[i],
			])
			for _ in range(3):      # TODO extract `3` to `constants.py`
				print_tensors(sess, [
					cfr_step,
					cumulative_infoset_strategies[i],
					ops_cumulate_infoset_strategies[i],
					cumulative_infoset_strategies[i],
				])
				sess.run(tf.assign_add(ref=cfr_step, value=1, name="increment_cfr_step_op"))  # simulate increasing `crf_step`
