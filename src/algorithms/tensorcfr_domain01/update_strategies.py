import tensorflow as tf

from src.algorithms.tensorcfr_domain01.regrets import update_positive_cumulative_regrets
from src.algorithms.tensorcfr_domain01.strategy_matched_to_regrets import get_strategy_matched_to_regrets
from src.algorithms.tensorcfr_domain01.topdown_reach_probabilities import get_infoset_reach_probabilities
from src.domains.domain01.domain_definitions import levels, get_infoset_acting_players, acting_depth, \
	cumulative_infoset_strategies, averaging_delay, cfr_step, current_infoset_strategies, infosets_of_non_chance_player, \
	current_updating_player, current_opponent
from src.utils.tensor_utils import print_tensors, masked_assign, expanded_multiply, normalize


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png

def update_strategy_of_updating_player(acting_player=current_updating_player):  # TODO unittest
	infoset_strategies_matched_to_regrets = get_strategy_matched_to_regrets()
	infoset_acting_players = get_infoset_acting_players()
	ops_update_infoset_strategies = [None] * acting_depth
	with tf.variable_scope("update_strategy_of_updating_player"):
		for level in range(acting_depth):
			infosets_of_acting_player = tf.reshape(   # `tf.reshape` to force "shape of 2D tensor" == [number of infosets, 1]
					tf.equal(infoset_acting_players[level], acting_player),
					shape=[current_infoset_strategies[level].shape[0]],
					name="infosets_of_updating_player_lvl{}".format(level)
			)
			ops_update_infoset_strategies[level] = masked_assign(
					ref=current_infoset_strategies[level],
					mask=infosets_of_acting_player,
					value=infoset_strategies_matched_to_regrets[level],
					name="op_update_infoset_strategies_lvl{}".format(level)
			)
		return ops_update_infoset_strategies


def get_weighted_averaging_factor(delay=averaging_delay):  # see https://arxiv.org/pdf/1407.5042.pdf (Section 2)
	with tf.variable_scope("weighted_averaging_factor"):
		if delay is None:   # when `delay` is None, no weighted averaging is used
			return tf.constant(
					1.0,
					name="weighted_averaging_factor"
			)
		else:
			return tf.to_float(
					tf.maximum(cfr_step - delay, 0),
					name="weighted_averaging_factor",
			)


def cumulate_strategy_of_opponent(opponent=current_opponent):  # TODO unittest
	infoset_acting_players = get_infoset_acting_players()
	infoset_reach_probabilities = get_infoset_reach_probabilities()
	with tf.variable_scope("cumulate_strategy_of_opponent"):
		cumulate_infoset_strategies_ops = [None] * acting_depth
		for level in range(acting_depth):
			infosets_of_opponent = tf.reshape(   # `tf.reshape` to force "shape of 2D tensor" == [number of infosets, 1]
					tf.equal(infoset_acting_players[level], opponent),
					shape=[current_infoset_strategies[level].shape[0]],
					name="infosets_of_opponent_lvl{}".format(level)
			)
			averaging_factor = get_weighted_averaging_factor()
			cumulate_infoset_strategies_ops[level] = masked_assign(
					# TODO implement and use `masked_assign_add` here
					ref=cumulative_infoset_strategies[level],
					mask=infosets_of_opponent,
					value=cumulative_infoset_strategies[level] + averaging_factor * expanded_multiply(
							expandable_tensor=infoset_reach_probabilities[level],
							expanded_tensor=current_infoset_strategies[level],
					),
					name="op_cumulate_infoset_strategies_lvl{}".format(level)
			)
		return cumulate_infoset_strategies_ops


def process_strategies(acting_player=current_updating_player, opponent=current_opponent):
	update_ops = update_strategy_of_updating_player(acting_player=acting_player)
	cumulate_ops = cumulate_strategy_of_opponent(opponent=opponent)
	with tf.variable_scope("process_strategies"):
		ops_process_strategies = [
			op
			for sublist in map(list, zip(update_ops, cumulate_ops))
			for op in sublist
		]
		return tf.group(ops_process_strategies, name="process_strategies")


def get_average_infoset_strategies():
	# TODO Do not normalize over imaginary nodes. <- Do we need to solve this? Or is it already ok (cf. `bottomup-*.py`)
	with tf.variable_scope("average_strategies"):
		average_infoset_strategies = [
			tf.where(
					condition=infosets_of_non_chance_player[level],
					x=normalize(cumulative_infoset_strategies[level]),
					y=current_infoset_strategies[level],
					name="average_infoset_strategies_lvl{}".format(level)
			)
			for level in range(acting_depth)
		]
		return average_infoset_strategies


if __name__ == '__main__':
	infoset_acting_players_ = get_infoset_acting_players()
	infoset_strategies_matched_to_regrets_ = get_strategy_matched_to_regrets()
	infoset_reach_probabilities_ = get_infoset_reach_probabilities()
	update_infoset_strategies = update_strategy_of_updating_player()
	ops_cumulate_infoset_strategies = cumulate_strategy_of_opponent()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.assign(ref=averaging_delay, value=0))
		print("########## Update strategy ##########")
		for i in range(levels - 1):
			print("########## Level {} ##########".format(i))
			print_tensors(sess, [
				current_infoset_strategies[i],
				infoset_acting_players_[i],
				current_updating_player,
				infoset_strategies_matched_to_regrets_[i],
				update_infoset_strategies[i],
				current_infoset_strategies[i]
			])
		print("########## Cumulate strategy ##########")
		for i in range(levels - 1):
			print("########## Level {} ##########".format(i))
			# TODO add to the unittest of ops_cumulate_infoset_strategies()
			print_tensors(sess, [
				infoset_acting_players_[i],
				current_opponent,
				infoset_reach_probabilities_[i],
				current_infoset_strategies[i],
			])
			for _ in range(3):
				print_tensors(sess, [
					cfr_step,
					cumulative_infoset_strategies[i],
					ops_cumulate_infoset_strategies[i],
					cumulative_infoset_strategies[i],
				])
				sess.run(tf.assign_add(ref=cfr_step, value=1, name="increment_cfr_step_op"))  # simulate increasing `crf_step`
		print("########## Average strategy ##########")
		print_tensors(sess, get_average_infoset_strategies())