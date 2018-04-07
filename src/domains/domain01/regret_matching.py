import tensorflow as tf

from src.domains.domain01.domain_01 import levels, positive_cumulative_regrets
from src.domains.domain01.regrets import update_positive_cumulative_regrets
from src.domains.domain01.uniform_strategies import get_infoset_uniform_strategies
from src.utils.tensor_utils import print_tensors


# custom-made game: see doc/domain_01_via_drawing.png and doc/domain_01_via_gambit.png

def get_strategy_matched_to_regrets():  # TODO unittest
	infoset_uniform_strategies = get_infoset_uniform_strategies()
	infoset_strategies = [None] * (levels - 1)
	for level in range(levels - 1):
		sums_of_regrets = tf.reduce_sum(
			positive_cumulative_regrets[level],
			axis=-1,
			keepdims=True,
			name="sums_of_regrets_lvl{}".format(level)
		)
		normalized_regrets = tf.divide(
			positive_cumulative_regrets[level],
			sums_of_regrets,
			name="normalized_regrets_lvl{}".format(level)
		)
		zero_sums = tf.squeeze(tf.equal(sums_of_regrets, 0), name="zero_sums_lvl{}".format(level))
		# Note: An all-0's row cannot be normalized. Thus, when PCRegrets sum to 0, a uniform strategy is used instead.
		# TODO verify uniform strategy is created (mix of both tf.where branches)
		infoset_strategies[level] = tf.where(
			condition=zero_sums,
			x=infoset_uniform_strategies[level],
			y=normalized_regrets,
			name="strategies_matched_to_regrets_lvl{}".format(level)
		)
	return infoset_strategies


if __name__ == '__main__':
	infoset_strategies_matched_to_regrets_ = get_strategy_matched_to_regrets()
	update_regrets = update_positive_cumulative_regrets()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(levels - 1):
			print("########## Level {} ##########".format(i))
			print_tensors(sess, [infoset_strategies_matched_to_regrets_[i],
			                     infoset_strategies_matched_to_regrets_[i],
			                     update_regrets[i],
			                     infoset_strategies_matched_to_regrets_[i],
			                     infoset_strategies_matched_to_regrets_[i],
			                     update_regrets[i],
			                     infoset_strategies_matched_to_regrets_[i],
			                     infoset_strategies_matched_to_regrets_[i],
			                     ])
