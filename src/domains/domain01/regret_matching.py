import tensorflow as tf

from domains.domain01.domain_01 import levels, positive_cumulative_regrets
from domains.domain01.regrets import update_positive_cumulative_regrets
from domains.domain01.uniform_strategies import get_IS_uniform_strategies
from utils.tensor_utils import print_tensors


# custom-made game: doc/domain_01.png (https://gitlab.com/beyond-deepstack/TensorCFR/blob/master/doc/domain_01.png)

def get_strategy_matched_to_regrets():  # TODO unittest
	IS_uniform_strategies = get_IS_uniform_strategies()
	IS_strategies = [None] * (levels - 1)
	for level in range(levels - 1):
		sums_of_regrets = tf.reduce_sum(positive_cumulative_regrets[level], axis=-1, keepdims=True,
		                                name="sums_of_regrets_lvl{}".format(level))
		normalized_regrets = tf.divide(positive_cumulative_regrets[level], sums_of_regrets,
		                               name="normalized_regrets_lvl{}".format(level))
		zero_sums = tf.squeeze(tf.equal(sums_of_regrets, 0), name="zero_sums_lvl{}".format(level))
		# Note: An all-0's row cannot be normalized. Thus, when PCRegrets sum to 0, a uniform strategy is used instead.
		IS_strategies[level] = tf.where(condition=zero_sums, x=IS_uniform_strategies[level],
		                                y=normalized_regrets, name="strategies_matched_to_regrets_lvl{}".format(level))
		# with tf.Session() as sess:
		# 	print("########## Level {} ##########".format(level))
		# 	sess.run(tf.global_variables_initializer())
		# 	print_tensors(sess, [sums_of_regrets, normalized_regrets, zero_sums, IS_uniform_strategies[level],
		# 	                     IS_strategies[level]])
	return IS_strategies


if __name__ == '__main__':
	IS_strategies_matched_to_regrets_ = get_strategy_matched_to_regrets()
	update_regrets = update_positive_cumulative_regrets()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(levels - 1):
			print("########## Level {} ##########".format(i))
			print_tensors(sess, [IS_strategies_matched_to_regrets_[i], update_regrets[i], IS_strategies_matched_to_regrets_[i]])

