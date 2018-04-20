import tensorflow as tf

from src.domains.domain01.domain_definitions import levels, positive_cumulative_regrets, acting_depth
from src.algorithms.tensorcfr_domain01.regrets import update_positive_cumulative_regrets, \
	get_updated_values_of_cumulative_regrets
from src.algorithms.tensorcfr_domain01.uniform_strategies import get_infoset_uniform_strategies
from src.utils.tensor_utils import print_tensors


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png

def get_strategy_matched_to_regrets():  # TODO unittest
	infoset_uniform_strategies = get_infoset_uniform_strategies()
	updated_values_of_cumulative_regrets = get_updated_values_of_cumulative_regrets()
	with tf.variable_scope("strategies_matched_to_regrets"):
		strategies_matched_to_regrets = [None] * acting_depth
		for level in range(levels - 1):
			sums_of_regrets = tf.reduce_sum(
				updated_values_of_cumulative_regrets[level],
				axis=-1,
				keepdims=True,
				name="sums_of_regrets_lvl{}".format(level)
			)
			# TODO use `normalize()` here
			normalized_regrets = tf.divide(
				updated_values_of_cumulative_regrets[level],
				sums_of_regrets,
				name="normalized_regrets_lvl{}".format(level)
			)
			rows_summing_to_0 = tf.squeeze(tf.equal(sums_of_regrets, 0), name="zero_sums_lvl{}".format(level))
			# Note: An all-0's row cannot be normalized. Thus, when PCRegrets sum to 0, a uniform strategy is used instead.
			# TODO verify uniform strategy is created (mix of both tf.where branches)
			strategies_matched_to_regrets[level] = tf.where(
				condition=rows_summing_to_0,
				x=infoset_uniform_strategies[level],
				y=normalized_regrets,
				name="strategies_matched_to_regrets_lvl{}".format(level)
			)
		return strategies_matched_to_regrets


if __name__ == '__main__':
	strategies_matched_to_regrets_ = get_strategy_matched_to_regrets()
	updated_values_of_cumulative_regrets_ = get_updated_values_of_cumulative_regrets()
	ops_update_regrets_ = update_positive_cumulative_regrets()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(levels - 1):
			print("########## Level {} ##########".format(i))
			print_tensors(sess, [
				positive_cumulative_regrets[i], updated_values_of_cumulative_regrets_[i], strategies_matched_to_regrets_[i],
			])

			print("___________________________________\n")
			print_tensors(sess, [ops_update_regrets_[i]])
			print("___________________________________\n")

			print_tensors(sess, [
				positive_cumulative_regrets[i], updated_values_of_cumulative_regrets_[i], strategies_matched_to_regrets_[i],
			])

			print("___________________________________\n")
			print_tensors(sess, [ops_update_regrets_[i]])
			print("___________________________________\n")

			print_tensors(sess, [
				positive_cumulative_regrets[i], updated_values_of_cumulative_regrets_[i], strategies_matched_to_regrets_[i],
			])
