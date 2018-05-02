import tensorflow as tf

from src.algorithms.tensorcfr.TensorCFR import TensorCFR
from src.domains.domain01.Domain01 import domain01
from src.domains.matching_pennies.MatchingPennies import matching_pennies
from src.utils.tensor_utils import print_tensors


if __name__ == '__main__':
	for tensorcfr in [TensorCFR(domain01), TensorCFR(matching_pennies)]:
		strategies_matched_to_regrets = tensorcfr.get_strategy_matched_to_regrets()
		update_regrets = tensorcfr.update_positive_cumulative_regrets()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			print(">>>>>>>>>> {} <<<<<<<<<<".format(tensorcfr.domain.domain_name))
			for i in range(tensorcfr.domain.acting_depth):
				print("########## Level {} ##########".format(i))
				print_tensors(sess, [
					tensorcfr.domain.positive_cumulative_regrets[i],
					strategies_matched_to_regrets[i],
					strategies_matched_to_regrets[i],
					update_regrets[i],
					tensorcfr.domain.positive_cumulative_regrets[i],
					strategies_matched_to_regrets[i],
					strategies_matched_to_regrets[i],
					update_regrets[i],
					tensorcfr.domain.positive_cumulative_regrets[i],
					strategies_matched_to_regrets[i],
					strategies_matched_to_regrets[i],
				])
