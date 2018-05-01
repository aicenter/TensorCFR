import tensorflow as tf

from src.algorithms.tensorcfr.TensorCFR import TensorCFR
from src.domains.domain01.Domain01 import domain01
from src.domains.matching_pennies.MatchingPennies import matching_pennies

if __name__ == '__main__':
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for tensorcfr in [TensorCFR(domain01), TensorCFR(matching_pennies)]:
			# TODO extract following lines to a UnitTest
			tensorcfr.show_reach_probabilities(sess)
			print("-----------Swap players-----------\n")
			sess.run(tensorcfr.swap_players())
			tensorcfr.show_reach_probabilities(sess)
