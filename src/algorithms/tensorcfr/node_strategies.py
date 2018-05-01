import tensorflow as tf

from src.algorithms.tensorcfr.TensorCFR import TensorCFR
from src.domains.domain01.Domain01 import domain01

if __name__ == '__main__':
	for tensorcfr in [TensorCFR(domain01)]:
		node_strategies = tensorcfr.get_node_strategies()
		node_cf_strategies = tensorcfr.get_node_cf_strategies()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			# TODO extract following lines to a UnitTest
			tensorcfr.show_strategies(session=sess)
			print("-----------Swap players-----------\n")
			sess.run(tensorcfr.swap_players())
			tensorcfr.show_strategies(session=sess)
