import tensorflow as tf

from src.algorithms.tensorcfr.TensorCFR import TensorCFR
from src.domains.domain01.Domain01 import domain01
from src.domains.matching_pennies.MatchingPennies import matching_pennies
from src.utils.tensor_utils import print_tensors


if __name__ == '__main__':
	for tensorcfr in [TensorCFR(domain01), TensorCFR(matching_pennies)]:
		print(">>>>>>>>>> {} <<<<<<<<<<".format(tensorcfr.domain.domain_name))
		infoset_uniform_strategies = tensorcfr.get_infoset_uniform_strategies()
		infoset_children_types = tensorcfr.get_infoset_children_types()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for i in range(tensorcfr.domain.acting_depth):
				print("########## Level {} ##########".format(i))
				print_tensors(sess, [
					tensorcfr.domain.node_types[i],
					infoset_children_types[i],
					infoset_uniform_strategies[i]
				])
