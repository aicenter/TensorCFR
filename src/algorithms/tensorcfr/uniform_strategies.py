import tensorflow as tf

from src.algorithms.tensorcfr.TensorCFR import TensorCFR
from src.domains.available_domains import get_domain_by_name
from src.utils.tensor_utils import print_tensors

if __name__ == '__main__':
	domains = [
		get_domain_by_name("domain01"),
		get_domain_by_name("matching_pennies"),
		get_domain_by_name("hunger_games")
	]
	for tensorcfr in map(TensorCFR, domains):
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
