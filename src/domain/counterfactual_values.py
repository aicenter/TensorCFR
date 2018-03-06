import tensorflow as tf

from domain.bottomup_expected_values_domain01 import get_expected_values
from domain.domain_01 import levels
from domain.topdown_reach_probabilities_domain01 import get_reach_probabilities
from utils.tensor_utils import print_tensors


# custom-made game: doc/domain_01.png (https://gitlab.com/beyond-deepstack/TensorCFR/blob/master/doc/domain_01.png)

def get_counterfactual_values_of_nodes():   # TODO verify and write a test
	_expected_values = get_expected_values()
	_reach_probabilities = get_reach_probabilities()
	counterfactual_values_of_nodes = [tf.multiply(_reach_probabilities[level], _expected_values[level],
	                                              name="node_cf_val_lvl{}".format(level)) for level in range(levels)]
	return counterfactual_values_of_nodes


if __name__ == '__main__':
	reach_probabilities = get_reach_probabilities()
	expected_values = get_expected_values()
	node_cf_values = get_counterfactual_values_of_nodes()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		i = 0
		for node_cf_val in node_cf_values:
			print("########## Level {} ##########".format(i))
			print_tensors(sess, [reach_probabilities[i], expected_values[i], node_cf_val])
			i += 1
