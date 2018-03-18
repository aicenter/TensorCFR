import tensorflow as tf

from domains.domain01.domain_01 import levels, get_node_types, get_node_to_IS, get_IS_strategies
from utils.tensor_utils import print_tensors


# custom-made game: doc/domain_01.png (https://gitlab.com/beyond-deepstack/TensorCFR/blob/master/doc/domain_01.png)


def get_IS_children_types():  # TODO unittest
	node_to_IS = get_node_to_IS()
	node_types = get_node_types()
	IS_strategies = get_IS_strategies()
	IS_children_types = [None] * (levels - 1)
	for level in range(levels - 1):
		if level == 0:
			IS_children_types[0] = tf.expand_dims(node_types[1], axis=0, name="IS_children_types_lvl0")
		else:
			IS_children_types[level] = tf.scatter_nd_update(ref=tf.Variable(tf.zeros_like(IS_strategies[level],
			                                                                              dtype=node_types[level + 1].dtype)),
			                                                indices=tf.expand_dims(node_to_IS[level], axis=-1),
			                                                updates=node_types[level + 1],
			                                                name="IS_children_types_lvl{}".format(level))
	return IS_children_types


# def get_IS_uniform_strategies():
# 	# TODO
# 	return [None] * (levels - 1)


if __name__ == '__main__':
	node_types_ = get_node_types()
	# IS_uniform_strategies_ = get_IS_uniform_strategies()
	IS_children_types_ = get_IS_children_types()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# for i in range(levels - 1):
		for i in range(0, levels - 1):
			print("########## Level {} ##########".format(i))
			# print_tensors(sess, [node_types_[i], node_uniform_strategies_[i], IS_uniform_strategies_[i]])
			print_tensors(sess, [node_types_[i], IS_children_types_[i]])
