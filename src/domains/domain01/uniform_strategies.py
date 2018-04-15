import tensorflow as tf

from src.commons.constants import IMAGINARY_NODE
from src.domains.domain01.domain01 import levels, node_types, node_to_infoset, current_infoset_strategies, \
	infosets_of_non_chance_player, acting_depth
from src.utils.tensor_utils import print_tensors, masked_assign


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png

def get_infoset_children_types():  # TODO unittest
	infoset_children_types = [None] * (levels - 1)
	for level in range(levels - 1):
		if level == 0:
			infoset_children_types[0] = tf.expand_dims(node_types[1], axis=0, name="infoset_children_types_lvl0")
		else:
			infoset_children_types[level] = tf.scatter_nd_update(
				ref=tf.Variable(
					tf.zeros_like(
						current_infoset_strategies[level],
						dtype=node_types[level + 1].dtype
					)
				),
				indices=tf.expand_dims(node_to_infoset[level], axis=-1),
				updates=node_types[level + 1],
				name="infoset_children_types_lvl{}".format(level))
	return infoset_children_types


def get_infoset_uniform_strategies():  # TODO unittest
	infoset_children_types = get_infoset_children_types()
	infoset_uniform_strategies = [None] * (levels - 1)
	for level in range(levels - 1):
		infoset_uniform_strategies[level] = tf.to_float(tf.not_equal(infoset_children_types[level], IMAGINARY_NODE))
		# Note: An all-0's row cannot be normalized. This is caused when IS has only imaginary children. As of now,
		#  `tf.divide` produces `nan` in the entire row.
		infoset_uniform_strategies[level] = tf.divide(
			infoset_uniform_strategies[level],
			tf.reduce_sum(infoset_uniform_strategies[level], axis=-1, keepdims=True),
			name="infoset_uniform_strategies_lvl{}".format(level))
	return infoset_uniform_strategies


def assign_uniform_strategies_to_players():
	uniform_strategies = get_infoset_uniform_strategies()
	return [
		masked_assign(
				ref=current_infoset_strategies[level],
				mask=infosets_of_non_chance_player[level],
				value=uniform_strategies[level])
		for level in range(acting_depth)
	]


if __name__ == '__main__':
	infoset_uniform_strategies_ = get_infoset_uniform_strategies()
	infoset_children_types_ = get_infoset_children_types()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(levels - 1):
			print("########## Level {} ##########".format(i))
			print_tensors(sess, [node_types[i], infoset_children_types_[i], infoset_uniform_strategies_[i]])