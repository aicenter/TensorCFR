import tensorflow as tf

from src.commons.constants import IMAGINARY_NODE
from src.domains.domain01.domain_definitions import levels, node_types, node_to_infoset, \
	current_infoset_strategies, infosets_of_non_chance_player, acting_depth
from src.utils.tensor_utils import print_tensors


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png

def get_infoset_children_types():  # TODO unittest
	with tf.variable_scope("infoset_children_types"):
		infoset_children_types = [None] * (levels - 1)
		for level in range(levels - 1):
			with tf.variable_scope("level{}".format(level)):
				if level == 0:
					infoset_children_types[0] = tf.expand_dims(
							node_types[1],
							axis=0,
							name="infoset_children_types_lvl0"
					)
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
							name="infoset_children_types_lvl{}".format(level)
					)
		return infoset_children_types


def get_infoset_uniform_strategies():  # TODO unittest
	with tf.variable_scope("infoset_uniform_strategies"):
		infoset_children_types = get_infoset_children_types()
		infoset_uniform_strategies = [None] * (levels - 1)
		for level in range(acting_depth):
			with tf.variable_scope("level{}".format(level)):
				infoset_uniform_strategies[level] = tf.to_float(tf.not_equal(infoset_children_types[level], IMAGINARY_NODE))
				# Note: An all-0's row cannot be normalized. This is caused when an infoset has only imaginary children. As of
				#       now, an all-0's row is kept without normalizing.
				count_of_actions = tf.reduce_sum(
						infoset_uniform_strategies[level],
						axis=-1,
						keepdims=True,
						name="count_of_actions_lvl{}".format(level),
				)
				infosets_with_no_actions = tf.squeeze(
						tf.equal(count_of_actions, 0.0),
						name="rows_summing_to_zero_lvl{}".format(level)
				)
				infoset_uniform_strategies[level] = tf.where(
						condition=infosets_with_no_actions,
						x=infoset_uniform_strategies[level],
						y=tf.divide(
								infoset_uniform_strategies[level],
								count_of_actions,
						),
						name="normalize_where_nonzero_sum_lvl{}".format(level),
				)
				infoset_uniform_strategies[level] = tf.where(
						condition=infosets_of_non_chance_player[level],
						x=infoset_uniform_strategies[level],
						y=current_infoset_strategies[level],
						name="infoset_uniform_strategies_lvl{}".format(level),
				)
	return infoset_uniform_strategies


if __name__ == '__main__':
	infoset_uniform_strategies_ = get_infoset_uniform_strategies()
	infoset_children_types_ = get_infoset_children_types()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(levels - 1):
			print("########## Level {} ##########".format(i))
			print_tensors(sess, [node_types[i], infoset_children_types_[i], infoset_uniform_strategies_[i]])
