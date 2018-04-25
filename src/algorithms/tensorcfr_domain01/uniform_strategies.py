import tensorflow as tf

from src.commons.constants import IMAGINARY_NODE, INT_DTYPE
from src.domains.domain01.domain_definitions import levels, node_to_infoset, current_infoset_strategies, \
	infosets_of_non_chance_player, acting_depth, get_node_types
from src.utils.tensor_utils import print_tensors


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png

def get_infoset_children_types():
	with tf.variable_scope("infoset_children_types", reuse=tf.AUTO_REUSE):
		return [
			tf.get_variable(
					name="infoset_children_types_lvl{}".format(level),
					shape=current_infoset_strategies[level].shape,
					dtype=INT_DTYPE,
			) for level in range(acting_depth)
		]


def set_infoset_children_types():  # TODO unittest
	nodal_types = get_node_types()
	infoset_children_types = get_infoset_children_types()
	with tf.variable_scope("set_infoset_children_types", reuse=tf.AUTO_REUSE):
		ops_set_infoset_children_types = [None] * acting_depth
		for level in range(acting_depth):
			if level == 0:
				ops_set_infoset_children_types[0] = tf.assign(
						ref=infoset_children_types[0],
						value=tf.expand_dims(nodal_types[1], axis=0),
						name="set_infoset_children_types_lvl0"
				)
			else:
				ops_set_infoset_children_types[level] = tf.scatter_nd_update(
						ref=infoset_children_types[level],
						indices=tf.expand_dims(node_to_infoset[level], axis=-1),
						updates=nodal_types[level + 1],
						name="set_infoset_children_types_lvl{}".format(level)
				)
		return ops_set_infoset_children_types


def get_infoset_uniform_strategies():  # TODO unittest
	ops_set_infoset_children_types = set_infoset_children_types()
	infoset_children_types = get_infoset_children_types()
	with tf.variable_scope("infoset_uniform_strategies"):
		infoset_uniform_strategies = [None] * (levels - 1)
		for level in range(levels - 1):
			with tf.variable_scope("uniform_strategies_lvl{}".format(level)):
				# TODO The next line could even be `[ops_set_infoset_children_types[level]]`, but doesn't work! Why?
				with tf.control_dependencies(ops_set_infoset_children_types):
					ones_at_non_imaginary_children = tf.to_float(
							tf.not_equal(
									infoset_children_types[level].read_value(),
									tf.constant(IMAGINARY_NODE, name="IMAGINARY_NODE"),
									name="non_imaginary_children_lvl{}".format(level),
							),
							name="ones_at_non_imaginary_children_lvl{}".format(level),
					)
					# Note: An all-0's row cannot be normalized. This is caused when IS has only imaginary children. As of now,
					#  `tf.divide` produces `nan` in the entire row.
					infoset_uniform_strategies[level] = tf.divide(
							ones_at_non_imaginary_children,
							tf.reduce_sum(
									ones_at_non_imaginary_children,
									axis=-1,
									keepdims=True,
									name="count_of_non_imaginary_children_lvl{}".format(level),
							),
							name="normalization_lvl{}".format(level),
					)
		return [tf.where(
				condition=infosets_of_non_chance_player[level],
				x=infoset_uniform_strategies[level],
				y=current_infoset_strategies[level],
				name="infoset_uniform_strategies_lvl{}".format(level),
		) for level in range(acting_depth)]


if __name__ == '__main__':
	infoset_uniform_strategies_ = get_infoset_uniform_strategies()
	infoset_children_types_ = get_infoset_children_types()
	nodal_types_ = get_node_types()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(levels - 1):
			print("########## Level {} ##########".format(i))
			print_tensors(sess, [
				nodal_types_[i],
				infoset_children_types_[i],
				infoset_uniform_strategies_[i],
			])
