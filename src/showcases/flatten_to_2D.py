from pprint import pprint

import numpy as np
import tensorflow as tf

from src.utils.tensor_utils import scatter_nd_sum, print_tensors

INT_DTYPE_NUMPY = np.int32
INT_DTYPE = tf.as_dtype(INT_DTYPE_NUMPY)
FLOAT_DTYPE = tf.float32


action_counts_ = [
	[2],
	[1, 6],
	[4, 0, 0, 0, 0, 0, 0],
	[3, 3, 2, 2],
	[2] * 10,
	[0] * 20
]

l3children = [31.1, 32, 33, 34, 35, 36, 37, 38, 39, 40]
parent_IS_map = [0, 0, 1, 1]
strategy = [[0.1, 0.3, 0.6], [.2, .8, .0]]


def get_parent_x_actions_from_action_counts(action_counts, children, name="reshape_CFVs"):
	"""
  Reshape data related to children (e.g., CFVs) to a 2D tensor of shape (parent x action).

  Args:
    :param action_counts: A 1-D array containing number of actions of each node.
    :param children: Data for the children to reshape.

  Returns: A corresponding TensorFlow operation (from the computation graph) that computes the (parent x action)
  tensor with the provided data distributed to the correct positions.
	"""
	mask_children = tf.sequence_mask(
			action_counts,
			dtype=INT_DTYPE,
			name="initial_mask_in_{}".format(name)
	)
	first_column = tf.expand_dims(
			tf.cumsum(action_counts,
			          exclusive=True,
			          name="first_column_indices_in_{}".format(name)),
			dim=1
	)
	replaced = tf.concat(
			[first_column, mask_children[:, 1:]], 1,
			name="replacing_col0_in_{}".format(name)
	)
	summed = tf.expand_dims(
			tf.cumsum(replaced, axis=1, name="computing_indices_in_{}".format(name)),
			dim=2
	)
	final = tf.multiply(
			tf.cast(mask_children, dtype=FLOAT_DTYPE),
			tf.gather_nd(children, summed),
			name="cleaned_up_result_in_{}".format(name)
	)
	return final


def get_action_and_IS_cfvs(children_values, action_counts, parent_IS_map, strategy):
	"""
  Compute counterfactual values of actions and information sets for one level.

  Args:
    :param children_values: A 1-D tensor of counterfactual values for the children.
    :param action_counts: A 1-D array containing number of actions of each node.
    :param parent_IS_map: A 1-D array indicating the index if the IS for each parent of the children.
    :param strategy: A 2-D representation of probability of playing an action in an IS.

  Returns: A pair of counterfactual values for all actions in ISs (2-D) and the cfvs of ISs expanded to (2-D) to be
  able to subtract for the first return value.
	"""
	parent_x_action = get_parent_x_actions_from_action_counts(action_counts, children_values)
	cfvs_IS_action = scatter_nd_sum(
			tf.expand_dims(parent_IS_map, axis=1),
			parent_x_action,
			tf.shape(strategy),
			name="sums_cfvs_over_IS"
	)
	cfvs_IS = tf.reduce_sum(
			tf.multiply(cfvs_IS_action, strategy),
			axis=1,
			name="IS_cfvs_from_action_cfvs"
	)
	return cfvs_IS_action, tf.expand_dims(cfvs_IS, dim=1)


print("action_counts:")
pprint(action_counts_, indent=1, width=80)
out = get_action_and_IS_cfvs(l3children, action_counts_[3], parent_IS_map, tf.Variable(strategy))
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print_tensors(sess, out)
