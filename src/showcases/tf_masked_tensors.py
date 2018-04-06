#!/usr/bin/env python3

import tensorflow as tf

from src.utils.tensor_utils import print_tensors, masked_assign


def masked_assign_via_tf_where():
	tensor = tf.reshape(tf.range(1, 7), [3, 2], name="tensor")
	mask = tf.less(tensor, 4, name="mask")
	zero_tensor = tf.zeros_like(tensor, name="zero_tensor")
	original_tensor = tensor
	tensor = tf.where(mask, zero_tensor, tensor, name="tensor_v2")
	with tf.Session() as sess_via_tf_where:
		sess_via_tf_where.run(tf.global_variables_initializer())
		print("##########\tvia tf.where\t##########")
		print_tensors(sess_via_tf_where, [original_tensor, mask, tensor])


def masked_assign_via_equal_operator():
	shape2x2x2 = [2, 2, 2]
	tensor = tf.Variable(tf.reshape(tf.range(1, 9), shape2x2x2), name="tensor")
	original_tensor = tf.Variable(tensor, name="original_tensor")
	mask = tf.cast(tf.less(tensor, 6), tensor.dtype, name="mask")
	masked_values = tf.fill(shape2x2x2, -1000)
	new_tensor = tf.add(tf.multiply(tf.ones_like(tensor) - mask, tensor), tf.multiply(mask, masked_values),
	                    name="new_tensor")
	tensor = new_tensor
	with tf.Session() as sess_via_tf_assign:
		sess_via_tf_assign.run(tf.global_variables_initializer())
		print("##########\tvia operator '='\t##########")
		print_tensors(sess_via_tf_assign, [original_tensor, mask, original_tensor, tensor])


def masked_assign_via_tf_assign():
	shape2x2x2 = [2, 2, 2]
	tensor = tf.Variable(tf.reshape(tf.range(1, 9), shape2x2x2), name="tensor")
	original_tensor = tf.Variable(tensor, name="original_tensor")
	mask = tf.cast(tf.less(tensor, 6), tensor.dtype, name="mask")
	new_values = tf.fill(shape2x2x2, -1000)
	new_tensor_value = tf.add(tf.multiply(tf.ones_like(tensor) - mask, tensor), tf.multiply(mask, new_values),
	                          name="new_tensor_value")
	masked_assign = tf.assign(ref=tensor, value=new_tensor_value)
	with tf.Session() as sess_via_tf_assign:
		sess_via_tf_assign.run(tf.global_variables_initializer())
		print("##########\tvia tf.assign\t##########")
		print_tensors(sess_via_tf_assign, [original_tensor, mask])
		sess_via_tf_assign.run(masked_assign)
		print_tensors(sess_via_tf_assign, [new_tensor_value, original_tensor, tensor])


def masked_assign_via_scatter_nd_update():
	# tensor = tf.Variable(tf.reshape(tf.range(1, 9), [2, 2, 2]), name="tensor")
	# original_tensor = tensor
	# mask = tf.less(tensor, 6, name="mask")
	# where_mask = tf.Variable(tf.where(mask), name="where_mask")
	# update_tensor = tf.scatter_nd_update(ref=tensor, indices=where_mask, updates=tf.zeros_like(tensor),
	#                                      name="update_tensor")
	# with tf.Session() as sess_via_tf_scatter_nd_update:
	# 	sess_via_tf_scatter_nd_update.run(tf.global_variables_initializer())
	# 	print_tensors(sess_via_tf_scatter_nd_update, [tensor, where_mask])
	# 	sess_via_tf_scatter_nd_update.run(update_tensor)
	# 	print_tensor(sess_via_tf_scatter_nd_update, [tensor, original_tensor])
	raise NotImplementedError


def masked_assign_via_tensor_utils():
	shape2x2x2 = [2, 2, 2]
	tensor = tf.Variable(tf.reshape(tf.range(1, 9), shape2x2x2), name="tensor")
	mask = tf.less(tensor, 6, name="mask")
	new_values = tf.fill(shape2x2x2, -1000)
	masked_assignment = masked_assign(ref=tensor, mask=mask, value=new_values)
	with tf.Session() as sess_via_tensor_utils:
		sess_via_tensor_utils.run(tf.global_variables_initializer())
		print("##########\tvia tensor_utils.masked_assign\t##########")
		print_tensors(sess_via_tensor_utils, [tensor, mask, masked_assignment, tensor])


if __name__ == '__main__':
	masked_assign_via_tf_where()
	masked_assign_via_equal_operator()
	masked_assign_via_tf_assign()
# masked_assign_via_scatter_nd_update()
	masked_assign_via_tensor_utils()
