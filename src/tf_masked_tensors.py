#!/usr/bin/env python3

import tensorflow as tf

# 1-D example
from utils.tensor_utils import print_tensor


def masked_assign_via_tf_where():
	tensor = tf.reshape(tf.range(1, 7), [3, 2], name="tensor")
	mask = tf.less(tensor, 4, name="mask")
	zero_tensor = tf.zeros_like(tensor, name="zero_tensor")
	original_tensor = tensor
	tensor = tf.where(mask, zero_tensor, tensor, name="tensor_v2")
	with tf.Session() as sess_via_tf_where:
		sess_via_tf_where.run(tf.global_variables_initializer())
		print_tensor(sess_via_tf_where, original_tensor)
		print_tensor(sess_via_tf_where, mask)
		print_tensor(sess_via_tf_where, tensor)


def masked_assign_via_equal_operator():
	shape2x2x2 = [2, 2, 2]
	tensor = tf.Variable(tf.reshape(tf.range(1, 9), shape2x2x2), name="tensor")
	original_tensor = tensor
	mask = tf.cast(tf.less(tensor, 6), tensor.dtype, name="mask")
	masked_values = tf.fill(shape2x2x2, -1000)
	new_tensor = tf.add(tf.multiply(tf.ones_like(tensor) - mask, tensor), tf.multiply(mask, masked_values), name="new_tensor")
	tensor = new_tensor
	with tf.Session() as sess_via_tf_assign:
		sess_via_tf_assign.run(tf.global_variables_initializer())
		print_tensor(sess_via_tf_assign, original_tensor)
		print_tensor(sess_via_tf_assign, mask)
		print("...using operator '='...\n")
		print("original_tensor: {}\n".format(sess_via_tf_assign.run(original_tensor)))
		print_tensor(sess_via_tf_assign, tensor)


def masked_assign_via_tf_assign():
	shape2x2x2 = [2, 2, 2]
	tensor = tf.Variable(tf.reshape(tf.range(1, 9), shape2x2x2), name="tensor")
	original_tensor = tensor
	mask = tf.cast(tf.less(tensor, 6), tensor.dtype, name="mask")
	masked_values = tf.fill(shape2x2x2, -1000)
	new_tensor = tf.add(tf.multiply(tf.ones_like(tensor) - mask, tensor), tf.multiply(mask, masked_values), name="new_tensor")
	masked_assign = tf.assign(ref=tensor, value=new_tensor)
	tensor = new_tensor
	with tf.Session() as sess_via_tf_assign:
		sess_via_tf_assign.run(tf.global_variables_initializer())
		print_tensor(sess_via_tf_assign, original_tensor)
		print_tensor(sess_via_tf_assign, mask)
		sess_via_tf_assign.run(masked_assign)
		print("...masked_assign...\n")
		print("original_tensor: {}\n".format(sess_via_tf_assign.run(original_tensor)))
		print_tensor(sess_via_tf_assign, tensor)


# TODO resolve or delete this method
def masked_assign_via_scatter_nd_update():
	raise NotImplementedError
	tensor = tf.Variable(tf.reshape(tf.range(1, 9), [2, 2, 2]), name="tensor")
	original_tensor = tensor
	mask = tf.less(tensor, 6, name="mask")
	where_mask = tf.Variable(tf.where(mask), name="where_mask")
	update_tensor = tf.scatter_nd_update(ref=tensor, indices=where_mask, updates=tf.zeros_like(tensor), name="update_tensor")
	with tf.Session() as sess_via_tf_scatter_nd_update:
		sess_via_tf_scatter_nd_update.run(tf.global_variables_initializer())
		print_tensor(sess_via_tf_scatter_nd_update, tensor)
		print_tensor(sess_via_tf_scatter_nd_update, where_mask)
		sess_via_tf_scatter_nd_update.run(update_tensor)
		print_tensor(sess_via_tf_scatter_nd_update, tensor)
		print_tensor(sess_via_tf_scatter_nd_update, original_tensor)


if __name__ == '__main__':
	masked_assign_via_tf_where()
	masked_assign_via_equal_operator()
	masked_assign_via_tf_assign()
	# masked_assign_via_scatter_nd_update()

