#!/usr/bin/env python3
import itertools

import tensorflow as tf


def print_tensor(sess, tensor):
	print('"{}"\n {}\n'.format(tensor.name, sess.run(tensor)))


def print_tensors(sess, tensors_to_print):
	for tensor_to_print in tensors_to_print:
		print_tensor(sess, tensor_to_print)


def masked_assign(ref, mask, value, name="masked_assign"):
	"""Update 'ref' by assigning 'value' to 'ref[mask]'.

	This operation outputs a corresponding TensorFlow operation (from the computation graph).

	Args:
		:param ref: A mutable `Tensor`.
			Should be from a `Variable` node. May be uninitialized.
    :param mask:  A boolean tensor. Must have the same shape as `ref`.
		:param value: A `Tensor`. Must have the same type and shape as `ref`.
			The value to be assigned to the variable.
		:param name: A name for the operation (optional).

	Returns:
		A corresponding TensorFlow operation (from the computation graph).
	"""
	# check: all 3 shapes must match
	shapes_of_args = [tensor.shape for tensor in [ref, mask, value]]
	for combination in itertools.combinations(shapes_of_args, 2):
		assert combination[0] == combination[1], \
			"masked_assign(): mismatched shapes {} and {}!".format(combination[0], combination[1])

	return tf.assign(ref=ref, value=tf.where(mask, value, ref), name=name)


def expanded_multiply(expandable_tensor, expanded_tensor, name):
	"""Multiply 'expandable_tensor' by 'expanded_tensor' element-wise. If N-dimensional 'expanded_tensor' has a shape
	'(d_1, d_2, ..., d_n)', then (N-1)-dimensional 'expandable_tensor' needs to have the shape '(d_1, d_2, ..., d_(n-1))'.
	The 'expandable_tensor' is "uplifted" to a 1-higher-dimensional expansion, where it has the shape
	'(d_1, ..., d_(n-1), 1)'. This is done in order to support broadcasting in tf.multiply.


	This operation outputs a corresponding TensorFlow operation (from the computation graph).

	Args:
		:param expandable_tensor: An (N-1)-D tensor of the shape '(d_1, d_2, ..., d_(n-1))'.
		:param expanded_tensor: An N-D tensor of the shape '(d_1, d_2, ..., d_n)'.
		:param name: A string to name the resulting tensor operation.

	Returns:
		A corresponding TensorFlow operation (from the computation graph).
	"""
	return tf.multiply(tf.expand_dims(expandable_tensor, axis=-1), expanded_tensor, name=name)