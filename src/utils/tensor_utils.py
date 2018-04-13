#!/usr/bin/env python3

import tensorflow as tf


def print_tensor(sess, tensor):
	print('"{}"\n {}\n'.format(tensor.name, sess.run(tensor)))


def print_tensors(sess, tensors_to_print):
	for tensor_to_print in tensors_to_print:
		print_tensor(sess, tensor_to_print)


# noinspection PySimplifyBooleanCheck
def masked_assign(ref, mask, value, name="masked_assign"):
	"""Update 'ref' by assigning 'value' to 'ref[mask]'.

	This operation outputs a corresponding TensorFlow operation (from the computation graph).

	Args:
		:param ref: A mutable `Tensor`.
			Should be from a `Variable` node. May be uninitialized.
    :param mask:  Either a boolean tensor of the same shape as `ref`, or a vector of the size of 'ref.shape[0]'
      (i.e. row mask).
		:param value: Either a `Tensor` of the same type and shape as `ref`, or a scalar (i.e. broadcasting a scalar
			value). The value to be assigned to the variable.
		:param name: A name for the operation (optional), by default `masked_assign`.

	Returns:
		A corresponding TensorFlow operation (from the computation graph).
	"""
	value = tf.to_float(value)  # make sure `value` is TensorFlow scalar of type `float`

	# check: all 3 shapes must match, "row-mask" can be a vector of the size of 'ref.shape[0]'
	assert value.shape == [] or ref.shape == value.shape, \
		"masked_assign(): value needs to be a scalar or a tensor of shape equal to ref.shape == {}!".format(ref.shape)
	assert ref.shape == mask.shape or mask.shape == [ref.shape[0]], \
		"masked_assign(): mismatched ref.shape {} and mask.shape {}!".format(ref.shape, mask.shape)

	if value.shape == []:
		tensor_of_values = tf.fill(dims=ref.shape, value=value)
	else:
		tensor_of_values = value
	return tf.assign(ref=ref, value=tf.where(mask, tensor_of_values, ref), name=name)


def expanded_multiply(expandable_tensor, expanded_tensor, name="expanded_multiply"):
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


def normalize(tensor, axis=-1, order=1, name="normalize"):
	"""
	Normalize the input tensor along an arbitrary axis with an arbitrary norm.

	Args:
		:param tensor - Input tf.Tensor.
		:param axis - Along which axis is to be the input tensor normalized, default is -1 (the last axis).
		:param order - Which norm will be used as in `numpy.linalg.norm`. Default is 1 (L1 norm).
		:param name: A string to name the resulting tensor operation.

	Returns:
		Normalized tensor.
	"""
	return tf.divide(tensor, tf.norm(tensor, axis=axis, keepdims=True, ord=order), name=name)


def scatter_nd_sum(indices, updates, shape, name="scatter_nd_sum"):
	# TODO unittest
	# TODO write a docstring
	return tf.scatter_nd(indices=indices, updates=updates, shape=shape, name=name)