#!/usr/bin/env python3
import itertools
import tensorflow as tf


def print_tensor(sess, tensor):
	print('"{}"\n {}\n'.format(tensor.name, sess.run(tensor)))


def print_tensors(sess, tensors_to_print):
	for tensor_to_print in tensors_to_print:
		print_tensor(sess, tensor_to_print)


def masked_assign(ref, mask, value):
	"""Update 'ref' by assigning 'value' to 'ref[mask]'.

	This operation outputs a corresponding TensorFlow operation (from the computation graph)

	Args:
		ref: A mutable `Tensor`.
			Should be from a `Variable` node. May be uninitialized.
    mask:  A boolean tensor. Must have the same shape as `ref`.
		value: A `Tensor`. Must have the same type and shape as `ref`.
			The value to be assigned to the variable.

	Returns:
		A corresponding TensorFlow operation (from the computation graph).
	"""
	# check: all 3 shapes must match
	shapes_of_args = [tensor.shape for tensor in [ref, mask, value]]
	for combination in itertools.combinations(shapes_of_args, 2):
		assert combination[0] == combination[1], \
			"masked_assign(): mismatched shapes {} and {}!".format(combination[0], combination[1])

	return tf.assign(ref=ref, value=tf.where(mask, value, ref), name="masked_assign")
