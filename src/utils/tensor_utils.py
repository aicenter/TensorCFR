#!/usr/bin/python


def print_tensor(sess, tensor):
	print("{}: {}\n".format(tensor.name, sess.run(tensor)))


def print_tensors(sess, tensors_to_print):
	for tensor_to_print in tensors_to_print:
		print_tensor(sess, tensor_to_print)