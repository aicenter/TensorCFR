#!/usr/bin/python

def print_tensor(sess, tensor):
	print("{}: {}\n".format(tensor.name, sess.run(tensor)))
