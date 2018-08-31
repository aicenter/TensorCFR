#!/usr/bin/env python3

# from https://www.tensorflow.org/api_docs/python/tf/set_random_seed

import tensorflow as tf

from src.commons.constants import SEED_FOR_TESTING
from src.utils.tf_utils import get_default_config_proto

if __name__ == '__main__':
	tf.set_random_seed(SEED_FOR_TESTING)
	a = tf.random_uniform([1])
	b = tf.random_normal([1])
	config = get_default_config_proto()

	# Repeatedly running this block with the same graph will generate the same
	# sequences of 'a' and 'b'.
	print("Session 1")
	with tf.Session(config=config) as sess1:
		print(sess1.run(a))  # generates 'A1'
		print(sess1.run(a))  # generates 'A2'
		print(sess1.run(b))  # generates 'B1'
		print(sess1.run(b))  # generates 'B2'

	print("Session 2")
	with tf.Session(config=config) as sess2:
		print(sess2.run(a))  # generates 'A1'
		print(sess2.run(a))  # generates 'A2'
		print(sess2.run(b))  # generates 'B1'
		print(sess2.run(b))  # generates 'B2'
