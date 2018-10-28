#!/usr/bin/env python3

# from https://www.tensorflow.org/api_docs/python/tf/contrib/distributions/bijectors/Permute#class_permute

import tensorflow as tf

from src.utils.tf_utils import print_tensors

tfd = tf.contrib.distributions

reverse = tfd.bijectors.Permute(permutation=[2, 1, 0])

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print_tensors(sess, [
		reverse.forward([-1., 0., 1.]),   # ==> [1., 0., -1]
		reverse.inverse([1., 0., -1])     # ==> [-1., 0., 1.]
	])
