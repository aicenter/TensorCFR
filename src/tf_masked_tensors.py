#!/usr/bin/env python3

import tensorflow as tf

# 1-D example
from utils.tensor_utils import print_tensor

tensor = tf.reshape(tf.range(1, 7), [3, 2], name="tensor")
mask = tf.less(tensor, 3, name="mask")
zero_tensor = tf.zeros_like(tensor, name="zero_tensor")
original_tensor = tensor
tensor = tf.where(mask, zero_tensor, tensor, name="tensor_v2")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print_tensor(sess, tensor)
    # print_tensor(sess, zero_tensor)
    print_tensor(sess, mask)
    print_tensor(sess, tensor)
    print_tensor(sess, original_tensor)
