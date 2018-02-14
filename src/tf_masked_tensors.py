#!/usr/bin/env python3

import tensorflow as tf

# 1-D example
from utils.tensor_utils import print_tensor

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
