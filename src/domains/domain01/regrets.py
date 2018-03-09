#!/usr/bin/env python3

import tensorflow as tf

from domains.domain01.counterfactual_values import get_cf_values_IS_actions, get_cf_values_IS
from domains.domain01.domain_01 import levels
from utils.tensor_utils import print_tensors


# custom-made game: doc/domain_01.png (https://gitlab.com/beyond-deepstack/TensorCFR/blob/master/doc/domain_01.png)

def get_regrets():  # TODO verify and write a unittest
	cf_values_IS_actions = get_cf_values_IS_actions()
	cf_values_IS = get_cf_values_IS()
	return [tf.subtract(cf_values_IS_actions[level], cf_values_IS[level], name="regrets_lvl{}".format(level))
	        for level in range(levels - 1)]


if __name__ == '__main__':
	cf_values_IS_actions_ = get_cf_values_IS_actions()
	cf_values_IS_ = get_cf_values_IS()
	regrets = get_regrets()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(levels - 1):
			print("########## Level {} ##########".format(i))
			print_tensors(sess, [cf_values_IS_actions_[i], cf_values_IS_[i], regrets[i]])
