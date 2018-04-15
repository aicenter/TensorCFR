from unittest import TestCase

import tensorflow as tf

from src.commons.constants import PLAYER1, PLAYER2, CHANCE_PLAYER, NO_ACTING_PLAYER
from src.utils.tensor_utils import masked_assign, print_tensors


class TestMaskedAssign(TestCase):
	def setUp(self):
		self.shape2x2x2 = [2, 2, 2]
		self.tensor = tf.Variable(tf.reshape(tf.range(1.0, 9.0), self.shape2x2x2), name="tensor")
		self.mask = tf.less(self.tensor, 6.0, name="mask")
		self.new_values = tf.fill(self.shape2x2x2, -0.5)
		self.masked_assignment = masked_assign(ref=self.tensor, mask=self.mask, value=self.new_values)

	def test_range_2x2x2(self):
		expected_result = tf.constant([[[-0.5, -0.5],
		                                [-0.5, -0.5]],

		                               [[-0.5,  6.0],
		                                [ 7.0,  8.0]]], dtype=self.tensor.dtype, name="expected_result")
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			sess.run(self.masked_assignment)
			print_tensors(sess, [self.tensor, self.mask, self.masked_assignment, expected_result])
			tf.assert_equal(self.tensor, expected_result)

	def test_masked_strategies_domain_01_lvl1(self):
		acting_players = tf.Variable([PLAYER1,          # I1,0
		                              PLAYER2,          # I1,1
		                              PLAYER2,          # I1,2
		                              CHANCE_PLAYER],   # I1,3
		                             name="acting_players")
		strategies = tf.Variable([[0.5, 0.4, 0.1],   # of I1,0
		                          [0.1, 0.9, 0.0],   # of I1,1
		                          [0.2, 0.8, 0.0],   # of I1,2
		                          [0.3, 0.3, 0.3]],  # of I1,3
		                         name="strategies")
		mask_of_resolving_player = tf.equal(acting_players, PLAYER1, name="mask_of_resolving_player")
		masked_strategies = masked_assign(ref=strategies, mask=mask_of_resolving_player,
		                                  value=1.0)
		expected_result = tf.constant([[1.0, 1.0, 1.0],   # counterfactual strategy of I1,0
		                               [0.1, 0.9, 0.0],   # of I1,1
		                               [0.2, 0.8, 0.0],   # of I1,2
		                               [0.3, 0.3, 0.3]],  # of I1,3
		                              dtype=self.tensor.dtype, name="expected_result")
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			print_tensors(sess, [strategies, acting_players, mask_of_resolving_player,
			                     masked_strategies,
			                     expected_result])
			tf.assert_equal(masked_strategies, expected_result)

	def test_masked_strategies_domain_01_lvl2(self):
		acting_players = tf.Variable([PLAYER1,            # of I2,0
		                              PLAYER2,            # of I2,1
		                              PLAYER1,            # of I2,2
		                              PLAYER2,            # of I2,3
		                              CHANCE_PLAYER,      # of I2,4
		                              PLAYER1,            # of I2,5
		                              PLAYER2,            # of I2,6
		                              NO_ACTING_PLAYER],  # of I2,t ... pseudo-infoset of terminal/imaginary nodes
		                             name="acting_players")
		strategies = tf.Variable([[0.15, 0.85],   # of I2,0
		                          [0.70, 0.30],   # of I2,1
		                          [0.25, 0.75],   # of I2,2
		                          [0.50, 0.50],   # of I2,3
		                          [0.10, 0.90],   # of I2,4
		                          [0.45, 0.55],   # of I2,5
		                          [0.40, 0.60],   # of I2,6
		                          [0.00, 0.00]],  # of I2,t ... terminal/imaginary nodes <- mock-up zero strategy
		                         name="strategies")
		mask_of_resolving_player = tf.equal(acting_players, PLAYER1, name="mask_of_resolving_player")
		masked_strategies = masked_assign(ref=strategies, mask=mask_of_resolving_player,
		                                  value=1)
		expected_result = tf.constant([[1.0, 1.0],   # counterfactual strategy of the resolving player
		                               [0.7, 0.3],
		                               [1.0, 1.0],   # counterfactual strategy of the resolving player
		                               [0.5, 0.5],
		                               [0.1, 0.9],
		                               [1.0, 1.0],   # counterfactual strategy of the resolving player
		                               [0.4, 0.6],
		                               [0.0, 0.0]],  # mock-up zero strategy of terminal/imaginary nodes
		                              dtype=self.tensor.dtype, name="expected_result")
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			print_tensors(sess, [strategies, acting_players, mask_of_resolving_player,
			                     masked_strategies,
			                     expected_result])
			tf.assert_equal(masked_strategies, expected_result)

	def test_mismatched_shape(self):
		mismatched_shapes = [[1, 8], [8, 1], [2, 4], [4, 2]]
		for shape in mismatched_shapes:
			reshaped_values = tf.reshape(self.new_values, shape)
			self.assertRaises(AssertionError, masked_assign, self.tensor, self.mask, reshaped_values)


if __name__ == '__main__':
	tf.test.main()
