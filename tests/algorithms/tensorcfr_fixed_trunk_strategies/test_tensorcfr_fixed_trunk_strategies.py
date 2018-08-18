from pprint import pprint

import numpy as np
import tensorflow as tf

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.commons.constants import SMALL_ERROR_TOLERANCE
from src.domains.flattened_domain01_gambit.domain_from_gambit_loader import get_flattened_domain01_from_gambit
from src.utils.tensor_utils import print_tensor, print_tensors


class TestNodalExpectedValuesAtTrunkDepth(tf.test.TestCase):
	def setUp(self):
		self.error_tolerance = SMALL_ERROR_TOLERANCE
		self.flattened_domain01 = get_flattened_domain01_from_gambit()
		self.tensorcfr_domain01_td2 = TensorCFRFixedTrunkStrategies(self.flattened_domain01, trunk_depth=2)

	@staticmethod
	def print_debug_information(expected_output, sess, tensorcfr_instance, tf_expected_values):
		print_tensors(sess, tensorcfr_instance.domain.current_infoset_strategies)
		print("___________________________________\n")
		print_tensor(sess, tf_expected_values)
		pprint(expected_output.tolist())

	def compare_with_expected_output(self, expected_output, np_expected_values):
		np.testing.assert_allclose(
			np_expected_values,
			expected_output,
			rtol=self.error_tolerance,
			equal_nan=True
		)

	def run_test_nodal_expected_values_given_domain_level_seed(self, flattened_domain, level, seed, expected_output):
		tensorcfr_instance = TensorCFRFixedTrunkStrategies(
			domain=flattened_domain,
			trunk_depth=level
		)
		nodal_expected_values = tensorcfr_instance.get_nodal_expected_values_at_trunk_depth()
		setup_messages, feed_dictionary = tensorcfr_instance.set_up_feed_dictionary(
			method="random",
			seed=seed
		)
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer(), feed_dict=feed_dictionary)
			print(setup_messages)
			self.print_debug_information(expected_output, sess, tensorcfr_instance, nodal_expected_values)
			self.compare_with_expected_output(expected_output, sess.run(nodal_expected_values))

	def test_domain01_lvl2_for_uniform_strategies(self):
		expected_output = np.array(
			[15., -35., 75., 95., -135., np.nan, -195., np.nan, 275., -295.]
		)
		nodal_expected_values = self.tensorcfr_domain01_td2.get_nodal_expected_values_at_trunk_depth()
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			self.print_debug_information(expected_output, sess, self.tensorcfr_domain01_td2, nodal_expected_values)
			self.compare_with_expected_output(expected_output, sess.run(nodal_expected_values))

	def test_domain01_lvl2_for_seed42(self):
		self.run_test_nodal_expected_values_given_domain_level_seed(
			flattened_domain=self.flattened_domain01,
			level=2,
			seed=42,
			expected_output=np.array([15., -35., 75., 95., -135., np.nan, -195., np.nan, 275., -295.])
		)
