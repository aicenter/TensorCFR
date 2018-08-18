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

	def compare_with_expected_output(self, expected_output, np_expected_values):
		np.testing.assert_allclose(
			np_expected_values,
			expected_output,
			rtol=self.error_tolerance,
			equal_nan=True
		)

	@staticmethod
	def print_debug_information(expected_output, sess, tensorcfr_instance, tf_expected_values):
		print_tensors(sess, tensorcfr_instance.domain.current_infoset_strategies)
		print("___________________________________\n")
		print_tensor(sess, tf_expected_values)
		pprint(expected_output.tolist())

	def test_domain01_lvl2_for_uniform_strategies(self):
		expected_output = np.array(
			[15., -35., 75., 95., -135., np.nan, -195., np.nan, 275., -295.]
		)
		tensorcfr_instance = TensorCFRFixedTrunkStrategies(
			self.flattened_domain01,
			trunk_depth=2
		)
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			nodal_expected_values = tensorcfr_instance.get_nodal_expected_values_at_trunk_depth()
			self.print_debug_information(expected_output, sess, tensorcfr_instance, nodal_expected_values)
			self.compare_with_expected_output(expected_output, sess.run(nodal_expected_values))
