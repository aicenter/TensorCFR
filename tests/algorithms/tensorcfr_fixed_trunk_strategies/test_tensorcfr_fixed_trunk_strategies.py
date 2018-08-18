import tensorflow as tf

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.commons.constants import SMALL_ERROR_TOLERANCE
from src.domains.flattened_domain01_gambit.domain_from_gambit_loader import get_flattened_domain01_from_gambit
from src.utils.tensor_utils import print_tensor, print_tensors


class TestNodalExpectedValuesAtTrunkDepth(tf.test.TestCase):
	def setUp(self):
		self.error_tolerance = SMALL_ERROR_TOLERANCE
		self.flattened_domain01 = get_flattened_domain01_from_gambit()

	def test_domain01_lvl2_seed42(self):
		# seed = 42
		#  [  15.  -35.   75.   95. -135.   nan -195.   nan  275. -295.]
		# expected_output = np.array([1])
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			tensorcfr = TensorCFRFixedTrunkStrategies(
				self.flattened_domain01,
				trunk_depth=2
			)
			tf_expected_values = tensorcfr.get_nodal_expected_values_at_trunk_depth()
			print_tensors(sess, tensorcfr.domain.current_infoset_strategies)
			print("___________________________________\n")
			print_tensor(sess, tf_expected_values)
			# np_expected_values = sess.run(tf_expected_values)
			# self.assertNDArrayNear(np_expected_values, expected_output, self.error_tolerance)
