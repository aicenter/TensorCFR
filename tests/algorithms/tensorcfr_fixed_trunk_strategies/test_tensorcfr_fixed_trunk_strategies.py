from pprint import pprint

import numpy as np
import tensorflow as tf

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.commons.constants import SMALL_ERROR_TOLERANCE, DEFAULT_TOTAL_STEPS
from src.domains.flattened_domain01_gambit.domain_from_gambit_loader import get_flattened_domain01_from_gambit
from src.domains.flattened_goofspiel3.domain_from_gambit_loader import get_flattened_goofspiel3
from src.utils.tensor_utils import print_tensor, print_tensors


class TestNodalExpectedValuesAtTrunkDepth(tf.test.TestCase):
	def setUp(self):
		self.total_steps = DEFAULT_TOTAL_STEPS
		self.error_tolerance = SMALL_ERROR_TOLERANCE
		self.flattened_domain01 = get_flattened_domain01_from_gambit()
		# self.gs2 = get_flattened_goofspiel2()   # TODO use the 2 card version
		self.ii_gs3_3cards = get_flattened_goofspiel3()
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

	def run_cfr_and_assign_average_strategies(self, sess, tensorcfr_instance):
		cfr_step_op = tensorcfr_instance.do_cfr_step()
		for _ in range(self.total_steps):
			sess.run(cfr_step_op)
		sess.run(tensorcfr_instance.assign_avg_strategies_to_current_strategies())

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
			self.run_cfr_and_assign_average_strategies(sess, tensorcfr_instance)
			self.print_debug_information(expected_output, sess, tensorcfr_instance, nodal_expected_values)
			self.compare_with_expected_output(expected_output, sess.run(nodal_expected_values))

	def test_domain01_lvl2_for_seed42(self):
		"""
		Strategies after 1000 CFR iterations should converge as follows:

			```
			"flattened_domain01_gambit/current_infoset_strategies_lvl0:0"
			 [[0.5  0.25 0.1  0.1  0.05]]

			"flattened_domain01_gambit/current_infoset_strategies_lvl1:0"
			 [[0.39268968 0.2793436  0.32796666]
			 [0.61359006 0.38641    0.        ]
			 [0.61939037 0.3806096  0.        ]
			 [0.33333334 0.33333334 0.33333334]]

			"flattened_domain01_gambit/current_infoset_strategies_lvl2:0"
			 [[0.  1. ]
			 [1.  0. ]
			 [0.  1. ]
			 [1.  0. ]
			 [0.1 0.9]
			 [0.  1. ]
			 [1.  0. ]]
			```
		"""
		self.run_test_nodal_expected_values_given_domain_level_seed(
			flattened_domain=self.flattened_domain01,
			level=2,
			seed=42,
			expected_output=np.array([20, -30, 80, 100, -130, np.nan, -190, np.nan, 280, -290])
		)

	def test_domain01_lvl2_for_seed1337(self):
		"""
		Strategies after 1000 CFR iterations should converge as follows:

			```
			"flattened_domain01_gambit/current_infoset_strategies_lvl0:0"
			 [[0.5  0.25 0.1  0.1  0.05]]

			"flattened_domain01_gambit/current_infoset_strategies_lvl1:0"
			 [[0.15991694 0.6740532  0.16602989]
			 [0.6694928  0.3305072  0.        ]
			 [0.85365707 0.14634295 0.        ]
			 [0.33333334 0.33333334 0.33333334]]

			"flattened_domain01_gambit/current_infoset_strategies_lvl2:0"
			 [[0.  1. ]
			 [1.  0. ]
			 [0.  1. ]
			 [1.  0. ]
			 [0.1 0.9]
			 [0.  1. ]
			 [1.  0. ]]
			```
		"""
		self.run_test_nodal_expected_values_given_domain_level_seed(
			flattened_domain=self.flattened_domain01,
			level=2,
			seed=1337,
			expected_output=np.array([20, -30, 80, 100, -130, np.nan, -190, np.nan, 280, -290])
		)

	def test_domain01_lvl2_for_uniform_strategies(self):
		"""
		Strategies after 1000 CFR iterations should converge as follows:

			```
			"flattened_domain01_gambit/current_infoset_strategies_lvl0:0"
			 [[0.5  0.25 0.1  0.1  0.05]]

			"flattened_domain01_gambit/current_infoset_strategies_lvl1:0"
			 [[0.33333334 0.33333334 0.33333334]
			 [0.5        0.5        0.        ]
			 [0.5        0.5        0.        ]
			 [0.33333334 0.33333334 0.33333334]]

			"flattened_domain01_gambit/current_infoset_strategies_lvl2:0"
			 [[0.  1. ]
			 [1.  0. ]
			 [0.  1. ]
			 [1.  0. ]
			 [0.1 0.9]
			 [0.  1. ]
			 [1.  0. ]]
			```
		"""
		expected_output = np.array(
			[20, -30, 80, 100, -130, np.nan, -190, np.nan, 280, -290]
		)
		nodal_expected_values = self.tensorcfr_domain01_td2.get_nodal_expected_values_at_trunk_depth()
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			self.run_cfr_and_assign_average_strategies(sess, self.tensorcfr_domain01_td2)
			self.print_debug_information(expected_output, sess, self.tensorcfr_domain01_td2, nodal_expected_values)
			self.compare_with_expected_output(expected_output, sess.run(nodal_expected_values))

	def test_ii_goofspiel3_3cards_lvl8_for_seed42(self):
		"""
		Strategies at level 5 after 1000 CFR iterations should converge as follows:

			```
			"II-GS3_gambit_flattened/current_infoset_strategies_lvl5:0"
			 [[1.  0. ]    # infoset 2:32
			 [0.5 0.5]     # infoset 2:28
			 [1.  0. ]     # infoset 2:15
			 [1.  0. ]     # infoset 2:11
			 [1.  0. ]     # infoset 2:20
			 [0.5 0.5]     # infoset 2:7
			 [1.  0. ]]    # infoset 2:2
			```

		(see `doc/goofspiel/II-GS3_solved_via_gambit.png` for the infoset labels)
		"""
		self.run_test_nodal_expected_values_given_domain_level_seed(
			flattened_domain=self.ii_gs3_3cards,
			level=5,
			seed=42,
			expected_output=np.array(
				# see `doc/goofspiel/II-GS3_solved_via_gambit.png` for the infoset labels
				[0, 1,    # infoset 2:32
				 -1, -1,  # infoset 2:28
				 0, -1,   # infoset 2:15
				 1, 1,    # infoset 2:11
				 0, 1,    # infoset 2:20
				 -1, -1,  # infoset 2:15
				 0, 1,    # infoset 2:11
				 1, 1,    # infoset 2:7
				 0, 1]    # infoset 2:2
			)
		)
