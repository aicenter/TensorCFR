from pprint import pprint
from unittest import TestCase

import numpy as np
import tensorflow as tf

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.commons.constants import SMALL_ERROR_TOLERANCE, DEFAULT_TOTAL_STEPS
from src.domains.flattened_domain01_gambit.domain_from_gambit_loader import get_flattened_domain01_from_gambit
from src.domains.flattened_goofspiel3.domain_from_gambit_loader import get_flattened_goofspiel3
from src.utils.tensor_utils import print_tensor, print_tensors


class TestNodalExpectedValuesAtTrunkDepth(TestCase):
	def setUp(self):
		self.total_steps = DEFAULT_TOTAL_STEPS
		self.error_tolerance = SMALL_ERROR_TOLERANCE
		self.flattened_domain01 = get_flattened_domain01_from_gambit()
		# self.gs2 = get_flattened_goofspiel2()   # TODO use the 2 card version
		self.ii_gs3_3cards = get_flattened_goofspiel3()
		self.tensorcfr_domain01_td2 = TensorCFRFixedTrunkStrategies(self.flattened_domain01, trunk_depth=2)

	@staticmethod
	def print_debug_information(expected_output, sess, tensorcfr_instance, tf_expected_values):
		print_tensors(sess, tensorcfr_instance.domain.initial_infoset_strategies)
		print("___________________________________\n")
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

	def run_test_nodal_expected_values_given_domain_level_seed(self, flattened_domain, level, initial_strategies,
	                                                           expected_output):
		tensorcfr_instance = TensorCFRFixedTrunkStrategies(
			domain=flattened_domain,
			trunk_depth=level
		)
		nodal_expected_values = tensorcfr_instance.get_nodal_expected_values_at_trunk_depth()
		setup_messages, feed_dictionary = tensorcfr_instance.set_up_feed_dictionary(
			method="custom",
			initial_strategy_values=initial_strategies
		)
		with tf.Session(
			# config=tf.ConfigProto(device_count={'GPU': 0})  # uncomment to run on CPU
		) as sess:
			sess.run(tf.global_variables_initializer(), feed_dict=feed_dictionary)
			print(setup_messages)
			self.print_debug_information(expected_output, sess, tensorcfr_instance, nodal_expected_values)
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
			[[0.6176899  0.21680082 0.16550928]
			 [0.09548762 0.9045124  0.        ]
			 [0.68187284 0.31812713 0.        ]
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
		initial_infoset_strategies = [
			[
				[0.5, .25, 0.1, 0.1, .05]
			],
			[
				[0.6176899, 0.21680082, 0.16550928],
				[0.09548762, 0.9045124, 0],
				[0.68187284, 0.31812713, 0],
				[0.33333334, 0.33333334, 0.33333334]
			],
			[
				[.5, .5],
				[.5, .5],
				[.5, .5],
				[.5, .5],
				[.1, .9],
				[.5, .5],
				[.5, .5]
			]
		]
		self.run_test_nodal_expected_values_given_domain_level_seed(
			flattened_domain=self.flattened_domain01,
			level=2,
			initial_strategies=initial_infoset_strategies,
			expected_output=np.array([20, -30, 80, 100, -130, np.nan, -190, np.nan, 280, -290])
		)

	def test_domain01_lvl2_for_seed1337(self):
		"""
		Strategies after 1000 CFR iterations should converge as follows:

			```
			"flattened_domain01_gambit/current_infoset_strategies_lvl0:0"
			 [[0.5  0.25 0.1  0.1  0.05]]

			"flattened_domain01_gambit/current_infoset_strategies_lvl1:0"
			 [[0.25356135 0.5085074  0.23793127]
			 [0.5893186  0.4106815  0.        ]
			 [0.47731555 0.5226844  0.        ]
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
		initial_infoset_strategies = [
			[
				[0.5, .25, 0.1, 0.1, .05]
			],
			[
				[0.25356138, 0.5085074, 0.23793125],
				[0.5893185, 0.4106815, 0],
				[0.4773155, 0.5226845, 0],
				[0.33333334, 0.33333334, 0.33333334]
			],
			[
				[.5, .5],
				[.5, .5],
				[.5, .5],
				[.5, .5],
				[.1, .9],
				[.5, .5],
				[.5, .5]
			]
		]
		self.run_test_nodal_expected_values_given_domain_level_seed(
			flattened_domain=self.flattened_domain01,
			level=2,
			initial_strategies=initial_infoset_strategies,
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

	def test_ii_goofspiel3_3cards_lvl8_for_seed1337(self):
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
			seed=1337,
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
