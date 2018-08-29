#!/usr/bin/env python3
import os

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.domains.available_domains import get_domain_by_name
from src.utils.other_utils import get_current_timestamp

# TODO: Get rid of `ACTIVATE_FILE` hotfix
ACTIVATE_FILE = True


if __name__ == '__main__' and ACTIVATE_FILE:
	domain = get_domain_by_name("IIGS6_gambit_flattened")
	tensorcfr = TensorCFRFixedTrunkStrategies(
		domain,
		trunk_depth=10
	)
	script_directory = os.path.dirname(os.path.abspath(__file__))
	# print(get_current_timestamp())
	# tensorcfr.generate_dataset_at_trunk_depth(
	# 	# dataset_for_nodes=False,
	# 	dataset_size=2,
	# 	dataset_directory=script_directory + "/out",
	# 	seed=SEED_FOR_TESTING
	# )
	print(get_current_timestamp())
	tensorcfr.generate_dataset_single_session(
		# dataset_for_nodes=False,
		dataset_size=200,
		dataset_directory=script_directory + "/out/IIGS6/50_datapoints",
		#seed=SEED_FOR_TESTING
	)
	print(get_current_timestamp())
	# tensorcfr.generate_dataset_tf_while_loop(
	# 	# dataset_for_nodes=False,
	# 	dataset_size=3,
	# 	dataset_directory=script_directory + "/out",
	# 	seed=SEED_FOR_TESTING
	# )
	# print(get_current_timestamp())
