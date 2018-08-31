#!/usr/bin/env python3
import os

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.domains.available_domains import get_domain_by_name
from src.utils.other_utils import get_current_timestamp, print_dataset_parameters, get_dataset_parameters

# TODO: Get rid of `ACTIVATE_FILE` hotfix
ACTIVATE_FILE = True


if __name__ == '__main__' and ACTIVATE_FILE:
	domain_name = "IIGS6_gambit_flattened"
	script_directory = os.path.dirname(os.path.abspath(__file__))
	dataset_parameters = get_dataset_parameters()
	print_dataset_parameters(domain_name, dataset_parameters["starting_seed"], dataset_parameters["dataset_size"])

	domain = get_domain_by_name(domain_name)
	print(get_current_timestamp())
	tensorcfr = TensorCFRFixedTrunkStrategies(
		domain,
		trunk_depth=10
	)
	print(get_current_timestamp())
	tensorcfr.generate_dataset_at_trunk_depth(
		dataset_size=1,
		dataset_directory=script_directory + "/out/{}/{}_datasets/{}".format(
			domain_name,
			dataset_parameters["dataset_size"],
			get_current_timestamp()
		),
		dataset_seed_to_start=dataset_parameters["starting_seed"]
	)
	print(get_current_timestamp())

	# tensorcfr.generate_dataset_single_session(
	# 	# dataset_for_nodes=False,
	# 	dataset_size=50,
	# 	dataset_directory=script_directory + "/out/IIGS6/50_datapoints",
	# 	#seed=SEED_FOR_TESTING
	# )
	# print(get_current_timestamp())

	# tensorcfr.generate_dataset_tf_while_loop(
	# 	# dataset_for_nodes=False,
	# 	dataset_size=3,
	# 	dataset_directory=script_directory + "/out",
	# 	seed=SEED_FOR_TESTING
	# )
	# print(get_current_timestamp())
