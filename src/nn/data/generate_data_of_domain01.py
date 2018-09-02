#!/usr/bin/env python3
import os

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.commons.constants import DEFAULT_DATAGEN_METHOD, DATAGEN_MULTISESSIONS, DATAGEN_SINGLESESSIONS
from src.domains.available_domains import get_domain_by_name
from src.utils.other_utils import get_current_timestamp, print_dataset_parameters, get_dataset_parameters

if __name__ == '__main__':
	script_directory = os.path.dirname(os.path.abspath(__file__))
	dataset_parameters = get_dataset_parameters("flattened_domain01_via_gambit")
	print_dataset_parameters(dataset_parameters)

	domain = get_domain_by_name(dataset_parameters["domain_name"])
	print(get_current_timestamp())
	tensorcfr = TensorCFRFixedTrunkStrategies(
		domain,
		trunk_depth=2
	)

	print(get_current_timestamp())
	if DEFAULT_DATAGEN_METHOD == DATAGEN_MULTISESSIONS:  # TODO else branch
		tensorcfr.generate_dataset_multiple_sessions(
			dataset_size=dataset_parameters["dataset_size"],
			dataset_directory=script_directory + "/out/{}/{}_datasets".format(
				dataset_parameters["domain_name"],
				dataset_parameters["dataset_size"],
			),
			dataset_seed_to_start=dataset_parameters["starting_seed"]
		)
	elif DEFAULT_DATAGEN_METHOD == DATAGEN_SINGLESESSIONS:
		tensorcfr.generate_dataset_single_session(
			dataset_size=dataset_parameters["dataset_size"],
			dataset_directory=script_directory + "/out/{}/{}_datasets".format(
				dataset_parameters["domain_name"],
				dataset_parameters["dataset_size"],
			),
			dataset_seed_to_start=dataset_parameters["starting_seed"]
		)
	print(get_current_timestamp())

	# tensorcfr.generate_dataset_tf_while_loop(
	# 	# dataset_for_nodes=False,
	# 	dataset_size=3,
	# 	dataset_directory=script_directory + "/out",
	# 	seed=SEED_FOR_TESTING
	# )
	# print(get_current_timestamp())
