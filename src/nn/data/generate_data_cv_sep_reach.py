#!/usr/bin/env python3
from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.commons.constants import DEFAULT_DATAGEN_METHOD, DATAGEN_MULTISESSIONS, DATAGEN_SINGLESESSIONS
from src.domains.available_domains import get_domain_by_name
from src.utils.other_utils import get_dataset_parameters, print_dataset_parameters, get_current_timestamp


def generate_data_cv_sep_reach(domain_name, script_directory, trunk_depth):
	dataset_parameters = get_dataset_parameters(domain_name)
	print_dataset_parameters(dataset_parameters)
	domain = get_domain_by_name(dataset_parameters["domain_name"])
	print(get_current_timestamp())

	tensorcfr = TensorCFRFixedTrunkStrategies(
		domain,
		trunk_depth=trunk_depth
	)
	print(get_current_timestamp())

	dataset_generation_method = DEFAULT_DATAGEN_METHOD
	if dataset_generation_method == DATAGEN_MULTISESSIONS:
		tensorcfr.generate_dataset_multiple_sessions_cf_sep_reach(
			dataset_size=dataset_parameters["dataset_size"],
			dataset_directory=script_directory + "/out/{}/{}_datasets".format(
				dataset_parameters["domain_name"],
				dataset_parameters["dataset_size"],
			),
			dataset_seed_to_start=dataset_parameters["starting_seed"]
		)
	elif dataset_generation_method == DATAGEN_SINGLESESSIONS:
		tensorcfr.generate_dataset_single_session_cf_sep_reach(
			dataset_size=dataset_parameters["dataset_size"],
			dataset_directory=script_directory + "/out/{}/{}_datasets".format(
				dataset_parameters["domain_name"],
				dataset_parameters["dataset_size"],
			),
			dataset_seed_to_start=dataset_parameters["starting_seed"]
		)
	# elif dataset_generation_method == DATAGEN_TF_WHILELOOP:   # TODO implement for CFR via `tf.while_loop`
	else:
		raise ValueError("Invalid value {} for 'dataset_generation_method'.".format(dataset_generation_method))
	print(get_current_timestamp())
