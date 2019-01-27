#!/usr/bin/env python3
from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
#from src.commons.constants import DEFAULT_DATAGEN_METHOD, DATAGEN_MULTISESSIONS, DATAGEN_SINGLESESSIONS
from src.domains.available_domains import get_domain_by_name
from src.utils.other_utils import get_dataset_parameters, print_dataset_parameters, get_current_timestamp


def print_strategies_cv_sep_reach(domain_name, trunk_depth=10):
	dataset_parameters = get_dataset_parameters(domain_name)
	print_dataset_parameters(dataset_parameters)
	domain = get_domain_by_name(dataset_parameters["domain_name"])
	print(get_current_timestamp())

	tensorcfr = TensorCFRFixedTrunkStrategies(
		domain,
		trunk_depth=trunk_depth
	)
	print(get_current_timestamp())

	tensorcfr.print_strategies_single_session_cf_sep_reach()

if __name__ == "__main__":
	domain = "IIGS6_gambit_flattened"
	print_strategies_cv_sep_reach(domain_name=domain)