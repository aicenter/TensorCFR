#!/usr/bin/env python3
import os

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.commons.constants import SEED_FOR_TESTING
from src.domains.available_domains import get_domain_by_name

if __name__ == '__main__':
	domain = get_domain_by_name("flattened_domain01_via_gambit")
	tensorcfr = TensorCFRFixedTrunkStrategies(
		domain,
		trunk_depth=2
	)
	script_directory = os.path.dirname(os.path.abspath(__file__))
	tensorcfr.generate_dataset_single_session(
		# dataset_for_nodes=False,
		dataset_size=4,
		dataset_directory=script_directory + "/out",
		seed=SEED_FOR_TESTING
	)
