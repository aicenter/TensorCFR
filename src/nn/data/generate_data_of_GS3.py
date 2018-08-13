#!/usr/bin/env python3
import os

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.domains.available_domains import get_domain_by_name

if __name__ == '__main__':
	domain = get_domain_by_name("II-GS3_gambit_flattened")
	tensorcfr = TensorCFRFixedTrunkStrategies(
		domain,
		trunk_depth=4
	)
	script_directory = os.path.dirname(os.path.abspath(__file__))
	tensorcfr.generate_dataset_at_trunk_depth(
		dataset_size=5,
		dataset_directory=script_directory + "/out",
		seed=SEED_FOR_TESTING
	)
